'''Unrestricted molecular HF-DFT gradients built on GPU4PySCF UKS.'''

import cupy as cp
import numpy as np
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import uks as uks_grad
from gpu4pyscf.lib import logger

from dcforge.cpu.disp.d2c import evaluate_d2c, normalize_dispersion
from dcforge.cpu.energy.rhf_dft import _check_source_state, _check_xc
from dcforge.gpu.grad.skeleton import skeleton_gradient_uhf
from dcforge.gpu.grad.uhf_nv import (
    get_z_response_oo as get_z_response_oo_ao,
)
from dcforge.gpu.grad.uhf_nv import (
    grad_fock_direct,
    grad_fock_make_h1,
    grad_ovlp_response,
    make_s1_oo_and_grad_ovlp_energy,
    nonvariational_correction,
)
from dcforge.gpu.grad.uhf_zvec import gen_vind as gen_vind_ao
from dcforge.gpu.grad.uhf_zvec import solve_z_vector
from dcforge.tools.zvec import warn_z_residual


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None):
    r'''Evaluate the unrestricted target-DFT-on-UHF GPU gradient.

    The implemented equation is

    .. math::

        \frac{dE}{d\lambda}
        =g_{\mathrm{skel}}^\lambda
        -2\sum_s Z^sF^{\mathrm{UHF},s,\lambda}_{vo}
        +2\sum_s Z^sS^{s,\lambda}_{vo}\epsilon_o^s
        +g_{\mathrm{ovlp-response}}^\lambda.

    GPU4PySCF owns the target UKS grid and provider-specific explicit
    derivative. DC-Forge adds the noncanonical target ``F_oo`` correction and
    the source-UHF adjoint terms.

    Args:
        mf_grad : dcforge.gpu.grad.uhf_dft.Gradients
            Active GPU unrestricted gradient object.

    Kwargs:
        mo_energy : cupy.ndarray of shape (2, nmo) or None
            Source-UHF orbital energies.
        mo_coeff : cupy.ndarray of shape (2, nao, nmo) or None
            Source-UHF MO coefficients.
        mo_occ : cupy.ndarray of shape (2, nmo) or None
            Exact source-UHF occupations.
        atmlst : sequence of int or None
            Nuclear centers included in the gradient. The current GPU
            capability requires all atoms.

    Returns:
        ndarray of shape (N_atom, 3)
            Electronic ``dE/dR`` in Ha/Bohr. Nuclear repulsion is added by
            the inherited GPU gradient kernel.
    '''
    mf = mf_grad.base
    mf_hf = mf.mf_hf
    mol = mf_grad.mol
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    if mf_hf.converged:
        _check_source_state(mf)
    if mf.e_dft is None:
        mf.kernel(run_hf=not mf_hf.converged)
    _check_source_state(mf)
    _check_xc(mf)
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    if atmlst != list(range(mol.natm)):
        raise NotImplementedError('GPU UHF-DFT gradients currently require all atoms')

    mo_energy = cp.asarray(mo_energy)
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    if cp.iscomplexobj(mo_coeff):
        raise NotImplementedError('complex molecular GPU UHF gradients are not supported')

    Ca, Cb = mo_coeff
    occa, occb = mo_occ
    for occ in (occa, occb):
        if not bool(cp.all((occ == 1) | (occ == 0))):
            raise ValueError('GPU UHF-DFT gradients require exact 1/0 occupations')
        nocc = int(cp.count_nonzero(occ > 0).get())
        if not bool(cp.all(occ[:nocc] == 1)) or not bool(cp.all(occ[nocc:] == 0)):
            raise ValueError('occupied GPU UHF orbitals must precede virtual orbitals')

    soa, sob = occa > 0, occb > 0
    sva, svb = occa == 0, occb == 0
    Co = (Ca[:, soa], Cb[:, sob])
    Cv = (Ca[:, sva], Cb[:, svb])
    dm0 = mf_hf.make_rdm1(mo_coeff, mo_occ)

    fock_ks = mf.get_fock(dm=dm0)
    fock_mo_a = Ca.conj().T @ fock_ks[0] @ Ca
    fock_mo_b = Cb.conj().T @ fock_ks[1] @ Cb
    fvo = (fock_mo_a[sva][:, soa], fock_mo_b[svb][:, sob])
    foo = (fock_mo_a[soa][:, soa], fock_mo_b[sob][:, sob])

    fvind = mf_hf.gen_response(
        mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=1)
    vind = mf_grad.gen_vind(Co, Cv, fvind)
    zvec, z_info = solve_z_vector(
        mo_energy, mo_occ, fvo, vind,
        conv_tol=mf_grad.cphf_conv_tol,
        max_cycle=mf_grad.cphf_max_cycle,
        log=log,
    )
    warn_z_residual(log, z_info, mf_grad.cphf_residual_warn_tol)
    mf_grad.zvec = zvec
    mf_grad.z_rhs_max_abs = z_info['rhs_max_abs']
    mf_grad.z_rhs_max_abs_spin = z_info['rhs_max_abs_spin']
    mf_grad.z_residual_max_abs = z_info['residual_max_abs']
    mf_grad.z_residual_max_abs_spin = z_info['residual_max_abs_spin']
    mf_grad.z_residual_relative = z_info['residual_relative']
    mf_grad.z_residual_warning = z_info['physical_residual_warning']
    mf_grad.z_solver = z_info['solver']

    # The target DF skeleton releases the shared device CDERI.  Evaluate the
    # small spin-resolved DF OO response while that CDERI is still live, then
    # retain only the occupied-space blocks through the derivative stages.
    # Conventional AO response keeps its established late schedule.
    vzoo = None
    if mf_grad.response_backend == 'mo_direct':
        vzoo = mf_grad.get_z_response_oo(Co, Cv, zvec, fvind)
        mf_grad.z_response_oo_schedule_used = 'early_after_z_solve'
    else:
        mf_grad.z_response_oo_schedule_used = 'late_after_derivatives'

    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    e_occ = (mo_energy[0][soa], mo_energy[1][sob])
    s1oo, de_ovlp_energy = make_s1_oo_and_grad_ovlp_energy(
        mol, Co, Cv, e_occ, zvec, s1, atmlst)
    # GPU4PySCF's full UKS grid-response helper does not recognize the
    # exact-exchange-only ``HF`` xctype.  Its XC grid derivative is exactly
    # zero, so bypass only that nonexistent contribution.  J/K and DF
    # auxiliary response continue through the active gradient provider.
    grid_response = mf_grad.grid_response
    if mf._numint._xc_type(mf.xc) == 'HF':
        mf_grad.grid_response = False
    try:
        de_skel = skeleton_gradient_uhf(
            mf_grad, foo, e_occ, s1oo,
            mo_energy, mo_coeff, mo_occ, atmlst)
    finally:
        mf_grad.grid_response = grid_response

    de_fock = mf_grad.grad_fock(
        Co, Cv, zvec, dm0, mo_coeff, mo_occ, atmlst)
    if vzoo is None:
        vzoo = mf_grad.get_z_response_oo(Co, Cv, zvec, fvind)
    de_ovlp_response = grad_ovlp_response(s1oo, vzoo)
    de_nvar = nonvariational_correction(
        mf_grad, s1oo, de_fock, de_ovlp_energy, de_ovlp_response)

    # The same geometry-only D2C derivative applies to RHF and UHF sources.
    # Its host evaluation is separate from both GPU spin-response channels.
    mf.dispersion = normalize_dispersion(mf.dispersion)
    mf_grad.d2c_result = None
    mf_grad.de_d2c = np.zeros_like(de_skel)
    if mf.dispersion == 'd2c':
        mf_grad.d2c_result = evaluate_d2c(mol, mf.xc, gradient=True)
        mf_grad.de_d2c = mf_grad.d2c_result.gradient[atmlst]
        mf.d2c_result = mf_grad.d2c_result
        mf.e_d2c = mf_grad.d2c_result.energy
        mf.e_tot = mf.e_dft + mf.e_d2c
    return de_skel + de_nvar + mf_grad.de_d2c


class Gradients(uks_grad.Gradients):
    '''Unrestricted target-DFT-on-UHF analytic GPU gradient driver.'''

    _keys = uks_grad.Gradients._keys | {
        'cphf_conv_tol', 'cphf_max_cycle', 'cphf_residual_warn_tol',
        'fock_deriv_method', 'z_residual_warning',
        'response_backend', 'z_response_oo_schedule_used',
        'de_d2c', 'd2c_result',
    }

    def __init__(self, mf):
        from dcforge.gpu.energy.uhf_dft import UHFDFT

        if not isinstance(mf, UHFDFT):
            raise TypeError(
                'GPU UHF Gradients requires a dcforge.gpu.energy.UHFDFT object'
            )
        uks_grad.Gradients.__init__(self, mf)
        self.grid_response = True
        self.cphf_conv_tol = 1e-9
        self.cphf_max_cycle = 200
        self.cphf_residual_warn_tol = 1e-8
        self.z_residual_warning = None
        self.fock_deriv_method = 'direct'
        self.response_backend = 'ao'
        self.z_response_oo_schedule_used = None
        self.de_d2c = None
        self.d2c_result = None

    def dump_flags(self, verbose=None):
        '''Log GPU grid and source-UHF response controls.'''
        uks_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'grid_response = %s', self.grid_response)
        logger.info(self, 'cphf_conv_tol = %g', self.cphf_conv_tol)
        logger.info(self, 'cphf_max_cycle = %d', self.cphf_max_cycle)
        logger.info(
            self, 'cphf_residual_warn_tol = %s',
            self.cphf_residual_warn_tol)
        logger.info(self, 'fock_deriv_method = %s', self.fock_deriv_method)
        logger.info(self, 'response_backend = %s', self.response_backend)
        logger.info(self, 'dispersion = %s', self.base.dispersion)
        return self

    def grad_fock(self, Co, Cv, zvec, dm0, mo_coeff, mo_occ, atmlst):
        '''Return the source-UHF Fock term for the active GPU provider.'''
        method = self.fock_deriv_method
        self.fock_deriv_method_used = method
        if method == 'make_h1':
            return grad_fock_make_h1(
                self, Co, Cv, zvec, mo_coeff, mo_occ, atmlst)
        if method == 'direct':
            return grad_fock_direct(self, Co, Cv, zvec, dm0, atmlst)
        raise ValueError(
            "fock_deriv_method must be 'make_h1' or 'direct'; "
            f'got {method!r}'
        )

    def gen_vind(self, Co, Cv, fvind):
        '''Return the conventional AO spin-coupled response action.'''
        if self.response_backend != 'ao':
            raise ValueError(
                "conventional GPU response_backend must be 'ao'; "
                f'got {self.response_backend!r}')
        self.response_backend_used = 'ao'
        return gen_vind_ao(Co, Cv, fvind)

    def get_z_response_oo(self, Co, Cv, zvec, fvind):
        '''Return conventional AO adjoint response OO blocks.'''
        return get_z_response_oo_ao(Co, Cv, zvec, fvind)

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        '''Evaluate all-atom ``dE/dR`` with complete target-grid response.'''
        if self.base.mf_hf.converged:
            _check_source_state(self.base)
        if not self.grid_response:
            raise NotImplementedError('GPU UHF gradient requires grid_response=True')
        if atmlst is not None and list(atmlst) != list(range(self.mol.natm)):
            raise NotImplementedError('GPU UHF-DFT gradients currently require all atoms')
        de = rhf_grad.GradientsBase.kernel(
            self, mo_energy, mo_coeff, mo_occ, atmlst)
        self.de_nuc = np.asarray(self.grad_nuc())
        self.de_skel = self.de_skel_elec + self.de_nuc
        return de

    def _finalize(self):
        from dcforge.tools.logger import gradient_summary
        gradient_summary(self, unrestricted=True)

    grad_elec = grad_elec

    def as_scanner(self):
        raise NotImplementedError('GPU UHF-DFT gradient scanner is not implemented')

    def to_cpu(self):
        raise NotImplementedError('GPU UHF gradient state conversion is not implemented')


Grad = Gradients
