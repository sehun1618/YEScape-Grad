'''Unrestricted molecular HF-DFT gradients built on PySCF UKS gradients.'''

import numpy as np
from pyscf.grad import uks as uks_grad
from pyscf.lib import logger

from dcforge.tools.logger import log_kernel_time
from dcforge.cpu.disp.d2c import evaluate_d2c, normalize_dispersion
from dcforge.cpu.energy.rhf_dft import _check_source_state, _check_xc
from dcforge.cpu.grad import grid_response
from dcforge.cpu.grad.skeleton import skeleton_gradient_uhf
from dcforge.cpu.grad.uhf_nv import (
    grad_fock_direct,
    grad_fock_make_h1,
    nonvariational_correction,
)
from dcforge.cpu.grad.uhf_zvec import solve_z_vector
from dcforge.tools.zvec import warn_z_residual


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None):
    r'''Evaluate the electronic unrestricted target-DFT-on-UHF gradient.

    The unrestricted extension of the molecular PDF equation is

    .. math::

        \begin{aligned}
        \frac{dE}{d\lambda}={}&
        \sum_{s\mu\nu}P^s_{\mu\nu}h^\lambda_{\mu\nu}
        +E_{\mathrm{Hxc}}^\lambda[P^\alpha,P^\beta]
        +E_{\mathrm{NN}}^\lambda\\
        &-\sum_{sij}F^{\mathrm{KS},s}_{ij}S^{\lambda,s}_{ij}
        +2\,\mathrm{Re}\sum_{sai}(Z^s_{ai})^*V^{\lambda,s}_{ai}.
        \end{aligned}

    Each spin orbital has occupation one, so the nonvariational prefactor is
    2 rather than the restricted spatial-orbital prefactor 4.

    Args:
        mf_grad : dcforge.cpu.grad.uhf_dft.Gradients
            Active UHF-DFT gradient object.

    Kwargs:
        mo_energy : tuple of two ndarray or None
            Source-UHF alpha/beta orbital energies.
        mo_coeff : tuple of two ndarray or None
            Source-UHF alpha/beta MO coefficients.
        mo_occ : tuple of two ndarray or None
            Source-UHF alpha/beta occupations.
        atmlst : sequence of int or None
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Electronic ``dE/dR`` in Ha/Bohr. PySCF adds nuclear repulsion.
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
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if mo_occ is None:    mo_occ = mf.mo_occ
    if atmlst is None:    atmlst = range(mol.natm)
    atmlst = list(atmlst)

    Ca, Cb = mo_coeff
    occa, occb = mo_occ
    for occ in (occa, occb):
        if not np.all((occ == 1) | (occ == 0)):
            raise ValueError('UHF-DFT gradients require exact 1/0 occupations')
        nocc = np.count_nonzero(occ > 0)
        if not np.all(occ[:nocc] == 1) or not np.all(occ[nocc:] == 0):
            raise ValueError('occupied UHF orbitals must precede virtual orbitals')

    soa, sob = occa > 0, occb > 0
    sva, svb = occa == 0, occb == 0
    Co = (Ca[:, soa], Cb[:, sob])
    Cv = (Ca[:, sva], Cb[:, svb])
    dm0 = mf_hf.make_rdm1(mo_coeff, mo_occ)

    # Compute the target Fock only once, then retain both VO and OO blocks.
    # AO-to-MO projection is split into two matrix products per spin; this is
    # faster and more predictable than a four-index einsum on CPU and GPU.
    fock_ks = mf.get_fock(dm=dm0)
    fock_mo_a = Ca.conj().T @ fock_ks[0] @ Ca
    fock_mo_b = Cb.conj().T @ fock_ks[1] @ Cb
    fvo = (fock_mo_a[sva][:, soa], fock_mo_b[svb][:, sob])
    foo = (fock_mo_a[soa][:, soa], fock_mo_b[sob][:, sob])

    fvind = mf_hf.gen_response(
        mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=1)
    zvec, z_info = solve_z_vector(
        mo_energy, mo_occ, Co, Cv, fvo, fvind,
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

    s1 = mf_grad.get_ovlp(mol)
    de_fock = mf_grad.grad_fock(
        Co, Cv, zvec, dm0, mo_coeff, mo_occ, atmlst)

    # PySCF 2.14 UKS full-grid response assumes an XC-grid array and fails for
    # an exact-exchange-only target (XC type HF), for which no grid term exists.
    # Disable only that nonexistent term while retaining conventional/DF J/K
    # and, in the DF route, auxiliary-basis response.
    grid_response = mf_grad.grid_response
    if mf._numint._xc_type(mf.xc) == 'HF':
        mf_grad.grid_response = False
    try:
        de_skel = skeleton_gradient_uhf(
            mf_grad, dm0, foo, Co, s1, atmlst)
    finally:
        mf_grad.grid_response = grid_response
    de_nvar = nonvariational_correction(
        mf_grad, Co, Cv, mo_energy, zvec, s1, de_fock, fvind, atmlst)

    # The same geometry-only D2C derivative applies to RHF and UHF sources.
    # It remains separate from both spin CPHF channels.
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
    '''Unrestricted target-DFT-on-UHF analytic nuclear gradient driver.

    grid_response_backend : str
        'becke_omp' (default) or 'pyscf_default'; same semantics as RHF.
    '''

    _keys = uks_grad.Gradients._keys | {
        'cphf_conv_tol', 'cphf_max_cycle', 'cphf_residual_warn_tol',
        'fock_deriv_method', 'z_residual_warning',
        'de_d2c', 'd2c_result', 'grid_response_backend',
        'grid_response_backend_used',
    }

    def __init__(self, mf):
        from dcforge.cpu.energy.uhf_dft import UHFDFT
        if not isinstance(mf, UHFDFT):
            raise TypeError('Gradients requires a dcforge.cpu.energy.UHFDFT object')
        uks_grad.Gradients.__init__(self, mf)
        self.grid_response = True
        self.grid_response_backend = 'becke_omp'
        self.grid_response_backend_used = None
        self.cphf_conv_tol = 1e-9
        self.cphf_max_cycle = 200
        self.cphf_residual_warn_tol = 1e-8
        self.z_residual_warning = None
        self.fock_deriv_method = 'direct'
        self.de_d2c = None
        self.d2c_result = None

    def get_veff(self, mol=None, dm=None):
        return grid_response.get_veff(self, uks_grad.get_veff, mol, dm)

    def dump_flags(self, verbose=None):
        '''Log inherited UKS flags and source-UHF UCPHF controls.'''
        uks_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'cphf_conv_tol = %g', self.cphf_conv_tol)
        logger.info(self, 'cphf_max_cycle = %d', self.cphf_max_cycle)
        logger.info(
            self, 'cphf_residual_warn_tol = %s',
            self.cphf_residual_warn_tol)
        logger.info(self, 'fock_deriv_method = %s', self.fock_deriv_method)
        logger.info(self, 'dispersion = %s', self.base.dispersion)
        return self

    def grad_fock(self, Co, Cv, zvec, dm0, mo_coeff, mo_occ, atmlst):
        '''Return the spin-resolved source-Fock term by the selected route.'''
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

    @log_kernel_time('Gradient workflow')
    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        '''Evaluate ``dE/dR`` and retain the unrestricted term ledger.'''
        if self.base.mf_hf.converged:
            _check_source_state(self.base)
        if atmlst is not None:
            atmlst = np.asarray(atmlst, dtype=int)
        de = uks_grad.Gradients.kernel(
            self, mo_energy, mo_coeff, mo_occ, atmlst)
        self.de_nuc = self.grad_nuc(atmlst=self.atmlst)
        self.de_skel = self.de_skel_elec + self.de_nuc
        return de

    def _finalize(self):
        from dcforge.tools.logger import gradient_summary
        gradient_summary(self, unrestricted=True)

    grad_elec = grad_elec

    def as_scanner(self):
        raise NotImplementedError('UHF-DFT gradient scanner is not implemented')


Grad = Gradients
