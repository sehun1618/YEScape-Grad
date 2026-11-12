# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Unrestricted molecular non-SCF gradients built on PySCF UKS gradients.'''

import numpy as np
from pyscf.grad import uks as uks_grad
from pyscf.lib import logger

from yescape_grad.tools.logger import log_kernel_time, gradient_stage, run_gradient_kernel
from yescape_grad.tools.sc_limit import sc_gradient_if_equivalent
from yescape_grad.cpu.disp.d2c import evaluate_d2c, normalize_dispersion
from yescape_grad.cpu.energy.rnsc import _check_source_state, _check_xc
from yescape_grad.cpu.grad import grid_response
from yescape_grad.cpu.grad.skeleton import skeleton_gradient_uhf
from yescape_grad.cpu.source.uhf import grad_fock_direct, grad_fock_make_h1
from yescape_grad.cpu.grad.u_nv import (
    nonvariational_correction,
)
from yescape_grad.cpu.grad.u_zvec import solve_z_vector
from yescape_grad.tools.zvec import warn_z_residual


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None):
    r'''Evaluate the electronic unrestricted target-on-source gradient.

    Without optional dispersion, the unrestricted gradient equation is

    .. math::

        \begin{aligned}
        \frac{dE}{d\lambda}={}&
        \sum_{\sigma \mu\nu}P^{\sigma}_{\mu\nu}h^\lambda_{\mu\nu}
        +E_{\mathrm{Hxc}}^\lambda[P^\alpha,P^\beta]
        +E_{\mathrm{NN}}^\lambda\\
        &-\sum_{\sigma ij}F^{\mathrm{target},\sigma}_{ij}S^{\lambda,\sigma}_{ij}
        +2\,\mathrm{Re}\sum_{\sigma ai}(Z^{\sigma}_{ai})^*V^{\lambda,\sigma}_{ai}.
        \end{aligned}

    P^σ is the source density. F^target,σ is the target Fock
    evaluated at both source spin densities, in the source MO basis.
    E_Hxc belongs to the target; V^lambda uses the source Fock derivative
    and source response kernel. Here σ denotes alpha or beta spin.

    Each spin orbital has occupation one, so the nonvariational prefactor is
    2 rather than the restricted spatial-orbital prefactor 4.

    Args:
        mf_grad : yescape_grad.cpu.grad.unsc.Gradients
            Active unrestricted non-SCF gradient object.

    Kwargs:
        mo_energy : tuple of two ndarray or None
            Unrestricted source-SCF alpha/beta orbital energies.
        mo_coeff : tuple of two ndarray or None
            Unrestricted source-SCF alpha/beta MO coefficients.
        mo_occ : tuple of two ndarray or None
            Unrestricted source-SCF alpha/beta occupations.
        atmlst : sequence of int or None
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Electronic ``dE/dR`` in Ha/Bohr. PySCF adds nuclear repulsion.
    '''
    mf = mf_grad.base
    mf_source = mf.mf_source
    mol = mf_grad.mol
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    if mf_source.converged:
        _check_source_state(mf)
    if mf.e_dft is None:
        mf.kernel(run_source=not mf_source.converged)
    _check_source_state(mf)
    _check_xc(mf)
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if mo_occ is None:    mo_occ = mf.mo_occ
    if atmlst is None:    atmlst = range(mol.natm)
    atmlst = list(atmlst)

    Ca, Cb = mo_coeff
    if hasattr(mf.mf_source, 'xc') and np.iscomplexobj(mo_coeff):
        raise NotImplementedError('DFA-source gradients require real orbitals')
    occa, occb = mo_occ
    for occ in (occa, occb):
        if not np.all((occ == 1) | (occ == 0)):
            raise ValueError('unrestricted non-SCF gradients require exact 1/0 occupations')
        nocc = np.count_nonzero(occ > 0)
        if not np.all(occ[:nocc] == 1) or not np.all(occ[nocc:] == 0):
            raise ValueError('occupied source orbitals must precede virtual orbitals')

    soa, sob = occa > 0, occb > 0
    sva, svb = occa == 0, occb == 0
    Co = (Ca[:, soa], Cb[:, sob])
    Cv = (Ca[:, sva], Cb[:, svb])
    dm0 = mf_source.make_rdm1(mo_coeff, mo_occ)

    sc_result = sc_gradient_if_equivalent(
        mf_grad, mo_energy, mo_coeff, mo_occ, dm0, Co, atmlst)
    if sc_result is not None:
        return sc_result

    # Compute the target Fock only once, then retain both VO and OO blocks.
    # AO-to-MO projection is split into two matrix products per spin; this is
    # faster and more predictable than a four-index einsum on CPU and GPU.
    # F^target[P^source], projected below into the source MO basis.
    # The source Fock/kernel instead determines the orbital response.
    with gradient_stage(mf_grad, 'Target Fock evaluation and MO projection'):
        fock_target = mf.get_fock(dm=dm0)
        fock_mo_a = Ca.conj().T @ fock_target[0] @ Ca
        fock_mo_b = Cb.conj().T @ fock_target[1] @ Cb
        fvo = (fock_mo_a[sva][:, soa], fock_mo_b[svb][:, sob])
        foo = (fock_mo_a[soa][:, soa], fock_mo_b[sob][:, sob])

    with gradient_stage(mf_grad, 'Z-vector response solve'):
        fvind = mf_source.gen_response(
            mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=1)
        response = mf_grad.make_response(Co, Cv)
        zvec, z_info = solve_z_vector(
            mo_energy, mo_occ, Co, Cv, fvo, fvind,
            conv_tol=mf_grad.cphf_conv_tol,
            max_cycle=mf_grad.cphf_max_cycle,
            log=log, vind=None if response is None else response.vind,
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
    with gradient_stage(mf_grad, 'Source Fock derivative (including source XC)'):
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
        with gradient_stage(mf_grad, 'Target explicit gradient (including target grid)'):
            de_skel = skeleton_gradient_uhf(
                mf_grad, dm0, foo, Co, s1, atmlst)
    finally:
        mf_grad.grid_response = grid_response
    with gradient_stage(mf_grad, 'NSC overlap response and assembly'):
        de_nvar = nonvariational_correction(
            mf_grad, Co, Cv, mo_energy, zvec, s1, de_fock, fvind, atmlst, response=response)

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
    '''Unrestricted unrestricted target-on-source analytic nuclear gradient driver.

    grid_response_backend : str
        'becke_omp' (default) or 'pyscf_default'; same semantics as RHF.
    '''

    _keys = uks_grad.Gradients._keys | {
        'cphf_conv_tol', 'cphf_max_cycle', 'cphf_residual_warn_tol',
        'fock_deriv_method', 'z_residual_warning',
        'sc_limit', 'sc_limit_used', 'gradient_stage_times', 'source_grid_response_backend_used', 'source_xc_stage_times',
        'de_d2c', 'd2c_result', 'source_grid_response', 'grid_response_backend',
        'grid_response_backend_used', 'response_backend', 'response_backend_used',
        'response_mo_cache_bytes', 'response_mo_setup_s',
    }

    def __init__(self, mf):
        from yescape_grad.cpu.energy.unsc import UNSC
        if not isinstance(mf, UNSC):
            raise TypeError('Gradients requires a yescape_grad.cpu.energy.UNSC object')
        uks_grad.Gradients.__init__(self, mf)
        self.grid_response = True
        self.source_grid_response = True
        self.source_xc_stage_times = {}
        self.sc_limit = True
        self.sc_limit_used = False
        self.grid_response_backend = 'becke_omp'
        self.grid_response_backend_used = None
        self.cphf_conv_tol = 1e-9
        self.cphf_max_cycle = 200
        self.cphf_residual_warn_tol = 1e-8
        self.z_residual_warning = None
        self.fock_deriv_method = 'direct'
        self.response_backend = 'ao'
        self.response_backend_used = None
        self.response_mo_cache_bytes = 0
        self.response_mo_setup_s = 0.
        self.de_d2c = None
        self.d2c_result = None

    def get_veff(self, mol=None, dm=None):
        return grid_response.get_veff(self, uks_grad.get_veff, mol, dm)

    def dump_flags(self, verbose=None):
        '''Log inherited UKS flags and unrestricted source-SCF UCPHF controls.'''
        uks_grad.Gradients.dump_flags(self, verbose)
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
        '''Return the spin-resolved source-Fock term by the selected route.'''
        if hasattr(self.base.mf_source, 'xc'):
            from yescape_grad.cpu.source.uks import grad_fock
            return grad_fock(self, Co, Cv, zvec, dm0, atmlst)
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

    def make_response(self, Co, Cv):
        """Conventional/source-DFA response uses the existing AO provider."""
        self.response_backend_used = None
        if self.response_backend != 'ao':
            raise ValueError("conventional CPU response_backend must be 'ao'")
        self.response_backend_used = 'ao'
        return None

    @log_kernel_time('Gradient workflow')
    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        '''Evaluate ``dE/dR`` and retain the unrestricted term ledger.'''
        if self.base.mf_source.converged:
            _check_source_state(self.base)
        if atmlst is not None:
            atmlst = np.asarray(atmlst, dtype=int)
        de = run_gradient_kernel(uks_grad.Gradients.kernel,
            self, mo_energy, mo_coeff, mo_occ, atmlst)
        self.de_nuc = self.grad_nuc(atmlst=self.atmlst)
        self.de_skel = self.de_skel_elec + self.de_nuc
        return de

    def _finalize(self):
        from yescape_grad.tools.logger import gradient_summary
        gradient_summary(self, unrestricted=True)

    grad_elec = grad_elec

    def as_scanner(self):
        from yescape_grad.tools.scanner import gradient_scanner
        return gradient_scanner(self)


Grad = Gradients
