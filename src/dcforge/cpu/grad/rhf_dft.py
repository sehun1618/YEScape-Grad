'''Restricted molecular HF-DFT gradients built on PySCF RKS gradients.'''

import numpy as np
from pyscf.grad import rks as rks_grad
from pyscf.lib import logger

from dcforge.cpu.energy.rhf_dft import _check_xc
from dcforge.cpu.grad.rhf_nv import (
    grad_fock_direct,
    grad_fock_make_h1,
    nonvariational_correction,
)
from dcforge.cpu.grad.rhf_zvec import solve_z_vector
from dcforge.cpu.grad.skeleton import skeleton_gradient


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None):
    r'''Evaluate the electronic restricted target-DFT-on-HF gradient.

    The implementation follows the complete Z-vector equation from the PDF,

    .. math::

        \begin{aligned}
        \frac{dE_{\mathrm{HF-DFT}}}{d\lambda}
        ={}&\sum_{\mu\nu}P_{\mu\nu}^{\mathrm{HF}}h_{\mu\nu}^{\lambda}\\
        &+\frac12\sum_{\mu\nu\rho\sigma}
          P_{\mu\nu}^{\mathrm{HF}}P_{\rho\sigma}^{\mathrm{HF}}
          \left[(\mu\nu|\rho\sigma)^\lambda
          -\frac{c_x}{2}(\mu\rho|\sigma\nu)^\lambda\right]\\
        &+E_{\mathrm{xc}}^\lambda[P_{\mathrm{HF}}]
          +E_{\mathrm{NN}}^\lambda
          -2\sum_{ij}F_{ij}^{\mathrm{KS}}S_{ij}^{\lambda}
          +4\,\mathrm{Re}\sum_{ai}Z_{ai}^{*}V_{ai}^{\lambda}.
        \end{aligned}

    In code attributes, the same equation is

    ``de = de_hcore + de_veff + de_extra + de_ovlp + de_nuc``
    ``     + de_fock + de_ovlp_energy + de_ovlp_response``.

    ``de_extra = de_grid + de_aux``.  The ``de_aux`` term is zero for the
    conventional integral route and contains the target auxiliary-basis
    response for a density-fitted workflow.

    ``grad_elec`` returns every term above except ``de_nuc``;
    ``Gradients.kernel`` adds the inherited nuclear-repulsion derivative.

    Source orbitals, occupations and the response Hessian come from RHF.  The
    energy, ``F^KS_vo``, ``F^KS_oo`` and explicit DFT grid derivatives come
    from the target RKS object evaluated at the fixed HF density.

    Args:
        mf_grad : dcforge.cpu.grad.rhf_dft.Gradients
            Active HF-DFT gradient object.

    Kwargs:
        mo_energy : ndarray of shape (nmo,) or None
            Source-RHF orbital energies. The converged reference values are
            used when omitted.
        mo_coeff : ndarray of shape (nao, nmo) or None
            Source-RHF MO coefficients.
        mo_occ : ndarray of shape (nmo,) or None
            Restricted source-RHF occupations.
        atmlst : sequence of int or None
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Electronic gradient ``dE/dR`` in Ha/Bohr. Nuclear repulsion is
            added by the inherited PySCF ``Gradients.kernel`` flow.
    '''
    mf = mf_grad.base
    mf_hf = mf.mf_hf
    mol = mf_grad.mol
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    if mf.e_dft is None:
        mf.kernel(run_hf=not mf_hf.converged)
    _check_xc(mf)
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if mo_occ is None:    mo_occ = mf.mo_occ
    if atmlst is None:    atmlst = range(mol.natm)
    atmlst = list(atmlst)

    so = mo_occ > 0
    sv = mo_occ == 0
    if not np.all((mo_occ == 2) | sv):
        raise ValueError('restricted HF-DFT gradients require exact 2/0 occupations')
    Co = mo_coeff[:, so]
    Cv = mo_coeff[:, sv]
    dm0 = mf_hf.make_rdm1(mo_coeff, mo_occ)

    # The HF MO basis diagonalizes F^HF, not the target KS Fock.  Its VO block
    # F^KS_vo is therefore the nonvariational Z-vector RHS, while F^KS_oo
    # builds the target energy-weighted density in the skeleton term.
    fock_ks = mf.get_fock(dm=dm0)
    fock_mo = mo_coeff.conj().T @ fock_ks @ mo_coeff
    fvo = fock_mo[sv][:, so]
    foo = fock_mo[so][:, so]

    # The left-hand side is always the source-HF orbital Hessian.  Target-XC
    # kernels do not enter [(A+B)^HF]^T Z = F^KS_vo.
    fvind = mf_hf.gen_response(mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=1)
    zvec, z_info = solve_z_vector(
        mo_energy, mo_occ, Co, Cv, fvo, fvind,
        conv_tol=mf_grad.cphf_conv_tol,
        max_cycle=mf_grad.cphf_max_cycle,
        log=log,
    )
    mf_grad.zvec = zvec
    mf_grad.z_rhs_max_abs = z_info['rhs_max_abs']
    mf_grad.z_residual_max_abs = z_info['residual_max_abs']
    mf_grad.z_residual_relative = z_info['residual_relative']
    mf_grad.z_solver = z_info['solver']

    # These are perturbation-specific geometric quantities.  S^lambda does
    # not belong in the perturbation-independent adjoint solve above; together
    # with F^HF,lambda it forms V^lambda for each nuclear coordinate here.
    s1 = mf_grad.get_ovlp(mol)
    de_fock = mf_grad.grad_fock(
        Co, Cv, zvec, dm0, mo_coeff, mo_occ, atmlst)
    de_skel = skeleton_gradient(mf_grad, dm0, foo, Co, s1, atmlst)
    de_nvar = nonvariational_correction(
        mf_grad, Co, Cv, mo_energy, zvec, s1, de_fock, fvind, atmlst)
    return de_skel + de_nvar


class Gradients(rks_grad.Gradients):
    '''Restricted target-DFT-on-HF analytic nuclear gradient driver.

    The class inherits PySCF RKS derivative providers and replaces only the
    electronic assembly that is nonvariational with respect to the target
    functional.  Public options retain the PySCF names ``cphf_conv_tol``,
    ``cphf_max_cycle``, ``grid_response`` and ``atmlst``.
    '''

    _keys = rks_grad.Gradients._keys | {
        'cphf_conv_tol', 'cphf_max_cycle', 'fock_deriv_method',
    }

    def __init__(self, mf):
        from dcforge.cpu.energy.rhf_dft import RHFDFT
        if not isinstance(mf, RHFDFT):
            raise TypeError('Gradients requires a dcforge.cpu.energy.RHFDFT object')
        rks_grad.Gradients.__init__(self, mf)
        self.grid_response = True
        self.cphf_conv_tol = 1e-9
        self.cphf_max_cycle = 200
        self.fock_deriv_method = 'direct'

    def dump_flags(self, verbose=None):
        '''Log inherited RKS flags and the source-HF CPHF controls.'''
        rks_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'cphf_conv_tol = %g', self.cphf_conv_tol)
        logger.info(self, 'cphf_max_cycle = %d', self.cphf_max_cycle)
        logger.info(self, 'fock_deriv_method = %s', self.fock_deriv_method)
        return self

    def grad_fock(self, Co, Cv, zvec, dm0, mo_coeff, mo_occ, atmlst):
        '''Return ``-4 Z F^HF,lambda_vo`` with the selected exact route.'''
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

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        '''Evaluate ``dE/dR`` and retain the term-resolved gradient ledger.'''
        if atmlst is not None:
            atmlst = np.asarray(atmlst, dtype=int)
        de = rks_grad.Gradients.kernel(
            self, mo_energy, mo_coeff, mo_occ, atmlst)
        self.de_nuc = self.grad_nuc(atmlst=self.atmlst)
        self.de_skel = self.de_skel_elec + self.de_nuc
        return de

    grad_elec = grad_elec

    def as_scanner(self):
        raise NotImplementedError('HF-DFT gradient scanner is not implemented')


Grad = Gradients
