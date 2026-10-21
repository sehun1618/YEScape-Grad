# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Skeleton terms of molecular CPU HF-DFT gradients.'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger


def grad_hcore(hcore_deriv, dm0, atmlst):
    r'''Return the one-electron term of the HF-DFT gradient.

    This function evaluates the first term of the complete gradient equation,

    .. math::

        g_{\mathrm{hcore}}^\lambda
        = \sum_{\mu\nu} P_{\mu\nu}^{\mathrm{HF}}
          h_{\mu\nu}^{\lambda}.

    Here ``h^lambda`` contains the kinetic and electron-nuclear attraction
    integral derivatives supplied by PySCF's ``hcore_generator``.

    Args:
        hcore_deriv : callable
            PySCF atom-resolved core-Hamiltonian derivative generator. Calling
            ``hcore_deriv(ia)`` returns an array of shape ``(3, nao, nao)``.
        dm0 : ndarray of shape (nao, nao)
            Total HF density: restricted ``P=2 C_o C_o^dagger`` or the sum
            of the two unrestricted spin densities.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            One-electron skeleton contribution in Ha/Bohr.
    '''
    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        de[k] = lib.einsum('xij,ij->x', hcore_deriv(ia), dm0)
    return de


def grad_veff(mol, vhf, dm0, atmlst):
    r'''Return the two-electron and AO-centered XC derivative terms.

    In the global-hybrid notation of the molecular-gradient PDF, this function
    supplies

    .. math::

        g_{\mathrm{veff}}^\lambda
        = \frac12\sum_{\mu\nu\rho\sigma}
          P_{\mu\nu}^{\mathrm{HF}}P_{\rho\sigma}^{\mathrm{HF}}
          \left[(\mu\nu|\rho\sigma)^\lambda
          -\frac{c_x}{2}(\mu\rho|\sigma\nu)^\lambda\right]
          + E_{\mathrm{xc,AO}}^\lambda[P_{\mathrm{HF}}].

    ``E_xc,AO^lambda`` denotes the AO-center/basis-function part returned in
    the derivative matrix ``vhf``.  Grid and, in the DF route, auxiliary-basis
    response are not included here; ``grad_extra`` adds them.  For
    range-separated hybrids, PySCF ``RKS.get_veff`` generalizes the exchange
    term using its ``(omega, alpha, hyb)`` coefficients.

    Args:
        mol : pyscf.gto.Mole
            Molecule defining atom-local AO slices.
        vhf : ndarray of shape (3, nao, nao)
            PySCF target-RKS effective-potential derivative evaluated with the
            supplied HF density.
        dm0 : ndarray of shape (nao, nao)
            Closed-shell HF density.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Combined Coulomb-XC skeleton contribution in Ha/Bohr.
    '''
    de = np.zeros((len(atmlst), 3))
    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        # nabla acts on the bra; factor 2 adds the ket contribution.
        de[k] = lib.einsum('xij,ij->x', vhf[:, p0:p1], dm0[p0:p1]) * 2
    return de


def grad_veff_uhf(mol, vhf, dm0, atmlst):
    r'''Return spin-summed two-electron and AO-centered XC derivatives.

    This is the unrestricted counterpart of ``grad_veff``:

    .. math::

        g_{\mathrm{veff}}^{I x}
        = 2\sum_{\sigma \in\{\alpha,\beta\}}
          \sum_{\mu\in I,\nu}P^{\sigma}_{\mu\nu}
          (\mathrm{vhf}^{\sigma}_{x})_{\mu\nu}.

    PySCF UKS supplies ``vhf`` with shape ``(2,3,nao,nao)``. The factor 2
    completes the bra/ket AO-center derivative, not spin occupancy.
    ``vhf`` is the upstream one-center energy-derivative intermediate, not
    the full nuclear derivative of the Fock matrix. Contracting the latter
    with P would double-count the quadratic Hartree/exchange energy.

    Args:
        mol : pyscf.gto.Mole
            Molecule defining atom-local AO slices.
        vhf : ndarray of shape (2, 3, nao, nao)
            Spin-resolved target-UKS effective-potential derivative.
        dm0 : ndarray of shape (2, nao, nao)
            Alpha/beta source-UHF density matrices.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Spin-summed Coulomb-XC contribution in Ha/Bohr.
    '''
    de = np.zeros((len(atmlst), 3))
    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de[k] = lib.einsum(
            'sxij,sij->x', vhf[:, :, p0:p1], dm0[:, p0:p1]) * 2
    return de


def grad_ovlp(mol, s1, dme0, atmlst):
    r'''Return the occupied-occupied moving-AO overlap term.

    This function evaluates

    .. math::

        g_{\mathrm{ovlp}}^\lambda
        = -2\sum_{ij}F_{ij}^{\mathrm{KS}}S_{ij}^{\lambda}
        = -\operatorname{Tr}(W_{\mathrm{AO}}S^\lambda),

    with

    .. math::

        W_{\mathrm{AO}}=2C_oF_{oo}^{\mathrm{KS}}C_o^\dagger.

    It is the known occupied-occupied metric response in the complete
    HF-DFT gradient, distinct from the VO response represented by the
    Z-vector contraction.

    Args:
        mol : pyscf.gto.Mole
            Molecule defining atom-local AO slices.
        s1 : ndarray of shape (3, nao, nao)
            PySCF AO overlap derivative.
        dme0 : ndarray of shape (nao, nao)
            Target energy-weighted density ``W = 2 C_o F^KS_oo C_o^dagger``.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Skeleton overlap/Pulay contribution in Ha/Bohr.
    '''
    de = np.zeros((len(atmlst), 3))
    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de[k] = -lib.einsum('xij,ij->x', s1[:, p0:p1], dme0[p0:p1]) * 2
    return de


def grad_extra(mf_grad, vhf, atmlst, log=None):
    r'''Return the target-XC grid and provider representation derivative.

    With ``grid_response=True``, PySCF stores the grid-weight, partition and
    coordinate response in ``vhf.exc1_grid``.  This function evaluates

    .. math::

        g_{\mathrm{grid}}^\lambda
        = E_{\mathrm{xc,grid}}^\lambda[P_{\mathrm{HF}}],

    so that

    .. math::

        E_{\mathrm{xc}}^\lambda[P_{\mathrm{HF}}]
        =E_{\mathrm{xc,AO}}^\lambda[P_{\mathrm{HF}}]
        +E_{\mathrm{xc,grid}}^\lambda[P_{\mathrm{HF}}].

    For a density-fitted target, PySCF's DF-RKS ``extra_force`` adds

    .. math::

        g_{\mathrm{aux}}^\lambda
        = E_{\mathrm{DF,aux}}^\lambda[P_{\mathrm{HF}}],

    the response of the atom-centered auxiliary basis and Coulomb metric.
    Thus this function returns ``g_grid + g_aux`` in the DF route and only
    ``g_grid`` in the conventional route.

    Args:
        mf_grad : dcforge.cpu.grad.rhf_dft.Gradients
            Active target-RKS gradient provider.
        vhf : ndarray
            Effective-potential derivative passed through the PySCF
            ``extra_force`` environment.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Kwargs:
        log : pyscf.lib.logger.Logger or None
            Logger exposed to ``extra_force``.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Provider-specific skeleton contribution in Ha/Bohr.
    '''
    if log is None:
        log = logger.new_logger(mf_grad)
    de = np.zeros((len(atmlst), 3))
    envs = {'vhf': vhf, 'log': log}
    for k, ia in enumerate(atmlst):
        de[k] = mf_grad.extra_force(ia, envs)
    return de


def split_extra(vhf, de_extra, atmlst):
    '''Split PySCF ``extra_force`` into grid and auxiliary-basis ledgers.

    Args:
        vhf : ndarray
            Tagged target effective-potential derivative.  PySCF RKS stores
            grid response in ``exc1_grid``; PySCF DF-RKS additionally stores
            auxiliary response in ``aux``.
        de_extra : ndarray of shape (len(atmlst), 3)
            Complete contribution returned through ``extra_force``.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        de_grid : ndarray of shape (len(atmlst), 3)
            Numerical-grid representation response.
        de_aux : ndarray of shape (len(atmlst), 3)
            DF auxiliary-basis response; zero for conventional integrals.

    Notes:
        ``de_extra`` remains the authoritative PySCF extension-hook result.
        The tagged arrays are used only to expose a readable term ledger.
    '''
    de_aux = np.zeros_like(de_extra)
    if hasattr(vhf, 'aux'):
        de_aux = np.asarray(vhf.aux)[atmlst]
    # PySCF DF-RKS extra_force adds the auxiliary term in-place to the view
    # returned from vhf.exc1_grid.  Recover the original grid term from the
    # authoritative total instead of reading that possibly mutated tag.
    de_grid = de_extra - de_aux
    return de_grid, de_aux


def skeleton_gradient(mf_grad, dm0, foo, Co, s1, atmlst=None):
    r'''Assemble all electronic terms except ``4 Z^dagger V^lambda``.

    In the global-hybrid notation of the PDF, this function returns

    .. math::

        \begin{aligned}
        g_{\mathrm{skel,elec}}^\lambda
        ={}&\sum_{\mu\nu}P_{\mu\nu}^{\mathrm{HF}}h_{\mu\nu}^{\lambda}\\
        &+\frac12\sum_{\mu\nu\rho\sigma}
          P_{\mu\nu}^{\mathrm{HF}}P_{\rho\sigma}^{\mathrm{HF}}
          \left[(\mu\nu|\rho\sigma)^\lambda
          -\frac{c_x}{2}(\mu\rho|\sigma\nu)^\lambda\right]\\
        &+E_{\mathrm{xc}}^\lambda[P_{\mathrm{HF}}]
          -2\sum_{ij}F_{ij}^{\mathrm{KS}}S_{ij}^{\lambda}.
        \end{aligned}

    The term ledger is

    ``de_skel_elec = de_hcore + de_veff + de_extra + de_ovlp``,

    where ``de_extra = de_grid + de_aux`` for density fitting and
    ``de_aux = 0`` for conventional four-center integrals.

    Nuclear repulsion ``E_NN^lambda`` is added later by the inherited PySCF
    ``Gradients.kernel``.  The remaining orbital-response contribution is
    assembled by ``nonvariational_correction`` as ``4 Z^dagger V^lambda``.

    Args:
        mf_grad : dcforge.cpu.grad.rhf_dft.Gradients
            Active gradient object and target-RKS derivative provider.
        dm0 : ndarray of shape (nao, nao)
            Closed-shell source-HF density.
        foo : ndarray of shape (nocc, nocc)
            Target-KS Fock occupied-occupied block ``F^KS_oo``.
        Co : ndarray of shape (nao, nocc)
            Occupied source-RHF MO coefficients.
        s1 : ndarray of shape (3, nao, nao)
            PySCF AO overlap derivative.

    Kwargs:
        atmlst : sequence of int or None
            Nuclear centers included in the gradient. All atoms are used when
            omitted.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Electronic skeleton contribution in Ha/Bohr.
    '''
    mol = mf_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    vhf = mf_grad.get_veff(mol, dm0)
    hcore_deriv = mf_grad.hcore_generator(mol)

    # The target energy-weighted density differs from the ordinary RHF one:
    # the orbitals and density are HF, while the OO Fock block is target KS.
    dme0 = 2 * Co @ foo @ Co.conj().T

    de_hcore = grad_hcore(hcore_deriv, dm0, atmlst)
    de_veff = grad_veff(mol, vhf, dm0, atmlst)
    de_ovlp = grad_ovlp(mol, s1, dme0, atmlst)
    de_extra = grad_extra(mf_grad, vhf, atmlst, log)
    de_grid, de_aux = split_extra(vhf, de_extra, atmlst)

    mf_grad.de_hcore = de_hcore
    mf_grad.de_veff = de_veff
    mf_grad.de_ovlp = de_ovlp
    mf_grad.de_grid = de_grid
    mf_grad.de_aux = de_aux
    mf_grad.de_extra = de_extra
    mf_grad.de_skel_elec = de_hcore + de_veff + de_ovlp + de_extra
    return mf_grad.de_skel_elec


def skeleton_gradient_uhf(mf_grad, dm0, foo, Co, s1, atmlst=None):
    r'''Assemble unrestricted electronic terms except the Z correction.

    The spin-resolved occupied-occupied metric term is

    .. math::

        -\sum_{\sigma \in\{\alpha,\beta\}}\sum_{ij}
         F^{\mathrm{KS},\sigma}_{ij}S^{\lambda,\sigma}_{ij},

    represented by ``dme0 = sum_σ C_o^σ F^KS,σ_oo (C_o^σ)^dagger``.
    Target grid and optional DF auxiliary responses enter through the same
    PySCF ``extra_force`` hook used by the restricted route.

    Args:
        mf_grad : dcforge.cpu.grad.uhf_dft.Gradients
            Active target-UKS derivative provider.
        dm0 : ndarray of shape (2, nao, nao)
            Alpha/beta source-UHF density.
        foo : tuple of two ndarray
            Target-UKS occupied-occupied Fock blocks.
        Co : tuple of two ndarray
            Alpha/beta occupied source-UHF MO coefficients.
        s1 : ndarray of shape (3, nao, nao)
            PySCF one-center AO overlap derivative.

    Kwargs:
        atmlst : sequence of int or None
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Unrestricted electronic skeleton contribution in Ha/Bohr.
    '''
    mol = mf_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    vhf = mf_grad.get_veff(mol, dm0)
    hcore_deriv = mf_grad.hcore_generator(mol)
    dm_sf = dm0[0] + dm0[1]
    # Each unrestricted spin orbital has occupation one.  The spin sum is
    # therefore explicit and carries no restricted factor 2.
    dme0 = Co[0] @ foo[0] @ Co[0].conj().T
    dme0 += Co[1] @ foo[1] @ Co[1].conj().T

    de_hcore = grad_hcore(hcore_deriv, dm_sf, atmlst)
    de_veff = grad_veff_uhf(mol, vhf, dm0, atmlst)
    de_ovlp = grad_ovlp(mol, s1, dme0, atmlst)
    de_extra = grad_extra(mf_grad, vhf, atmlst, log)
    de_grid, de_aux = split_extra(vhf, de_extra, atmlst)

    mf_grad.de_hcore = de_hcore
    mf_grad.de_veff = de_veff
    mf_grad.de_ovlp = de_ovlp
    mf_grad.de_grid = de_grid
    mf_grad.de_aux = de_aux
    mf_grad.de_extra = de_extra
    mf_grad.de_skel_elec = de_hcore + de_veff + de_ovlp + de_extra
    return mf_grad.de_skel_elec
