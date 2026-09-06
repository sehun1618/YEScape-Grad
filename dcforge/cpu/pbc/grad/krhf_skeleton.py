'''Explicit sampled-k semilocal-DFA@KRHF nuclear-gradient terms.'''

import numpy as np
from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.pbc.grad import krks

from dcforge.cpu.pbc.df.gaussian.jk import grad_j_energy_kpts
from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.grad.skeleton import get_mgga_vxc_deriv


def real_gradient_term(mf_grad, value, name, expect_real=True):
    '''Check and return the real part of one full-BZ gradient term.

    Complete Hermitian traces must have a negligible imaginary component.
    AO-row derivatives instead become real only after their conjugate side is
    restored, so callers mark those half-contractions with
    ``expect_real=False`` while still retaining the raw diagnostic.
    '''
    value = np.asarray(value)
    error = float(np.max(np.abs(value.imag))) if value.size else 0.0
    mf_grad.term_raw_imaginary_components[name] = error
    mf_grad.term_raw_imaginary_component_max_abs = max(
        mf_grad.term_raw_imaginary_component_max_abs, error)
    if expect_real:
        mf_grad.term_imaginary_residuals[name] = error
        mf_grad.term_imaginary_residual_max_abs = max(
            mf_grad.term_imaginary_residual_max_abs, error)
        if error > mf_grad.imaginary_residual_tol:
            raise ValueError(
                f'{name} imaginary residual {error:.3e} exceeds '
                f'{mf_grad.imaginary_residual_tol:.3e}')
    return value.real


def grad_hcore(mf_grad, dm0, h1ao, nkpts):
    r'''Return :math:`N_k^{-1}\sum_k\mathrm{Tr}[P_kh_k^\lambda]`.

    ``h1ao`` contains complete matrix derivatives ordered as
    ``(atom,axis,k,AO,AO)``; both AO sides are already present.
    '''
    value = lib.einsum('axkij,kji->ax', h1ao, dm0) / nkpts
    return real_gradient_term(mf_grad, value, 'de_hcore')


def get_vxc_deriv(mf_grad, dm0):
    '''Return the k-resolved AO-row target-DFA potential derivative.'''
    mf = mf_grad.base
    kpts = np.asarray(mf.kpts)
    grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    ni = numint.KNumInt(kpts)
    mf_grad.xc_deriv_backend = 'uniform_grid'
    if ni._xc_type(mf.xc) == 'MGGA':
        return get_mgga_vxc_deriv(mf_grad, dm0, ni, kpts)
    return np.asarray(krks.get_vxc(
        ni, mf.cell, grids, mf.xc, dm0, kpts,
        max_memory=mf_grad.max_memory, verbose=mf_grad.verbose))


def grad_vxc(mf_grad, dm0, atmlst):
    '''Return the fixed-density target-DFA XC derivative.'''
    mf = mf_grad.base
    nkpts = len(mf.kpts)
    v1 = get_vxc_deriv(mf_grad, dm0)
    de = []
    aoslices = mf.cell.aoslice_by_atom()
    for ia in atmlst:
        p0, p1 = aoslices[ia, 2:]
        de.append(2 * lib.einsum(
            'xkij,kji->x', v1[:, :, p0:p1],
            dm0[:, :, p0:p1]) / nkpts)
    return real_gradient_term(
        mf_grad, de, 'de_vxc', expect_real=False)


def grad_veff(mf_grad, dm0, atmlst):
    '''Return sampled-k Hartree plus uniform-grid target-DFA derivatives.'''
    mf = mf_grad.base
    nkpts = len(mf.kpts)
    de_vxc = grad_vxc(mf_grad, dm0, atmlst)
    if isinstance(mf.with_df, GDF):
        de_j_source = grad_j_energy_kpts(
            dm0, mf_grad.vj1_source)[atmlst]
        de_j_metric = grad_j_energy_kpts(
            dm0, mf_grad.vj1_metric)[atmlst]
        de_j = real_gradient_term(
            mf_grad, de_j_source + de_j_metric, 'de_j')
        mf_grad.de_j_source = de_j_source
        mf_grad.de_j_metric = de_j_metric
        mf_grad.de_j = de_j
        mf_grad.de_vxc = de_vxc
        return de_j + de_vxc

    vj1 = np.asarray(mf.with_df.get_j_e1(dm0, mf.kpts))
    de_j = []
    aoslices = mf.cell.aoslice_by_atom()
    for ia in atmlst:
        p0, p1 = aoslices[ia, 2:]
        de_j.append(2 * lib.einsum(
            'xkij,kji->x', vj1[:, :, p0:p1],
            dm0[:, :, p0:p1]) / nkpts)
    de_j = real_gradient_term(
        mf_grad, de_j, 'de_j', expect_real=False)
    mf_grad.de_j = de_j
    mf_grad.de_vxc = de_vxc
    return de_j + de_vxc


def grad_ovlp(mf_grad, Co, foo, s1, cell, atmlst, nkpts):
    r'''Return the target-Fock overlap/Pulay contribution.

    For every k-point,

    .. math::

        D^{E,\mathrm{target}}_k=2C_{o,k}F^{\mathrm{DFA}}_{oo,k}C_{o,k}^H.

    PySCF ``s1`` differentiates the atom-owned AO row.  The leading factor
    two restores the conjugate AO side; ``1/nkpts`` is the full-BZ weight.
    '''
    dme0 = np.asarray([
        2 * lib.dot(lib.dot(Co_k, foo_k), Co_k.conj().T)
        for Co_k, foo_k in zip(Co, foo, strict=True)
    ])
    de = []
    aoslices = cell.aoslice_by_atom()
    for ia in atmlst:
        p0, p1 = aoslices[ia, 2:]
        de.append(-2 * lib.einsum(
            'kxij,kji->x', s1[:, :, p0:p1],
            dme0[:, :, p0:p1]) / nkpts)
    return real_gradient_term(
        mf_grad, de, 'de_ovlp', expect_real=False)


def skeleton_gradient(mf_grad, dm0, foo, Co, h1ao, s1, atmlst):
    '''Assemble the explicit electronic semilocal-DFA@KRHF gradient terms.

    The returned fixed-orbital term is

    ``de_skel_elec = de_hcore + de_veff + de_ovlp``.

    The uniform periodic quadrature is fixed to the Cell, so it has no
    atom-centered grid-motion term.  Ewald nuclear repulsion is added by the
    inherited PySCF gradient kernel.
    '''
    nkpts = len(mf_grad.base.kpts)
    de_hcore = grad_hcore(mf_grad, dm0, h1ao[atmlst], nkpts)
    de_veff = grad_veff(mf_grad, dm0, atmlst)
    de_ovlp = grad_ovlp(
        mf_grad, Co, foo, s1, mf_grad.cell, atmlst, nkpts)
    mf_grad.de_hcore = de_hcore
    mf_grad.de_veff = de_veff
    mf_grad.de_grid = np.zeros_like(de_hcore)
    mf_grad.de_extra = mf_grad.de_grid
    mf_grad.de_ovlp = de_ovlp
    mf_grad.de_skel_elec = de_hcore + de_veff + de_ovlp
    return mf_grad.de_skel_elec
