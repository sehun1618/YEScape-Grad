'''Explicit sampled-k semilocal-DFA@KRHF nuclear-gradient terms.'''

import numpy as np
from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.pbc.grad import krks

from dcforge.cpu.pbc.df.gaussian.jk import grad_j_energy_kpts
from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.df.rsdf import RSDF
from dcforge.cpu.pbc.grad.skeleton import get_mgga_vxc_deriv


def real_gradient_term(mf_grad, value, name, expect_real=True):
    '''Check and return the real part of one full-BZ gradient term.

    The k-point sum is completed before this operation.  ``expect_real`` is
    used for complete Hermitian traces, whose imaginary part is a genuine
    numerical residual.  It is disabled for AO-row derivatives and terms
    defined as an explicit ``Re`` inner product: their unpaired half
    contraction can legitimately be complex before the conjugate AO side is
    restored.  That raw imaginary component is still retained separately and
    is never silently confused with a physical residual.
    '''
    value = np.asarray(value)
    if value.size:
        error = float(np.max(np.abs(value.imag)))
    else:
        error = 0.0
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
    r'''Return :math:`N_k^{-1}\sum_k\mathrm{Tr}[P_k h_k^\lambda]`.

    ``h1ao`` contains complete matrix derivatives with ordering
    ``(atom,axis,k,AO,AO)``.  Both AO sides are already differentiated, so no
    additional row factor is applied.
    '''
    value = lib.einsum('axkij,kji->ax', h1ao, dm0) / nkpts
    return real_gradient_term(mf_grad, value, 'de_hcore')


def get_vxc_deriv(mf_grad, dm0):
    '''Return sampled-k AO-row derivatives of target semilocal ``V_xc``.

    PySCF's public periodic ``krks.get_vxc`` owns the uniform-grid GGA
    derivative.  Its output ordering is ``(axis,k,AO-row,AO-column)`` and it
    retains the same full-BZ k representation as the K1 target energy.
    '''
    mf = mf_grad.base
    kpts = np.asarray(mf.kpts)
    grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    ni = numint.KNumInt(kpts)
    mf_grad.xc_deriv_backend = 'uniform_grid'
    if ni._xc_type(mf.xc) == 'MGGA':
        return get_mgga_vxc_deriv(mf_grad, dm0, ni, kpts)
    vxc = krks.get_vxc(
        ni, mf.cell, grids, mf.xc, dm0, kpts,
        max_memory=mf_grad.max_memory, verbose=mf_grad.verbose)
    return np.asarray(vxc)


def grad_vxc(mf_grad, dm0, atmlst):
    '''Return the fixed-density target semilocal XC derivative.'''
    mf = mf_grad.base
    nkpts = len(mf.kpts)
    v1 = get_vxc_deriv(mf_grad, dm0)
    de = []
    aoslices = mf.cell.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de.append(
            2 * lib.einsum(
                'xkij,kji->x', v1[:, :, p0:p1],
                dm0[:, :, p0:p1]) / nkpts)
    return real_gradient_term(
        mf_grad, de, 'de_vxc', expect_real=False)


def grad_veff(mf_grad, dm0, atmlst):
    '''Return sampled-k target Hartree plus uniform-grid XC derivatives.'''
    mf = mf_grad.base
    nkpts = len(mf.kpts)
    de_vxc = grad_vxc(mf_grad, dm0, atmlst)
    if isinstance(mf.with_df, (GDF, RSDF)):
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
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de_j.append(
            2 * lib.einsum(
                'xkij,kji->x', vj1[:, :, p0:p1],
                dm0[:, :, p0:p1]) / nkpts)
    de_j = real_gradient_term(
        mf_grad, de_j, 'de_j', expect_real=False)
    mf_grad.de_j = de_j
    mf_grad.de_vxc = de_vxc
    return de_j + de_vxc


def grad_ovlp(mf_grad, Co, foo, s1, cell, atmlst, nkpts):
    r'''Return the target-Fock overlap/Pulay term.

    For every k-point,

    ``D^E_target,k = 2 C_o,k F^DFA_oo,k C_o,k^H``.

    PySCF ``s1`` differentiates the atom-owned AO row.  The leading factor two
    restores its conjugate AO side, and ``1/nkpts`` applies the equal full-BZ
    energy weight.
    '''
    Co = tuple(np.asarray(block) for block in Co)
    foo = tuple(np.asarray(block) for block in foo)
    dme0 = np.asarray([
        2 * lib.dot(lib.dot(Co_k, foo_k), Co_k.conj().T)
        for Co_k, foo_k in zip(Co, foo, strict=True)
    ])
    de = []
    aoslices = cell.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de.append(
            -2 * lib.einsum(
                'kxij,kji->x', s1[:, :, p0:p1],
                dme0[:, :, p0:p1]) / nkpts)
    return real_gradient_term(
        mf_grad, de, 'de_ovlp', expect_real=False)


def skeleton_gradient(mf_grad, dm0, foo, Co, h1ao, s1, atmlst):
    '''Assemble sampled-k explicit electronic DFA@KRHF gradient terms.

    The returned fixed-orbital contribution is

    ``de_skel_elec = de_hcore + de_veff + de_ovlp``.

    The periodic uniform grid has no atom-centered grid-motion term.  Ewald
    nuclear repulsion is added by ``Gradients.kernel`` after this function.
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
