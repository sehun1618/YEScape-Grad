'''Explicit Gamma-point PBE@HF gradient terms for supported DF providers.'''

import numpy as np
from pyscf import lib
from pyscf.pbc.dft import multigrid, numint
from pyscf.pbc.grad import krhf, krks

from dcforge.cpu.pbc.df.gaussian.jk import grad_j_energy
from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.df.rsdf import RSDF
from dcforge.cpu.pbc.grad._pseudo import get_vppnl_deriv
from dcforge.cpu.pbc.grad.multigrid import (
    get_vxc_deriv as get_multigrid_vxc_deriv,
)


class _HCoreGrad:
    @staticmethod
    def get_hcore(cell=None, kpts=None):
        return krhf.get_hcore(cell, kpts)


def get_hcore_deriv(cell, kpts):
    '''Return complete kinetic plus local/nonlocal GTH derivatives.'''
    hcore_gen = krhf.hcore_generator(_HCoreGrad(), cell, kpts)
    h1ao = np.asarray([hcore_gen(ia) for ia in range(cell.natm)])
    vppnl1 = get_vppnl_deriv(cell, kpts).transpose(0, 2, 1, 3, 4)
    return h1ao + vppnl1


def get_ovlp_deriv(cell, kpts):
    '''Return PySCF AO-row overlap derivatives ``-<nabla chi|chi>``.'''
    return np.asarray(krhf.get_ovlp(cell, kpts))


def grad_hcore(dm0, h1ao):
    '''Return ``Tr[P_HF h^lambda]`` in Ha/Bohr.'''
    return lib.einsum('axkij,ji->ax', h1ao, dm0).real


def get_vxc_deriv(mf_grad, dm0):
    '''Return target-PBE AO-row XC derivatives on the selected XC grid.

    Atom-centered PySCF grids use ``pyscf.pbc.grad.krks.get_vxc``.  When the
    target was configured with ``RHFDFT.multigrid_numint()``, the XC-only
    adapter removes multigrid Hartree before returning the same canonical
    ``(3, 1, nao, nao)`` AO-row derivative layout.  Consequently
    ``grad_vxc`` and the term-resolved skeleton assembly are independent of
    the target XC integration backend.
    '''
    mf = mf_grad.base
    if isinstance(mf._numint, multigrid.MultiGridNumInt2):
        mf_grad.xc_deriv_backend = 'multigrid'
        return get_multigrid_vxc_deriv(mf_grad, dm0)
    mf_grad.xc_deriv_backend = 'uniform_grid'
    kpts = np.zeros((1, 3))
    grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    ni = numint.KNumInt(kpts)
    vxc = krks.get_vxc(
        ni, mf.cell, grids, mf.xc, dm0[None], kpts,
        max_memory=mf_grad.max_memory, verbose=mf_grad.verbose,
    )
    return np.asarray(vxc)


def grad_vxc(mf_grad, dm0, atmlst):
    '''Return the fixed-density target XC derivative.'''
    mf = mf_grad.base
    v1 = get_vxc_deriv(mf_grad, dm0)
    de = np.zeros((len(atmlst), 3))
    aoslices = mf.cell.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de[k] = 2 * lib.einsum(
            'xij,ji->x', v1[:, 0, p0:p1], dm0[:, p0:p1],
        ).real
    return de


def grad_veff(mf_grad, dm0, atmlst):
    '''Return fixed-density target Hartree plus XC derivatives.'''
    mf = mf_grad.base
    de_vxc = grad_vxc(mf_grad, dm0, atmlst)
    if isinstance(mf.with_df, (GDF, RSDF)):
        if mf_grad.vj1 is None:
            de_j_source = mf_grad._de_j_source_contracted[atmlst]
            de_j_metric = mf_grad._de_j_metric_contracted[atmlst]
            de_j = de_j_source + de_j_metric
        else:
            de_j = grad_j_energy(dm0, mf_grad.vj1)[atmlst]
            de_j_source = grad_j_energy(
                dm0, mf_grad.vj1_source)[atmlst]
            de_j_metric = grad_j_energy(
                dm0, mf_grad.vj1_metric)[atmlst]
        mf_grad.de_j_source = de_j_source
        mf_grad.de_j_metric = de_j_metric
    else:
        kpts = np.zeros((1, 3))
        vj1 = np.asarray(mf.with_df.get_j_e1(dm0[None], kpts))
        de_j = np.zeros((len(atmlst), 3))
        aoslices = mf.cell.aoslice_by_atom()
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            de_j[k] = 2 * lib.einsum(
                'xij,ji->x', vj1[:, 0, p0:p1], dm0[:, p0:p1],
            ).real
        mf_grad.de_j_source = None
        mf_grad.de_j_metric = None
    mf_grad.de_j = de_j
    mf_grad.de_vxc = de_vxc
    return de_j + de_vxc


def grad_ovlp(Co, foo, s1, cell, atmlst):
    '''Return ``-2 Tr[S_row^lambda D^E_target]`` in Ha/Bohr.'''
    dme0 = 2 * Co @ foo @ Co.conj().T
    de = np.zeros((len(atmlst), 3))
    aoslices = cell.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de[k] = -2 * lib.einsum(
            'xij,ji->x', s1[0, :, p0:p1], dme0[:, p0:p1],
        ).real
    return de


def skeleton_gradient(mf_grad, dm0, foo, Co, h1ao, s1, atmlst,
                      de_hcore=None):
    '''Assemble all explicit electronic PBC gradient terms.

    The returned quantity is

    ``E_hcore^lambda + E_Hxc^lambda - 2 F^PBE_oo S^lambda_oo``.

    The fixed uniform periodic grid has no atom-centered grid-motion term.
    Nuclear Ewald repulsion is added by ``Gradients.kernel``.
    '''
    if de_hcore is None:
        de_hcore = grad_hcore(dm0, h1ao[atmlst])
    de_veff = grad_veff(mf_grad, dm0, atmlst)
    de_ovlp = grad_ovlp(Co, foo, s1, mf_grad.cell, atmlst)
    mf_grad.de_hcore = de_hcore
    mf_grad.de_veff = de_veff
    mf_grad.de_grid = np.zeros_like(de_hcore)
    mf_grad.de_extra = mf_grad.de_grid
    mf_grad.de_ovlp = de_ovlp
    mf_grad.de_skel_elec = de_hcore + de_veff + de_ovlp
    return mf_grad.de_skel_elec
