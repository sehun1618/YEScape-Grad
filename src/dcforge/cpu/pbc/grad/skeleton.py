'''Explicit Gamma-point semilocal-DFA@HF gradient terms.'''

import numpy as np
from pyscf import lib
from pyscf.grad import rks as molecular_rks_grad
from pyscf.pbc.dft import multigrid, numint
from pyscf.pbc.grad import krhf, krks

from dcforge.cpu.pbc.df.gaussian.jk import grad_j_energy
from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.df.rsdf import RSDF
from dcforge.cpu.pbc.grad._pseudo import get_vppnl_deriv
from dcforge.cpu.pbc.grad.multigrid import (
    get_veff_deriv as get_multigrid_veff_deriv,
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
    '''Return target-DFA AO-row XC derivatives on the selected XC grid.

    Atom-centered PySCF grids use ``pyscf.pbc.grad.krks.get_vxc``.  When the
    target was configured with ``RHFDFT.multigrid_numint()``, the XC-only
    adapter removes multigrid Hartree before returning the same canonical
    ``(3, 1, nao, nao)`` AO-row derivative layout.  Consequently
    ``grad_vxc`` and the term-resolved skeleton assembly are independent of
    the target XC integration backend.
    '''
    mf = mf_grad.base
    if isinstance(mf._numint, multigrid.MultiGridNumInt2):
        if mf.multigrid_with_j:
            raise ValueError(
                'grad_vxc cannot split a fused multigrid J+XC derivative; '
                'use grad_veff')
        mf_grad.xc_deriv_backend = 'multigrid'
        return get_multigrid_veff_deriv(mf_grad, dm0)
    mf_grad.xc_deriv_backend = 'uniform_grid'
    kpts = np.zeros((1, 3))
    grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    ni = numint.KNumInt(kpts)
    if ni._xc_type(mf.xc) == 'MGGA':
        return get_mgga_vxc_deriv(mf_grad, dm0, ni, kpts)
    vxc = krks.get_vxc(
        ni, mf.cell, grids, mf.xc, dm0[None], kpts,
        max_memory=mf_grad.max_memory, verbose=mf_grad.verbose,
    )
    return np.asarray(vxc)


def get_mgga_vxc_deriv(mf_grad, dm0, ni, kpts):
    r'''Return the periodic MGGA AO-row derivative including the tau channel.

    PySCF 2.14.0's periodic ``krks.get_vxc`` implements LDA and GGA but
    raises for MGGA.  The missing term is the response of

    .. math::

       \tau(\mathbf r)=\frac12\sum_{\mu\nu}P_{\mu\nu}
       \nabla\phi_\mu(\mathbf r)\cdot\nabla\phi_\nu(\mathbf r).

    This helper retains PySCF's ``KNumInt`` density, block loop, libxc
    weights, and molecular AO derivative contractions.  The factor one half
    applied to ``v_tau`` follows PySCF's definition of ``tau``.  r2SCAN does
    not request a density-Laplacian channel in this convention.

    Args:
        mf_grad : dcforge.cpu.pbc.grad.rhf_dft.Gradients
            Active fixed-density periodic gradient object.
        dm0 : ndarray of shape (nao, nao) or (nkpts,nao,nao)
            Closed-shell source RHF or KRHF density.
        ni : pyscf.pbc.dft.numint.KNumInt
            K-resolved numerical integrator used by the target derivative.
        kpts : ndarray of shape (nkpts, 3)
            Explicit full-BZ k-point axis.

    Returns:
        ndarray of shape (3, nkpts, nao, nao)
            Nuclear AO-row derivative of the target MGGA potential.
    '''
    mf = mf_grad.base
    cell = mf.cell
    grids = mf.grids
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    density = np.asarray(dm0)
    if density.ndim == 2:
        density = density[None]
    if density.shape != (len(kpts), cell.nao_nr(), cell.nao_nr()):
        raise ValueError(
            'periodic MGGA density must have shape (nkpts,nao,nao)')
    make_rho, nset, nao = ni._gen_rho_evaluator(
        cell, density, hermi=1)
    if nset != 1:
        raise ValueError('restricted periodic MGGA requires one density set')
    ao_loc = cell.ao_loc_nr()
    vmat = np.zeros(
        (3, nset, len(kpts), nao, nao), dtype=density.dtype)
    for ao_k1, ao_k2, mask, weight, _coords in ni.block_loop(
            cell, grids, nao, 2, kpts, None,
            mf_grad.max_memory):
        ao_k1 = np.asarray(ao_k1)
        ao_k2 = np.asarray(ao_k2)
        rho = make_rho(0, ao_k2[:, :4], mask, 'MGGA')
        vxc = ni.eval_xc_eff(
            mf.xc, rho, deriv=1, xctype='MGGA', spin=0)[1]
        weighted = vxc * weight
        weighted[0] *= .5
        weighted[4] *= .5
        for k in range(len(kpts)):
            molecular_rks_grad._gga_grad_sum_(
                vmat[:, 0, k], cell, ao_k1[k], weighted[:4],
                mask, ao_loc)
            molecular_rks_grad._tau_grad_dot_(
                vmat[:, 0, k], cell, ao_k1[k], weighted[4],
                mask, ao_loc, True)
    return -vmat[:, 0]


def grad_vxc(mf_grad, dm0, atmlst):
    '''Return the fixed-density target-DFA XC derivative.'''
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
    if (isinstance(mf._numint, multigrid.MultiGridNumInt2)
            and mf.multigrid_with_j):
        v1 = get_multigrid_veff_deriv(mf_grad, dm0)
        de_veff = np.zeros((len(atmlst), 3))
        aoslices = mf.cell.aoslice_by_atom()
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            de_veff[k] = 2 * lib.einsum(
                'xij,ji->x', v1[:, 0, p0:p1], dm0[:, p0:p1],
            ).real
        # PySCF forms J and XC on one reciprocal potential before the AO
        # derivative transform.  Keep that honest combined term instead of
        # attributing it to either physical component without a second pass.
        mf_grad.de_j = None
        mf_grad.de_vxc = None
        mf_grad.de_j_source = None
        mf_grad.de_j_metric = None
        mf_grad.de_j_xc_multigrid = de_veff
        mf_grad.xc_deriv_backend = 'multigrid_j_xc'
        return de_veff
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
    mf_grad.de_j_xc_multigrid = None
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

    ``E_hcore^lambda + E_Hxc^lambda - 2 F^DFA_oo S^lambda_oo``.

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
