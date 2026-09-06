'''Explicit GPU sampled-k semilocal-DFA@KRHF skeleton terms.'''

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.dft import multigrid, multigrid_v2
from gpu4pyscf.pbc.grad import krks as gpu_krks_grad
from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad
from gpu4pyscf.pbc.gto import int1e
from pyscf import lib

from dcforge.cpu.pbc.df.gaussian.hcore import (
    contract_hcore_deriv as contract_gaussian_hcore_deriv,
)
from dcforge.cpu.pbc.df.gaussian.hcore import (
    get_hcore_deriv as get_gaussian_hcore_deriv,
)
from dcforge.gpu.pbc.df.gdf import GDF
from dcforge.gpu.pbc.grad.multigrid import get_veff_deriv


def real_gradient_term(mf_grad, value, name, expect_real=True):
    '''Check one complete full-BZ gradient term and return its real part.'''
    value = cp.asarray(value)
    error = float(cp.max(cp.abs(value.imag)).get()) if value.size else 0.0
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


def get_ovlp_deriv(mf_grad):
    '''Return native GPU AO-row overlap derivatives at every k-point.'''
    return -int1e.int1e_ipovlp(mf_grad.cell, mf_grad.base.kpts)


def prepare_gaussian_hcore_pair(mf_grad, dm0, dz, atmlst):
    r'''Contract one sampled-k Gaussian Hcore derivative with ``P`` and ``D_Z``.

    The same one-electron derivative ``h^lambda`` appears in the skeleton
    term ``Tr[P h^lambda]`` and in the nonvariational source-Fock term
    ``-2 Tr[D_Z h^lambda]``.  Production passes both densities through the
    CPU provider primitive once and transfers only two ``(natm_selected,3)``
    scalars.  ``gaussian_hcore_method='full'`` retains the complete matrix
    reference for small-system parity tests.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.krhf_dft.Gradients
            Active sampled-k gradient object.
        dm0, dz : cupy.ndarray of shape (nkpts,nao,nao)
            Source and Hermitian Z-transition densities.
        atmlst : sequence of int
            Requested nuclear centers.
    '''
    mf = mf_grad.base
    if not isinstance(mf.with_df, GDF):
        raise TypeError('Gaussian Hcore pair requires GDF')
    densities = cp.asnumpy(cp.stack((dm0, dz)))
    nkpts = len(mf.kpts)
    if mf_grad.gaussian_hcore_method == 'contracted':
        total, components = contract_gaussian_hcore_deriv(
            mf.cell, mf.kpts, densities, atmlst=atmlst)
        total = np.asarray(total).real / nkpts
        components = {
            name: np.asarray(value).real / nkpts
            for name, value in components.items()
        }
        full_matrix_bytes = int(
            mf.cell.natm * 3 * nkpts * mf.cell.nao_nr() ** 2
            * np.dtype(np.complex128).itemsize)
        mf_grad.gaussian_hcore_host_workspace_bytes = 0
        mf_grad.gaussian_hcore_full_matrix_bytes_avoided = full_matrix_bytes
        mf_grad.gaussian_hcore_host_owner = (
            'dcforge CPU sampled-k Gaussian Hcore contracted pair')
    elif mf_grad.gaussian_hcore_method == 'full':
        h1ao = get_gaussian_hcore_deriv(mf.cell, mf.kpts)
        total = lib.einsum(
            'axkij,skji->sax', h1ao[np.asarray(atmlst)], densities
        ).real / nkpts
        components = None
        mf_grad.gaussian_hcore_host_workspace_bytes = int(h1ao.nbytes)
        mf_grad.gaussian_hcore_full_matrix_bytes_avoided = 0
        mf_grad.gaussian_hcore_host_owner = (
            'dcforge CPU sampled-k Gaussian Hcore full-matrix reference')
    else:
        raise ValueError(
            "gaussian_hcore_method must be 'contracted' or 'full'")
    mf_grad.gaussian_hcore_pair = cp.asarray(total)
    mf_grad.gaussian_hcore_components = components
    mf_grad.gaussian_hcore_host_calls += 1
    mf_grad.gaussian_hcore_contraction_calls += 2
    mf_grad.gaussian_hcore_host_input_bytes += int(densities.nbytes)
    mf_grad.gaussian_hcore_host_result_bytes += int(total.nbytes)


def grad_hcore_energy(mf_grad, density, atmlst, name):
    r'''Contract the source-consistent GPU Hcore derivative with a density.

    GPU4PySCF's FFTDF KRHF source splits the local GTH derivative in two.
    ``eval_vpplocG_SI_gradient`` differentiates the nuclear structure
    factor, while ``get_veff_ip1(..., with_j=False)`` restores the AO-orbital
    derivative of the same local potential.  Both pieces are required for
    the complete derivative of ``Tr[P h_core]``.  The kinetic AO-row term is
    added explicitly.  The nonlocal GTH projector half integrals are the
    upstream PySCF host primitive for non-Gamma k-points; their contraction
    is performed on the device and only the final ``(natm,3)`` result crosses
    to the host inside GPU4PySCF.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.krhf_dft.Gradients
            Active sampled-k gradient object.
        density : cupy.ndarray of shape (nkpts,nao,nao)
            Hermitian AO density held fixed during the Hcore derivative.
        atmlst : sequence of int
            Requested nuclear centers.
        name : str
            Term-ledger label for imaginary-residual diagnostics.

    Returns:
        cupy.ndarray of shape (len(atmlst),3)
            Complete Hcore energy derivative in Ha/Bohr.
    '''
    mf = mf_grad.base
    mf_hf = mf.mf_hf
    cell = mf.cell
    kpts = mf.kpts
    nkpts = len(kpts)
    density = cp.asarray(density, order='C')
    if isinstance(mf.with_df, GDF):
        if mf_grad.gaussian_hcore_pair is None:
            raise RuntimeError(
                'prepare the sampled-k Gaussian Hcore density pair first')
        if name == 'de_hcore':
            value = mf_grad.gaussian_hcore_pair[0]
        elif name == 'de_fock_hcore_density':
            value = mf_grad.gaussian_hcore_pair[1]
        else:
            raise ValueError(f'unknown Gaussian Hcore contraction {name}')
        return real_gradient_term(mf_grad, value, name)

    ni = mf_hf._numint
    if not isinstance(ni, multigrid_v2.MultiGridNumInt):
        raise NotImplementedError(
            'K8.2 FFTDF Hcore requires GPU4PySCF MultiGridNumInt source')

    rho_g = multigrid_v2.evaluate_density_on_g_mesh(
        ni, density, kpts)[0, 0]
    local = multigrid.eval_vpplocG_SI_gradient(
        cell, ni.mesh, rho_g)
    local_orbital = multigrid_v2.get_veff_ip1(
        ni, 'HF', density, kpts=kpts, with_j=False,
        with_pseudo_vloc_orbital_derivative=True) / nkpts

    hkin = int1e.int1e_ipkin(cell, kpts)
    kinetic = cp.empty((cell.natm, 3), dtype=cp.float64)
    for ia, aoslice in enumerate(cell.aoslice_by_atom()):
        p0, p1 = aoslice[2:]
        kinetic[ia] = -2 * cp.einsum(
            'kxij,kji->x', hkin[:, :, p0:p1],
            density[:, :, p0:p1]).real / nkpts

    nonlocal_host = np.asarray(vppnl_nuc_grad(cell, density, kpts=kpts))
    mf_grad.nonlocal_pseudo_host_result_bytes += int(nonlocal_host.nbytes)
    nonlocal_term = cp.asarray(nonlocal_host) / nkpts
    value = cp.asarray(local) + local_orbital + kinetic + nonlocal_term
    return real_gradient_term(mf_grad, value[atmlst], name)


def grad_j_energy(mf_grad, dm0, vj1, atmlst):
    r'''Return :math:`(2N_k)^{-1}\sum_k Tr[P_k J_k^\lambda]`.'''
    nkpts = len(mf_grad.base.kpts)
    value = .5 * cp.einsum(
        'akxij,kji->ax', vj1[atmlst], dm0) / nkpts
    return real_gradient_term(mf_grad, value, 'de_j')


def grad_vxc(mf_grad, dm0, atmlst):
    '''Return native GPU uniform-grid or XC-only multigrid derivative.'''
    mf = mf_grad.base
    if isinstance(mf._numint, multigrid_v2.MultiGridNumInt):
        if mf.multigrid_with_j:
            raise ValueError(
                'grad_vxc cannot split a fused multigrid J+XC derivative; '
                'use skeleton_gradient')
        return get_veff_deriv(
            mf_grad, dm0, kpts=mf.kpts)[atmlst]
    grids = mf.grids
    if grids.coords is None:
        grids.build()
    host = np.asarray(gpu_krks_grad.get_vxc(
        mf._numint, mf.cell, grids, mf.xc, dm0, mf.kpts)) * 2
    mf_grad.xc_derivative_host_result_bytes += int(host.nbytes)
    mf_grad.xc_deriv_backend = 'gpu4pyscf_uniform_grid'
    return cp.asarray(host[atmlst])


def grad_ovlp(mf_grad, Co, foo, s1, atmlst):
    r'''Return the target-Fock overlap/Pulay term at sampled k-points.'''
    Co = cp.asarray(Co)
    foo = cp.asarray(foo)
    nkpts = len(Co)
    dme0 = cp.asarray([
        2 * Co_k @ foo_k @ Co_k.conj().T
        for Co_k, foo_k in zip(Co, foo, strict=True)])
    output = []
    aoslices = mf_grad.cell.aoslice_by_atom()
    for ia in atmlst:
        p0, p1 = aoslices[ia, 2:]
        output.append(-2 * cp.einsum(
            'kxij,kji->x', s1[:, :, p0:p1],
            dme0[:, :, p0:p1]) / nkpts)
    return real_gradient_term(
        mf_grad, cp.asarray(output), 'de_ovlp', expect_real=False)


def skeleton_gradient(
        mf_grad, dm0, foo, Co, s1, vj1, atmlst):
    '''Assemble explicit GPU sampled-k semilocal-DFA@KRHF skeleton terms.'''
    de_hcore = grad_hcore_energy(
        mf_grad, dm0, atmlst, 'de_hcore')
    mf = mf_grad.base
    if (isinstance(mf._numint, multigrid_v2.MultiGridNumInt)
            and mf.multigrid_with_j):
        de_veff = get_veff_deriv(
            mf_grad, dm0, kpts=mf.kpts)[atmlst]
        de_j = None
        de_vxc = None
        mf_grad.de_j_xc_multigrid = de_veff
    else:
        if vj1 is None:
            if not isinstance(mf.with_df, GDF):
                raise ValueError('a non-GDF skeleton requires vj1')
            de_j = cp.asarray(mf_grad.de_j)
        else:
            de_j = grad_j_energy(mf_grad, dm0, vj1, atmlst)
        de_vxc = grad_vxc(mf_grad, dm0, atmlst)
        de_veff = de_j + de_vxc
        mf_grad.de_j_xc_multigrid = None
    de_ovlp = grad_ovlp(mf_grad, Co, foo, s1, atmlst)
    mf_grad.de_hcore = de_hcore
    mf_grad.de_j = de_j
    mf_grad.de_vxc = de_vxc
    mf_grad.de_veff = de_veff
    mf_grad.de_ovlp = de_ovlp
    mf_grad.de_grid = cp.zeros_like(de_hcore)
    mf_grad.de_extra = mf_grad.de_grid
    mf_grad.de_skel_elec = de_hcore + de_veff + de_ovlp
    return mf_grad.de_skel_elec
