# Copyright 2021-2026 The GPU4PySCF Developers.
# Modifications Copyright 2026 Sehun.
# SPDX-License-Identifier: Apache-2.0

"""Development GPU Fourier-source contractions for analytic stress.

The low-level kernel calls follow GPU4PySCF's Apache-2.0 AFT stress driver,
but expose only the AO-pair strain contraction needed to validate a future
GDF/RSDF reverse-source path.  No Coulomb-kernel derivative is included here.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from time import perf_counter

import cupy as cp
import numpy as np

from gpu4pyscf.lib.cupy_helper import get_avail_mem
from gpu4pyscf.pbc.df.aft_jk import (
    _estimate_max_shm_size,
    _generate_shl_pairs,
)
from gpu4pyscf.pbc.df.ft_ao import FTOpt, libpbc
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from pyscf import lib

from pyscf_pbc_dcdft_stress.gdf_metric_strain import matrix_tensor_to_voigt
from pyscf_pbc_dcdft_stress.gdf_source_strain import (
    _contract_vbar_source_adjoint_strain,
    auxiliary_fourier_strain,
    kpoint_short_range_source_strain,
    weighted_coulomb_strain,
)


_AUTO_AO_PAIR_WORK_BUDGET_BYTES = 2 << 30
_AUTO_GRID_BLOCK_ALIGNMENT = 16
_AUTO_GRID_BLOCK_MAX = 16384


@dataclass(frozen=True)
class _GammaFTSourceContext:
    ft_opt: object
    sorted_cell: object
    pair_indices: cp.ndarray
    image_indices: cp.ndarray
    shell_pair_offsets: cp.ndarray
    shell_pair_batches: int
    shared_memory: int
    aft_envs: object
    ft_kernel: object


def _sampled_k_ft_source_context(cell, kpts):
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    ft_opt = FTOpt(cell, kpts_to_kmesh(cell, kpts))
    ft_opt.permutation_symmetry = False
    ft_opt.build()
    sorted_cell = ft_opt.cell
    pair_indices, image_indices, shell_pair_offsets = _generate_shl_pairs(
        ft_opt
    )
    return _GammaFTSourceContext(
        ft_opt=ft_opt,
        sorted_cell=sorted_cell,
        pair_indices=pair_indices,
        image_indices=image_indices,
        shell_pair_offsets=shell_pair_offsets,
        shell_pair_batches=len(shell_pair_offsets)-1,
        shared_memory=_estimate_max_shm_size(sorted_cell, (1, 0)),
        aft_envs=ft_opt.aft_envs,
        ft_kernel=ft_opt.gen_ft_kernel(
            transform_ao=True, kpts=kpts
        ),
    )


@dataclass(frozen=True)
class GammaLongRangeSourceAdjointResult:
    values: np.ndarray
    reciprocal_block_size: int
    grid_points: int
    estimated_block_work_bytes: int
    wall_seconds: float


@dataclass(frozen=True)
class GammaThreeCenterSourceAdjointResult:
    values: np.ndarray
    reciprocal_block_size: int
    reciprocal_grid_points: int
    estimated_block_work_bytes: int
    timings_s: dict[str, float]
    memory_estimate: dict[str, int]


def _gamma_ft_source_context(cell):
    ft_opt = FTOpt(cell, np.ones(3, dtype=int)).build()
    sorted_cell = ft_opt.cell
    pair_indices, image_indices, shell_pair_offsets = _generate_shl_pairs(
        ft_opt
    )
    return _GammaFTSourceContext(
        ft_opt=ft_opt,
        sorted_cell=sorted_cell,
        pair_indices=pair_indices,
        image_indices=image_indices,
        shell_pair_offsets=shell_pair_offsets,
        shell_pair_batches=len(shell_pair_offsets)-1,
        shared_memory=_estimate_max_shm_size(sorted_cell, (1, 0)),
        aft_envs=ft_opt.aft_envs,
        ft_kernel=ft_opt.gen_ft_kernel(
            transform_ao=True, kpts=np.zeros((1, 3))
        ),
    )


def _automatic_block_size(sorted_nao, ngrid, available_bytes):
    """Bound the two dominant complex AO-pair work arrays for one G block."""

    sorted_nao = int(sorted_nao)
    ngrid = int(ngrid)
    available_bytes = int(available_bytes)
    if sorted_nao < 1 or ngrid < 1:
        raise ValueError("sorted_nao and ngrid must be positive")
    complex_bytes = np.dtype(np.complex128).itemsize
    bytes_per_grid = 2*sorted_nao*sorted_nao*complex_bytes
    budget = min(
        max(0, available_bytes//2),
        _AUTO_AO_PAIR_WORK_BUDGET_BYTES,
    )
    raw = budget//bytes_per_grid
    aligned = max(
        _AUTO_GRID_BLOCK_ALIGNMENT,
        raw//_AUTO_GRID_BLOCK_ALIGNMENT*_AUTO_GRID_BLOCK_ALIGNMENT,
    )
    return min(aligned, ngrid, _AUTO_GRID_BLOCK_MAX)


def _resolve_block_size(context, ngrid, requested):
    if requested is not None:
        resolved = int(requested)
        if resolved < 1:
            raise ValueError("block_size must be positive")
        return min(resolved, int(ngrid))
    return _automatic_block_size(
        context.sorted_cell.nao,
        ngrid,
        get_avail_mem(exclude_memory_pool=True),
    )


def _accumulate_general_strain(context, weight_gij, vectors, strain):
    """Accumulate one G block of a general AO-pair weight on device."""

    # The source weight multiplies I_ij, while apply_C_mat_CT and the kernel
    # consume density-like (j,i) coefficients.
    density_like = weight_gij.transpose(0, 2, 1)
    sorted_density_like = context.sorted_cell.apply_C_mat_CT(density_like)
    sorted_density_like = cp.asarray(
        sorted_density_like.transpose(1, 2, 0), order="C"
    )
    vectors_device = cp.asarray(np.asarray(vectors).T.ravel())
    atom_gradient = cp.zeros((context.sorted_cell.natm, 3))
    error = libpbc.PBC_ft_aopair_ek_strain_deriv(
        ctypes.cast(atom_gradient.data.ptr, ctypes.c_void_p),
        ctypes.cast(strain.data.ptr, ctypes.c_void_p),
        ctypes.cast(sorted_density_like.data.ptr, ctypes.c_void_p),
        ctypes.cast(vectors_device.data.ptr, ctypes.c_void_p),
        ctypes.byref(context.aft_envs),
        ctypes.c_int(context.shell_pair_batches),
        ctypes.c_int(len(vectors)),
        ctypes.c_int(context.shared_memory),
        ctypes.cast(context.pair_indices.data.ptr, ctypes.c_void_p),
        ctypes.cast(context.image_indices.data.ptr, ctypes.c_void_p),
        ctypes.cast(context.shell_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.c_int(int(context.ft_opt.permutation_symmetry)),
    )
    if error != 0:
        raise RuntimeError("PBC_ft_aopair_ek_strain_deriv failed")


def _accumulate_sampled_k_general_strain(context, weight_gij, momentum,
                                         kptj, strain):
    """Accumulate one ordered sampled-k AO-pair strain block."""

    density_like = weight_gij.transpose(0, 2, 1)
    sorted_density_like = context.sorted_cell.apply_C_mat_CT(density_like)
    phase = cp.exp(
        1j*cp.asarray(context.ft_opt.bvkmesh_Ls).dot(cp.asarray(kptj))
    )
    bvk_weight = cp.asarray(
        phase[:, None, None, None]
        * sorted_density_like[None].transpose(0, 2, 3, 1),
        order="C",
    )
    vectors_device = cp.asarray(np.asarray(momentum).T.ravel())
    atom_gradient = cp.zeros((context.sorted_cell.natm, 3))
    error = libpbc.PBC_ft_aopair_ek_strain_deriv(
        ctypes.cast(atom_gradient.data.ptr, ctypes.c_void_p),
        ctypes.cast(strain.data.ptr, ctypes.c_void_p),
        ctypes.cast(bvk_weight.data.ptr, ctypes.c_void_p),
        ctypes.cast(vectors_device.data.ptr, ctypes.c_void_p),
        ctypes.byref(context.aft_envs),
        ctypes.c_int(context.shell_pair_batches),
        ctypes.c_int(len(momentum)),
        ctypes.c_int(context.shared_memory),
        ctypes.cast(context.pair_indices.data.ptr, ctypes.c_void_p),
        ctypes.cast(context.image_indices.data.ptr, ctypes.c_void_p),
        ctypes.cast(context.shell_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.c_int(int(context.ft_opt.permutation_symmetry)),
    )
    if error != 0:
        raise RuntimeError("PBC_ft_aopair_ek_strain_deriv failed")


def gamma_separable_ao_pair_strain(cell, density, reciprocal_weight, grid_vectors,
                                    *, block_size=None):
    """Contract Gamma AO-pair strain for a separable ``D(ij) v(G)`` weight.

    This is a low-level exact prototype.  It deliberately excludes auxiliary
    and Coulomb-kernel strain derivatives, which remain owned by the existing
    GDF/RSDF source implementation.
    """

    density = np.asarray(density)
    nao = int(cell.nao_nr())
    if density.shape != (nao, nao):
        raise ValueError(
            f"density must have shape {(nao, nao)}, got {density.shape}"
        )
    grid_vectors = np.asarray(grid_vectors, dtype=float)
    if grid_vectors.ndim != 2 or grid_vectors.shape[1] != 3:
        raise ValueError(
            "grid_vectors must have shape (nG,3), got "
            f"{grid_vectors.shape}"
        )
    reciprocal_weight = np.asarray(reciprocal_weight)
    if reciprocal_weight.shape != (len(grid_vectors),):
        raise ValueError(
            "reciprocal_weight must have shape "
            f"{(len(grid_vectors),)}, got {reciprocal_weight.shape}"
        )
    if np.max(np.abs(density.imag)) > 1e-12:
        raise ValueError("Gamma prototype currently requires a real density")

    ft_opt = FTOpt(cell, np.ones(3, dtype=int)).build()
    sorted_cell = ft_opt.cell
    sorted_density = sorted_cell.apply_C_mat_CT(
        cp.asarray(density.real[None])
    )
    density_bvk = cp.asarray(sorted_density.real, order="C")
    sorted_nao = int(sorted_cell.nao)
    ngrid = len(grid_vectors)
    if block_size is None:
        available = int(0.5*get_avail_mem(exclude_memory_pool=True))
        estimated_per_grid = max(1, 2*sorted_nao*sorted_nao*16)
        block_size = max(16, available//estimated_per_grid//16*16)
        block_size = min(block_size, ngrid, 16384)
    else:
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be positive")

    pair_indices, image_indices, shell_pair_offsets = _generate_shl_pairs(
        ft_opt
    )
    shell_pair_batches = len(shell_pair_offsets)-1
    shared_memory = _estimate_max_shm_size(sorted_cell, (1, 0))
    kernel = libpbc.PBC_ft_aopair_ej_strain_deriv
    atom_gradient = cp.zeros((sorted_cell.natm, 3))
    strain = cp.zeros((3, 3))
    aft_envs = ft_opt.aft_envs

    for start, stop in lib.prange(0, ngrid, block_size):
        vectors = cp.asarray(grid_vectors[start:stop].T.ravel())
        weights = cp.asarray(reciprocal_weight[start:stop])
        error = kernel(
            ctypes.cast(atom_gradient.data.ptr, ctypes.c_void_p),
            ctypes.cast(strain.data.ptr, ctypes.c_void_p),
            ctypes.cast(density_bvk.data.ptr, ctypes.c_void_p),
            ctypes.cast(weights.data.ptr, ctypes.c_void_p),
            ctypes.cast(vectors.data.ptr, ctypes.c_void_p),
            ctypes.byref(aft_envs),
            ctypes.c_int(shell_pair_batches),
            ctypes.c_int(stop-start),
            ctypes.c_int(shared_memory),
            ctypes.cast(pair_indices.data.ptr, ctypes.c_void_p),
            ctypes.cast(image_indices.data.ptr, ctypes.c_void_p),
            ctypes.cast(shell_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.c_int(int(ft_opt.permutation_symmetry)),
        )
        if error != 0:
            raise RuntimeError("PBC_ft_aopair_ej_strain_deriv failed")

    # The kernel visits one triangular shell-pair representative when
    # permutation symmetry is enabled.  This is the same normalization used
    # by GPU4PySCF's get_ej_strain_deriv driver.
    strain *= 2.0
    return matrix_tensor_to_voigt(cp.asnumpy(strain))


def gamma_general_ao_pair_strain(cell, ao_grid_weight, grid_vectors, *,
                                 block_size=None):
    """Contract Gamma AO-pair strain with a general ``T(ij,G)`` weight.

    ``ao_grid_weight[i,j,g]`` is the coefficient multiplying the derivative
    of the Fourier AO pair ``I_ij(G)``.  The GPU4PySCF exchange-strain kernel
    accepts the equivalent G-dependent density layout after AO sorting.
    """

    nao = int(cell.nao_nr())
    grid_vectors = np.asarray(grid_vectors, dtype=float)
    if grid_vectors.ndim != 2 or grid_vectors.shape[1] != 3:
        raise ValueError(
            "grid_vectors must have shape (nG,3), got "
            f"{grid_vectors.shape}"
        )
    ao_grid_weight = np.asarray(ao_grid_weight)
    expected = (nao, nao, len(grid_vectors))
    if ao_grid_weight.shape != expected:
        raise ValueError(
            f"ao_grid_weight must have shape {expected}, "
            f"got {ao_grid_weight.shape}"
        )

    context = _gamma_ft_source_context(cell)
    ngrid = len(grid_vectors)
    block_size = _resolve_block_size(context, ngrid, block_size)
    strain = cp.zeros((3, 3))

    for start, stop in lib.prange(0, ngrid, block_size):
        block_weight = cp.asarray(
            ao_grid_weight[:, :, start:stop].transpose(2, 0, 1)
        )
        _accumulate_general_strain(
            context, block_weight, grid_vectors[start:stop], strain
        )

    strain *= 2.0
    return matrix_tensor_to_voigt(cp.asnumpy(strain))


def kpoint_general_ao_pair_strain(cell, ao_grid_weight, grid_vectors, kpts,
                                  kpti, kptj, *, block_size=None):
    """Contract one ordered sampled-k AO-pair strain with ``T(ij,G)``."""

    nao = int(cell.nao_nr())
    grid_vectors = np.asarray(grid_vectors, dtype=float)
    ao_grid_weight = np.asarray(ao_grid_weight)
    expected = (nao, nao, len(grid_vectors))
    if ao_grid_weight.shape != expected:
        raise ValueError(
            f"ao_grid_weight must have shape {expected}, "
            f"got {ao_grid_weight.shape}"
        )
    kpti = np.asarray(kpti, dtype=float)
    kptj = np.asarray(kptj, dtype=float)
    transfer = kptj-kpti
    context = _sampled_k_ft_source_context(cell, kpts)
    block_size = _resolve_block_size(context, len(grid_vectors), block_size)
    strain = cp.zeros((3, 3))
    for start, stop in lib.prange(0, len(grid_vectors), block_size):
        block_weight = cp.asarray(
            ao_grid_weight[:, :, start:stop].transpose(2, 0, 1)
        )
        _accumulate_sampled_k_general_strain(
            context, block_weight, grid_vectors[start:stop]+transfer,
            kptj, strain,
        )
    return matrix_tensor_to_voigt(cp.asnumpy(strain))


def gamma_long_range_source_adjoint_strain(cell, auxcell, omega, mesh,
                                           source_adjoints, *, block_size=None):
    """Prototype complete Gamma reciprocal source-adjoint contraction.

    AO-pair values and their strain contraction run through GPU4PySCF FTOpt.
    Auxiliary-function and Coulomb-kernel strain factors retain the established
    host formulas in this validation increment.
    """

    started = perf_counter()
    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError(f"mesh must contain three positive integers: {mesh}")
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    adjoints = np.asarray(source_adjoints)
    if adjoints.ndim == 2:
        adjoints = adjoints[None]
    expected = (nao*nao, naux)
    if adjoints.ndim != 3 or tuple(adjoints.shape[1:]) != expected:
        raise ValueError(
            "source_adjoints must have shape (nsource,nao**2,naux); "
            f"expected (*,{expected[0]},{expected[1]}), got {adjoints.shape}"
        )

    vectors = cell.get_Gv(mesh)
    context = _gamma_ft_source_context(cell)
    block_size = _resolve_block_size(context, len(vectors), block_size)
    adjoints_device = cp.asarray(adjoints)
    scalar_result = cp.zeros((len(adjoints), 6), dtype=cp.complex128)
    pair_strain = [cp.zeros((3, 3)) for _ in adjoints]

    for start, stop in lib.prange(0, len(vectors), block_size):
        block_vectors = vectors[start:stop]
        auxiliary = auxiliary_fourier_strain(auxcell, block_vectors)
        kernel = weighted_coulomb_strain(
            cell, block_vectors, omega=float(omega)
        )
        weighted_auxiliary = (auxiliary.value*kernel.value).T
        weighted_device = cp.asarray(weighted_auxiliary)
        response_weight = cp.einsum(
            "gp,sqp->sgq", weighted_device.conj(), adjoints_device,
            optimize=True,
        )
        for source in range(len(adjoints)):
            _accumulate_general_strain(
                context,
                response_weight[source].reshape(-1, nao, nao),
                block_vectors, pair_strain[source],
            )

        pair_value = context.ft_kernel(block_vectors)[0]
        pair_flat = pair_value.reshape(len(block_vectors), nao*nao)
        pair_values = cp.einsum(
            "gq,sqp->sgp", pair_flat, adjoints_device, optimize=True
        )
        auxiliary_derivative = matrix_tensor_to_voigt(
            auxiliary.raw_derivative
        )
        kernel_derivative = matrix_tensor_to_voigt(kernel.raw_derivative)
        derivative_weight = cp.asarray(
            (
                auxiliary_derivative*kernel.value[None, None, :]
                + auxiliary.value[None, :, :]
                * kernel_derivative[:, None, :]
            ).transpose(0, 2, 1).conj()
        )
        scalar_result += cp.einsum(
            "sgp,xgp->sx", pair_values, derivative_weight, optimize=True
        )

    pair_values_voigt = np.stack([
        matrix_tensor_to_voigt(cp.asnumpy(value*2.0))
        for value in pair_strain
    ])
    values = np.real(pair_values_voigt+cp.asnumpy(scalar_result))
    complex_bytes = np.dtype(np.complex128).itemsize
    estimated_block_bytes = int(complex_bytes*block_size*(
        (2+2*len(adjoints))*nao*nao
        + (6+len(adjoints))*naux
    ))
    return GammaLongRangeSourceAdjointResult(
        values=values,
        reciprocal_block_size=int(block_size),
        grid_points=int(len(vectors)),
        estimated_block_work_bytes=estimated_block_bytes,
        wall_seconds=float(perf_counter()-started),
    )


def kpoint_long_range_source_adjoint_strain(cell, auxcell, omega, mesh, kpts,
                                            kpti, kptj, source_adjoints, *,
                                            block_size=None):
    """Complete reciprocal source-adjoint contraction for one sampled-k pair."""

    started = perf_counter()
    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError(f"mesh must contain three positive integers: {mesh}")
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    kpti = np.asarray(kpti, dtype=float)
    kptj = np.asarray(kptj, dtype=float)
    transfer = kptj-kpti
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    adjoints = np.asarray(source_adjoints)
    if adjoints.ndim == 2:
        adjoints = adjoints[None]
    expected = (nao*nao, naux)
    if adjoints.ndim != 3 or tuple(adjoints.shape[1:]) != expected:
        raise ValueError(
            "source_adjoints must have shape (nsource,nao**2,naux); "
            f"expected (*,{expected[0]},{expected[1]}), got {adjoints.shape}"
        )

    vectors = cell.get_Gv(mesh)
    context = _sampled_k_ft_source_context(cell, kpts)
    block_size = _resolve_block_size(context, len(vectors), block_size)
    adjoints_device = cp.asarray(adjoints)
    scalar_result = cp.zeros((len(adjoints), 6), dtype=cp.complex128)
    pair_strain = [cp.zeros((3, 3)) for _ in adjoints]

    for start, stop in lib.prange(0, len(vectors), block_size):
        block_vectors = vectors[start:stop]
        momentum = block_vectors+transfer
        auxiliary = auxiliary_fourier_strain(auxcell, momentum)
        kernel = weighted_coulomb_strain(
            cell, momentum, omega=float(omega)
        )
        weighted_auxiliary = (auxiliary.value*kernel.value).T
        weighted_device = cp.asarray(weighted_auxiliary)
        response_weight = cp.einsum(
            "gp,sqp->sgq", weighted_device.conj(), adjoints_device,
            optimize=True,
        )
        for source in range(len(adjoints)):
            _accumulate_sampled_k_general_strain(
                context,
                response_weight[source].reshape(-1, nao, nao),
                momentum, kptj, pair_strain[source],
            )

        pair_value = context.ft_kernel(
            block_vectors, q=transfer, kpts=kptj.reshape(1, 3)
        )[0]
        pair_flat = pair_value.reshape(len(block_vectors), nao*nao)
        pair_values = cp.einsum(
            "gq,sqp->sgp", pair_flat, adjoints_device, optimize=True
        )
        auxiliary_derivative = matrix_tensor_to_voigt(
            auxiliary.raw_derivative
        )
        kernel_derivative = matrix_tensor_to_voigt(kernel.raw_derivative)
        derivative_weight = cp.asarray(
            (
                auxiliary_derivative*kernel.value[None, None, :]
                + auxiliary.value[None, :, :]
                * kernel_derivative[:, None, :]
            ).transpose(0, 2, 1).conj()
        )
        scalar_result += cp.einsum(
            "sgp,xgp->sx", pair_values, derivative_weight, optimize=True
        )

    pair_values_voigt = np.stack([
        matrix_tensor_to_voigt(cp.asnumpy(value)) for value in pair_strain
    ])
    values = np.real(pair_values_voigt+cp.asnumpy(scalar_result))
    complex_bytes = np.dtype(np.complex128).itemsize
    bvk_cells = int(np.prod(context.ft_opt.bvk_kmesh))
    estimated_block_bytes = int(complex_bytes*block_size*(
        (2+bvk_cells+2*len(adjoints))*nao*nao
        + (6+len(adjoints))*naux
    ))
    return GammaLongRangeSourceAdjointResult(
        values=values,
        reciprocal_block_size=int(block_size),
        grid_points=int(len(vectors)),
        estimated_block_work_bytes=estimated_block_bytes,
        wall_seconds=float(perf_counter()-started),
    )


def gamma_three_center_source_adjoint_strain(cell, auxcell, omega, mesh,
                                             source_adjoints, *, precision=None,
                                             block_size=None):
    """Complete Gamma source-adjoint strain with GPU reciprocal AO pairs."""

    adjoints = np.asarray(source_adjoints)
    if adjoints.ndim == 2:
        adjoints = adjoints[None]
    started = perf_counter()
    short_range = kpoint_short_range_source_strain(
        cell, auxcell, omega, np.zeros(3), np.zeros(3), precision=precision
    )
    short_seconds = perf_counter()-started
    short_voigt = matrix_tensor_to_voigt(short_range.raw_derivative)
    values = np.real(np.einsum(
        "sqp,xqp->sx", adjoints, short_voigt, optimize=True
    ))

    reciprocal = gamma_long_range_source_adjoint_strain(
        cell, auxcell, omega, mesh, adjoints, block_size=block_size
    )
    values += reciprocal.values
    started = perf_counter()
    values += _contract_vbar_source_adjoint_strain(
        cell, auxcell, omega, np.zeros(3), np.zeros(3), adjoints
    )
    vbar_seconds = perf_counter()-started
    short_bytes = int(
        np.asarray(short_range.raw_derivative).nbytes
        + np.asarray(short_range.value).nbytes
    )
    return GammaThreeCenterSourceAdjointResult(
        values=values,
        reciprocal_block_size=reciprocal.reciprocal_block_size,
        reciprocal_grid_points=reciprocal.grid_points,
        estimated_block_work_bytes=reciprocal.estimated_block_work_bytes,
        timings_s={
            "short_range_full_tensor": float(short_seconds),
            "reciprocal_direct_adjoint": float(reciprocal.wall_seconds),
            "vbar_direct_adjoint": float(vbar_seconds),
            "total": float(
                short_seconds+reciprocal.wall_seconds+vbar_seconds
            ),
        },
        memory_estimate={
            "source_adjoint_bytes": int(adjoints.nbytes),
            "short_range_tensor_bytes": short_bytes,
            "reciprocal_full_response_avoided_bytes": int(
                6*cell.nao_nr()**2*auxcell.nao_nr()
                * np.dtype(np.complex128).itemsize
            ),
            "retained_source_derivative_bytes": 0,
        },
    )


def kpoint_three_center_source_adjoint_strain(cell, auxcell, omega, mesh, kpts,
                                              kpti, kptj, source_adjoints, *,
                                              precision=None, block_size=None):
    """Complete sampled-k source-adjoint strain with GPU reciprocal AO pairs."""

    adjoints = np.asarray(source_adjoints)
    if adjoints.ndim == 2:
        adjoints = adjoints[None]
    started = perf_counter()
    short_range = kpoint_short_range_source_strain(
        cell, auxcell, omega, kpti, kptj, precision=precision
    )
    short_seconds = perf_counter()-started
    values = np.real(np.einsum(
        "sqp,xqp->sx", adjoints,
        matrix_tensor_to_voigt(short_range.raw_derivative), optimize=True,
    ))
    reciprocal = kpoint_long_range_source_adjoint_strain(
        cell, auxcell, omega, mesh, kpts, kpti, kptj, adjoints,
        block_size=block_size,
    )
    values += reciprocal.values
    started = perf_counter()
    values += _contract_vbar_source_adjoint_strain(
        cell, auxcell, omega, kpti, kptj, adjoints
    )
    vbar_seconds = perf_counter()-started
    short_bytes = int(
        np.asarray(short_range.raw_derivative).nbytes
        + np.asarray(short_range.value).nbytes
    )
    return GammaThreeCenterSourceAdjointResult(
        values=values,
        reciprocal_block_size=reciprocal.reciprocal_block_size,
        reciprocal_grid_points=reciprocal.grid_points,
        estimated_block_work_bytes=reciprocal.estimated_block_work_bytes,
        timings_s={
            "short_range_full_tensor": float(short_seconds),
            "reciprocal_direct_adjoint": float(reciprocal.wall_seconds),
            "vbar_direct_adjoint": float(vbar_seconds),
            "total": float(
                short_seconds+reciprocal.wall_seconds+vbar_seconds
            ),
        },
        memory_estimate={
            "source_adjoint_bytes": int(adjoints.nbytes),
            "short_range_tensor_bytes": short_bytes,
            "reciprocal_full_response_avoided_bytes": int(
                6*cell.nao_nr()**2*auxcell.nao_nr()
                * np.dtype(np.complex128).itemsize
            ),
            "retained_source_derivative_bytes": 0,
        },
    )
