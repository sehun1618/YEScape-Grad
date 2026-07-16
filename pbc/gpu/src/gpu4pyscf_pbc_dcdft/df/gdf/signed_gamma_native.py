# Copyright 2025 The PySCF Developers. All Rights Reserved.
# Modifications Copyright 2026 Sehun.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from GPU4PySCF's periodic GDF RHF derivative implementation. This
# version fuses signed density states for the DC-DFT response contraction.

"""One-pass Gamma GDF derivative for a signed pair of RHF densities.

This module follows the native GPU4PySCF periodic GDF decomposition, but
combines the quadratic sources for ``D + T`` and ``D - T`` before applying
the nuclear derivative kernels.  The expensive SR, LR, metric, auxiliary
centre, and Ewald derivative traversals are therefore each executed once.

The route is deliberately private and Gamma-only.  Sampled k-points require
momentum-pair bookkeeping and must pass their own parity gate before this
implementation can replace the authoritative two-pass production route.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass

import cupy as cp
import numpy as np
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.df.grad.rhf import factorize_dm
from gpu4pyscf.gto.mole import groupby
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    asarray,
    contract,
    empty_aligned,
    get_avail_mem,
    ndarray,
)
from gpu4pyscf.pbc.df import aft_jk, ft_ao
from gpu4pyscf.pbc.df.grad.rhf import (
    _split_l_ctr_pattern,
    get_ao_pair_loc,
    int3c2e_scheme,
)
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.df.int3c2e import MAX_IMGS_PER_TASK, POOL_SIZE, libpbc
from gpu4pyscf.pbc.df.rsdf_builder import _weighted_coulG_kpts
from gpu4pyscf.pbc.grad.rhf import contract_h1e_dm
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.tools.pbc import madelung
from pyscf import lib
from pyscf.gto import ATOM_OF


@dataclass
class _QuadraticState:
    """Zero-order factors for one signed quadratic density."""

    coulomb_weight: float
    exchange_weight: float
    factor_l: cp.ndarray
    factor_r: cp.ndarray
    original_density: cp.ndarray
    density: cp.ndarray
    j3c_oo: cp.ndarray | None = None
    auxvec: cp.ndarray | None = None

    @property
    def nocc(self) -> int:
        return int(self.factor_l.shape[1])


def gamma_signed_energy_derivative(
    int3c2e_opt,
    reference_density,
    response_density,
    *,
    exxdiv="ewald",
    omega=0.0,
    verbose=None,
):
    """Return ``Q(D+T)-Q(D-T)`` with one native derivative traversal.

    ``Q`` is the fixed-density restricted ``J-K/2`` nuclear derivative.  The
    function consumes the same built :class:`SRInt3c2eOpt` used by the
    upstream Gamma GDF gradient, and returns ``(natm, 3)`` on the host.
    """

    reference = _real_symmetric_density(reference_density, "reference_density")
    response = _real_symmetric_density(response_density, "response_density")
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )

    cell = int3c2e_opt.cell
    states = _signed_states(cell, reference, response)
    return _weighted_energy_derivative(
        int3c2e_opt,
        states,
        exxdiv=exxdiv,
        omega=omega,
        verbose=verbose,
    )


def gamma_target_j_plus_signed_energy_derivative(
    int3c2e_opt,
    reference_density,
    response_density,
    *,
    exxdiv="ewald",
    omega=0.0,
    verbose=None,
):
    """Return target-J plus the RHF bilinear derivative in one traversal.

    The first weighted state contributes only the quadratic Coulomb
    derivative at ``D``.  The remaining two states are the exact signed
    polarization pair ``D+T`` and ``D-T`` for the restricted HF kernel.
    Their derivative sources are added before every SR, LR, metric and Ewald
    kernel call, so the expensive integral traversal is executed once.
    """

    reference = _real_symmetric_density(reference_density, "reference_density")
    response = _real_symmetric_density(response_density, "response_density")
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    cell = int3c2e_opt.cell
    states = [
        _prepare_state(
            cell,
            reference,
            coulomb_weight=1.0,
            exchange_weight=0.0,
        ),
        *_signed_states(cell, reference, response),
    ]
    return _weighted_energy_derivative(
        int3c2e_opt,
        states,
        exxdiv=exxdiv,
        omega=omega,
        verbose=verbose,
    )


def _signed_states(cell, reference, response):
    return [
        _prepare_state(
            cell,
            reference + response,
            coulomb_weight=+1.0,
            exchange_weight=+1.0,
        ),
        _prepare_state(
            cell,
            reference - response,
            coulomb_weight=-1.0,
            exchange_weight=-1.0,
        ),
    ]


def _weighted_energy_derivative(
    int3c2e_opt,
    states,
    *,
    exxdiv,
    omega,
    verbose,
):
    """Apply one native derivative traversal to weighted quadratic states."""

    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    reference = states[0].original_density
    nao = int(reference.shape[0])
    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False
    )[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    aux_loc = auxcell.ao_loc
    naux = int(aux_loc[-1])
    log = logger.new_logger(cell, verbose)

    batch_size, block_size = _working_block_sizes(
        states,
        nao_pair=nao_pair,
        naux=naux,
        bvk_ncells=len(int3c2e_opt.bvkmesh_Ls),
    )
    _build_short_range_zero_order(
        int3c2e_opt,
        states,
        pair_addresses,
        i_addr,
        j_addr,
        batch_size=batch_size,
        block_size=block_size,
    )

    precision = auxcell.precision * 1.0e-6
    auxcell.rcut = _estimate_sr_2c2e_rcut(
        auxcell, int3c2e_opt.omega, precision
    )
    int2c2e_opt = Int2c2eOpt(auxcell)
    j2c = int2c2e_opt.int2c2e(sort_output=False)

    reciprocal = _ReciprocalContext.build(
        int3c2e_opt,
        states,
        j2c,
        i_addr=i_addr,
        j_addr=j_addr,
        omega=omega,
    )
    metric = _metric_inverse_in_primitive_basis(auxcell, reciprocal.j2c)
    for state in states:
        state.j3c_oo = cp.einsum(
            "uv,vij->uij", metric, state.j3c_oo, optimize=True
        )
        state.auxvec = state.j3c_oo.trace(axis1=1, axis2=2)
    metric_source = _signed_metric_source(states, naux)
    result = -int2c2e_opt.energy_ip1_per_atom(metric_source)

    result += _long_range_derivative(
        int3c2e_opt,
        reciprocal,
        states,
        metric_source,
        nao=nao,
        naux=naux,
    )
    result += _short_range_derivative(
        int3c2e_opt,
        states,
        pair_addresses,
        batch_size=batch_size,
        block_size=block_size,
        nao=nao,
        naux=naux,
    )
    result += _ewald_derivative(
        cell,
        states,
        exxdiv=exxdiv,
        omega=omega,
    )
    log.debug1("completed one-pass Gamma signed GDF derivative")
    return np.asarray(result, dtype=float)


def _prepare_state(
    cell,
    density,
    *,
    coulomb_weight,
    exchange_weight,
):
    original_density = cp.asarray(density, dtype=cp.float64)
    factor_l, factor_r = factorize_dm(density, hermi=1)
    factor_l = cell.apply_C_dot(factor_l, axis=0)
    if factor_r is None:
        factor_r = factor_l
    else:
        factor_r = cell.apply_C_dot(factor_r, axis=0)
    if factor_l.dtype != np.float64 or factor_r.dtype != np.float64:
        raise NotImplementedError("Gamma signed GDF factors must be real")
    return _QuadraticState(
        coulomb_weight=float(coulomb_weight),
        exchange_weight=float(exchange_weight),
        factor_l=factor_l,
        factor_r=factor_r,
        original_density=original_density,
        density=factor_l @ factor_r.T,
    )


def _working_block_sizes(states, *, nao_pair, naux, bvk_ncells):
    free = int(get_avail_mem(exclude_memory_pool=True))
    occupied_storage = sum(naux * state.nocc**2 * 8 for state in states)
    available = max(1, free - occupied_storage)
    batch = max(
        1,
        min(naux, int(available * 0.45 / (nao_pair * 8 * bvk_ncells))),
    )
    block = max(
        1,
        min(
            naux,
            int(
                available
                * 0.35
                / (len(states) * states[0].density.size * 8)
            ),
        ),
    )
    block = max(1, block // 8 * 8)
    return int(batch), int(block)


def _build_short_range_zero_order(
    int3c2e_opt,
    states,
    pair_addresses,
    i_addr,
    j_addr,
    *,
    batch_size,
    block_size,
):
    nao = states[0].density.shape[0]
    naux = int(int3c2e_opt.auxcell.ao_loc[-1])
    evaluator, _, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size,
        cart=True,
    )
    for state in states:
        state.j3c_oo = cp.empty((naux, state.nocc, state.nocc))

    j3c_full = cp.zeros((nao, nao, block_size))
    compressed_buffer = cp.empty((batch_size, len(pair_addresses)))
    aux0 = 0
    for batch_id in range(len(aux_offsets) - 1):
        compressed = evaluator(
            aux_batch_id=batch_id,
            out=compressed_buffer,
        )[:, 0, :]
        naux_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_batch, block_size):
            width = k1 - k0
            aux1 = aux0 + width
            j3c = j3c_full[:, :, :width]
            j3c[j_addr, i_addr] = j3c[i_addr, j_addr] = compressed[:, k0:k1]
            for state in states:
                tmp = cp.empty((state.nocc, nao, width))
                contract("pqr,pi->iqr", j3c, state.factor_r, out=tmp)
                contract(
                    "iqr,qj->rij",
                    tmp,
                    state.factor_l,
                    out=state.j3c_oo[aux0:aux1],
                )
            aux0 = aux1


@dataclass(frozen=True)
class _ReciprocalContext:
    j2c: cp.ndarray
    ft_opt: object
    evaluator: object
    coulomb: cp.ndarray
    vectors: cp.ndarray
    block_size: int

    @classmethod
    def build(
        cls,
        int3c2e_opt,
        states,
        j2c,
        *,
        i_addr,
        j_addr,
        omega,
    ):
        cell = int3c2e_opt.cell
        auxcell = int3c2e_opt.auxcell
        mesh = int3c2e_opt.mesh
        ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
        evaluator = ft_opt.ft_evaluator(
            compressing=False,
            cart=True,
            original_ao_order=False,
        )[0]
        omega_abs = abs(float(omega or 0.0))
        if omega_abs >= int3c2e_opt.omega:
            if cell.dimension != 3:
                raise NotImplementedError(
                    "Gamma signed route requires a 3D cell without LR splitting"
                )
            mesh = [1, 1, 1]
        vectors = asarray(auxcell.get_Gv(mesh))
        coulomb = _gamma_coulomb_vector(
            _weighted_coulG_kpts(
                cell,
                mesh,
                omega_abs,
                int3c2e_opt.omega,
            ),
            ngrids=len(vectors),
        )
        free = int(get_avail_mem(exclude_memory_pool=True))
        nao = states[0].density.shape[0]
        nocc_max = max(state.nocc for state in states)
        block = max(32, int(free // max(1, (nao + nocc_max) * nao * 16)))
        block = min(block // 32 * 32, len(vectors))
        if block <= 0:
            raise MemoryError("insufficient GPU memory for Gamma GDF LR block")

        for p0, p1 in lib.prange(0, len(vectors), block):
            ngrid = p1 - p0
            aux_g = ft_ao.ft_ao(auxcell, vectors[p0:p1]).T
            aux_weighted = aux_g * coulomb[p0:p1]
            contract(
                "iG,jG->ij",
                aux_g.view(np.float64),
                aux_weighted.view(np.float64),
                beta=1,
                out=j2c,
            )
            pair_g = evaluator(vectors[p0:p1])
            pair_g = pair_g.view(np.float64).reshape(nao, nao, ngrid * 2)
            pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]
            for state in states:
                tmp = cp.empty((state.nocc, nao, ngrid * 2))
                occupied_g = cp.empty(
                    (state.nocc, state.nocc, ngrid * 2)
                )
                contract("pqG,pi->iqG", pair_g, state.factor_r, out=tmp)
                contract("iqG,qj->ijG", tmp, state.factor_l, out=occupied_g)
                contract(
                    "rG,ijG->rij",
                    aux_weighted.view(np.float64),
                    occupied_g,
                    beta=1,
                    out=state.j3c_oo,
                )
        return cls(
            j2c=j2c,
            ft_opt=ft_opt,
            evaluator=evaluator,
            coulomb=coulomb,
            vectors=vectors,
            block_size=int(block),
        )


def _gamma_coulomb_vector(coulomb, *, ngrids):
    """Remove the singleton k-point axis from Gamma Coulomb weights.

    GPU4PySCF's ``_weighted_coulG_kpts`` keeps a leading k-point axis even
    when only Gamma is requested.  The reciprocal-space loops in this module
    slice along the G-vector axis, so they require a one-dimensional vector.
    Keeping the ``(1, nG)`` result happens to broadcast for a single full-grid
    block, but fails as soon as a larger calculation splits the grid.
    """

    vector = asarray(coulomb)
    if vector.ndim == 2:
        if vector.shape[0] != 1:
            raise ValueError(
                "Gamma GDF expected one Coulomb-weight k-point row, "
                f"got shape {vector.shape}"
            )
        vector = vector[0]
    if vector.ndim != 1 or vector.size != ngrids:
        raise ValueError(
            "Gamma GDF Coulomb weights do not match the reciprocal grid: "
            f"weights shape {vector.shape}, ngrids {ngrids}"
        )
    return vector


def _metric_inverse_in_primitive_basis(auxcell, j2c):
    transformed = auxcell.apply_CT_mat_C(j2c)
    if auxcell.cell.cart:
        raise NotImplementedError("Cartesian auxiliary GDF metric is unsupported")
    coefficients = cp.asarray(auxcell.ctr_coeff)
    return coefficients @ cp.linalg.solve(transformed, coefficients.T)


def _signed_metric_source(states, naux):
    source = cp.zeros((naux, naux))
    for state in states:
        source += state.coulomb_weight * (
            state.auxvec[:, None] * state.auxvec[None, :]
        )
        contract(
            "rij,sji->rs",
            state.j3c_oo,
            state.j3c_oo,
            alpha=-0.5 * state.exchange_weight,
            beta=1.0,
            out=source,
        )
    return source


def _long_range_derivative(
    int3c2e_opt,
    reciprocal,
    states,
    metric_source,
    *,
    nao,
    naux,
):
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    vectors = reciprocal.vectors
    evaluator = reciprocal.evaluator
    coulomb = reciprocal.coulomb
    ft_opt = reciprocal.ft_opt
    block_size = reciprocal.block_size

    bas_ij_idx, bas_ij_img_idx, shell_offsets = aft_jk._generate_shl_pairs(
        ft_opt
    )
    aft_envs = ft_opt.aft_envs
    shared_memory = aft_jk._estimate_max_shm_size(cell, (1, 0))
    kernel = libpbc.PBC_ft_aopair_ek_ip1
    ao_derivative = cp.zeros((cell.natm, 3))
    auxiliary_partial = cp.zeros((3, naux))

    for p0, p1 in lib.prange(0, len(vectors), block_size):
        ngrid = p1 - p0
        pair_g = evaluator(vectors[p0:p1])
        pair_g = pair_g.view(np.float64).reshape(nao, nao, ngrid * 2)
        pair_addresses = ft_opt.pair_and_diag_indices(
            cart=True, original_ao_order=False
        )[0]
        i_addr, j_addr = divmod(pair_addresses, nao)
        pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]

        aux_g = ft_ao.ft_ao(auxcell, vectors[p0:p1]).T
        derivative_aux_source = cp.zeros((naux, ngrid * 2))
        ao_source = cp.zeros((nao, nao, ngrid * 2))

        for state in states:
            rho_g = cp.einsum(
                "pqG,qp->G", pair_g, state.density, optimize=True
            )
            derivative_aux_source += (
                state.coulomb_weight
                * state.auxvec[:, None]
                * rho_g[None, :]
            )

            tmp = cp.empty((state.nocc, nao, ngrid * 2))
            occupied_g = cp.empty(
                (state.nocc, state.nocc, ngrid * 2)
            )
            contract("pqG,pi->iqG", pair_g, state.factor_r, out=tmp)
            contract("iqG,qj->ijG", tmp, state.factor_l, out=occupied_g)
            contract(
                "rji,ijG->rG",
                state.j3c_oo,
                occupied_g,
                alpha=-0.5 * state.exchange_weight,
                beta=1.0,
                out=derivative_aux_source,
            )

            aux_weighted_conj = (
                aux_g.conj() * coulomb[p0:p1]
            ).view(np.float64)
            occupied_aux_g = (
                state.j3c_oo.reshape(naux, -1).T @ aux_weighted_conj
            ).reshape(state.nocc, state.nocc, ngrid * 2)
            contract("jiG,qi->jqG", occupied_aux_g, state.factor_r, out=tmp)
            exchange_source = cp.empty((nao, nao, ngrid * 2))
            contract("jqG,pj->pqG", tmp, state.factor_l, out=exchange_source)
            potential_g = state.auxvec @ aux_weighted_conj
            ao_source += (
                state.coulomb_weight
                * state.density[:, :, None]
                * potential_g[None, None, :]
                - 0.5 * state.exchange_weight * exchange_source
            )

        contract(
            "sr,sG->rG",
            metric_source,
            aux_g.view(np.float64),
            alpha=-1.0,
            beta=1.0,
            out=derivative_aux_source,
        )
        derivative_aux_source = derivative_aux_source.view(np.complex128)
        derivative_aux_source *= coulomb[p0:p1]
        derivative_aux_source = derivative_aux_source.view(np.float64)

        derivative_aux_g = cp.empty((naux, ngrid), dtype=np.complex128)
        for axis in range(3):
            cp.multiply(
                aux_g,
                -1j * vectors[p0:p1, axis],
                out=derivative_aux_g,
            )
            auxiliary_partial[axis] += cp.einsum(
                "ag,ag->a",
                derivative_aux_g.view(np.float64),
                derivative_aux_source,
            )

        vector_transpose = cp.asarray(vectors[p0:p1].T.ravel())
        error = kernel(
            ctypes.cast(ao_derivative.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_source.data.ptr, ctypes.c_void_p),
            ctypes.cast(vector_transpose.data.ptr, ctypes.c_void_p),
            ctypes.byref(aft_envs),
            ctypes.c_int(len(shell_offsets) - 1),
            ctypes.c_int(ngrid),
            ctypes.c_int(shared_memory),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(shell_offsets.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ft_opt.permutation_symmetry),
        )
        if error != 0:
            raise RuntimeError("PBC_ft_aopair_ek_ip1 failed")

    dimensions = auxcell.ao_loc[1:] - auxcell.ao_loc[:-1]
    atom_for_aux = np.repeat(auxcell._bas[:, ATOM_OF], dimensions)
    auxiliary_grouped = groupby(
        atom_for_aux,
        auxiliary_partial.T.get(),
        op="sum",
    )
    result = ao_derivative.get() * 2.0
    if len(auxiliary_grouped) < cell.natm:
        result[np.unique(atom_for_aux)] += auxiliary_grouped
    else:
        result += auxiliary_grouped
    return result


def _short_range_derivative(
    int3c2e_opt,
    states,
    pair_addresses,
    *,
    batch_size,
    block_size,
    nao,
    naux,
):
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    auxiliary_shell_limit = max(
        1,
        min(int(auxcell.nbas), POOL_SIZE),
    )
    shell_pair_limit = max(1, POOL_SIZE // auxiliary_shell_limit)
    _, gout_stride, shared_sizes = int3c2e_scheme(-1, 54)
    lmax = cell.uniq_l_ctr[:, 0].max()
    laux = auxcell.uniq_l_ctr[:, 0].max()
    shared_memory = shared_sizes[: laux + 1, : lmax + 1, : lmax + 1].max()
    bas_ij_idx, shell_pair_offsets = cell.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache,
        shell_pair_limit,
    )
    ao_pair_loc = get_ao_pair_loc(
        cell.uniq_l_ctr[:, 0],
        int3c2e_opt.bas_ij_cache,
        cart=True,
    )
    aux_shell_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
    aux_shell_offsets, aux_patterns = _split_l_ctr_pattern(
        aux_shell_offsets,
        auxcell.uniq_l_ctr,
        min(batch_size, auxiliary_shell_limit),
    )
    aux_shell_offsets_gpu = cp.asarray(aux_shell_offsets, dtype=np.int32)
    pool = cp.empty(
        gpu_specs["multiProcessorCount"]
        * POOL_SIZE
        * (MAX_IMGS_PER_TASK + 2)
        + 1,
        dtype=np.uint32,
    )
    head = pool[-1:]
    task_pool = empty_aligned(
        (gpu_specs["multiProcessorCount"], POOL_SIZE * 16),
        np.int32,
        alignment=128,
    )
    ao_derivative = cp.zeros((cell.natm, 3))
    auxiliary_derivative = cp.zeros((cell.natm, 3))
    diffuse_exponents = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefficients = cp.asarray(int3c2e_opt.diffuse_coefs)
    log_cutoff = math.log(int3c2e_opt.cutoff)
    aux0 = 0

    for batch_id, _ in enumerate(aux_patterns[:, 0]):
        aux_ao_offset = auxcell.ao_loc[aux_shell_offsets[batch_id]]
        naux_batch = (
            auxcell.ao_loc[aux_shell_offsets[batch_id + 1]] - aux_ao_offset
        )
        compressed = cp.empty((len(pair_addresses), naux_batch))
        for k0, k1 in lib.prange(0, naux_batch, block_size):
            width = k1 - k0
            aux1 = aux0 + width
            source = cp.zeros((nao, nao, width))
            for state in states:
                source += state.coulomb_weight * (
                    state.density[:, :, None]
                    * state.auxvec[None, None, aux0:aux1]
                )
                tmp = cp.empty((state.nocc, nao, width))
                contract(
                    "rji,qj->iqr",
                    state.j3c_oo[aux0:aux1],
                    state.factor_l,
                    out=tmp,
                )
                contract(
                    "iqr,pi->pqr",
                    tmp,
                    state.factor_r,
                    alpha=-0.5 * state.exchange_weight,
                    beta=1.0,
                    out=source,
                )
            cp.take(
                source.reshape(-1, width),
                pair_addresses,
                axis=0,
                out=compressed[:, k0:k1],
            )
            aux0 = aux1

        error = libpbc.PBCsr_ejk_int3c2e_ip1(
            ctypes.cast(ao_derivative.data.ptr, ctypes.c_void_p),
            ctypes.cast(auxiliary_derivative.data.ptr, ctypes.c_void_p),
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.byref(int3c2e_opt.int3c2e_envs),
            ctypes.cast(pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(task_pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(head.data.ptr, ctypes.c_void_p),
            ctypes.c_int(shared_memory),
            ctypes.c_int(len(shell_pair_offsets) - 1),
            ctypes.c_int(1),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(shell_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(aux_shell_offsets_gpu[batch_id:].data.ptr, ctypes.c_void_p),
            ctypes.cast(int3c2e_opt.img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(int3c2e_opt.img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(aux_ao_offset),
            ctypes.c_int(auxcell.nbas),
            ctypes.c_int(naux_batch),
            ctypes.cast(diffuse_exponents.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefficients.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff),
        )
        if error != 0:
            raise RuntimeError("PBCsr_ejk_int3c2e_ip1 failed")

    ao_derivative *= 2.0
    auxiliary_derivative *= 2.0
    return (ao_derivative + auxiliary_derivative).get()


def _ewald_derivative(cell, states, *, exxdiv, omega):
    if exxdiv != "ewald" or not (
        cell.dimension == 3
        or (cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum")
    ):
        return np.zeros((cell.natm, 3))
    overlap = int1e.int1e_ovlp(cell)
    overlap_derivative = int1e.int1e_ipovlp(cell)
    signed_product = cp.zeros_like(states[0].original_density)
    for state in states:
        signed_product += state.exchange_weight * (
            state.original_density @ overlap @ state.original_density
        )
    result = contract_h1e_dm(
        cell.cell,
        overlap_derivative,
        signed_product,
        hermi=1,
    )
    correction = madelung(cell, np.zeros((1, 3)), omega=-float(omega))
    return np.asarray(result * (0.5 * correction), dtype=float)


def _real_symmetric_density(value, name):
    density = cp.asarray(value)
    if density.ndim == 3 and density.shape[0] == 1:
        density = density[0]
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError(f"{name} must be a square Gamma matrix, got {density.shape}")
    imaginary = float(cp.max(cp.abs(density.imag)).item())
    if imaginary > 1.0e-10:
        raise NotImplementedError(
            f"{name} must be real at Gamma; max imaginary component is "
            f"{imaginary:.3e}"
        )
    density = cp.asarray(density.real, dtype=cp.float64)
    asymmetry = float(cp.max(cp.abs(density - density.T)).item())
    if asymmetry > 1.0e-9:
        raise ValueError(
            f"{name} must be symmetric; max asymmetry is {asymmetry:.3e}"
        )
    return density
