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
# Adapted from GPU4PySCF's periodic GDF KRHF derivative implementation. This
# version fuses signed density states for the DC-DFT response contraction.

"""One-pass sampled-k GDF derivative for a signed pair of RHF densities.

The implementation follows GPU4PySCF's periodic KRHF GDF derivative
decomposition.  Two quadratic states, ``D + T`` and ``D - T``, share the
zero-order integral evaluation and contribute with opposite signs to one
SR, LR, metric, auxiliary-centre, and Ewald derivative traversal.

This module is private until the sampled-k full-gradient and timing gates
have passed.  It intentionally does not materialize nuclear ``dCDERI``.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass

import cupy as cp
import numpy as np
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.gto.mole import groupby
from gpu4pyscf.lib.cupy_helper import (
    asarray,
    contract,
    empty_aligned,
    get_avail_mem,
)
from gpu4pyscf.pbc.df import aft_jk, ft_ao
from gpu4pyscf.pbc.df.grad.rhf import (
    _split_l_ctr_pattern,
    factorize_dm,
    get_ao_pair_loc,
    int3c2e_scheme,
)
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.df.int3c2e import MAX_IMGS_PER_TASK, POOL_SIZE, libpbc
from gpu4pyscf.pbc.df.rsdf_builder import _weighted_coulG_kpts
from gpu4pyscf.pbc.grad.krhf import contract_h1e_dm
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter
from gpu4pyscf.pbc.tools.pbc import _Gv_wrap_around, madelung
from pyscf import lib
from pyscf.gto import ATOM_OF
from pyscf.pbc.tools.k2gamma import double_translation_indices


@dataclass
class _KPointState:
    """Factorized zero-order data for one signed quadratic state."""

    coulomb_weight: float
    exchange_weight: float
    original_density: cp.ndarray
    factor_l: cp.ndarray
    factor_r: cp.ndarray
    density: cp.ndarray
    j3c_oo: cp.ndarray | None = None
    auxvec: cp.ndarray | None = None

    @property
    def nocc(self) -> int:
        return int(self.factor_l.shape[-1])


@dataclass(frozen=True)
class _MomentumContext:
    """Immutable momentum-transfer maps shared by all signed states."""

    iterations: tuple
    unique_kpts: np.ndarray
    ijk_conserv: cp.ndarray
    order_ki: cp.ndarray
    order_kj: cp.ndarray
    exp_lk: cp.ndarray
    exp_lk_conj: cp.ndarray


@dataclass(frozen=True)
class _ReciprocalContext:
    """Long-range values reused by the zero- and first-order contractions."""

    mesh: tuple[int, int, int]
    vectors: np.ndarray
    coulomb: cp.ndarray
    ft_opt: object
    ft_kernel: object
    block_size: int


def kpoint_signed_energy_derivative(
    int3c2e_opt,
    reference_density,
    response_density,
    kpts,
    *,
    exxdiv="ewald",
    omega=0.0,
):
    """Return ``Q(D+T)-Q(D-T)`` with one sampled-k derivative traversal."""

    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    if len(points) <= 1:
        raise ValueError("sampled-k signed GDF derivative requires nkpts > 1")
    reference = _hermitian_kpoint_density(
        reference_density, len(points), "reference_density"
    )
    response = _hermitian_kpoint_density(
        response_density, len(points), "response_density"
    )
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )

    states = _signed_states(int3c2e_opt.cell, reference, response)
    return _weighted_energy_derivative(
        int3c2e_opt,
        states,
        points,
        exxdiv=exxdiv,
        omega=omega,
    )


def kpoint_target_j_plus_signed_energy_derivative(
    int3c2e_opt,
    reference_density,
    response_density,
    kpts,
    *,
    exxdiv="ewald",
    omega=0.0,
):
    """Return target-J plus RHF bilinear response in one sampled-k pass."""

    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    if len(points) <= 1:
        raise ValueError("sampled-k fused GDF derivative requires nkpts > 1")
    reference = _hermitian_kpoint_density(
        reference_density, len(points), "reference_density"
    )
    response = _hermitian_kpoint_density(
        response_density, len(points), "response_density"
    )
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    cell = int3c2e_opt.cell
    states = (
        _prepare_state(
            cell,
            reference,
            coulomb_weight=1.0,
            exchange_weight=0.0,
        ),
        *_signed_states(cell, reference, response),
    )
    return _weighted_energy_derivative(
        int3c2e_opt,
        states,
        points,
        exxdiv=exxdiv,
        omega=omega,
    )


def _signed_states(cell, reference, response):
    return (
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
    )


def _weighted_energy_derivative(
    int3c2e_opt,
    states,
    points,
    *,
    exxdiv,
    omega,
):
    """Apply one sampled-k derivative traversal to weighted states."""

    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    reference = states[0].original_density
    nkpts = int(reference.shape[0])
    nao = _state_ao_count(states, nkpts=nkpts)
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
    if bvk_ncells != nkpts:
        raise ValueError(
            "sampled-k native derivative requires a complete BvK mesh: "
            f"{bvk_ncells} cells for {nkpts} k-points"
        )

    pair_addresses, diagonal = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False
    )
    naux = int(auxcell.ao_loc[-1])
    momentum = _build_momentum_context(int3c2e_opt, points)
    auxiliary_batch, auxiliary_block = _working_block_sizes(
        states,
        nao_pair=len(pair_addresses),
        nao=nao,
        naux=naux,
        nkpts=nkpts,
        bvk_ncells=bvk_ncells,
    )
    _build_short_range_zero_order(
        int3c2e_opt,
        states,
        momentum,
        pair_addresses,
        diagonal,
        auxiliary_batch=auxiliary_batch,
        auxiliary_block=auxiliary_block,
        nao=nao,
    )

    precision = auxcell.precision * 1.0e-6
    auxcell.rcut = _estimate_sr_2c2e_rcut(
        auxcell, int3c2e_opt.omega, precision
    )
    int2c2e_opt = Int2c2eOpt(auxcell, int3c2e_opt.bvk_kmesh)
    j2c = int2c2e_opt.int2c2e(momentum.unique_kpts, sort_output=False)
    if j2c.dtype == np.float64:
        j2c = j2c.astype(np.complex128)

    reciprocal = _build_reciprocal_zero_order(
        int3c2e_opt,
        states,
        momentum,
        points,
        j2c,
        omega=float(omega or 0.0),
        nao=nao,
        naux=naux,
    )
    metric_sources, result = _apply_metrics_and_metric_derivative(
        auxcell,
        int2c2e_opt,
        states,
        momentum,
        j2c,
        nkpts=nkpts,
        naux=naux,
    )
    result += _long_range_derivative(
        int3c2e_opt,
        states,
        momentum,
        reciprocal,
        metric_sources,
        points,
        nao=nao,
        naux=naux,
        nkpts=nkpts,
    )
    result += _short_range_derivative(
        int3c2e_opt,
        states,
        momentum,
        pair_addresses,
        auxiliary_batch=auxiliary_batch,
        auxiliary_block=auxiliary_block,
        nao=nao,
        naux=naux,
        nkpts=nkpts,
    )
    result += _ewald_derivative(
        cell,
        states,
        points,
        int3c2e_opt.bvk_kmesh,
        exxdiv=exxdiv,
        omega=float(omega or 0.0),
    )
    return np.asarray(result, dtype=float)


def _state_ao_count(states, *, nkpts):
    """Return the sorted-AO dimension shared by all sampled-k states."""

    shapes = {tuple(state.density.shape) for state in states}
    if len(shapes) != 1:
        raise ValueError(
            f"sampled-k signed states have inconsistent shapes: {shapes}"
        )
    shape = shapes.pop()
    if len(shape) != 3 or shape[0] != nkpts or shape[1] != shape[2]:
        raise ValueError(
            "sampled-k sorted density must have shape "
            f"({nkpts},nao,nao), got {shape}"
        )
    return int(shape[1])


def _hermitian_kpoint_density(value, nkpts, name):
    density = cp.asarray(value)
    if density.ndim != 3 or density.shape[0] != nkpts:
        raise ValueError(
            f"{name} must have shape ({nkpts},nao,nao), got {density.shape}"
        )
    if density.shape[1] != density.shape[2]:
        raise ValueError(f"{name} AO dimensions must be square")
    error = cp.max(cp.abs(density - density.swapaxes(-1, -2).conj()))
    if float(error.item()) > 1.0e-10:
        raise ValueError(f"{name} must be Hermitian at every k-point")
    return cp.ascontiguousarray(density)


def _prepare_state(
    cell,
    density,
    *,
    coulomb_weight,
    exchange_weight,
):
    original = cp.asarray(density)
    factor_l, factor_r = factorize_dm(original, hermi=1)
    factor_l = cell.apply_C_dot(factor_l, axis=1)
    if factor_r is None:
        factor_r = factor_l.conj()
    else:
        factor_r = cell.apply_C_dot(factor_r, axis=1)
    sorted_density = contract("kpi,kqi->kpq", factor_l, factor_r)
    return _KPointState(
        coulomb_weight=float(coulomb_weight),
        exchange_weight=float(exchange_weight),
        original_density=original,
        factor_l=factor_l,
        factor_r=factor_r,
        density=sorted_density,
    )


def _build_momentum_context(int3c2e_opt, kpts):
    nkpts = len(kpts)
    exp_lk = cp.exp(
        1j * cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T))
    )
    ijk_conserv = cp.asarray(
        double_translation_indices(int3c2e_opt.bvk_kmesh)
    )
    order_ki = cp.argsort(
        (ijk_conserv * nkpts + cp.arange(nkpts)[:, None]).ravel()
    )
    order_kj = (
        ijk_conserv * nkpts + cp.arange(nkpts)
    ).ravel()
    iterations = tuple(kk_adapted_iter(int3c2e_opt.bvk_kmesh))
    unique_kpts = kpts[[item[0] for item in iterations]]
    return _MomentumContext(
        iterations=iterations,
        unique_kpts=unique_kpts,
        ijk_conserv=ijk_conserv,
        order_ki=order_ki,
        order_kj=order_kj,
        exp_lk=exp_lk,
        exp_lk_conj=exp_lk.conj(),
    )


def _working_block_sizes(
    states,
    *,
    nao_pair,
    nao,
    naux,
    nkpts,
    bvk_ncells,
):
    free = int(get_avail_mem(exclude_memory_pool=True))
    occupied = sum(
        naux * nkpts**2 * state.nocc**2 * 16 for state in states
    )
    available = max(1, free - occupied)
    auxiliary_batch = max(
        1,
        min(
            naux,
            int(available * 0.20 / max(1, nao_pair * bvk_ncells * 8)),
        ),
    )
    bytes_per_auxiliary = max(
        1,
        2 * nkpts**2 * nao**2 * 16
        + sum(nkpts**2 * state.nocc * nao * 16 for state in states),
    )
    auxiliary_block = max(
        1, min(naux, int(available * 0.25 / bytes_per_auxiliary))
    )
    return int(auxiliary_batch), int(auxiliary_block)


def _build_short_range_zero_order(
    int3c2e_opt,
    states,
    momentum,
    pair_addresses,
    diagonal,
    *,
    auxiliary_batch,
    auxiliary_block,
    nao,
):
    nkpts = len(momentum.exp_lk.T)
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
    naux = int(int3c2e_opt.auxcell.ao_loc[-1])
    evaluator, _, _, offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=auxiliary_batch,
        cart=True,
    )
    for state in states:
        state.j3c_oo = cp.empty(
            (naux, nkpts, nkpts, state.nocc, state.nocc),
            dtype=np.complex128,
        )

    compressed_buffer = cp.empty(
        (bvk_ncells * auxiliary_batch, len(pair_addresses))
    )
    phase = momentum.exp_lk_conj.view(np.float64).reshape(
        bvk_ncells, nkpts, 2
    )
    aux0 = 0
    for batch_id in range(len(offsets) - 1):
        compressed = evaluator(
            aux_batch_id=batch_id,
            out=compressed_buffer,
        )
        compressed = contract("tLr,LKz->trKz", compressed, phase)
        compressed = compressed.view(np.complex128)[:, :, :, 0]
        compressed[diagonal] *= 0.5
        batch_width = int(compressed.shape[1])
        for k0, k1 in lib.prange(0, batch_width, auxiliary_block):
            width = k1 - k0
            aux1 = aux0 + width
            j3c = cp.zeros(
                (nao * bvk_ncells * nao, width, nkpts),
                dtype=np.complex128,
            )
            j3c[pair_addresses] = compressed[:, k0:k1]
            j3c = j3c.reshape(
                nao, bvk_ncells, nao, width, nkpts
            )
            j3c_ki = contract(
                "jLikK,LI->KIijk", j3c, momentum.exp_lk_conj
            )
            j3c_kj = contract(
                "iLjkK,LJ->KJijk", j3c, momentum.exp_lk
            )
            j3c_crystal = cp.empty(
                (nkpts * nkpts, nao * nao * width),
                dtype=np.complex128,
            )
            j3c_crystal[momentum.order_ki] = j3c_ki.reshape(
                nkpts**2, -1
            )
            j3c_crystal[momentum.order_kj] += j3c_kj.reshape(
                nkpts**2, -1
            )
            j3c_crystal = j3c_crystal.reshape(
                nkpts, nkpts, nao, nao, width
            )
            for state in states:
                temporary = contract(
                    "IJpqr,Ipi->IJiqr",
                    j3c_crystal,
                    state.factor_r,
                )
                state.j3c_oo[aux0:aux1] = contract(
                    "IJiqr,Jqj->rIJij",
                    temporary,
                    state.factor_l,
                )
            aux0 = aux1


def _build_reciprocal_zero_order(
    int3c2e_opt,
    states,
    momentum,
    kpts,
    j2c,
    *,
    omega,
    nao,
    naux,
):
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    mesh = tuple(int(x) for x in int3c2e_opt.mesh)
    if abs(omega) >= int3c2e_opt.omega:
        if cell.dimension != 3:
            raise NotImplementedError(
                "sampled-k signed route requires a 3D cell without LR splitting"
            )
        mesh = (1, 1, 1)
    ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
    if not ft_opt.permutation_symmetry:
        raise NotImplementedError(
            "sampled-k signed route requires permutation-symmetric FT pairs"
        )
    ft_kernel = ft_opt.gen_ft_kernel(transform_ao=False)
    coulomb = _weighted_coulG_kpts(
        cell,
        mesh,
        abs(omega),
        int3c2e_opt.omega,
        momentum.unique_kpts,
    )
    vectors = auxcell.get_Gv(mesh)
    free = int(get_avail_mem(exclude_memory_pool=True))
    nocc_max = max(state.nocc for state in states)
    denominator = max(1, (nao * 2 + nocc_max) * nao * 16 * len(kpts))
    block = max(32, int(free * 0.25 // denominator) // 32 * 32)
    block = min(block, len(vectors))
    if block <= 0:
        raise MemoryError("insufficient GPU memory for sampled-k GDF LR block")

    for p0, p1 in lib.prange(0, len(vectors), block):
        width = p1 - p0
        shifted = (
            vectors[p0:p1] + momentum.unique_kpts[:, None]
        ).reshape(-1, 3)
        auxiliary_g = ft_ao.ft_ao(auxcell, shifted).T.reshape(
            naux, len(momentum.unique_kpts), width
        )
        auxiliary_weighted = auxiliary_g.conj() * coulomb[:, p0:p1]
        contract(
            "iKG,jKG->Kij",
            auxiliary_weighted,
            auxiliary_g,
            beta=1.0,
            out=j2c,
        )
        for q_index, (kp, kp_conj, ki_idx, kj_idx) in enumerate(
            momentum.iterations
        ):
            pair_g = ft_kernel(vectors[p0:p1], kpts[kp], kpts, kj_idx)
            pair_g = pair_g.transpose(0, 2, 3, 1)[kj_idx]
            for state in states:
                temporary = contract(
                    "kpqG,kpi->kiqG", pair_g, state.factor_r
                )
                occupied_g = contract(
                    "kiqG,kqj->kijG",
                    temporary,
                    state.factor_l[kj_idx],
                )
                state.j3c_oo[:, ki_idx, kj_idx] += contract(
                    "rG,kijG->rkij",
                    auxiliary_weighted[:, q_index],
                    occupied_g,
                )
                if kp != kp_conj:
                    temporary = contract(
                        "kqpG,kpi->kiqG",
                        pair_g.conj(),
                        state.factor_r[kj_idx],
                    )
                    occupied_g = contract(
                        "kiqG,kqj->kijG",
                        temporary,
                        state.factor_l,
                    )
                    state.j3c_oo[:, kj_idx, ki_idx] += contract(
                        "rG,kijG->rkij",
                        auxiliary_weighted[:, q_index].conj(),
                        occupied_g,
                    )
    return _ReciprocalContext(
        mesh=mesh,
        vectors=vectors,
        coulomb=coulomb,
        ft_opt=ft_opt,
        ft_kernel=ft_kernel,
        block_size=int(block),
    )


def _apply_metrics_and_metric_derivative(
    auxcell,
    int2c2e_opt,
    states,
    momentum,
    j2c,
    *,
    nkpts,
    naux,
):
    transformed_metric = auxcell.apply_CT_mat_C(j2c)
    metric_derivative = auxcell.pbc_intor(
        "int2c2e_ip1", kpts=momentum.unique_kpts
    )
    coefficients = cp.asarray(auxcell.ctr_coeff)
    weight = 1.0 / nkpts**2
    sources = cp.zeros(
        (len(momentum.iterations), naux, naux),
        dtype=np.complex128,
    )
    result = np.zeros((auxcell.cell.natm, 3))

    for q_index, (kp, kp_conj, ki_idx, kj_idx) in enumerate(
        momentum.iterations
    ):
        inverse = coefficients @ cp.linalg.solve(
            transformed_metric[q_index], coefficients.T
        )
        for state in states:
            direct = state.j3c_oo[:, ki_idx, kj_idx]
            fitted = contract("uv,vnij->unij", inverse, direct)
            state.j3c_oo[:, ki_idx, kj_idx] = fitted
            if kp == 0:
                conjugate = fitted
            elif kp == kp_conj:
                conjugate = state.j3c_oo[:, kj_idx, ki_idx]
            else:
                direct_conjugate = state.j3c_oo[:, kj_idx, ki_idx]
                conjugate = contract(
                    "vu,vnij->unij", inverse, direct_conjugate
                )
                state.j3c_oo[:, kj_idx, ki_idx] = conjugate

            if kp == 0:
                state.auxvec = cp.einsum("unii->u", fitted)
                sources[q_index] += (
                    state.coulomb_weight
                    * weight
                    * state.auxvec[:, None]
                    * state.auxvec.conj()[None, :]
                )
            contract(
                "rkij,skji->rs",
                fitted,
                conjugate,
                alpha=-0.5 * state.exchange_weight * weight,
                beta=1.0,
                out=sources[q_index],
            )

        derivative = asarray(metric_derivative[q_index])
        factor = 1.0 if kp == kp_conj else 2.0
        result += factor * contract_h1e_dm(
            auxcell, derivative, sources[q_index], hermi=0
        )
    return sources, result * 0.5


def _long_range_derivative(
    int3c2e_opt,
    states,
    momentum,
    reciprocal,
    metric_sources,
    kpts,
    *,
    nao,
    naux,
    nkpts,
):
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    vectors = reciprocal.vectors
    ngrid = len(vectors)
    unique_count = len(momentum.unique_kpts)
    shifted = (
        asarray(vectors) + asarray(momentum.unique_kpts)[:, None]
    ).reshape(-1, 3)
    shifted = _Gv_wrap_around(
        auxcell, shifted, cp.zeros(3), reciprocal.mesh
    ).reshape(unique_count, ngrid, 3)

    bas_ij_idx, bas_ij_img_idx, shell_offsets = aft_jk._generate_shl_pairs(
        reciprocal.ft_opt
    )
    aft_envs = reciprocal.ft_opt.aft_envs
    shared_memory = aft_jk._estimate_max_shm_size(cell, (1, 0))
    kernel = libpbc.PBC_ft_aopair_ek_ip1
    ao_derivative = cp.zeros((cell.natm, 3))
    auxiliary_partial = cp.zeros((3, naux))
    weight = 1.0 / nkpts**2

    for p0, p1 in lib.prange(0, ngrid, reciprocal.block_size):
        width = p1 - p0
        auxiliary_g = ft_ao.ft_ao(
            auxcell,
            shifted[:, p0:p1].reshape(-1, 3),
        ).T.reshape(naux, unique_count, width)
        for q_index, (kp, kp_conj, ki_idx, kj_idx) in enumerate(
            momentum.iterations
        ):
            pair_g = reciprocal.ft_kernel(
                vectors[p0:p1], kpts[kp], kpts, kj_idx
            )
            pair_g = pair_g.transpose(0, 2, 3, 1)[kj_idx]
            auxiliary_source = cp.zeros((naux, width), dtype=np.complex128)

            for state in states:
                if kp == 0:
                    rho_g = cp.einsum(
                        "kpqG,kqp->G", pair_g, state.density
                    )
                    auxiliary_source += (
                        state.coulomb_weight
                        * weight
                        * state.auxvec[:, None]
                        * rho_g[None, :]
                    )
                temporary = contract(
                    "kpqG,kpi->kiqG", pair_g, state.factor_r
                )
                occupied_g = contract(
                    "kiqG,kqj->kijG",
                    temporary,
                    state.factor_l[kj_idx],
                )
                contract(
                    "rkji,kijG->rG",
                    state.j3c_oo[:, kj_idx, ki_idx],
                    occupied_g,
                    alpha=-0.5 * state.exchange_weight * weight,
                    beta=1.0,
                    out=auxiliary_source,
                )

            contract(
                "sr,sG->rG",
                metric_sources[q_index],
                auxiliary_g[:, q_index],
                alpha=-1.0,
                beta=1.0,
                out=auxiliary_source,
            )
            auxiliary_source *= reciprocal.coulomb[q_index, p0:p1]
            source_real = auxiliary_source.view(np.float64)
            for axis in range(3):
                derivative_auxiliary = (
                    auxiliary_g[:, q_index]
                    * (-1j * shifted[q_index, p0:p1, axis])
                ).view(np.float64)
                factor = 2.0 if kp != kp_conj else 1.0
                auxiliary_partial[axis] += factor * cp.einsum(
                    "ag,ag->a", derivative_auxiliary, source_real
                )

            auxiliary_weighted = (
                auxiliary_g[:, q_index].conj()
                * reciprocal.coulomb[q_index, p0:p1]
            )
            real_space_source = None
            for state in states:
                fitted_g = contract(
                    "rkji,rG->kijG",
                    state.j3c_oo[:, kj_idx, ki_idx],
                    auxiliary_weighted,
                )
                temporary = contract(
                    "kijG,kpi->kpjG", fitted_g, state.factor_r
                )
                ao_g = contract(
                    "kpjG,kqj->kpqG",
                    temporary,
                    state.factor_l[kj_idx],
                    alpha=-0.5 * state.exchange_weight * weight,
                )
                contribution = contract(
                    "Lk,kpqG->LqpG", momentum.exp_lk[:, kj_idx], ao_g
                )
                if reciprocal.ft_opt.permutation_symmetry:
                    contribution += contract(
                        "Lk,kpqG->LpqG", momentum.exp_lk.conj(), ao_g
                    )

                if kp == 0:
                    potential_g = state.auxvec @ auxiliary_weighted
                    if reciprocal.ft_opt.permutation_symmetry:
                        potential_g *= 2.0
                    bvk_density = contract(
                        "Lk,kpq->Lpq", momentum.exp_lk, state.density
                    )
                    contribution += (
                        state.coulomb_weight
                        * weight
                        * bvk_density[:, :, :, None]
                        * potential_g
                    )
                if real_space_source is None:
                    real_space_source = contribution
                else:
                    real_space_source += contribution

            if kp != kp_conj:
                real_space_source *= 2.0
            real_space_source = cp.asarray(real_space_source, order="C")
            vector_transpose = cp.asarray(
                (vectors[p0:p1] + kpts[kp]).T.ravel()
            )
            error = kernel(
                ctypes.cast(ao_derivative.data.ptr, ctypes.c_void_p),
                ctypes.cast(real_space_source.data.ptr, ctypes.c_void_p),
                ctypes.cast(vector_transpose.data.ptr, ctypes.c_void_p),
                ctypes.byref(aft_envs),
                ctypes.c_int(len(shell_offsets) - 1),
                ctypes.c_int(width),
                ctypes.c_int(shared_memory),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(shell_offsets.data.ptr, ctypes.c_void_p),
                ctypes.c_int(reciprocal.ft_opt.permutation_symmetry),
            )
            if error != 0:
                raise RuntimeError("PBC_ft_aopair_ek_ip1 failed")

    dimensions = auxcell.ao_loc[1:] - auxcell.ao_loc[:-1]
    atom_for_auxiliary = np.repeat(
        auxcell._bas[:, ATOM_OF], dimensions
    )
    grouped = groupby(
        atom_for_auxiliary, auxiliary_partial.T.get(), op="sum"
    )
    result = ao_derivative.get()
    if len(grouped) < cell.natm:
        result[np.unique(atom_for_auxiliary)] += grouped
    else:
        result += grouped
    return result


def _short_range_derivative(
    int3c2e_opt,
    states,
    momentum,
    pair_addresses,
    *,
    auxiliary_batch,
    auxiliary_block,
    nao,
    naux,
    nkpts,
):
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
    auxiliary_shell_limit = max(
        1, min(int(auxcell.nbas), POOL_SIZE // bvk_ncells)
    )
    shell_pair_limit = max(
        1, POOL_SIZE // (auxiliary_shell_limit * bvk_ncells)
    )
    _, gout_stride, shared_sizes = int3c2e_scheme(-1, 54)
    lmax = cell.uniq_l_ctr[:, 0].max()
    laux = auxcell.uniq_l_ctr[:, 0].max()
    shared_memory = shared_sizes[: laux + 1, : lmax + 1, : lmax + 1].max()
    bas_ij_idx, shell_pair_offsets = cell.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, shell_pair_limit
    )
    ao_pair_loc = get_ao_pair_loc(
        cell.uniq_l_ctr[:, 0], int3c2e_opt.bas_ij_cache, cart=True
    )
    auxiliary_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
    auxiliary_offsets, auxiliary_patterns = _split_l_ctr_pattern(
        auxiliary_offsets,
        auxcell.uniq_l_ctr,
        min(auxiliary_batch, auxiliary_shell_limit),
    )
    auxiliary_offsets_gpu = cp.asarray(auxiliary_offsets, dtype=np.int32)
    diffuse_exponents = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefficients = cp.asarray(int3c2e_opt.diffuse_coefs)
    log_cutoff = math.log(int3c2e_opt.cutoff)
    order_ki = (
        momentum.ijk_conserv.T * nkpts + cp.arange(nkpts)[:, None]
    ).ravel()
    ao_derivative = cp.zeros((cell.natm, 3))
    auxiliary_derivative = cp.zeros((cell.natm, 3))
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
    weight = 1.0 / nkpts**2
    kernel = libpbc.PBCsr_ejk_int3c2e_ip1
    aux0 = 0

    for batch_id, _ in enumerate(auxiliary_patterns[:, 0]):
        auxiliary_ao_offset = auxcell.ao_loc[auxiliary_offsets[batch_id]]
        batch_width = (
            auxcell.ao_loc[auxiliary_offsets[batch_id + 1]]
            - auxiliary_ao_offset
        )
        compressed = cp.empty(
            (len(pair_addresses), bvk_ncells, batch_width)
        )
        for k0, k1 in lib.prange(0, batch_width, auxiliary_block):
            width = k1 - k0
            aux1 = aux0 + width
            source = cp.zeros(
                (nkpts, nkpts, nao, nao, width),
                dtype=np.complex128,
            )
            for state in states:
                temporary = contract(
                    "rIJij,Jqj->IJiqr",
                    state.j3c_oo[aux0:aux1],
                    state.factor_r,
                    alpha=-0.5 * state.exchange_weight * weight,
                )
                contract(
                    "IJiqr,Ipi->IJpqr",
                    temporary,
                    state.factor_l,
                    beta=1.0,
                    out=source,
                )
            swapped = cp.empty_like(source).reshape(
                nkpts**2, nao, nao, width
            )
            swapped[order_ki] = source.reshape(
                nkpts**2, nao, nao, width
            )
            swapped = swapped.reshape(nkpts, nkpts, nao, nao, width)
            for state in states:
                swapped[0] += (
                    state.coulomb_weight
                    * weight
                    * state.auxvec[aux0:aux1]
                    * state.density[:, :, :, None]
                )
            mixed = contract(
                "KJpqr,LK->JpqLr", swapped, momentum.exp_lk_conj
            )
            real_space = contract(
                "JpqLr,NJ->qNpLr", mixed, momentum.exp_lk
            )
            real_space = real_space.reshape(-1, bvk_ncells, width).real
            cp.take(
                real_space,
                pair_addresses,
                axis=0,
                out=compressed[:, :, k0:k1],
            )
            aux0 = aux1

        error = kernel(
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
            ctypes.cast(auxiliary_offsets_gpu[batch_id:].data.ptr, ctypes.c_void_p),
            ctypes.cast(int3c2e_opt.img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(int3c2e_opt.img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(auxiliary_ao_offset),
            ctypes.c_int(auxcell.nbas),
            ctypes.c_int(batch_width),
            ctypes.cast(diffuse_exponents.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefficients.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff),
        )
        if error != 0:
            raise RuntimeError("PBCsr_ejk_int3c2e_ip1 failed")
    return (ao_derivative + auxiliary_derivative).get() * 2.0


def _ewald_derivative(
    cell,
    states,
    kpts,
    bvk_kmesh,
    *,
    exxdiv,
    omega,
):
    if exxdiv != "ewald":
        return np.zeros((cell.natm, 3))
    if not (
        cell.dimension == 3
        or (cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum")
    ):
        return np.zeros((cell.natm, 3))
    overlap = int1e.int1e_ovlp(cell, kpts, bvk_kmesh)
    overlap_derivative = int1e.int1e_ipovlp(cell, kpts, bvk_kmesh)
    exchange_density = cp.zeros_like(states[0].original_density)
    for state in states:
        temporary = contract(
            "kpq,kqr->kpr", state.original_density, overlap
        )
        exchange_density += state.exchange_weight * contract(
            "kpr,krs->kps", temporary, state.original_density
        )
    result = contract_h1e_dm(
        cell.cell, overlap_derivative, exchange_density, hermi=1
    )
    result *= (
        0.5
        / len(kpts)
        * madelung(cell, kpts, omega=-abs(omega))
    )
    return result
