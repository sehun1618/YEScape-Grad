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

The route is deliberately private and Gamma-only.  Sampled k-points use a
separate momentum-pair implementation, while the two-pass polarization route
is retained as an independent reference for parity checks.
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

from .reciprocal_memory import (
    execute_adaptive_reciprocal_blocks,
    plan_gamma_reciprocal_blocks,
)


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


class _DevicePhaseTimeline:
    """Record non-synchronizing CUDA-event boundaries for one traversal."""

    def __init__(self, enabled):
        self.enabled = bool(enabled)
        self._events = []

    def mark(self, name):
        if not self.enabled:
            return
        event = cp.cuda.Event()
        event.record()
        self._events.append((str(name), event))

    def metadata(self):
        phases = {}
        for (_previous_name, previous), (name, current) in zip(
            self._events, self._events[1:]
        ):
            phases[name] = float(
                cp.cuda.get_elapsed_time(previous, current) * 1.0e-3
            )
        return {
            "measurement": "CUDA event elapsed time; host-only work excluded",
            "phases_s": phases,
            "total_s": float(sum(phases.values())),
        }


def gamma_signed_energy_derivative(
    int3c2e_opt,
    reference_density,
    response_density,
    *,
    exxdiv="ewald",
    omega=0.0,
    verbose=None,
    memory_diagnostics=None,
    component_diagnostics=None,
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
        memory_diagnostics=memory_diagnostics,
        component_diagnostics=component_diagnostics,
    )


def gamma_target_j_plus_signed_energy_derivative(
    int3c2e_opt,
    reference_density,
    response_density,
    *,
    exxdiv="ewald",
    omega=0.0,
    verbose=None,
    memory_diagnostics=None,
    component_diagnostics=None,
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
        memory_diagnostics=memory_diagnostics,
        component_diagnostics=component_diagnostics,
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
    memory_diagnostics,
    component_diagnostics,
):
    """Apply one native derivative traversal to weighted quadratic states."""

    timeline = _DevicePhaseTimeline(memory_diagnostics is not None)
    timeline.mark("start")
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    # GPU4PySCF evaluates SR/LR tensors in SortedCell's Cartesian AO basis.
    # Spherical-to-Cartesian expansion makes this basis larger when l >= 2
    # (Water64/DZVP has 1472 spherical AOs but 1536 sorted Cartesian AOs).
    # The transformed state, pair addresses, and FT evaluator all share this
    # internal dimension; the original density is retained only for the Ewald
    # correction in the physical AO basis.
    nao = _state_ao_count(states)
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
    timeline.mark("short_range_zero_order")

    precision = auxcell.precision * 1.0e-6
    auxcell.rcut = _estimate_sr_2c2e_rcut(
        auxcell, int3c2e_opt.omega, precision
    )
    int2c2e_opt = Int2c2eOpt(auxcell)
    j2c = int2c2e_opt.int2c2e(sort_output=False)
    timeline.mark("two_center_metric")

    reciprocal = _ReciprocalContext.build(
        int3c2e_opt,
        states,
        j2c,
        i_addr=i_addr,
        j_addr=j_addr,
        omega=omega,
    )
    timeline.mark("reciprocal_zero_order")
    metric = _metric_inverse_in_primitive_basis(auxcell, reciprocal.j2c)
    for state in states:
        state.j3c_oo = cp.einsum(
            "uv,vij->uij", metric, state.j3c_oo, optimize=True
        )
        state.auxvec = state.j3c_oo.trace(axis1=1, axis2=2)
    timeline.mark("metric_factorization_and_state_solve")
    metric_source = _signed_metric_source(states, naux)
    metric_derivative = -int2c2e_opt.energy_ip1_per_atom(metric_source)
    result = metric_derivative
    timeline.mark("metric_derivative")

    long_range, derivative_execution = _long_range_derivative(
        int3c2e_opt,
        reciprocal,
        states,
        metric_source,
        nao=nao,
        naux=naux,
    )
    result += long_range
    timeline.mark("long_range_derivative")
    short_range_device_timing = {}
    short_range = _short_range_derivative(
        int3c2e_opt,
        states,
        pair_addresses,
        batch_size=batch_size,
        block_size=block_size,
        nao=nao,
        naux=naux,
        device_timing=short_range_device_timing,
    )
    result += short_range
    timeline.mark("short_range_derivative")
    ewald = _ewald_derivative(
        cell,
        states,
        exxdiv=exxdiv,
        omega=omega,
    )
    result += ewald
    timeline.mark("ewald_derivative")
    log.debug1("completed one-pass Gamma signed GDF derivative")
    host_result = np.asarray(result, dtype=float)
    if component_diagnostics is not None:
        component_diagnostics.clear()
        component_diagnostics.update(
            {
                "metric": np.asarray(metric_derivative, dtype=float),
                "long_range_and_auxiliary_center": np.asarray(
                    long_range, dtype=float
                ),
                "short_range": np.asarray(short_range, dtype=float),
                "ewald": np.asarray(ewald, dtype=float),
            }
        )
    if memory_diagnostics is not None:
        memory_diagnostics.clear()
        memory_diagnostics.update(
            {
                "route": "gamma",
                "short_range_omega": float(int3c2e_opt.omega),
                "plan": reciprocal.memory_plan.metadata(),
                "zero_order_execution": (
                    reciprocal.zero_order_execution.metadata()
                ),
                "derivative_execution": derivative_execution.metadata(),
                "device_phase_timing": timeline.metadata(),
                "short_range_device_timing": short_range_device_timing,
            }
        )
    return host_result


def _state_ao_count(states):
    """Return and validate the common sorted-AO dimension of signed states."""

    shapes = {tuple(state.density.shape) for state in states}
    if len(shapes) != 1:
        raise ValueError(f"Gamma signed states have inconsistent shapes: {shapes}")
    shape = shapes.pop()
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Gamma sorted density must be square, got {shape}")
    return int(shape[0])


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
    memory_plan: object
    zero_order_execution: object

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
        max_block_size=None,
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
        nao = int(states[0].density.shape[0])
        naux = int(auxcell.ao_loc[-1])
        nocc_max = max(state.nocc for state in states)
        memory_plan = plan_gamma_reciprocal_blocks(
            len(vectors),
            nao,
            naux,
            nocc_max,
            free_device_bytes=free,
            max_block_size=max_block_size,
        )

        def allocate_block(p0, p1):
            ngrid = int(p1 - p0)
            aux_g = ft_ao.ft_ao(auxcell, vectors[p0:p1]).T
            aux_weighted = cp.empty_like(aux_g)
            cp.multiply(aux_g, coulomb[p0:p1], out=aux_weighted)
            pair_g = evaluator(vectors[p0:p1])
            pair_g = pair_g.view(np.float64).reshape(
                nao, nao, ngrid * 2
            )
            pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]
            return _ZeroOrderReciprocalBlock(
                aux_g=aux_g,
                aux_weighted=aux_weighted,
                pair_g=pair_g,
                tmp=cp.empty((nocc_max, nao, ngrid * 2)),
                occupied=cp.empty((nocc_max, nocc_max, ngrid * 2)),
            )

        def accept_block(_p0, _p1, workspace):
            contract(
                "iG,jG->ij",
                workspace.aux_g.view(np.float64),
                workspace.aux_weighted.view(np.float64),
                beta=1,
                out=j2c,
            )
            for state in states:
                tmp = workspace.tmp[: state.nocc]
                occupied_g = workspace.occupied[
                    : state.nocc, : state.nocc
                ]
                contract(
                    "pqG,pi->iqG",
                    workspace.pair_g,
                    state.factor_r,
                    out=tmp,
                )
                contract(
                    "iqG,qj->ijG",
                    tmp,
                    state.factor_l,
                    out=occupied_g,
                )
                contract(
                    "rG,ijG->rij",
                    workspace.aux_weighted.view(np.float64),
                    occupied_g,
                    beta=1,
                    out=state.j3c_oo,
                )

        execution = execute_adaptive_reciprocal_blocks(
            len(vectors),
            memory_plan.initial_block_size,
            allocate_block,
            accept_block,
            phase="zero_order",
            is_oom=_is_gpu_out_of_memory,
            block_alignment=memory_plan.block_alignment,
            on_oom=_release_cached_gpu_blocks,
        )
        return cls(
            j2c=j2c,
            ft_opt=ft_opt,
            evaluator=evaluator,
            coulomb=coulomb,
            vectors=vectors,
            block_size=execution.final_block_size,
            memory_plan=memory_plan,
            zero_order_execution=execution,
        )


@dataclass(frozen=True)
class _ZeroOrderReciprocalBlock:
    """Fully allocated workspace committed only after allocation succeeds."""

    aux_g: cp.ndarray
    aux_weighted: cp.ndarray
    pair_g: cp.ndarray
    tmp: cp.ndarray
    occupied: cp.ndarray


def _is_gpu_out_of_memory(error) -> bool:
    if isinstance(error, cp.cuda.memory.OutOfMemoryError):
        return True
    message = str(error).lower()
    return (
        error.__class__.__name__ == "CUDARuntimeError"
        and (
            "out of memory" in message
            or "cudaerrormemoryallocation" in message
        )
    )


def _release_cached_gpu_blocks(_start, _stop, _error):
    cp.get_default_memory_pool().free_all_blocks()


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
    primitive_source=None,
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
    pair_addresses = ft_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False
    )[0]
    i_addr, j_addr = divmod(pair_addresses, nao)

    def worker(p0, p1):
        if primitive_source is not None:
            return _evaluate_long_range_adjoint_block(
                cell,
                auxcell,
                primitive_source,
                metric_source,
                vectors,
                coulomb,
                evaluator,
                p0,
                p1,
                nao=nao,
                naux=naux,
                i_addr=i_addr,
                j_addr=j_addr,
                kernel=kernel,
                aft_envs=aft_envs,
                shell_offsets=shell_offsets,
                shared_memory=shared_memory,
                bas_ij_idx=bas_ij_idx,
                bas_ij_img_idx=bas_ij_img_idx,
                permutation_symmetry=ft_opt.permutation_symmetry,
            )
        return _evaluate_long_range_block(
            cell,
            auxcell,
            states,
            metric_source,
            vectors,
            coulomb,
            evaluator,
            p0,
            p1,
            nao=nao,
            naux=naux,
            i_addr=i_addr,
            j_addr=j_addr,
            kernel=kernel,
            aft_envs=aft_envs,
            shell_offsets=shell_offsets,
            shared_memory=shared_memory,
            bas_ij_idx=bas_ij_idx,
            bas_ij_img_idx=bas_ij_img_idx,
            permutation_symmetry=ft_opt.permutation_symmetry,
        )

    def accept_result(_p0, _p1, value):
        ao_derivative[:] += value.ao_derivative
        auxiliary_partial[:] += value.auxiliary_partial

    execution = execute_adaptive_reciprocal_blocks(
        len(vectors),
        block_size,
        worker,
        accept_result,
        phase="long_range_derivative",
        is_oom=_is_gpu_out_of_memory,
        block_alignment=reciprocal.memory_plan.block_alignment,
        on_oom=_release_cached_gpu_blocks,
    )

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
    return result, execution


def _evaluate_long_range_adjoint_block(
    cell,
    auxcell,
    primitive_source,
    metric_source,
    vectors,
    coulomb,
    evaluator,
    p0,
    p1,
    *,
    nao,
    naux,
    i_addr,
    j_addr,
    kernel,
    aft_envs,
    shell_offsets,
    shared_memory,
    bas_ij_idx,
    bas_ij_img_idx,
    permutation_symmetry,
):
    """Contract one reciprocal block with direct ``dB``/``dM`` adjoints."""

    ngrid = int(p1 - p0)
    pair_g = evaluator(vectors[p0:p1])
    pair_g = pair_g.view(np.float64).reshape(nao, nao, ngrid * 2)
    pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]

    aux_g = ft_ao.ft_ao(auxcell, vectors[p0:p1]).T
    aux_weighted_conj = cp.empty_like(aux_g)
    cp.conj(aux_g, out=aux_weighted_conj)
    aux_weighted_conj *= coulomb[p0:p1]
    aux_weighted_real = aux_weighted_conj.view(np.float64)

    derivative_aux_source = cp.empty((naux, ngrid * 2))
    contract(
        "rpq,pqG->rG",
        primitive_source,
        pair_g,
        out=derivative_aux_source,
    )
    ao_source = cp.empty((nao, nao, ngrid * 2))
    contract(
        "rpq,rG->pqG",
        primitive_source,
        aux_weighted_real,
        out=ao_source,
    )
    contract(
        "sr,sG->rG",
        metric_source,
        aux_g.view(np.float64),
        alpha=-1.0,
        beta=1.0,
        out=derivative_aux_source,
    )
    derivative_aux_complex = derivative_aux_source.view(np.complex128)
    derivative_aux_complex *= coulomb[p0:p1]

    derivative_aux_g = cp.empty((naux, ngrid), dtype=np.complex128)
    auxiliary_partial = cp.empty((3, naux))
    for axis in range(3):
        cp.multiply(
            aux_g,
            -1j * vectors[p0:p1, axis],
            out=derivative_aux_g,
        )
        auxiliary_partial[axis] = cp.einsum(
            "ag,ag->a",
            derivative_aux_g.view(np.float64),
            derivative_aux_source,
        )

    ao_derivative = cp.zeros((cell.natm, 3))
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
        ctypes.c_int(permutation_symmetry),
    )
    if error != 0:
        raise RuntimeError("PBC_ft_aopair_ek_ip1 failed")
    return _LongRangeBlockResult(
        ao_derivative=ao_derivative,
        auxiliary_partial=auxiliary_partial,
    )


@dataclass(frozen=True)
class _LongRangeBlockResult:
    """Block-local derivative increments committed after successful work."""

    ao_derivative: cp.ndarray
    auxiliary_partial: cp.ndarray


def _evaluate_long_range_block(
    cell,
    auxcell,
    states,
    metric_source,
    vectors,
    coulomb,
    evaluator,
    p0,
    p1,
    *,
    nao,
    naux,
    i_addr,
    j_addr,
    kernel,
    aft_envs,
    shell_offsets,
    shared_memory,
    bas_ij_idx,
    bas_ij_img_idx,
    permutation_symmetry,
):
    """Evaluate one reciprocal block without mutating global accumulators."""

    ngrid = int(p1 - p0)
    pair_g = evaluator(vectors[p0:p1])
    pair_g = pair_g.view(np.float64).reshape(nao, nao, ngrid * 2)
    pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]

    aux_g = ft_ao.ft_ao(auxcell, vectors[p0:p1]).T
    derivative_aux_source = cp.zeros((naux, ngrid * 2))
    ao_source = cp.zeros((nao, nao, ngrid * 2))
    aux_weighted_conj = cp.empty_like(aux_g)
    cp.conj(aux_g, out=aux_weighted_conj)
    aux_weighted_conj *= coulomb[p0:p1]
    aux_weighted_real = aux_weighted_conj.view(np.float64)

    nocc_max = max(state.nocc for state in states)
    tmp_workspace = cp.empty((nocc_max, nao, ngrid * 2))
    occupied_workspace = cp.empty(
        (nocc_max, nocc_max, ngrid * 2)
    )
    rho_g = cp.empty(ngrid * 2)
    potential_g = cp.empty(ngrid * 2)

    for state in states:
        tmp = tmp_workspace[: state.nocc]
        occupied_g = occupied_workspace[: state.nocc, : state.nocc]
        contract("pqG,qp->G", pair_g, state.density, out=rho_g)
        contract(
            "r,G->rG",
            state.auxvec,
            rho_g,
            alpha=state.coulomb_weight,
            beta=1.0,
            out=derivative_aux_source,
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

        contract(
            "rij,rG->ijG",
            state.j3c_oo,
            aux_weighted_real,
            out=occupied_g,
        )
        contract("jiG,qi->jqG", occupied_g, state.factor_r, out=tmp)
        contract(
            "jqG,pj->pqG",
            tmp,
            state.factor_l,
            alpha=-0.5 * state.exchange_weight,
            beta=1.0,
            out=ao_source,
        )
        contract(
            "r,rG->G",
            state.auxvec,
            aux_weighted_real,
            out=potential_g,
        )
        contract(
            "pq,G->pqG",
            state.density,
            potential_g,
            alpha=state.coulomb_weight,
            beta=1.0,
            out=ao_source,
        )

    contract(
        "sr,sG->rG",
        metric_source,
        aux_g.view(np.float64),
        alpha=-1.0,
        beta=1.0,
        out=derivative_aux_source,
    )
    derivative_aux_complex = derivative_aux_source.view(np.complex128)
    derivative_aux_complex *= coulomb[p0:p1]

    derivative_aux_g = cp.empty((naux, ngrid), dtype=np.complex128)
    auxiliary_partial = cp.empty((3, naux))
    for axis in range(3):
        cp.multiply(
            aux_g,
            -1j * vectors[p0:p1, axis],
            out=derivative_aux_g,
        )
        auxiliary_partial[axis] = cp.einsum(
            "ag,ag->a",
            derivative_aux_g.view(np.float64),
            derivative_aux_source,
        )

    ao_derivative = cp.zeros((cell.natm, 3))
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
        ctypes.c_int(permutation_symmetry),
    )
    if error != 0:
        raise RuntimeError("PBC_ft_aopair_ek_ip1 failed")
    return _LongRangeBlockResult(
        ao_derivative=ao_derivative,
        auxiliary_partial=auxiliary_partial,
    )


def _short_range_derivative(
    int3c2e_opt,
    states,
    pair_addresses,
    *,
    batch_size,
    block_size,
    nao,
    naux,
    device_timing=None,
    primitive_source=None,
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
    auxiliary_shell_counts = np.diff(aux_shell_offsets)
    maximum_auxiliary_shells = max(1, int(auxiliary_shell_counts.max()))
    shell_pair_counts = np.diff(cp.asnumpy(shell_pair_offsets))
    maximum_shell_pairs = max(1, int(shell_pair_counts.max()))
    maximum_tasks = maximum_shell_pairs * maximum_auxiliary_shells
    if maximum_tasks > POOL_SIZE:
        raise RuntimeError(
            "Gamma GDF SR derivative batch exceeds CUDA POOL_SIZE: "
            f"{maximum_tasks} > {POOL_SIZE}"
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
    timing_events = []

    for batch_id, _ in enumerate(aux_patterns[:, 0]):
        source_start = cp.cuda.Event() if device_timing is not None else None
        if source_start is not None:
            source_start.record()
        aux_ao_offset = auxcell.ao_loc[aux_shell_offsets[batch_id]]
        naux_batch = (
            auxcell.ao_loc[aux_shell_offsets[batch_id + 1]] - aux_ao_offset
        )
        compressed = cp.empty((len(pair_addresses), naux_batch))
        for k0, k1 in lib.prange(0, naux_batch, block_size):
            width = k1 - k0
            aux1 = aux0 + width
            if primitive_source is None:
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
            else:
                source = primitive_source[aux0:aux1].transpose(1, 2, 0)
            cp.take(
                source.reshape(-1, width),
                pair_addresses,
                axis=0,
                out=compressed[:, k0:k1],
            )
            aux0 = aux1

        source_end = cp.cuda.Event() if device_timing is not None else None
        if source_end is not None:
            source_end.record()

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
        if device_timing is not None:
            kernel_end = cp.cuda.Event()
            kernel_end.record()
            timing_events.append((source_start, source_end, kernel_end))

    ao_derivative *= 2.0
    auxiliary_derivative *= 2.0
    result = (ao_derivative + auxiliary_derivative).get()
    if device_timing is not None:
        source_seconds = sum(
            cp.cuda.get_elapsed_time(start, source_end) * 1.0e-3
            for start, source_end, _kernel_end in timing_events
        )
        kernel_seconds = sum(
            cp.cuda.get_elapsed_time(source_end, kernel_end) * 1.0e-3
            for _start, source_end, kernel_end in timing_events
        )
        device_timing.clear()
        device_timing.update(
            {
                "measurement": (
                    "CUDA event elapsed time; host-only work excluded"
                ),
                "batch_count": len(timing_events),
                "shell_pair_group_count": int(len(shell_pair_counts)),
                "shell_pair_limit": int(shell_pair_limit),
                "maximum_shell_pairs_per_group": maximum_shell_pairs,
                "maximum_auxiliary_shells_per_batch": (
                    maximum_auxiliary_shells
                ),
                "maximum_tasks_per_batch": int(maximum_tasks),
                "task_pool_size": int(POOL_SIZE),
                "source_contraction_s": float(source_seconds),
                "integral_kernel_s": float(kernel_seconds),
            }
        )
    return result


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
