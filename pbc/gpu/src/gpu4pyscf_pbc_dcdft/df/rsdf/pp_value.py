"""Bounded GPU pseudopotential value assembly for periodic RSDF.

The integral decomposition follows GPU4PySCF's RSDF PP builder.  This module
changes only the reciprocal-grid execution policy: local-part-1 Fourier
integrals are accumulated in bounded, transactional blocks instead of using
most of the currently free device memory in one allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.df.int3c2e import (
    SRInt3c2eOpt,
    fill_triu_bvk,
)
from gpu4pyscf.pbc.df.rsdf_builder import _weighted_coulG_LR
from gpu4pyscf.pbc.gto.pseudo.pp_int import get_pp_nl_gpu
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from pyscf.pbc.df import aft as aft_cpu
from pyscf.pbc.df.rsdf_builder import estimate_ke_cutoff_for_omega
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh

from ..gdf.reciprocal_memory import execute_adaptive_reciprocal_blocks


DEFAULT_PP_MEMORY_FRACTION = 0.20
DEFAULT_PP_MINIMUM_RESERVE_BYTES = 512 * 1024**2
DEFAULT_PP_MAX_GRID_BLOCK_SIZE = 4096
DEFAULT_PP_BLOCK_ALIGNMENT = 32


@dataclass(frozen=True)
class PPValueResult:
    """Pseudopotential AO matrix and JSON-safe execution metadata."""

    value: cp.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class _PPGridPlan:
    ngrids: int
    nao_pairs: int
    free_device_bytes: int
    budget_bytes: int
    reserve_bytes: int
    bytes_per_grid: int
    initial_block_size: int

    def metadata(self):
        return {
            "ngrids": self.ngrids,
            "nao_pairs": self.nao_pairs,
            "free_device_bytes_at_plan": self.free_device_bytes,
            "memory_fraction": DEFAULT_PP_MEMORY_FRACTION,
            "minimum_reserve_bytes": DEFAULT_PP_MINIMUM_RESERVE_BYTES,
            "max_grid_block_size": DEFAULT_PP_MAX_GRID_BLOCK_SIZE,
            "reserve_bytes": self.reserve_bytes,
            "budget_bytes": self.budget_bytes,
            "bytes_per_grid": self.bytes_per_grid,
            "block_alignment": DEFAULT_PP_BLOCK_ALIGNMENT,
            "initial_block_size": self.initial_block_size,
            "estimated_block_working_bytes": (
                self.initial_block_size * self.bytes_per_grid
            ),
        }


def build_pp_value(cell, kpts=None) -> PPValueResult:
    """Build the GTH pseudopotential matrix with bounded local-part-1 GPU work."""

    points = None if kpts is None else np.asarray(kpts, dtype=float)
    is_single_kpt = points is not None and points.ndim == 1
    local_part2 = aft_cpu._IntPPBuilder(cell, points).get_pp_loc_part2()
    value = cp.asarray(local_part2)
    value += cp.asarray(get_pp_nl_gpu(cell, points))
    local_part1, reciprocal_metadata = _get_pp_loc_part1(cell, points)
    value += local_part1
    if is_single_kpt and value.ndim == 3:
        value = value[0]
    return PPValueResult(
        value=value,
        metadata={
            "builder": "project-local bounded GPU4PySCF RSDF PP",
            "local_part1": reciprocal_metadata,
            "local_part2_owner": "PySCF host primitive",
            "nonlocal_owner": "GPU4PySCF device kernel",
        },
    )


def _get_pp_loc_part1(cell, kpts):
    is_single_kpt = kpts is not None and kpts.ndim == 1
    is_gamma_point = kpts is None or is_zero(kpts)
    if is_gamma_point:
        bvk_kmesh = np.ones(3, dtype=int)
        bvk_ncells = 1
    else:
        bvk_kmesh = kpts_to_kmesh(cell, kpts, bound_by_supmol=True)
        bvk_ncells = int(np.prod(bvk_kmesh))
    if is_single_kpt:
        kpts = kpts.reshape(1, 3)

    omega = 0.2
    fakenuc = aft_cpu._fake_nuc(cell, with_pseudo=True)
    optimizer = SRInt3c2eOpt(
        cell,
        fakenuc,
        omega=-omega,
        bvk_kmesh=bvk_kmesh,
    ).build()
    charges = -cp.asarray(cell.atom_charges(), dtype=np.float64)
    local = optimizer.contract_auxvec(charges, kpts)

    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    mesh = cell.symmetrize_mesh(cell.cutoff_to_mesh(ke_cutoff))
    vectors, _bases, weights = cell.get_Gv_weights(mesh)
    auxiliary_g = ft_ao.ft_ao(fakenuc, vectors).conj()
    weighted_charge = auxiliary_g.dot(charges)
    weighted_charge *= _weighted_coulG_LR(
        cell,
        vectors,
        omega,
        weights,
    )
    if cell.dimension == 3 or (
        cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum"
    ):
        exponents = cp.asarray(np.hstack(fakenuc.bas_exps()))
        weighted_charge[0] -= charges.dot(np.pi / exponents) / cell.vol

    ft_optimizer = ft_ao.FTOpt(cell, bvk_kmesh=bvk_kmesh).build()
    sorted_cell = ft_optimizer.cell
    pair_address = ft_optimizer.pair_and_diag_indices(
        cart=True,
        original_ao_order=False,
    )[0]
    evaluate = ft_optimizer.ft_evaluator(
        cart=True,
        original_ao_order=False,
    )[0]
    plan = _plan_grid_blocks(len(vectors), len(pair_address))
    compressed = cp.zeros(len(pair_address), dtype=cp.float64)

    def worker(start, stop):
        block_size = int(stop - start)
        buffer = cp.empty(
            len(pair_address) * block_size,
            dtype=cp.complex128,
        )
        pair_g = evaluate(vectors[start:stop], out=buffer)
        return contract(
            "pG,G->p",
            pair_g,
            weighted_charge[start:stop],
        ).real

    def accept(_start, _stop, block):
        compressed[:] += block

    execution = execute_adaptive_reciprocal_blocks(
        len(vectors),
        plan.initial_block_size,
        worker,
        accept,
        phase="rsdf_pp_local_part1",
        is_oom=_is_gpu_out_of_memory,
        block_alignment=DEFAULT_PP_BLOCK_ALIGNMENT,
        on_oom=_release_cached_blocks,
    )

    nao = int(sorted_cell.nao)
    raw = cp.zeros(nao * bvk_ncells * nao)
    raw[pair_address] = compressed
    raw = raw.reshape(nao, bvk_ncells, nao).transpose(1, 0, 2)
    raw = fill_triu_bvk(cp.asarray(raw, order="C"), nao, bvk_kmesh)
    raw = sorted_cell.apply_CT_mat_C(raw)
    if is_gamma_point:
        raw = raw[0]
    else:
        translations = translation_vectors_for_kmesh(
            sorted_cell,
            bvk_kmesh,
            True,
        )
        phase = cp.exp(1j * cp.asarray(translations.dot(kpts.T)))
        raw = contract("lk,lpq->kpq", phase, raw)
    local += raw
    if is_single_kpt and local.ndim == 3:
        local = local[0]
    return local, {
        "mesh": [int(value) for value in mesh],
        "omega": omega,
        "plan": plan.metadata(),
        "execution": execution.metadata(),
    }


def _plan_grid_blocks(ngrids, nao_pairs):
    free_bytes = int(get_avail_mem(exclude_memory_pool=True))
    fraction_budget = int(free_bytes * DEFAULT_PP_MEMORY_FRACTION)
    reserve_budget = max(
        0,
        free_bytes - DEFAULT_PP_MINIMUM_RESERVE_BYTES,
    )
    budget = min(fraction_budget, reserve_budget)
    bytes_per_grid = int(nao_pairs) * np.dtype(np.complex128).itemsize
    raw_block = min(
        int(ngrids),
        DEFAULT_PP_MAX_GRID_BLOCK_SIZE,
        budget // bytes_per_grid,
    )
    if raw_block < 1:
        raise MemoryError(
            "insufficient GPU memory for one RSDF PP reciprocal vector: "
            f"budget={budget} bytes, required={bytes_per_grid} bytes"
        )
    block_size = int(raw_block)
    if block_size >= DEFAULT_PP_BLOCK_ALIGNMENT:
        block_size = max(
            DEFAULT_PP_BLOCK_ALIGNMENT,
            block_size // DEFAULT_PP_BLOCK_ALIGNMENT
            * DEFAULT_PP_BLOCK_ALIGNMENT,
        )
    return _PPGridPlan(
        ngrids=int(ngrids),
        nao_pairs=int(nao_pairs),
        free_device_bytes=free_bytes,
        budget_bytes=budget,
        reserve_bytes=free_bytes - budget,
        bytes_per_grid=bytes_per_grid,
        initial_block_size=block_size,
    )


def _is_gpu_out_of_memory(error):
    if isinstance(error, cp.cuda.memory.OutOfMemoryError):
        return True
    message = str(error).lower()
    return error.__class__.__name__ == "CUDARuntimeError" and (
        "out of memory" in message
        or "cudaerrormemoryallocation" in message
    )


def _release_cached_blocks(_start, _stop, _error):
    cp.get_default_memory_pool().free_all_blocks()
