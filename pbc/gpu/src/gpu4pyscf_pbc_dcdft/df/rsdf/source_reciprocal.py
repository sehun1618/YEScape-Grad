"""Bounded reciprocal contractions for RSDF source derivatives."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import asarray, get_avail_mem
from gpu4pyscf.pbc.df import aft_jk, ft_ao
from gpu4pyscf.pbc.df.int3c2e import libpbc
from gpu4pyscf.pbc.df.rsdf_builder import _weighted_coulG_kpts
from gpu4pyscf.pbc.tools.pbc import _Gv_wrap_around
from pyscf.gto import ATOM_OF

from ..gdf.reciprocal_memory import (
    execute_adaptive_reciprocal_blocks,
    plan_gamma_reciprocal_blocks,
)
from .packed_gamma import (
    SortedPackedGammaSources,
    packed_pair_weights,
)


DEFAULT_SOURCE_RECIPROCAL_MEMORY_FRACTION = 0.15
DEFAULT_SOURCE_RECIPROCAL_MAX_BLOCK_SIZE = 512


@dataclass(frozen=True)
class _BlockResult:
    ao_derivative: cp.ndarray
    auxiliary_partial: cp.ndarray


def contract_gamma(optimizer, source):
    """Contract Gamma reciprocal and zero-transfer source derivatives."""

    return contract_gamma_batch(optimizer, cp.asarray(source)[None])[0]


def contract_gamma_batch(optimizer, sources, *, diagnostics=None):
    """Contract several Gamma sources with one Fourier-integral traversal."""

    sources = cp.asarray(sources)
    if sources.ndim != 4:
        raise ValueError(
            "Gamma sources must have shape (nsource,nao,nao,naux), "
            f"got {sources.shape}"
        )
    cell = optimizer.cell
    auxcell = optimizer.auxcell
    mesh = np.asarray(optimizer.mesh, dtype=int)
    vectors = asarray(auxcell.get_Gv(mesh))
    coulomb = _gamma_coulomb_vector(
        _weighted_coulG_kpts(
            cell,
            mesh,
            0.0,
            float(optimizer.omega),
        ),
        ngrids=len(vectors),
    )
    ft_opt = ft_ao.FTOpt.from_intopt(optimizer)
    evaluator = ft_opt.ft_evaluator(
        compressing=False,
        cart=True,
        original_ao_order=False,
    )[0]
    kernel_inputs = _fourier_kernel_inputs(cell, ft_opt)
    pair_addresses = ft_opt.pair_and_diag_indices(
        cart=True,
        original_ao_order=False,
    )[0]
    source_count = len(sources)
    nao = int(sources.shape[1])
    naux = int(sources.shape[3])
    i_addr, j_addr = divmod(pair_addresses, nao)
    plan = _memory_plan(
        len(vectors),
        nao,
        naux * source_count,
    )
    ao_derivative = cp.zeros((source_count, cell.natm, 3))
    auxiliary_partial = cp.zeros((source_count, 3, naux))

    def worker(start, stop):
        return _gamma_block(
            cell,
            auxcell,
            sources,
            vectors,
            coulomb,
            evaluator,
            start,
            stop,
            nao=nao,
            naux=naux,
            i_addr=i_addr,
            j_addr=j_addr,
            permutation_symmetry=ft_opt.permutation_symmetry,
            **kernel_inputs,
        )

    execution = _execute_blocks(
        len(vectors),
        plan,
        worker,
        ao_derivative,
        auxiliary_partial,
        phase="rsdf_source_reciprocal",
    )
    _record_diagnostics(diagnostics, "gamma", plan, execution)
    return ao_derivative * 2.0 + _group_auxiliary_batch(
        auxcell,
        auxiliary_partial,
    )


def contract_gamma_packed_batch(optimizer, sources, *, diagnostics=None):
    """Contract packed Gamma sources with block-local dense AO workspaces."""

    if not isinstance(sources, SortedPackedGammaSources):
        raise TypeError("sources must be SortedPackedGammaSources")
    values = cp.asarray(sources.values)
    cell = optimizer.cell
    auxcell = optimizer.auxcell
    mesh = np.asarray(optimizer.mesh, dtype=int)
    vectors = asarray(auxcell.get_Gv(mesh))
    coulomb = _gamma_coulomb_vector(
        _weighted_coulG_kpts(
            cell,
            mesh,
            0.0,
            float(optimizer.omega),
        ),
        ngrids=len(vectors),
    )
    ft_opt = ft_ao.FTOpt.from_intopt(optimizer)
    evaluator = ft_opt.ft_evaluator(
        compressing=False,
        cart=True,
        original_ao_order=False,
    )[0]
    kernel_inputs = _fourier_kernel_inputs(cell, ft_opt)
    pair_addresses = ft_opt.pair_and_diag_indices(
        cart=True,
        original_ao_order=False,
    )[0]
    nao = int(sources.nao)
    naux = int(values.shape[-1])
    i_addr, j_addr = divmod(pair_addresses, nao)
    plan = _memory_plan(len(vectors), nao, naux * len(values))
    ao_derivative = cp.zeros((len(values), cell.natm, 3))
    auxiliary_partial = cp.zeros((len(values), 3, naux))

    def worker(start, stop):
        return _gamma_packed_block(
            cell,
            auxcell,
            values,
            vectors,
            coulomb,
            evaluator,
            start,
            stop,
            nao=nao,
            naux=naux,
            i_addr=i_addr,
            j_addr=j_addr,
            permutation_symmetry=ft_opt.permutation_symmetry,
            **kernel_inputs,
        )

    execution = _execute_blocks(
        len(vectors),
        plan,
        worker,
        ao_derivative,
        auxiliary_partial,
        phase="rsdf_packed_source_reciprocal",
    )
    _record_diagnostics(diagnostics, "gamma_packed", plan, execution)
    return ao_derivative * 2.0 + _group_auxiliary_batch(
        auxcell,
        auxiliary_partial,
    )


def contract_kpoint(optimizer, ordered_source, momentum, kpts):
    """Contract sampled-k reciprocal and zero-transfer source derivatives."""

    return contract_kpoint_batch(
        optimizer,
        cp.asarray(ordered_source)[None],
        momentum,
        kpts,
    )[0]


def contract_kpoint_batch(
    optimizer,
    ordered_sources,
    momentum,
    kpts,
    *,
    diagnostics=None,
):
    """Contract sampled-k sources with one Fourier-integral traversal."""

    ordered_sources = cp.asarray(ordered_sources)
    if ordered_sources.ndim != 6:
        raise ValueError(
            "sampled-k sources must have shape "
            "(nsource,nkpts,nkpts,nao,nao,naux), "
            f"got {ordered_sources.shape}"
        )
    cell = optimizer.cell
    auxcell = optimizer.auxcell
    mesh = tuple(int(value) for value in optimizer.mesh)
    vectors = np.asarray(auxcell.get_Gv(mesh))
    unique_count = len(momentum.unique_kpts)
    shifted = _shifted_vectors(
        auxcell,
        vectors,
        momentum.unique_kpts,
        mesh,
    )
    coulomb = _weighted_coulG_kpts(
        cell,
        mesh,
        0.0,
        float(optimizer.omega),
        momentum.unique_kpts,
    )
    ft_opt = ft_ao.FTOpt.from_intopt(optimizer)
    if not ft_opt.permutation_symmetry:
        raise NotImplementedError(
            "sampled-k RSDF source contraction requires "
            "permutation-symmetric Fourier pairs"
        )
    ft_kernel = ft_opt.gen_ft_kernel(transform_ao=False)
    kernel_inputs = _fourier_kernel_inputs(cell, ft_opt)
    source_count = len(ordered_sources)
    nao = int(ordered_sources.shape[3])
    naux = int(ordered_sources.shape[-1])
    plan = _memory_plan(
        len(vectors),
        nao,
        naux * unique_count * source_count,
    )
    ao_derivative = cp.zeros((source_count, cell.natm, 3))
    auxiliary_partial = cp.zeros((source_count, 3, naux))

    def worker(start, stop):
        return _kpoint_block(
            cell,
            auxcell,
            ordered_sources,
            momentum,
            kpts,
            vectors,
            shifted,
            coulomb,
            ft_kernel,
            start,
            stop,
            naux=naux,
            permutation_symmetry=ft_opt.permutation_symmetry,
            **kernel_inputs,
        )

    execution = _execute_blocks(
        len(vectors),
        plan,
        worker,
        ao_derivative,
        auxiliary_partial,
        phase="rsdf_kpoint_source_reciprocal",
    )
    _record_diagnostics(diagnostics, "sampled_k", plan, execution)
    return ao_derivative + _group_auxiliary_batch(
        auxcell,
        auxiliary_partial,
    )


def _gamma_block(
    cell,
    auxcell,
    sources,
    vectors,
    coulomb,
    evaluator,
    start,
    stop,
    *,
    nao,
    naux,
    i_addr,
    j_addr,
    permutation_symmetry,
    **kernel_inputs,
):
    width = int(stop - start)
    block_vectors = vectors[start:stop]
    pair_real = evaluator(block_vectors).view(np.float64).reshape(
        nao,
        nao,
        width * 2,
    )
    pair_real[j_addr, i_addr] = pair_real[i_addr, j_addr]
    pair_complex = pair_real.view(np.complex128)
    auxiliary_g = ft_ao.ft_ao(auxcell, block_vectors).T
    auxiliary_weighted = auxiliary_g.conj() * coulomb[start:stop]
    auxiliary_source = cp.einsum(
        "spqr,pqG->srG",
        sources,
        pair_complex,
        optimize=True,
    )
    auxiliary_source *= coulomb[start:stop]
    auxiliary_source = cp.ascontiguousarray(auxiliary_source).view(np.float64)
    auxiliary_partial = _auxiliary_phase_derivative(
        auxiliary_g,
        block_vectors,
        auxiliary_source,
    )
    del pair_complex, pair_real, auxiliary_source
    ao_source = cp.einsum(
        "spqr,rG->spqG",
        sources,
        auxiliary_weighted,
        optimize=True,
    )
    ao_source = cp.ascontiguousarray(ao_source).view(np.float64)
    ao_derivative = cp.stack(
        [
            _run_fourier_kernel(
                cell,
                source,
                block_vectors,
                width,
                permutation_symmetry=permutation_symmetry,
                **kernel_inputs,
            )
            for source in ao_source
        ]
    )
    return _BlockResult(ao_derivative, auxiliary_partial)


def _gamma_packed_block(
    cell,
    auxcell,
    sources,
    vectors,
    coulomb,
    evaluator,
    start,
    stop,
    *,
    nao,
    naux,
    i_addr,
    j_addr,
    permutation_symmetry,
    **kernel_inputs,
):
    """Consume packed sources while limiting dense AO storage to one G block."""

    width = int(stop - start)
    block_vectors = vectors[start:stop]
    pair_real = evaluator(block_vectors).view(np.float64).reshape(
        nao,
        nao,
        width * 2,
    )
    pair_real[j_addr, i_addr] = pair_real[i_addr, j_addr]
    pair_complex = pair_real.view(np.complex128)
    row, column = cp.tril_indices(nao)
    pair_packed = pair_complex[row, column]
    pair_weights = packed_pair_weights(nao, dtype=pair_packed.real.dtype)

    auxiliary_g = ft_ao.ft_ao(auxcell, block_vectors).T
    auxiliary_weighted = auxiliary_g.conj() * coulomb[start:stop]
    auxiliary_source = cp.einsum(
        "spr,pG,p->srG",
        sources,
        pair_packed,
        pair_weights,
        optimize=True,
    )
    auxiliary_source *= coulomb[start:stop]
    auxiliary_source = cp.ascontiguousarray(auxiliary_source).view(np.float64)
    auxiliary_partial = _auxiliary_phase_derivative(
        auxiliary_g,
        block_vectors,
        auxiliary_source,
    )
    del pair_complex, pair_real, pair_packed, auxiliary_source

    ao_packed = cp.einsum(
        "spr,rG->spG",
        sources,
        auxiliary_weighted,
        optimize=True,
    )
    ao_source = cp.empty((len(sources), nao, nao, width), dtype=ao_packed.dtype)
    ao_source[:, row, column] = ao_packed
    ao_source[:, column, row] = ao_packed
    del ao_packed
    ao_source = cp.ascontiguousarray(ao_source).view(np.float64)
    ao_derivative = cp.stack(
        [
            _run_fourier_kernel(
                cell,
                source,
                block_vectors,
                width,
                permutation_symmetry=permutation_symmetry,
                **kernel_inputs,
            )
            for source in ao_source
        ]
    )
    return _BlockResult(ao_derivative, auxiliary_partial)


def _kpoint_block(
    cell,
    auxcell,
    ordered_sources,
    momentum,
    kpts,
    vectors,
    shifted,
    coulomb,
    ft_kernel,
    start,
    stop,
    *,
    naux,
    permutation_symmetry,
    **kernel_inputs,
):
    width = int(stop - start)
    auxiliary_g = ft_ao.ft_ao(
        auxcell,
        shifted[:, start:stop].reshape(-1, 3),
    ).T.reshape(naux, len(momentum.unique_kpts), width)
    source_count = len(ordered_sources)
    ao_derivative = cp.zeros((source_count, cell.natm, 3))
    auxiliary_partial = cp.zeros((source_count, 3, naux))
    for q_index, (kp, kp_conj, ki_idx, kj_idx) in enumerate(
        momentum.iterations
    ):
        pair_g = ft_kernel(
            vectors[start:stop],
            kpts[kp],
            kpts,
            kj_idx,
        )
        pair_g = pair_g.transpose(0, 2, 3, 1)[kj_idx]
        pair_sources = ordered_sources[:, ki_idx, kj_idx]
        auxiliary_source = cp.einsum(
            "skpqr,kpqG->srG",
            pair_sources,
            pair_g,
            optimize=True,
        )
        auxiliary_source *= coulomb[q_index, start:stop]
        auxiliary_source = cp.ascontiguousarray(auxiliary_source).view(
            np.float64
        )
        factor = 2.0 if kp != kp_conj else 1.0
        auxiliary_partial += factor * _auxiliary_phase_derivative(
            auxiliary_g[:, q_index],
            shifted[q_index, start:stop],
            auxiliary_source,
        )
        del pair_g, auxiliary_source
        ao_source = _kpoint_ao_source(
            pair_sources,
            auxiliary_g[:, q_index],
            coulomb[q_index, start:stop],
            momentum,
            kj_idx,
            permutation_symmetry=permutation_symmetry,
        )
        if kp != kp_conj:
            ao_source *= 2.0
        for source_id, source in enumerate(ao_source):
            ao_derivative[source_id] += _run_fourier_kernel(
                cell,
                cp.asarray(source, order="C"),
                vectors[start:stop] + kpts[kp],
                width,
                permutation_symmetry=permutation_symmetry,
                **kernel_inputs,
            )
    return _BlockResult(ao_derivative, auxiliary_partial)


def _kpoint_ao_source(
    pair_sources,
    auxiliary_g,
    coulomb,
    momentum,
    kj_idx,
    *,
    permutation_symmetry,
):
    weighted = auxiliary_g.conj() * coulomb
    ao_g = cp.einsum(
        "skpqr,rG->skpqG",
        pair_sources,
        weighted,
        optimize=True,
    )
    result = cp.einsum(
        "Lk,skpqG->sLqpG",
        momentum.exp_lk[:, kj_idx],
        ao_g,
        optimize=True,
    )
    if permutation_symmetry:
        result += cp.einsum(
            "Lk,skpqG->sLpqG",
            momentum.exp_lk.conj(),
            ao_g,
            optimize=True,
        )
    return result


def _auxiliary_phase_derivative(auxiliary_g, vectors, source_real):
    naux = auxiliary_g.shape[0]
    result = cp.empty((len(source_real), 3, naux))
    for axis in range(3):
        derivative = auxiliary_g * (-1j * vectors[:, axis])
        derivative = cp.ascontiguousarray(derivative).view(np.float64)
        result[:, axis] = cp.einsum(
            "rG,srG->sr",
            derivative,
            source_real,
            optimize=True,
        )
    return result


def _fourier_kernel_inputs(cell, ft_opt):
    bas_ij_idx, bas_ij_img_idx, shell_offsets = (
        aft_jk._generate_shl_pairs(ft_opt)
    )
    return {
        "kernel": libpbc.PBC_ft_aopair_ek_ip1,
        "aft_envs": ft_opt.aft_envs,
        "shell_offsets": shell_offsets,
        "shared_memory": aft_jk._estimate_max_shm_size(cell, (1, 0)),
        "bas_ij_idx": bas_ij_idx,
        "bas_ij_img_idx": bas_ij_img_idx,
    }


def _run_fourier_kernel(
    cell,
    source,
    vectors,
    width,
    *,
    kernel,
    aft_envs,
    shell_offsets,
    shared_memory,
    bas_ij_idx,
    bas_ij_img_idx,
    permutation_symmetry,
):
    output = cp.zeros((cell.natm, 3))
    vector_transpose = cp.asarray(vectors.T.ravel())
    error = kernel(
        ctypes.cast(output.data.ptr, ctypes.c_void_p),
        ctypes.cast(source.data.ptr, ctypes.c_void_p),
        ctypes.cast(vector_transpose.data.ptr, ctypes.c_void_p),
        ctypes.byref(aft_envs),
        ctypes.c_int(len(shell_offsets) - 1),
        ctypes.c_int(width),
        ctypes.c_int(shared_memory),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(shell_offsets.data.ptr, ctypes.c_void_p),
        ctypes.c_int(permutation_symmetry),
    )
    if error != 0:
        raise RuntimeError("PBC_ft_aopair_ek_ip1 failed")
    return output


def _execute_blocks(
    count,
    plan,
    worker,
    ao_derivative,
    auxiliary_partial,
    *,
    phase,
):
    def accept(_start, _stop, block):
        ao_derivative[:] += block.ao_derivative
        auxiliary_partial[:] += block.auxiliary_partial

    return execute_adaptive_reciprocal_blocks(
        count,
        plan.initial_block_size,
        worker,
        accept,
        phase=phase,
        is_oom=_is_gpu_out_of_memory,
        block_alignment=plan.block_alignment,
        on_oom=_release_cached_blocks,
    )


def _memory_plan(ngrids, nao, naux):
    return plan_gamma_reciprocal_blocks(
        ngrids,
        nao,
        naux,
        1,
        free_device_bytes=int(get_avail_mem(exclude_memory_pool=True)),
        memory_fraction=DEFAULT_SOURCE_RECIPROCAL_MEMORY_FRACTION,
        max_block_size=DEFAULT_SOURCE_RECIPROCAL_MAX_BLOCK_SIZE,
    )


def _record_diagnostics(diagnostics, route, plan, execution):
    if diagnostics is None:
        return
    diagnostics.clear()
    diagnostics.update(
        {
            "route": str(route),
            "policy": {
                "memory_fraction": (
                    DEFAULT_SOURCE_RECIPROCAL_MEMORY_FRACTION
                ),
                "max_block_size": (
                    DEFAULT_SOURCE_RECIPROCAL_MAX_BLOCK_SIZE
                ),
                "lifetime_shortening": (
                    "AO-pair and source-contracted grid tensors do not "
                    "overlap"
                ),
            },
            "plan": plan.metadata(),
            "execution": execution.metadata(),
        }
    )


def _group_auxiliary(auxcell, auxiliary_partial):
    dimensions = auxcell.ao_loc[1:] - auxcell.ao_loc[:-1]
    atom_ids = cp.asarray(
        np.repeat(auxcell._bas[:, ATOM_OF], dimensions),
        dtype=cp.int32,
    )
    grouped = cp.zeros((auxcell.natm, 3))
    cp.add.at(grouped, atom_ids, auxiliary_partial.T)
    return grouped


def _group_auxiliary_batch(auxcell, auxiliary_partials):
    return cp.stack(
        [
            _group_auxiliary(auxcell, partial)
            for partial in auxiliary_partials
        ]
    )


def _shifted_vectors(auxcell, vectors, unique_kpts, mesh):
    shifted = (
        asarray(vectors) + asarray(unique_kpts)[:, None]
    ).reshape(-1, 3)
    return _Gv_wrap_around(
        auxcell,
        shifted,
        cp.zeros(3),
        mesh,
    ).reshape(len(unique_kpts), len(vectors), 3)


def _gamma_coulomb_vector(value, *, ngrids):
    vector = asarray(value)
    if vector.ndim == 2:
        if vector.shape[0] != 1:
            raise ValueError(
                "Gamma RSDF expected one Coulomb-weight row, got "
                f"{vector.shape}"
            )
        vector = vector[0]
    if vector.ndim != 1 or vector.size != int(ngrids):
        raise ValueError(
            "Gamma RSDF Coulomb weights do not match the source mesh: "
            f"{vector.shape} and {ngrids}"
        )
    return vector


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
