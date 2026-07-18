"""Adjoints that contract RSDF source derivatives before CDERI assembly.

For one canonical k-point pair, RSDF stores

``C = L^-1 B.T``

where ``B`` has AO-pair-by-auxiliary ordering.  The DC-DFT response needs
only two real scalar derivatives.  This module moves their linear weights
backwards through the triangular solve, producing source adjoints ``W_B``
such that ``Re(sum(W_B * dB))`` equals the established ``dCDERI``
contraction.  A native integral kernel can therefore contract ``dB`` in
place without materializing one source derivative tensor per atom and axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas
from cupyx.scipy.linalg import solve_triangular

from ..gdf.jk_derivative import (
    auxiliary_density_contraction,
    pair_block,
)


DEFAULT_FUSED_ADJOINT_MEMORY_FRACTION = 0.20
DEFAULT_FUSED_ADJOINT_RESERVE_BYTES = 2 * 1024**3
DEFAULT_FUSED_ADJOINT_MAX_AUX_BLOCK = 32


@dataclass(frozen=True)
class PairSourceAdjoints:
    """Source-order adjoints for the two RSDF response observables."""

    target_coulomb: cp.ndarray
    restricted_hf_bilinear: cp.ndarray


def build_pair_source_adjoints(
    cderi,
    pair,
    cholesky_factor,
    reference_density,
    response_density,
    *,
    auxiliary_reference=None,
    auxiliary_response=None,
) -> PairSourceAdjoints:
    """Return AO-pair-by-auxiliary adjoints for one canonical pair.

    The returned arrays follow the same ``(nao**2, naux)`` ordering as the
    CPU RSDF source builder.  No derivative primitive is generated here.
    """

    reference = _density_array(reference_density)
    response = _density_array(response_density)
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    canonical = (int(pair[0]), int(pair[1]))
    central = pair_block(cderi, *canonical)
    factor = cp.asarray(cholesky_factor)
    if factor.shape != (central.shape[0], central.shape[0]):
        raise ValueError(
            "Cholesky factor and canonical CDERI auxiliary dimensions "
            f"differ: {factor.shape} and {central.shape}"
        )
    if auxiliary_reference is None:
        auxiliary_reference = auxiliary_density_contraction(
            cderi, reference
        )
    if auxiliary_response is None:
        auxiliary_response = auxiliary_density_contraction(cderi, response)

    target_cderi, bilinear_cderi = build_pair_cderi_adjoints(
        cderi,
        canonical,
        reference,
        response,
        cp.asarray(auxiliary_reference),
        cp.asarray(auxiliary_response),
    )
    return PairSourceAdjoints(
        target_coulomb=push_cderi_adjoint_through_source(
            factor, target_cderi
        ),
        restricted_hf_bilinear=push_cderi_adjoint_through_source(
            factor, bilinear_cderi
        ),
    )


def contract_source_derivative(source_adjoint, source_derivative):
    """Contract one source adjoint and ``dB`` into a real scalar."""

    adjoint = cp.asarray(source_adjoint)
    derivative = cp.asarray(source_derivative)
    if adjoint.shape != derivative.shape:
        raise ValueError(
            "source adjoint and derivative shapes differ: "
            f"{adjoint.shape} and {derivative.shape}"
        )
    return cp.real(cp.einsum("qp,qp->", adjoint, derivative))


def build_pair_cderi_adjoints(
    cderi,
    pair,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
):
    """Build plain-product adjoints for the canonical ``dC`` block."""

    first, second = pair
    central = pair_block(cderi, first, second)
    auxiliary_reference, auxiliary_response, dtype = _adjoint_dtype(
        pair,
        central,
        reference,
        response,
        auxiliary_reference,
        auxiliary_response,
    )
    target = cp.zeros(central.shape, dtype=dtype)
    bilinear = cp.zeros_like(target)
    nkpts = len(reference)
    normalization = float(nkpts**2)

    if first == second:
        target += cp.einsum(
            "p,ji->pij",
            auxiliary_reference,
            reference[first],
            optimize=True,
        ) / normalization
        bilinear += 2.0 * cp.einsum(
            "p,ji->pij",
            auxiliary_reference,
            response[first],
            optimize=True,
        ) / normalization
        bilinear += 2.0 * cp.einsum(
            "p,ji->pij",
            auxiliary_response,
            reference[first],
            optimize=True,
        ) / normalization

    exchange = cp.zeros_like(target)
    ordered_pairs = (
        (pair,) if first == second else (pair, (second, first))
    )
    for ki, kj in ordered_pairs:
        factor_ij = pair_block(cderi, ki, kj)
        factor_ji = pair_block(cderi, kj, ki)
        derivative_ij = _exchange_adjoint_for_first_factor(
            reference[ki],
            response[kj],
            factor_ji,
        )
        derivative_ij += _exchange_adjoint_for_first_factor(
            response[ki],
            reference[kj],
            factor_ji,
        )
        derivative_ji = _exchange_adjoint_for_second_factor(
            reference[ki],
            factor_ij,
            response[kj],
        )
        derivative_ji += _exchange_adjoint_for_second_factor(
            response[ki],
            factor_ij,
            reference[kj],
        )
        exchange += _return_to_canonical_orientation(
            derivative_ij, pair, ki, kj
        )
        exchange += _return_to_canonical_orientation(
            derivative_ji, pair, kj, ki
        )
    bilinear -= 0.5 * exchange / normalization
    return target, bilinear


def build_pair_fused_cderi_adjoint(
    cderi,
    pair,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
    *,
    memory_fraction=DEFAULT_FUSED_ADJOINT_MEMORY_FRACTION,
    minimum_reserve_bytes=DEFAULT_FUSED_ADJOINT_RESERVE_BYTES,
    max_auxiliary_block=DEFAULT_FUSED_ADJOINT_MAX_AUX_BLOCK,
    memory_diagnostics=None,
):
    """Build the production target-J plus HF-response adjoint in one array.

    The resolved route keeps the two observables separate for diagnostics.
    Production gradients consume only their sum, so allocating both dense
    ``(naux,nao,nao)`` arrays needlessly doubles the larger-basis peak.  This
    routine accumulates every Coulomb and exchange contribution directly into
    one output while retaining the established plain-product convention.
    """

    first, second = (int(pair[0]), int(pair[1]))
    canonical = (first, second)
    central = pair_block(cderi, first, second)
    auxiliary_reference, auxiliary_response, dtype = _adjoint_dtype(
        canonical,
        central,
        reference,
        response,
        auxiliary_reference,
        auxiliary_response,
    )
    fused = cp.empty(central.shape, dtype=dtype)
    normalization = float(len(reference) ** 2)
    plan = plan_fused_adjoint_auxiliary_blocks(
        central.shape[0],
        central.shape[1],
        dtype,
        memory_fraction=memory_fraction,
        minimum_reserve_bytes=minimum_reserve_bytes,
        maximum_block_size=max_auxiliary_block,
    )
    execution = _execute_fused_adjoint_blocks(
        fused,
        cderi,
        canonical,
        reference,
        response,
        auxiliary_reference,
        auxiliary_response,
        normalization=normalization,
        initial_block_size=plan["initial_block_size"],
    )
    if memory_diagnostics is not None:
        memory_diagnostics.clear()
        memory_diagnostics.update({"plan": plan, "execution": execution})
    return fused


def plan_fused_adjoint_auxiliary_blocks(
    naux,
    nao,
    dtype,
    *,
    memory_fraction=DEFAULT_FUSED_ADJOINT_MEMORY_FRACTION,
    minimum_reserve_bytes=DEFAULT_FUSED_ADJOINT_RESERVE_BYTES,
    maximum_block_size=DEFAULT_FUSED_ADJOINT_MAX_AUX_BLOCK,
    free_device_bytes=None,
):
    """Plan a bounded auxiliary tile for transactional adjoint assembly."""

    naux = int(naux)
    nao = int(nao)
    if naux < 1 or nao < 1:
        raise ValueError("naux and nao must be positive")
    fraction = float(memory_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    maximum = max(1, int(maximum_block_size))
    if free_device_bytes is None:
        free_device_bytes = int(cp.cuda.runtime.memGetInfo()[0])
    reserve = min(int(minimum_reserve_bytes), int(free_device_bytes))
    available = max(0, int(free_device_bytes) - reserve)
    budget = max(1, int(available * fraction))
    bytes_per_auxiliary = nao**2 * cp.dtype(dtype).itemsize
    # One output tile and at most one einsum/outer-product contribution are
    # live.  A third tile allowance covers contraction-engine workspace.
    working_bytes_per_auxiliary = 3 * bytes_per_auxiliary
    block_size = max(1, budget // working_bytes_per_auxiliary)
    block_size = min(naux, maximum, int(block_size))
    return {
        "naux": naux,
        "nao": nao,
        "dtype": cp.dtype(dtype).str,
        "free_device_bytes_at_plan": int(free_device_bytes),
        "memory_fraction": fraction,
        "reserve_bytes": reserve,
        "budget_bytes": budget,
        "bytes_per_auxiliary": int(bytes_per_auxiliary),
        "working_bytes_per_auxiliary": int(
            working_bytes_per_auxiliary
        ),
        "initial_block_size": int(block_size),
        "maximum_block_size": maximum,
    }


def _execute_fused_adjoint_blocks(
    output,
    cderi,
    pair,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
    *,
    normalization,
    initial_block_size,
):
    """Assemble and commit complete auxiliary tiles with OOM rebatching."""

    naux = int(output.shape[0])
    start = 0
    block_size = max(1, min(naux, int(initial_block_size)))
    minimum_block_size = block_size
    block_count = 0
    rebatch_count = 0
    while start < naux:
        stop = min(naux, start + block_size)
        try:
            block = _build_fused_adjoint_block(
                cderi,
                pair,
                reference,
                response,
                auxiliary_reference,
                auxiliary_response,
                start=start,
                stop=stop,
                normalization=normalization,
                dtype=output.dtype,
            )
        except Exception as error:
            if not _is_gpu_out_of_memory(error) or stop - start <= 1:
                raise
            block_size = max(1, (stop - start) // 2)
            minimum_block_size = min(minimum_block_size, block_size)
            rebatch_count += 1
            _release_unused_gpu_memory()
            continue
        output[start:stop] = block
        del block
        minimum_block_size = min(minimum_block_size, stop - start)
        block_count += 1
        start = stop
    return {
        "completed_auxiliary_count": naux,
        "initial_block_size": int(initial_block_size),
        "final_block_size": int(block_size),
        "minimum_block_size": int(minimum_block_size),
        "block_count": int(block_count),
        "rebatch_count": int(rebatch_count),
    }


def _build_fused_adjoint_block(
    cderi,
    pair,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
    *,
    start,
    stop,
    normalization,
    dtype,
):
    first, second = pair
    shape = (int(stop - start), reference.shape[-1], reference.shape[-1])
    block = cp.empty(shape, dtype=dtype)
    auxiliary_slice = slice(int(start), int(stop))
    if first == second:
        _assign_outer_product(
            block,
            auxiliary_reference[auxiliary_slice],
            reference[first],
            scale=1.0 / normalization,
        )
        _accumulate_outer_product(
            block,
            auxiliary_reference[auxiliary_slice],
            response[first],
            scale=2.0 / normalization,
        )
        _accumulate_outer_product(
            block,
            auxiliary_response[auxiliary_slice],
            reference[first],
            scale=2.0 / normalization,
        )
    else:
        block.fill(0)

    exchange_scale = -0.5 / normalization
    ordered_pairs = (
        (pair,) if first == second else (pair, (second, first))
    )
    for ki, kj in ordered_pairs:
        factor_ij = pair_block(cderi, ki, kj)[auxiliary_slice]
        factor_ji = pair_block(cderi, kj, ki)[auxiliary_slice]
        _accumulate_exchange(
            block,
            _exchange_adjoint_for_first_factor(
                reference[ki], response[kj], factor_ji
            ),
            pair,
            ki,
            kj,
            scale=exchange_scale,
        )
        _accumulate_exchange(
            block,
            _exchange_adjoint_for_first_factor(
                response[ki], reference[kj], factor_ji
            ),
            pair,
            ki,
            kj,
            scale=exchange_scale,
        )
        _accumulate_exchange(
            block,
            _exchange_adjoint_for_second_factor(
                reference[ki], factor_ij, response[kj]
            ),
            pair,
            kj,
            ki,
            scale=exchange_scale,
        )
        _accumulate_exchange(
            block,
            _exchange_adjoint_for_second_factor(
                response[ki], factor_ij, reference[kj]
            ),
            pair,
            kj,
            ki,
            scale=exchange_scale,
        )
    return block


def _is_gpu_out_of_memory(error):
    if isinstance(error, cp.cuda.memory.OutOfMemoryError):
        return True
    message = str(error).lower()
    return "out of memory" in message or "memoryallocation" in message


def _release_unused_gpu_memory():
    cp.get_default_memory_pool().free_all_blocks()
    pinned = cp.get_default_pinned_memory_pool()
    if hasattr(pinned, "free_all_blocks"):
        pinned.free_all_blocks()


def _adjoint_dtype(
    pair,
    central,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
):
    """Return exact-real Gamma auxiliaries without complex container growth."""

    auxiliary_reference = cp.asarray(auxiliary_reference)
    auxiliary_response = cp.asarray(auxiliary_response)
    core_arrays = (central, reference, response)
    exact_real_gamma = (
        int(pair[0]) == int(pair[1])
        and all(cp.asarray(value).dtype.kind != "c" for value in core_arrays)
    )
    if exact_real_gamma:
        auxiliary_reference = auxiliary_reference.real
        auxiliary_response = auxiliary_response.real
    dtype = cp.result_type(
        central.dtype,
        reference.dtype,
        response.dtype,
        auxiliary_reference.dtype,
        auxiliary_response.dtype,
    )
    return auxiliary_reference, auxiliary_response, dtype


def _assign_outer_product(output, auxiliary, density, *, scale):
    cp.multiply(
        auxiliary[:, None, None],
        density.T[None],
        out=output,
    )
    output *= float(scale)


def _accumulate_outer_product(output, auxiliary, density, *, scale):
    term = cp.multiply(auxiliary[:, None, None], density.T[None])
    term *= float(scale)
    output += term
    del term


def _accumulate_exchange(
    output,
    contribution,
    canonical,
    first,
    second,
    *,
    scale,
):
    contribution *= float(scale)
    output += _return_to_canonical_orientation(
        contribution,
        canonical,
        first,
        second,
    )


def _exchange_adjoint_for_first_factor(left, right, second_factor):
    return cp.einsum(
        "li,jk,pkl->pij",
        left,
        right,
        second_factor,
        optimize=True,
    )


def _exchange_adjoint_for_second_factor(left, first_factor, right):
    return cp.einsum(
        "li,pij,jk->pkl",
        left,
        first_factor,
        right,
        optimize=True,
    )


def _return_to_canonical_orientation(value, canonical, first, second):
    if (int(first), int(second)) == canonical:
        return value
    if (int(second), int(first)) == canonical:
        # Re[A * conj(dC.T)] = Re[conj(A.T) * dC].
        return value.conj().transpose(0, 2, 1)
    raise ValueError(
        f"ordered pair {(first, second)} is unrelated to {canonical}"
    )


def push_cderi_adjoint_through_source(
    factor,
    cderi_adjoint,
    *,
    overwrite=False,
):
    """Apply the transpose adjoint of ``C = solve(L, B.T)``.

    ``overwrite=True`` uses a right-side cuBLAS triangular solve on the
    C-contiguous adjoint storage.  Interpreting that storage as a Fortran
    ``(nao**2,naux)`` matrix gives the required
    ``A.T @ inv(L)`` directly, so no full-size C-to-Fortran RHS copy is made.
    """

    factor_array = cp.asarray(factor)
    adjoint = cp.asarray(cderi_adjoint)
    dtype = cp.result_type(factor_array.dtype, adjoint.dtype)
    factor_array = factor_array.astype(dtype, copy=False)
    adjoint = adjoint.astype(dtype, copy=False)
    if overwrite:
        if not adjoint.flags.c_contiguous:
            raise ValueError(
                "in-place source-adjoint solve requires C-contiguous storage"
            )
        return _right_triangular_source_solve(factor_array, adjoint)
    flat = adjoint.reshape(adjoint.shape[0], -1)
    solved = solve_triangular(
        factor_array.T,
        flat,
        lower=False,
        check_finite=False,
    )
    return solved.T


def _right_triangular_source_solve(factor, adjoint):
    """Overwrite ``adjoint`` with its source-order reverse solve."""

    factor = cp.ascontiguousarray(factor)
    naux = int(factor.shape[0])
    flat = adjoint.reshape(naux, -1)
    source_pair_count = int(flat.shape[1])
    dtype = cp.dtype(flat.dtype)
    alpha = np.array(1, dtype=dtype)
    kernels = {
        cp.dtype(cp.float32): cublas.strsm,
        cp.dtype(cp.float64): cublas.dtrsm,
        cp.dtype(cp.complex64): cublas.ctrsm,
        cp.dtype(cp.complex128): cublas.ztrsm,
    }
    try:
        triangular_solve = kernels[dtype]
    except KeyError as error:
        raise TypeError(f"unsupported source-adjoint dtype {dtype}") from error
    triangular_solve(
        cp.cuda.device.get_cublas_handle(),
        cublas.CUBLAS_SIDE_RIGHT,
        cublas.CUBLAS_FILL_MODE_UPPER,
        cublas.CUBLAS_OP_T,
        cublas.CUBLAS_DIAG_NON_UNIT,
        source_pair_count,
        naux,
        alpha.ctypes.data,
        factor.data.ptr,
        naux,
        flat.data.ptr,
        source_pair_count,
    )
    return flat.T


def _density_array(value):
    array = cp.asarray(value)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(
            "density must have shape (nkpts,nao,nao), got "
            f"{array.shape}"
        )
    return array
