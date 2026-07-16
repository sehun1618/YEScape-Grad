"""Bounded GPU execution for source-contracted RSDF derivatives.

This module owns only memory planning, component batching, and transactional
retry.  The higher-level RSDF response assembly lives in
``contracted_response`` so that numerical execution policy is independently
testable.
"""

from __future__ import annotations

import cupy as cp

from ..gdf.jk_derivative import contract_canonical_pair_derivative


DEFAULT_COMPONENT_MEMORY_FRACTION = 0.20
DEFAULT_COMPONENT_RESERVE_BYTES = 512 * 1024**2
DEFAULT_MAX_COMPONENT_BATCH_SIZE = 3


def plan_component_batch(
    component_count,
    *,
    component_upload_bytes,
    factor_bytes,
    cderi_bytes,
    memory_fraction,
    minimum_reserve_bytes,
    max_component_batch_size,
):
    """Choose a derivative-component batch inside the device budget."""

    if int(component_count) < 1 or int(component_upload_bytes) < 1:
        raise ValueError("RSDF component dimensions must be positive")
    if not 0.0 < float(memory_fraction) <= 1.0:
        raise ValueError("component_memory_fraction must lie in (0, 1]")
    if int(minimum_reserve_bytes) < 0:
        raise ValueError("component_minimum_reserve_bytes cannot be negative")
    if max_component_batch_size is not None and int(
        max_component_batch_size
    ) < 1:
        raise ValueError("max_component_batch_size must be positive")

    free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
    fraction_budget = int(free_bytes * float(memory_fraction))
    reserve_budget = max(0, int(free_bytes) - int(minimum_reserve_bytes))
    budget = min(fraction_budget, reserve_budget)
    # Additional transient allocations after factor/value construction:
    # dL-like metric work plus source, metric-RHS, and metric dC buffers.
    fixed_working = 2 * int(factor_bytes) + 3 * int(cderi_bytes)
    available_upload = budget - fixed_working
    if available_upload < int(component_upload_bytes):
        raise MemoryError(
            "insufficient GPU memory for one RSDF derivative component: "
            f"budget={budget} bytes, fixed_working={fixed_working} bytes, "
            f"component_upload={component_upload_bytes} bytes, "
            f"free={int(free_bytes)} bytes"
        )
    batch_size = min(
        int(component_count),
        available_upload // int(component_upload_bytes),
    )
    if max_component_batch_size is not None:
        batch_size = min(batch_size, int(max_component_batch_size))
    return max(1, int(batch_size)), int(budget)


def plan_central_pair_factorization(
    primitives,
    *,
    memory_fraction,
    minimum_reserve_bytes,
    free_device_bytes=None,
):
    """Reject a central pair factorization that cannot fit safely."""

    if not 0.0 < float(memory_fraction) <= 1.0:
        raise ValueError("component_memory_fraction must lie in (0, 1]")
    if int(minimum_reserve_bytes) < 0:
        raise ValueError("component_minimum_reserve_bytes cannot be negative")
    if free_device_bytes is None:
        free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
    else:
        free_bytes = int(free_device_bytes)
    fraction_budget = int(free_bytes * float(memory_fraction))
    reserve_budget = max(0, int(free_bytes) - int(minimum_reserve_bytes))
    budget = min(fraction_budget, reserve_budget)
    # Conservative complex128 ownership: source upload/cast/solve plus metric
    # upload/cast/factor/workspace. The retained provider CDERI is already
    # reflected in ``free_bytes`` and is not counted a second time.
    source_bytes = int(primitives.source.size) * 16
    metric_bytes = int(primitives.metric.size) * 16
    working_bytes = 3 * source_bytes + 5 * metric_bytes
    if working_bytes > budget:
        raise MemoryError(
            "insufficient GPU memory for RSDF central pair factorization: "
            f"budget={budget} bytes, required={working_bytes} bytes, "
            f"free={int(free_bytes)} bytes"
        )
    return int(working_bytes), int(budget)


def execute_component_batches(
    factorization,
    *,
    pair,
    central,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
    source_components,
    metric_components,
    target_coulomb,
    bilinear,
    initial_batch_size,
    explicit_working_bytes,
    additional_budget_bytes,
):
    """Execute batches transactionally, halving only after an OOM."""

    component_count = len(source_components)
    batch_size = min(int(initial_batch_size), component_count)
    rebatch_count = 0
    start = 0
    while start < component_count:
        stop = min(component_count, start + batch_size)
        try:
            target_batch, bilinear_batch = _component_batch_worker(
                factorization,
                pair=pair,
                central=central,
                reference=reference,
                response=response,
                auxiliary_reference=auxiliary_reference,
                auxiliary_response=auxiliary_response,
                source_components=source_components[start:stop],
                metric_components=metric_components[start:stop],
            )
        except Exception as error:
            if not _is_gpu_out_of_memory(error):
                raise
            cp.get_default_memory_pool().free_all_blocks()
            if batch_size == 1:
                atom_id, axis = divmod(start, 3)
                raise MemoryError(
                    "RSDF contracted response ran out of GPU memory at "
                    "minimum component batch size 1 for "
                    f"pair={pair}, atom={atom_id}, axis={axis}; explicit "
                    f"working estimate={explicit_working_bytes} bytes, "
                    f"additional budget={additional_budget_bytes} bytes"
                ) from error
            batch_size = max(1, batch_size // 2)
            rebatch_count += 1
            continue

        # Commit only after the entire batch succeeds. A failed batch can be
        # retried without double-counting any already computed component.
        for local_index, component_index in enumerate(range(start, stop)):
            atom_id, axis = divmod(component_index, 3)
            target_coulomb[atom_id, axis] += target_batch[local_index]
            bilinear[atom_id, axis] += bilinear_batch[local_index]
        start = stop
    return {
        "final_batch_size": int(batch_size),
        "rebatch_count": int(rebatch_count),
    }


def _component_batch_worker(
    factorization,
    *,
    pair,
    central,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
    source_components,
    metric_components,
):
    d_source = cp.asarray(source_components)
    d_metric = cp.asarray(metric_components)
    target = cp.empty(len(d_source), dtype=cp.float64)
    bilinear = cp.empty_like(target)
    for component in range(len(d_source)):
        parts = factorization.differentiate(
            d_source[component], d_metric[component]
        )
        derivative = parts.source
        derivative += parts.metric
        contracted = contract_canonical_pair_derivative(
            central,
            pair,
            derivative,
            reference,
            response,
            auxiliary_reference=auxiliary_reference,
            auxiliary_response=auxiliary_response,
        )
        target[component] = contracted.target_coulomb
        bilinear[component] = contracted.restricted_hf_bilinear
    cp.cuda.Stream.null.synchronize()
    return target, bilinear


def _is_gpu_out_of_memory(error):
    if isinstance(error, cp.cuda.memory.OutOfMemoryError):
        return True
    message = str(error).lower()
    return error.__class__.__name__ == "CUDARuntimeError" and (
        "out of memory" in message
        or "cudaerrormemoryallocation" in message
    )
