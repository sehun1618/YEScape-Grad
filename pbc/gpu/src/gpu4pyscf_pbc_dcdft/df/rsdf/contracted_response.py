"""Source-contracted production response for GPU RSDF gradients."""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from pyscf_pbc_dcdft.df.gdf.storage import canonical_pairs
from pyscf_pbc_dcdft.df.rsdf.parameters import (
    RSDFParameters,
    parameters_from_provider,
)

from ..gdf.cholesky import CDERIFactorization
from ..gdf.jk_derivative import auxiliary_density_contraction
from .component_stream import (
    DEFAULT_COMPONENT_MEMORY_FRACTION,
    DEFAULT_COMPONENT_RESERVE_BYTES,
    DEFAULT_MAX_COMPONENT_BATCH_SIZE,
    _is_gpu_out_of_memory,
    execute_component_batches,
    plan_central_pair_factorization,
    plan_component_batch,
)
from .response import (
    RSDFResponseProfile,
    _host_pair_primitives,
)


@dataclass(frozen=True)
class _ContractedPairExecution:
    """Measurements from one streamed canonical-pair response."""

    parity: float
    central_seconds: float
    derivative_seconds: float
    peak_explicit_bytes: int
    full_derivative_equivalent_bytes: int
    initial_component_batch_size: int
    final_component_batch_size: int
    component_rebatch_count: int


@dataclass(frozen=True)
class RSDFContractedResponse:
    """Bounded RSDF target-J and Z-bilinear nuclear response."""

    target_coulomb: cp.ndarray
    restricted_hf_bilinear: cp.ndarray
    parameters: RSDFParameters
    value_parity_max_abs: float
    host_primitive_seconds: float
    device_response_seconds: float
    profile: RSDFResponseProfile


def build_rsdf_contracted_response(
    provider,
    reference_density,
    response_density,
    *,
    value_tolerance: float = 1.0e-7,
    component_memory_fraction: float = DEFAULT_COMPONENT_MEMORY_FRACTION,
    component_minimum_reserve_bytes: int = (
        DEFAULT_COMPONENT_RESERVE_BYTES
    ),
    max_component_batch_size: int | None = (
        DEFAULT_MAX_COMPONENT_BATCH_SIZE
    ),
) -> RSDFContractedResponse:
    """Stream primitive derivatives into the two required force scalars.

    Central CDERI factors remain provider-owned. One canonical k-pair and a
    bounded set of Cartesian primitive derivatives are uploaded at a time;
    each ``dC`` is contracted and released immediately. No full derivative
    CDERI, J, or K tensor survives this function.
    """

    witness, kpts, parameters = _prepare_provider(provider)
    reference = _density_array(reference_density, len(kpts), "reference")
    response = _density_array(response_density, len(kpts), "response")
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    central = provider.central_cderi_blocks
    auxiliary_reference = auxiliary_density_contraction(central, reference)
    auxiliary_response = auxiliary_density_contraction(central, response)
    natm = int(provider.cell.natm)
    target_coulomb = cp.zeros((natm, 3), dtype=cp.float64)
    bilinear = cp.zeros_like(target_coulomb)

    host_totals = _empty_host_profile()
    device_central_seconds = 0.0
    device_derivative_seconds = 0.0
    peak_explicit_bytes = 0
    full_derivative_bytes = 0
    initial_component_batch_size = 0
    final_component_batch_size = None
    component_rebatch_count = 0
    parity = 0.0
    pairs = canonical_pairs(len(kpts))
    for pair in pairs:
        primitives = _host_pair_primitives(
            witness.cell,
            witness.auxcell,
            kpts,
            pair,
            parameters,
        )
        _accumulate_host_profile(host_totals, primitives)
        pair_response = _contract_device_pair(
            primitives,
            pair=pair,
            central=central,
            reference=reference,
            response=response,
            auxiliary_reference=auxiliary_reference,
            auxiliary_response=auxiliary_response,
            target_coulomb=target_coulomb,
            bilinear=bilinear,
            nao=int(provider.cell.nao_nr()),
            memory_fraction=component_memory_fraction,
            minimum_reserve_bytes=component_minimum_reserve_bytes,
            max_component_batch_size=max_component_batch_size,
        )
        parity = max(parity, pair_response.parity)
        device_central_seconds += pair_response.central_seconds
        device_derivative_seconds += pair_response.derivative_seconds
        peak_explicit_bytes = max(
            peak_explicit_bytes,
            pair_response.peak_explicit_bytes,
        )
        full_derivative_bytes += (
            pair_response.full_derivative_equivalent_bytes
        )
        initial_component_batch_size = max(
            initial_component_batch_size,
            pair_response.initial_component_batch_size,
        )
        final_component_batch_size = _minimum_batch_size(
            final_component_batch_size,
            pair_response.final_component_batch_size,
        )
        component_rebatch_count += pair_response.component_rebatch_count

    host_seconds = sum(
        host_totals[name]
        for name in (
            "source_value",
            "metric_value",
            "source_derivative",
            "metric_derivative",
        )
    )
    cp.cuda.Stream.null.synchronize()
    device_seconds = device_central_seconds + device_derivative_seconds
    if parity > float(value_tolerance):
        raise RuntimeError(
            "GPU RSDF source/metric convention does not reproduce CPU "
            f"storage: max gap {parity:.3e} > {float(value_tolerance):.3e}"
        )
    profile = _contracted_profile(
        pairs=pairs,
        natm=natm,
        host_totals=host_totals,
        device_central_seconds=device_central_seconds,
        device_derivative_seconds=device_derivative_seconds,
        peak_explicit_bytes=peak_explicit_bytes,
        full_derivative_bytes=full_derivative_bytes,
        final_component_batch_size=final_component_batch_size,
        initial_component_batch_size=initial_component_batch_size,
        component_rebatch_count=component_rebatch_count,
    )
    return RSDFContractedResponse(
        target_coulomb=target_coulomb,
        restricted_hf_bilinear=bilinear,
        parameters=parameters,
        value_parity_max_abs=parity,
        host_primitive_seconds=host_seconds,
        device_response_seconds=device_seconds,
        profile=profile,
    )


def _prepare_provider(provider):
    if getattr(provider, "backend_name", None) != "rsdf":
        raise TypeError("RSDF response requires a GPU RSDF provider")
    if getattr(provider, "_cderi", None) is None:
        provider.build(j_only=False)
    witness = provider.cpu_witness
    if bool(getattr(witness, "j2c_eig_always", False)):
        raise NotImplementedError(
            "GPU RSDF analytic response requires Cholesky metric "
            "factorization; j2c_eig_always=True is unsupported"
        )
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    return witness, kpts, parameters_from_provider(witness)


def _density_array(value, nkpts, name):
    array = cp.asarray(value)
    if array.ndim == 2:
        array = array[None]
    expected = (int(nkpts), array.shape[-1], array.shape[-1])
    if array.ndim != 3 or array.shape != expected:
        raise ValueError(
            f"{name} density must have shape (nkpts,nao,nao), got "
            f"{array.shape}"
        )
    return array


def _empty_host_profile():
    return {
        "source_value": 0.0,
        "metric_value": 0.0,
        "source_derivative": 0.0,
        "metric_derivative": 0.0,
        "bytes": 0,
        "peak_bytes": 0,
    }


def _accumulate_host_profile(totals, primitives):
    totals["source_value"] += primitives.source_value_seconds
    totals["metric_value"] += primitives.metric_value_seconds
    totals["source_derivative"] += primitives.source_derivative_seconds
    totals["metric_derivative"] += primitives.metric_derivative_seconds
    totals["bytes"] += primitives.transferred_bytes
    totals["peak_bytes"] = max(
        totals["peak_bytes"], primitives.transferred_bytes
    )


def _minimum_batch_size(current, candidate):
    if current is None:
        return int(candidate)
    return min(int(current), int(candidate))


def _contracted_profile(
    *,
    pairs,
    natm,
    host_totals,
    device_central_seconds,
    device_derivative_seconds,
    peak_explicit_bytes,
    full_derivative_bytes,
    final_component_batch_size,
    initial_component_batch_size,
    component_rebatch_count,
):
    return RSDFResponseProfile(
        canonical_pair_count=len(pairs),
        derivative_component_count=len(pairs) * natm * 3,
        host_source_value_seconds=host_totals["source_value"],
        host_metric_value_seconds=host_totals["metric_value"],
        host_source_derivative_seconds=host_totals["source_derivative"],
        host_metric_derivative_seconds=host_totals["metric_derivative"],
        host_primitive_bytes=host_totals["bytes"],
        host_to_device_bytes=host_totals["bytes"],
        device_cderi_output_bytes=0,
        device_derivative_output_bytes=0,
        device_central_upload_factorization_seconds=device_central_seconds,
        device_derivative_upload_response_seconds=device_derivative_seconds,
        response_mode="source_contracted_component_stream",
        host_peak_primitive_bytes=host_totals["peak_bytes"],
        device_peak_explicit_working_bytes=peak_explicit_bytes,
        full_derivative_equivalent_bytes=full_derivative_bytes,
        retained_device_derivative_bytes=0,
        component_batch_size=final_component_batch_size,
        initial_component_batch_size=initial_component_batch_size,
        component_rebatch_count=component_rebatch_count,
    )


def _contract_device_pair(
    primitives,
    *,
    pair,
    central,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
    target_coulomb,
    bilinear,
    nao,
    memory_fraction,
    minimum_reserve_bytes,
    max_component_batch_size,
):
    central_working_bytes, central_budget_bytes = (
        plan_central_pair_factorization(
            primitives,
            memory_fraction=memory_fraction,
            minimum_reserve_bytes=minimum_reserve_bytes,
        )
    )
    central_start = cp.cuda.Event()
    central_end = cp.cuda.Event()
    central_start.record()
    try:
        factorization = CDERIFactorization.build(
            cp.asarray(primitives.source),
            cp.asarray(primitives.metric),
            nao=nao,
        )
    except Exception as error:
        if not _is_gpu_out_of_memory(error):
            raise
        raise MemoryError(
            "RSDF central pair factorization ran out of GPU memory for "
            f"pair={pair}; explicit working estimate="
            f"{central_working_bytes} bytes, budget="
            f"{central_budget_bytes} bytes"
        ) from error
    central_end.record()
    central_end.synchronize()
    central_seconds = cp.cuda.get_elapsed_time(
        central_start, central_end
    ) / 1000.0
    parity = float(
        cp.max(cp.abs(factorization.value - central[pair])).get()
    )

    source_components = primitives.source_derivatives.reshape(
        -1, *primitives.source_derivatives.shape[2:]
    )
    metric_components = primitives.metric_derivatives.reshape(
        -1, *primitives.metric_derivatives.shape[2:]
    )
    derivative_dtype = cp.result_type(
        source_components.dtype, metric_components.dtype
    )
    factorization = factorization.astype(
        cp.result_type(
            factorization.factor.dtype,
            factorization.value.dtype,
            derivative_dtype,
        )
    )
    factor_bytes = int(factorization.factor.nbytes)
    cderi_bytes = int(factorization.value.nbytes)
    component_upload_bytes = int(source_components[0].nbytes) + int(
        metric_components[0].nbytes
    )
    initial_batch_size, additional_budget = plan_component_batch(
        len(source_components),
        component_upload_bytes=component_upload_bytes,
        factor_bytes=factor_bytes,
        cderi_bytes=cderi_bytes,
        memory_fraction=memory_fraction,
        minimum_reserve_bytes=minimum_reserve_bytes,
        max_component_batch_size=max_component_batch_size,
    )
    peak_explicit_bytes = max(
        central_working_bytes,
        3 * factor_bytes
        + 4 * cderi_bytes
        + initial_batch_size * component_upload_bytes,
    )

    derivative_start = cp.cuda.Event()
    derivative_end = cp.cuda.Event()
    derivative_start.record()
    execution = execute_component_batches(
        factorization,
        pair=pair,
        central=central,
        reference=reference,
        response=response,
        auxiliary_reference=auxiliary_reference,
        auxiliary_response=auxiliary_response,
        source_components=source_components,
        metric_components=metric_components,
        target_coulomb=target_coulomb,
        bilinear=bilinear,
        initial_batch_size=initial_batch_size,
        explicit_working_bytes=peak_explicit_bytes,
        additional_budget_bytes=additional_budget,
    )
    derivative_end.record()
    derivative_end.synchronize()
    natm = primitives.source_derivatives.shape[0]
    return _ContractedPairExecution(
        parity=parity,
        central_seconds=central_seconds,
        derivative_seconds=cp.cuda.get_elapsed_time(
            derivative_start, derivative_end
        ) / 1000.0,
        peak_explicit_bytes=peak_explicit_bytes,
        full_derivative_equivalent_bytes=natm * 3 * cderi_bytes,
        initial_component_batch_size=initial_batch_size,
        final_component_batch_size=execution["final_batch_size"],
        component_rebatch_count=execution["rebatch_count"],
    )
