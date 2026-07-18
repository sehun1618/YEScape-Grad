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
from .metric_adjoint import push_cderi_adjoint_through_metric
from .metric_native import build_metric_contraction_context
from .source_adjoint import (
    build_pair_cderi_adjoints,
    build_pair_fused_cderi_adjoint,
    push_cderi_adjoint_through_source,
)
from .source_native import (
    build_source_optimizer,
    contract_gamma_sources_profiled,
    contract_kpoint_sources_profiled,
)


@dataclass(frozen=True)
class _ContractedPairExecution:
    """Measurements from one streamed canonical-pair response."""

    parity: float | None
    central_seconds: float
    derivative_seconds: float
    peak_explicit_bytes: int
    full_derivative_equivalent_bytes: int
    initial_component_batch_size: int | None
    final_component_batch_size: int | None
    component_rebatch_count: int
    native_metric_seconds: float = 0.0
    source_adjoints: tuple[cp.ndarray, ...] | None = None
    fused_adjoint_memory: dict[str, object] | None = None


@dataclass(frozen=True)
class RSDFContractedResponse:
    """Bounded RSDF target-J and Z-bilinear nuclear response."""

    target_coulomb: cp.ndarray | None
    restricted_hf_bilinear: cp.ndarray | None
    coupled_target_j_bilinear: cp.ndarray | None
    parameters: RSDFParameters
    value_parity_max_abs: float | None
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
    native_source_derivatives: bool | str = "auto",
    native_metric_derivatives: bool | str = "auto",
    observable_mode: str = "resolved",
    validate_central_source: bool = True,
) -> RSDFContractedResponse:
    """Stream primitive derivatives into the two required force scalars.

    Central CDERI factors remain provider-owned. Native source and metric
    adjoints contract derivatives before full ``dC`` assembly. Explicit host
    derivative routes retain bounded component streaming for reference and
    mixed execution. No full derivative CDERI, J, or K tensor survives this
    function.
    """

    observable_mode = _normalize_observable_mode(observable_mode)
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
    output_shape = (natm, 3)
    if observable_mode == "fused":
        target_coulomb = None
        bilinear = None
        coupled = cp.zeros(output_shape, dtype=cp.float64)
        observable_outputs = (coupled,)
    else:
        target_coulomb = cp.zeros(output_shape, dtype=cp.float64)
        bilinear = cp.zeros_like(target_coulomb)
        coupled = None
        observable_outputs = (target_coulomb, bilinear)

    host_totals = _empty_host_profile()
    device_central_seconds = 0.0
    device_derivative_seconds = 0.0
    device_native_metric_seconds = 0.0
    device_native_source_seconds = 0.0
    device_native_source_setup_seconds = 0.0
    device_native_source_short_range_seconds = 0.0
    device_native_source_reciprocal_seconds = 0.0
    peak_explicit_bytes = 0
    full_derivative_bytes = 0
    initial_component_batch_size = None
    final_component_batch_size = None
    component_rebatch_count = 0
    validate_central_source = bool(validate_central_source)
    parity = 0.0 if validate_central_source else None
    reciprocal_memory = None
    fused_adjoint_memory = []
    pairs = canonical_pairs(len(kpts))
    source_optimizer = _native_source_optimizer(
        provider,
        witness,
        kpts,
        parameters,
        enabled=native_source_derivatives,
    )
    use_native_metric = _native_derivatives_enabled(
        native_metric_derivatives,
        name="native_metric_derivatives",
    )
    if observable_mode == "fused" and (
        source_optimizer is None or not use_native_metric
    ):
        raise NotImplementedError(
            "fused RSDF observables require native source and metric "
            "derivatives; use observable_mode='resolved' for mixed or "
            "host-reference execution"
        )
    source_adjoint_groups = tuple({} for _unused in observable_outputs)
    for pair in pairs:
        first, second = pair
        metric_context = (
            build_metric_contraction_context(
                witness.cell,
                witness.auxcell,
                parameters,
                q=kpts[second] - kpts[first],
            )
            if use_native_metric
            else None
        )
        primitives = _host_pair_primitives(
            witness.cell,
            witness.auxcell,
            kpts,
            pair,
            parameters,
            include_source_value=validate_central_source,
            include_source_derivatives=(source_optimizer is None),
            include_metric_derivatives=(metric_context is None),
        )
        _accumulate_host_profile(host_totals, primitives)
        if metric_context is not None:
            _accumulate_metric_context_profile(host_totals, metric_context)
        pair_response = _contract_device_pair(
            primitives,
            pair=pair,
            central=central,
            reference=reference,
            response=response,
            auxiliary_reference=auxiliary_reference,
            auxiliary_response=auxiliary_response,
            observable_outputs=observable_outputs,
            observable_mode=observable_mode,
            nao=int(provider.cell.nao_nr()),
            natm=natm,
            metric_context=metric_context,
            memory_fraction=component_memory_fraction,
            minimum_reserve_bytes=component_minimum_reserve_bytes,
            max_component_batch_size=max_component_batch_size,
        )
        if pair_response.parity is not None:
            parity = max(float(parity), pair_response.parity)
        device_central_seconds += pair_response.central_seconds
        device_derivative_seconds += pair_response.derivative_seconds
        device_native_metric_seconds += pair_response.native_metric_seconds
        peak_explicit_bytes = max(
            peak_explicit_bytes,
            pair_response.peak_explicit_bytes,
        )
        full_derivative_bytes += (
            pair_response.full_derivative_equivalent_bytes
        )
        initial_component_batch_size = _maximum_batch_size(
            initial_component_batch_size,
            pair_response.initial_component_batch_size,
        )
        final_component_batch_size = _minimum_batch_size(
            final_component_batch_size,
            pair_response.final_component_batch_size,
        )
        component_rebatch_count += pair_response.component_rebatch_count
        if pair_response.source_adjoints is not None:
            for group, adjoint in zip(
                source_adjoint_groups,
                pair_response.source_adjoints,
                strict=True,
            ):
                group[pair] = adjoint
        if pair_response.fused_adjoint_memory is not None:
            fused_adjoint_memory.append(
                {
                    "pair": [int(pair[0]), int(pair[1])],
                    **pair_response.fused_adjoint_memory,
                }
            )
        del pair_response

    central_release_memory = None
    if source_optimizer is not None:
        central_release_memory = _release_gamma_central_if_needed(
            provider,
            source_optimizer,
            kpts,
            source_adjoint_groups,
        )
        if central_release_memory.get("released", False):
            central = None
            _release_unused_device_memory()
            central_release_memory["free_device_bytes_after"] = int(
                cp.cuda.runtime.memGetInfo()[0]
            )
        source_result = _contract_native_sources(
            source_optimizer,
            kpts,
            source_adjoint_groups,
        )
        for output, value in zip(
            observable_outputs,
            source_result.values,
            strict=True,
        ):
            output += value
        device_native_source_seconds = source_result.total_seconds
        device_native_source_setup_seconds = source_result.setup_seconds
        device_native_source_short_range_seconds = (
            source_result.short_range_seconds
        )
        device_native_source_reciprocal_seconds = (
            source_result.reciprocal_seconds
        )
        reciprocal_memory = source_result.reciprocal_memory
        device_derivative_seconds += device_native_source_seconds

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
    if parity is not None and parity > float(value_tolerance):
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
        native_source=(source_optimizer is not None),
        native_metric=use_native_metric,
        observable_mode=observable_mode,
        device_native_source_seconds=device_native_source_seconds,
        device_native_metric_seconds=device_native_metric_seconds,
        device_native_source_setup_seconds=(
            device_native_source_setup_seconds
        ),
        device_native_source_short_range_seconds=(
            device_native_source_short_range_seconds
        ),
        device_native_source_reciprocal_seconds=(
            device_native_source_reciprocal_seconds
        ),
        central_source_validation=validate_central_source,
        device_native_source_reciprocal_memory=reciprocal_memory,
        device_fused_adjoint_memory=(
            {
                "pairs": fused_adjoint_memory,
                "central_release": central_release_memory,
            }
            if fused_adjoint_memory or central_release_memory is not None
            else None
        ),
    )
    return RSDFContractedResponse(
        target_coulomb=target_coulomb,
        restricted_hf_bilinear=bilinear,
        coupled_target_j_bilinear=coupled,
        parameters=parameters,
        value_parity_max_abs=parity,
        host_primitive_seconds=host_seconds,
        device_response_seconds=device_seconds,
        profile=profile,
    )


def _native_source_optimizer(
    provider,
    witness,
    kpts,
    parameters,
    *,
    enabled,
):
    if not _native_derivatives_enabled(
        enabled,
        name="native_source_derivatives",
    ):
        return None
    return build_source_optimizer(
        provider.cell,
        witness.auxcell,
        parameters,
        bvk_kmesh=parameters.bvk_kmesh,
    )


def _native_derivatives_enabled(value, *, name):
    if value == "auto":
        return True
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be True, False, or 'auto'")
    return bool(value)


def _normalize_observable_mode(value):
    normalized = str(value).strip().lower()
    if normalized not in {"fused", "resolved"}:
        raise ValueError(
            "observable_mode must be 'fused' or 'resolved', got "
            f"{value!r}"
        )
    return normalized


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


def _accumulate_metric_context_profile(totals, context):
    totals["metric_derivative"] += context.host_seconds
    totals["bytes"] += context.host_transfer_bytes
    totals["peak_bytes"] = max(
        totals["peak_bytes"], context.host_transfer_bytes
    )


def _minimum_batch_size(current, candidate):
    if candidate is None:
        return current
    if current is None:
        return int(candidate)
    return min(int(current), int(candidate))


def _maximum_batch_size(current, candidate):
    if candidate is None:
        return current
    if current is None:
        return int(candidate)
    return max(int(current), int(candidate))


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
    native_source,
    native_metric,
    observable_mode,
    device_native_source_seconds,
    device_native_metric_seconds,
    device_native_source_setup_seconds,
    device_native_source_short_range_seconds,
    device_native_source_reciprocal_seconds,
    central_source_validation,
    device_native_source_reciprocal_memory,
    device_fused_adjoint_memory,
):
    device_adjoint_seconds = max(
        0.0,
        device_derivative_seconds
        - device_native_source_seconds
        - device_native_metric_seconds,
    )
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
        response_mode=_response_mode(
            native_source,
            native_metric,
            observable_mode,
        ),
        host_peak_primitive_bytes=host_totals["peak_bytes"],
        device_peak_explicit_working_bytes=peak_explicit_bytes,
        full_derivative_equivalent_bytes=full_derivative_bytes,
        retained_device_derivative_bytes=0,
        component_batch_size=final_component_batch_size,
        initial_component_batch_size=initial_component_batch_size,
        component_rebatch_count=component_rebatch_count,
        device_native_source_derivative_seconds=(
            device_native_source_seconds
        ),
        device_native_metric_derivative_seconds=(
            device_native_metric_seconds
        ),
        device_adjoint_component_seconds=device_adjoint_seconds,
        device_native_source_setup_seconds=(
            device_native_source_setup_seconds
        ),
        device_native_source_short_range_seconds=(
            device_native_source_short_range_seconds
        ),
        device_native_source_reciprocal_seconds=(
            device_native_source_reciprocal_seconds
        ),
        central_source_validation=central_source_validation,
        device_native_source_reciprocal_memory=(
            device_native_source_reciprocal_memory
        ),
        device_fused_adjoint_memory=device_fused_adjoint_memory,
    )


def _response_mode(native_source, native_metric, observable_mode):
    if native_source and native_metric:
        if observable_mode == "fused":
            return "native_source_native_metric_fused_adjoint"
        return "native_source_native_metric_contracted_adjoint"
    if native_source:
        return "native_source_contracted_component_stream"
    if native_metric:
        return "native_metric_contracted_component_stream"
    return "source_contracted_component_stream"


def _contract_device_pair(
    primitives,
    *,
    pair,
    central,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
    observable_outputs,
    observable_mode,
    nao,
    natm,
    metric_context,
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
        if primitives.source is None:
            factorization = CDERIFactorization.from_value_and_metric(
                central[pair],
                cp.asarray(primitives.metric),
                nao=nao,
            )
        else:
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
    parity = (
        None
        if primitives.source is None
        else float(
            cp.max(cp.abs(factorization.value - central[pair])).get()
        )
    )

    derivative_start = cp.cuda.Event()
    derivative_end = cp.cuda.Event()
    derivative_start.record()
    source_components = _flatten_components(
        primitives.source_derivatives
    )
    metric_components = _flatten_components(
        primitives.metric_derivatives
    )
    source_adjoints, metric_adjoints, fused_adjoint_memory = (
        _build_native_pair_adjoints(
            central,
            pair,
            factorization,
            reference,
            response,
            auxiliary_reference,
            auxiliary_response,
            native_source=(source_components is None),
            native_metric=(metric_context is not None),
            observable_mode=observable_mode,
        )
    )
    context_bytes = (
        0
        if metric_context is None
        else metric_context.device_resident_bytes
    )
    peak_explicit_bytes = central_working_bytes + context_bytes
    initial_batch_size = None
    final_batch_size = None
    rebatch_count = 0
    if source_components is not None or metric_components is not None:
        stream = _execute_host_component_stream(
            factorization,
            pair=pair,
            central=central,
            reference=reference,
            response=response,
            auxiliary_reference=auxiliary_reference,
            auxiliary_response=auxiliary_response,
            source_components=source_components,
            metric_components=metric_components,
            target_coulomb=observable_outputs[0],
            bilinear=observable_outputs[1],
            memory_fraction=memory_fraction,
            minimum_reserve_bytes=minimum_reserve_bytes,
            max_component_batch_size=max_component_batch_size,
            retained_context_bytes=context_bytes,
        )
        initial_batch_size = stream["initial_batch_size"]
        final_batch_size = stream["final_batch_size"]
        rebatch_count = stream["rebatch_count"]
        peak_explicit_bytes = max(
            peak_explicit_bytes,
            stream["peak_explicit_bytes"],
        )
    if metric_context is not None:
        metric_result = metric_context.contract(cp.stack(metric_adjoints))
        for output, value in zip(
            observable_outputs,
            metric_result.values,
            strict=True,
        ):
            output += value
        peak_explicit_bytes = max(
            peak_explicit_bytes,
            metric_result.peak_explicit_bytes,
        )
        native_metric_seconds = (
            metric_context.device_setup_seconds
            + metric_result.device_seconds
        )
    else:
        native_metric_seconds = 0.0
    derivative_end.record()
    derivative_end.synchronize()
    cderi_bytes = int(factorization.value.nbytes)
    return _ContractedPairExecution(
        parity=parity,
        central_seconds=central_seconds,
        derivative_seconds=(
            cp.cuda.get_elapsed_time(derivative_start, derivative_end)
            / 1000.0
            + (
                0.0
                if metric_context is None
                else metric_context.device_setup_seconds
            )
        ),
        peak_explicit_bytes=peak_explicit_bytes,
        full_derivative_equivalent_bytes=natm * 3 * cderi_bytes,
        initial_component_batch_size=initial_batch_size,
        final_component_batch_size=final_batch_size,
        component_rebatch_count=rebatch_count,
        native_metric_seconds=native_metric_seconds,
        source_adjoints=source_adjoints,
        fused_adjoint_memory=fused_adjoint_memory,
    )


def _flatten_components(derivatives):
    if derivatives is None:
        return None
    return derivatives.reshape(-1, *derivatives.shape[2:])


def _build_native_pair_adjoints(
    central,
    pair,
    factorization,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
    *,
    native_source,
    native_metric,
    observable_mode,
):
    if not native_source and not native_metric:
        return None, None, None
    fused_adjoint_memory = None
    if observable_mode == "fused":
        fused_adjoint_memory = {}
        cderi_adjoints = (
            build_pair_fused_cderi_adjoint(
                central,
                pair,
                reference,
                response,
                auxiliary_reference,
                auxiliary_response,
                memory_diagnostics=fused_adjoint_memory,
            ),
        )
    else:
        cderi_adjoints = build_pair_cderi_adjoints(
            central,
            pair,
            reference,
            response,
            auxiliary_reference,
            auxiliary_response,
        )
    metric_adjoints = None
    if native_metric:
        metric_adjoints = tuple(
            push_cderi_adjoint_through_metric(
                factorization,
                adjoint,
            )
            for adjoint in cderi_adjoints
        )
    source_adjoints = None
    if native_source:
        source_adjoints = tuple(
            push_cderi_adjoint_through_source(
                factorization.factor,
                adjoint,
                overwrite=(observable_mode == "fused"),
            )
            for adjoint in cderi_adjoints
        )
    return source_adjoints, metric_adjoints, fused_adjoint_memory


def _release_gamma_central_if_needed(
    provider,
    optimizer,
    kpts,
    source_adjoint_groups,
    *,
    minimum_reserve_bytes=2 * 1024**3,
):
    """Evict rebuildable Gamma CDERI only when source sorting cannot fit."""

    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    groups = tuple(source_adjoint_groups)
    if (
        len(points) != 1
        or np.linalg.norm(points[0]) > 1.0e-10
        or len(groups) != 1
        or not groups[0]
    ):
        return {"released": False, "reason": "not_single_fused_gamma"}
    source = next(iter(groups[0].values()))
    itemsize = int(source.dtype.itemsize)
    source_bytes = int(source.nbytes)
    physical_naux = int(source.shape[1])
    sorted_nao = int(optimizer.cell.nao)
    sorted_naux = int(optimizer.auxcell.nao)
    ao_sorted_bytes = physical_naux * sorted_nao**2 * itemsize
    fully_sorted_bytes = sorted_naux * sorted_nao**2 * itemsize
    free_before = int(cp.cuda.runtime.memGetInfo()[0])
    reserve = int(minimum_reserve_bytes)
    phase_one_fits = free_before >= ao_sorted_bytes + reserve
    phase_two_free = free_before - ao_sorted_bytes + source_bytes
    phase_two_fits = phase_two_free >= fully_sorted_bytes + reserve
    metadata = {
        "released": False,
        "reason": "source_sorting_already_fits",
        "free_device_bytes_before": free_before,
        "source_adjoint_bytes": source_bytes,
        "ao_sorted_intermediate_bytes": int(ao_sorted_bytes),
        "fully_sorted_source_bytes": int(fully_sorted_bytes),
        "minimum_reserve_bytes": reserve,
    }
    if phase_one_fits and phase_two_fits:
        return metadata
    release = getattr(provider, "release_cderi_cache", None)
    if release is None:
        metadata["reason"] = "provider_does_not_support_cderi_eviction"
        return metadata
    provider_metadata = release(reason="native_gamma_source_sorting")
    metadata.update(provider_metadata)
    return metadata


def _release_unused_device_memory():
    cp.get_default_memory_pool().free_all_blocks()
    pinned = cp.get_default_pinned_memory_pool()
    if hasattr(pinned, "free_all_blocks"):
        pinned.free_all_blocks()


def _execute_host_component_stream(
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
    memory_fraction,
    minimum_reserve_bytes,
    max_component_batch_size,
    retained_context_bytes,
):
    arrays = tuple(
        array
        for array in (source_components, metric_components)
        if array is not None
    )
    component_count = _matching_component_count(arrays)
    derivative_dtype = cp.result_type(*(array.dtype for array in arrays))
    working = factorization.astype(
        cp.result_type(
            factorization.factor.dtype,
            factorization.value.dtype,
            derivative_dtype,
        )
    )
    factor_bytes = int(working.factor.nbytes)
    cderi_bytes = int(working.value.nbytes)
    component_upload_bytes = sum(int(array[0].nbytes) for array in arrays)
    initial_batch_size, additional_budget = plan_component_batch(
        component_count,
        component_upload_bytes=component_upload_bytes,
        factor_bytes=factor_bytes,
        cderi_bytes=cderi_bytes,
        memory_fraction=memory_fraction,
        minimum_reserve_bytes=minimum_reserve_bytes,
        max_component_batch_size=max_component_batch_size,
    )
    peak_explicit_bytes = (
        retained_context_bytes
        + 3 * factor_bytes
        + 4 * cderi_bytes
        + initial_batch_size * component_upload_bytes
    )
    execution = execute_component_batches(
        working,
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
    return {
        "initial_batch_size": initial_batch_size,
        "final_batch_size": execution["final_batch_size"],
        "rebatch_count": execution["rebatch_count"],
        "peak_explicit_bytes": peak_explicit_bytes,
    }


def _matching_component_count(arrays):
    counts = {len(array) for array in arrays}
    if len(counts) != 1:
        raise ValueError(
            "source and metric derivative component counts differ: "
            f"{sorted(counts)}"
        )
    return int(next(iter(counts)))


def _contract_native_sources(
    optimizer,
    kpts,
    source_adjoint_groups,
):
    groups = tuple(source_adjoint_groups)
    if not groups:
        raise ValueError("at least one source-adjoint group is required")
    if len(kpts) == 1 and np.linalg.norm(kpts[0]) <= 1.0e-10:
        pair = next(iter(groups[0]))
        return contract_gamma_sources_profiled(
            optimizer,
            [group.pop(pair) for group in groups],
            consume=True,
        )
    return contract_kpoint_sources_profiled(
        optimizer,
        kpts,
        groups,
    )
