"""RSDF host primitives and full-tensor reference response."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import cupy as cp
import numpy as np
from pyscf_pbc_dcdft.df.gdf.storage import Pair, canonical_pairs
from pyscf_pbc_dcdft.df.rsdf.metric import (
    rsdf_metric,
    rsdf_metric_derivatives,
)
from pyscf_pbc_dcdft.df.rsdf.parameters import (
    RSDFParameters,
    parameters_from_provider,
)
from pyscf_pbc_dcdft.df.rsdf.source import (
    rsdf_source,
    rsdf_source_derivatives,
)

from ..gdf.cholesky import CDERIFactorization


@dataclass(frozen=True)
class RSDFResponseProfile:
    """Aggregated host/device ownership and timing for one response build."""

    canonical_pair_count: int
    derivative_component_count: int
    host_source_value_seconds: float
    host_metric_value_seconds: float
    host_source_derivative_seconds: float
    host_metric_derivative_seconds: float
    host_primitive_bytes: int
    host_to_device_bytes: int
    device_cderi_output_bytes: int
    device_derivative_output_bytes: int
    device_central_upload_factorization_seconds: float
    device_derivative_upload_response_seconds: float
    response_mode: str = "full_tensor_reference"
    host_peak_primitive_bytes: int = 0
    device_peak_explicit_working_bytes: int = 0
    full_derivative_equivalent_bytes: int = 0
    retained_device_derivative_bytes: int = 0
    component_batch_size: int | None = None
    initial_component_batch_size: int | None = None
    component_rebatch_count: int = 0
    device_native_source_derivative_seconds: float = 0.0
    device_native_metric_derivative_seconds: float = 0.0
    device_adjoint_component_seconds: float = 0.0
    device_native_source_setup_seconds: float = 0.0
    device_native_source_short_range_seconds: float = 0.0
    device_native_source_reciprocal_seconds: float = 0.0
    central_source_validation: bool = True
    device_native_source_reciprocal_memory: Mapping[str, object] | None = None

    def metadata(self) -> dict[str, object]:
        """Return JSON-safe profile fields with explicit ownership labels."""

        component_stream = "component_stream" in self.response_mode
        native_source = "native_source" in self.response_mode
        native_metric = "native_metric" in self.response_mode
        observable_mode = (
            "fused" if "fused_adjoint" in self.response_mode else "resolved"
        )
        device_operations = ["host-to-device primitive transfer"]
        if self.central_source_validation:
            device_operations.append("metric Cholesky and central CDERI solve")
        else:
            device_operations.append(
                "provider CDERI reuse and metric Cholesky factorization"
            )
        if component_stream:
            device_operations.extend(
                [
                    "bounded component CDERI response",
                    "immediate target-J and Z-bilinear scalar contraction",
                ]
            )
        if native_source:
            device_operations.append(
                "native short-range and reciprocal source derivative contraction"
            )
        if native_metric:
            device_operations.append(
                "native reciprocal metric derivative and Cholesky-adjoint contraction"
            )
        if not native_source and not native_metric:
            device_operations.append(
                "source and metric full-CDERI derivative response"
            )
        return {
            "canonical_pair_count": self.canonical_pair_count,
            "derivative_component_count": self.derivative_component_count,
            "host_source_value_seconds": self.host_source_value_seconds,
            "host_metric_value_seconds": self.host_metric_value_seconds,
            "host_source_derivative_seconds": (
                self.host_source_derivative_seconds
            ),
            "host_metric_derivative_seconds": (
                self.host_metric_derivative_seconds
            ),
            "host_primitive_bytes": self.host_primitive_bytes,
            "host_to_device_bytes": self.host_to_device_bytes,
            "device_cderi_output_bytes": self.device_cderi_output_bytes,
            "device_derivative_output_bytes": (
                self.device_derivative_output_bytes
            ),
            "device_central_upload_factorization_seconds": (
                self.device_central_upload_factorization_seconds
            ),
            "device_derivative_upload_response_seconds": (
                self.device_derivative_upload_response_seconds
            ),
            "response_mode": self.response_mode,
            "observable_mode": observable_mode,
            "observable_count": 1 if observable_mode == "fused" else 2,
            "host_peak_primitive_bytes": self.host_peak_primitive_bytes,
            "device_peak_explicit_working_bytes": (
                self.device_peak_explicit_working_bytes
            ),
            "full_derivative_equivalent_bytes": (
                self.full_derivative_equivalent_bytes
            ),
            "retained_device_derivative_bytes": (
                self.retained_device_derivative_bytes
            ),
            "component_batch_size": self.component_batch_size,
            "initial_component_batch_size": (
                self.initial_component_batch_size
            ),
            "component_rebatch_count": self.component_rebatch_count,
            "device_native_source_derivative_seconds": (
                self.device_native_source_derivative_seconds
            ),
            "device_native_metric_derivative_seconds": (
                self.device_native_metric_derivative_seconds
            ),
            "device_adjoint_component_seconds": (
                self.device_adjoint_component_seconds
            ),
            "device_native_source_setup_seconds": (
                self.device_native_source_setup_seconds
            ),
            "device_native_source_short_range_seconds": (
                self.device_native_source_short_range_seconds
            ),
            "device_native_source_reciprocal_seconds": (
                self.device_native_source_reciprocal_seconds
            ),
            "central_source_validation": self.central_source_validation,
            "central_value_ownership": (
                "independently reconstructed from source and metric"
                if self.central_source_validation
                else "provider-authoritative CDERI with reconstructed metric"
            ),
            "device_native_source_reciprocal_memory": (
                None
                if self.device_native_source_reciprocal_memory is None
                else dict(self.device_native_source_reciprocal_memory)
            ),
            "host_owned_operations": _host_owned_operations(
                native_source=native_source,
                native_metric=native_metric,
                central_source_validation=self.central_source_validation,
            ),
            "device_owned_operations": device_operations,
        }


@dataclass(frozen=True)
class _HostPairPrimitives:
    """Host arrays and exclusive primitive timings for one k-point pair."""

    source: np.ndarray | None
    metric: np.ndarray
    source_derivatives: np.ndarray | None
    metric_derivatives: np.ndarray | None
    source_value_seconds: float
    metric_value_seconds: float
    source_derivative_seconds: float
    metric_derivative_seconds: float

    @property
    def transferred_bytes(self) -> int:
        """Bytes uploaded when this pair is evaluated on the device."""

        arrays = (
            self.source,
            self.metric,
            self.source_derivatives,
            self.metric_derivatives,
        )
        return sum(
            int(array.nbytes) for array in arrays if array is not None
        )


@dataclass(frozen=True)
class _DevicePairResponse:
    """Device response arrays and CUDA-event phase timings for one pair."""

    value: cp.ndarray
    total: cp.ndarray
    source: cp.ndarray
    metric: cp.ndarray
    central_seconds: float
    derivative_seconds: float


@dataclass(frozen=True)
class RSDFResponseTensors:
    """Device RSDF factors and derivatives for canonical k-point pairs."""

    cderi: Mapping[Pair, cp.ndarray]
    dcderi: Mapping[Pair, cp.ndarray]
    source_dcderi: Mapping[Pair, cp.ndarray]
    metric_dcderi: Mapping[Pair, cp.ndarray]
    parameters: RSDFParameters
    value_parity_max_abs: float
    host_primitive_seconds: float
    device_response_seconds: float
    profile: RSDFResponseProfile


def build_rsdf_response_tensors(
    provider,
    *,
    value_tolerance: float = 1.0e-7,
) -> RSDFResponseTensors:
    """Build the full analytic RSDF ``dCDERI`` reference tensors.

    This intentionally materializes the complete derivative tensors and is
    retained as a correctness witness. Production gradients use the bounded
    source-contracted builder in ``contracted_response``.
    """

    if getattr(provider, "backend_name", None) != "rsdf":
        raise TypeError("build_rsdf_response_tensors requires GPU RSDF")
    if getattr(provider, "_cderi", None) is None:
        provider.build(j_only=False)
    witness = provider.cpu_witness
    if bool(getattr(witness, "j2c_eig_always", False)):
        raise NotImplementedError(
            "GPU RSDF analytic response requires Cholesky metric "
            "factorization; j2c_eig_always=True is unsupported"
        )
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    parameters = parameters_from_provider(witness)

    host_start = perf_counter()
    primitives = {
        pair: _host_pair_primitives(
            witness.cell,
            witness.auxcell,
            kpts,
            pair,
            parameters,
        )
        for pair in canonical_pairs(len(kpts))
    }
    host_seconds = perf_counter() - host_start

    device_start = perf_counter()
    values = {}
    derivatives = {}
    source_parts = {}
    metric_parts = {}
    parity = 0.0
    central = provider.central_cderi_blocks
    device_central_seconds = 0.0
    device_derivative_seconds = 0.0
    device_cderi_output_bytes = 0
    device_derivative_output_bytes = 0
    for pair, pair_primitives in primitives.items():
        pair_response = _device_pair_response(
            pair_primitives,
            nao=int(provider.cell.nao_nr()),
        )
        value = pair_response.value
        values[pair] = value
        derivatives[pair] = pair_response.total
        source_parts[pair] = pair_response.source
        metric_parts[pair] = pair_response.metric
        device_central_seconds += pair_response.central_seconds
        device_derivative_seconds += pair_response.derivative_seconds
        device_cderi_output_bytes += int(value.nbytes)
        device_derivative_output_bytes += sum(
            int(array.nbytes)
            for array in (
                pair_response.total,
                pair_response.source,
                pair_response.metric,
            )
        )
        parity = max(
            parity,
            float(cp.max(cp.abs(value - central[pair])).get()),
        )
    cp.cuda.Stream.null.synchronize()
    device_seconds = perf_counter() - device_start
    if parity > float(value_tolerance):
        raise RuntimeError(
            "GPU RSDF source/metric convention does not reproduce CPU "
            f"storage: max gap {parity:.3e} > {float(value_tolerance):.3e}"
        )
    profile = _response_profile(
        primitives,
        device_cderi_output_bytes=device_cderi_output_bytes,
        device_derivative_output_bytes=device_derivative_output_bytes,
        device_central_seconds=device_central_seconds,
        device_derivative_seconds=device_derivative_seconds,
    )
    return RSDFResponseTensors(
        cderi=MappingProxyType(values),
        dcderi=MappingProxyType(derivatives),
        source_dcderi=MappingProxyType(source_parts),
        metric_dcderi=MappingProxyType(metric_parts),
        parameters=parameters,
        value_parity_max_abs=parity,
        host_primitive_seconds=host_seconds,
        device_response_seconds=device_seconds,
        profile=profile,
    )


def _host_pair_primitives(
    cell,
    auxcell,
    kpts,
    pair,
    parameters,
    *,
    include_source_value=True,
    include_source_derivatives=True,
    include_metric_derivatives=True,
):
    ki, kj = pair
    first = kpts[ki]
    second = kpts[kj]
    if include_source_value:
        start = perf_counter()
        source = np.asarray(
            rsdf_source(
                cell,
                auxcell,
                parameters,
                kpti=first,
                kptj=second,
            )
        )
        source_value_seconds = perf_counter() - start
    else:
        source = None
        source_value_seconds = 0.0

    start = perf_counter()
    metric = rsdf_metric(
        cell,
        auxcell,
        parameters,
        q=second - first,
    )
    metric_value_seconds = perf_counter() - start
    if include_source_derivatives:
        start = perf_counter()
        source_derivatives = np.asarray(
            rsdf_source_derivatives(
                cell,
                auxcell,
                parameters,
                kpti=first,
                kptj=second,
            ).total
        )
        source_derivative_seconds = perf_counter() - start
    else:
        source_derivatives = None
        source_derivative_seconds = 0.0
    if include_metric_derivatives:
        start = perf_counter()
        metric_derivatives = np.asarray(
            rsdf_metric_derivatives(
                cell,
                auxcell,
                parameters,
                q=second - first,
            ).total
        )
        metric_derivative_seconds = perf_counter() - start
    else:
        metric_derivatives = None
        metric_derivative_seconds = 0.0
    return _HostPairPrimitives(
        source=source,
        metric=np.asarray(metric),
        source_derivatives=source_derivatives,
        metric_derivatives=metric_derivatives,
        source_value_seconds=source_value_seconds,
        metric_value_seconds=metric_value_seconds,
        source_derivative_seconds=source_derivative_seconds,
        metric_derivative_seconds=metric_derivative_seconds,
    )


def _host_owned_operations(
    *,
    native_source,
    native_metric,
    central_source_validation=True,
):
    operations = []
    if central_source_validation:
        operations.append("RSDF three-center source value")
    operations.append("RSDF two-center metric value")
    if not native_source:
        operations.append("RSDF three-center source nuclear derivatives")
    if native_metric:
        operations.append(
            "exact screened short-range metric ip1/ip2 and reciprocal grid metadata"
        )
    else:
        operations.append("RSDF two-center metric nuclear derivatives")
    return operations


def _device_pair_response(primitives, *, nao):
    central_start = cp.cuda.Event()
    central_end = cp.cuda.Event()
    derivative_end = cp.cuda.Event()
    central_start.record()
    factorization = CDERIFactorization.build(
        cp.asarray(primitives.source),
        cp.asarray(primitives.metric),
        nao=nao,
    )
    central_end.record()

    natm = primitives.source_derivatives.shape[0]
    naux = factorization.value.shape[0]
    shape = (natm, 3, naux, nao, nao)
    total = cp.empty(shape, dtype=cp.complex128)
    source_part = cp.empty_like(total)
    metric_part = cp.empty_like(total)
    for atom_id in range(natm):
        d_source = cp.asarray(primitives.source_derivatives[atom_id])
        d_metric = cp.asarray(primitives.metric_derivatives[atom_id])
        for axis in range(3):
            response = factorization.differentiate(
                d_source[axis], d_metric[axis]
            )
            source_part[atom_id, axis] = response.source
            metric_part[atom_id, axis] = response.metric
            total[atom_id, axis] = response.total
    derivative_end.record()
    derivative_end.synchronize()
    return _DevicePairResponse(
        value=factorization.value,
        total=total,
        source=source_part,
        metric=metric_part,
        central_seconds=cp.cuda.get_elapsed_time(
            central_start, central_end
        ) / 1000.0,
        derivative_seconds=cp.cuda.get_elapsed_time(
            central_end, derivative_end
        ) / 1000.0,
    )


def _response_profile(
    primitives,
    *,
    device_cderi_output_bytes,
    device_derivative_output_bytes,
    device_central_seconds,
    device_derivative_seconds,
):
    blocks = tuple(primitives.values())
    return RSDFResponseProfile(
        canonical_pair_count=len(blocks),
        derivative_component_count=sum(
            int(block.source_derivatives.shape[0]) * 3
            for block in blocks
        ),
        host_source_value_seconds=sum(
            block.source_value_seconds for block in blocks
        ),
        host_metric_value_seconds=sum(
            block.metric_value_seconds for block in blocks
        ),
        host_source_derivative_seconds=sum(
            block.source_derivative_seconds for block in blocks
        ),
        host_metric_derivative_seconds=sum(
            block.metric_derivative_seconds for block in blocks
        ),
        host_primitive_bytes=sum(
            block.transferred_bytes for block in blocks
        ),
        host_to_device_bytes=sum(
            block.transferred_bytes for block in blocks
        ),
        device_cderi_output_bytes=int(device_cderi_output_bytes),
        device_derivative_output_bytes=int(
            device_derivative_output_bytes
        ),
        device_central_upload_factorization_seconds=float(
            device_central_seconds
        ),
        device_derivative_upload_response_seconds=float(
            device_derivative_seconds
        ),
    )
