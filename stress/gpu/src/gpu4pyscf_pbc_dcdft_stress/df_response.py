"""Bounded device GDF/RSDF strain response from analytic host primitives."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os
from time import perf_counter
from types import MappingProxyType

import cupy as cp
import numpy as np

from gpu4pyscf_pbc_dcdft.df.gdf.cholesky import CDERIFactorization
from gpu4pyscf_pbc_dcdft.df.gdf.jk_derivative import (
    auxiliary_density_contractions,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.metric_adjoint import (
    push_cderi_adjoint_through_metric,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.source_adjoint import (
    build_pair_cderi_adjoints,
    push_cderi_adjoint_through_source,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.central_exact import (
    _canonical_pair_mapping,
)
from pyscf_pbc_dcdft.df.gdf.storage import canonical_pairs
from pyscf_pbc_dcdft.df.rsdf import parameters_from_provider
from pyscf_pbc_dcdft_stress.gdf_metric_strain import (
    kpoint_auxiliary_metric_strain,
    matrix_tensor_to_voigt,
)
from pyscf_pbc_dcdft_stress.gdf_source_strain import (
    kpoint_three_center_source_adjoint_strain,
    kpoint_three_center_source_strain,
)
from pyscf_pbc_dcdft_stress.gradient_adapter import resolved_gdf_parameters
from pyscf_pbc_dcdft_stress.rsdf_metric_strain import (
    kpoint_rsdf_metric_strain,
)

from .ft_source import (
    _AUTO_AO_PAIR_WORK_BUDGET_BYTES,
    gamma_three_center_source_adjoint_strain as
    gamma_three_center_source_adjoint_strain_gpu,
    kpoint_three_center_source_adjoint_strain as
    kpoint_three_center_source_adjoint_strain_gpu,
)


@dataclass(frozen=True)
class PairStrainPrimitive:
    """Host source/metric derivatives and their device factorization."""

    source_derivative: np.ndarray
    metric_derivative: np.ndarray
    factorization: CDERIFactorization


@dataclass(frozen=True)
class DeviceJKVoigtStrain:
    """All six symmetric strain derivatives of sampled-k J and K."""

    coulomb: cp.ndarray
    exchange: cp.ndarray
    derivative_evaluations: int

    @property
    def restricted_hf(self):
        return self.coulomb-0.5*self.exchange


@dataclass(frozen=True)
class DeviceDFContractedVoigtStrain:
    """Analytic target-J and adjoint RHF derivatives without AO outputs."""

    target_coulomb: cp.ndarray
    adjoint_restricted_hf: cp.ndarray
    derivative_evaluations: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class DeviceDFStrainResponse:
    """Reusable central factors plus bounded component response machinery."""

    backend: str
    primitives: object
    nkpts: int
    nao: int
    parameters: dict[str, object]
    memory_estimate: dict[str, int]
    diagonal_response_bytes: int
    diagonal_cache_enabled: bool
    preflight_free_device_bytes: int
    device_memory_fraction: float
    host_primitive_seconds: float
    host_worker_count: int
    device_factorization_seconds: float
    central_parity_max_abs: float | None

    def contract_jk_voigt(self, density):
        """Build all six symmetric J/K strain derivatives in one pair pass."""

        dm = cp.asarray(density)
        expected = (self.nkpts, self.nao, self.nao)
        if dm.shape != expected:
            raise ValueError(f"density must have shape {expected}, got {dm.shape}")
        central = {
            pair: primitive.factorization.value
            for pair, primitive in self.primitives.items()
        }
        jaux = cp.sum(cp.stack([
            cp.einsum(
                "pij,ji->p", central[(k, k)], dm[k], optimize=True
            ) for k in range(self.nkpts)
        ]), axis=0)
        pairs = canonical_pairs(self.nkpts)
        shape = (6, self.nkpts, self.nao, self.nao)
        coulomb = cp.zeros(shape, dtype=cp.complex128)
        exchange = cp.zeros_like(coulomb)
        derivative_evaluations = 0

        for component in range(6):
            djaux = cp.zeros_like(jaux)
            for first, second in pairs:
                value = central[(first, second)]
                derivative = _differentiate_voigt(
                    self.primitives[(first, second)], component
                )
                derivative_evaluations += 1
                reverse_value = value.conj().transpose(0, 2, 1)
                reverse_derivative = derivative.conj().transpose(0, 2, 1)

                if first == second:
                    djaux += cp.einsum(
                        "pij,ji->p", derivative, dm[first], optimize=True
                    )
                    coulomb[component, first] = cp.einsum(
                        "p,pij->ij", jaux, derivative, optimize=True
                    )

                exchange[component, first] += cp.einsum(
                    "pij,jk,pkl->il", derivative, dm[second], reverse_value,
                    optimize=True,
                )
                exchange[component, first] += cp.einsum(
                    "pij,jk,pkl->il", value, dm[second], reverse_derivative,
                    optimize=True,
                )
                if first != second:
                    exchange[component, second] += cp.einsum(
                        "pij,jk,pkl->il", reverse_derivative, dm[first], value,
                        optimize=True,
                    )
                    exchange[component, second] += cp.einsum(
                        "pij,jk,pkl->il", reverse_value, dm[first], derivative,
                        optimize=True,
                    )
                del derivative, reverse_derivative

            for k in range(self.nkpts):
                coulomb[component, k] += cp.einsum(
                    "p,pij->ij", djaux, central[(k, k)], optimize=True
                )

        coulomb /= self.nkpts
        exchange /= self.nkpts
        return DeviceJKVoigtStrain(
            coulomb=coulomb, exchange=exchange,
            derivative_evaluations=derivative_evaluations,
        )

    def contract_dc_energy_voigt(self, density, transition_density):
        """Contract the six DF derivatives directly with the two densities.

        This evaluates the same algebra as ``contract_jk_voigt`` followed by
        the DC-DFT target-J and adjoint-RHF traces.  No ``(6,nk,nao,nao)`` J/K
        response arrays are allocated.
        """

        dm = cp.asarray(density)
        transition = cp.asarray(transition_density)
        expected = (self.nkpts, self.nao, self.nao)
        if dm.shape != expected or transition.shape != expected:
            raise ValueError(
                "density and transition_density must both have shape "
                f"{expected}; got {dm.shape} and {transition.shape}"
            )
        central = {
            pair: primitive.factorization.value
            for pair, primitive in self.primitives.items()
        }
        reference_aux = cp.sum(cp.stack([
            cp.einsum(
                "pij,ji->p", central[(k, k)], dm[k], optimize=True
            ) for k in range(self.nkpts)
        ]), axis=0)
        transition_aux = cp.sum(cp.stack([
            cp.einsum(
                "pij,ji->p", central[(k, k)], transition[k], optimize=True
            ) for k in range(self.nkpts)
        ]), axis=0)
        pairs = canonical_pairs(self.nkpts)
        target = cp.empty(6, dtype=cp.float64)
        adjoint = cp.empty(6, dtype=cp.float64)
        derivative_evaluations = 0
        scale = 1.0/(self.nkpts*self.nkpts)

        for component in range(6):
            reference_derivative_aux = cp.zeros_like(reference_aux)
            transition_derivative_aux = cp.zeros_like(transition_aux)
            exchange_bilinear = cp.asarray(0.0, dtype=cp.complex128)
            for first, second in pairs:
                value = central[(first, second)]
                derivative = _differentiate_voigt(
                    self.primitives[(first, second)], component
                )
                derivative_evaluations += 1
                reverse_value = value.conj().transpose(0, 2, 1)
                reverse_derivative = derivative.conj().transpose(0, 2, 1)

                if first == second:
                    reference_derivative_aux += cp.einsum(
                        "pij,ji->p", derivative, dm[first], optimize=True
                    )
                    transition_derivative_aux += cp.einsum(
                        "pij,ji->p", derivative, transition[first],
                        optimize=True,
                    )

                exchange_bilinear += cp.einsum(
                    "li,pij,jk,pkl->", transition[first], derivative,
                    dm[second], reverse_value, optimize=True,
                )
                exchange_bilinear += cp.einsum(
                    "li,pij,jk,pkl->", transition[first], value,
                    dm[second], reverse_derivative, optimize=True,
                )
                if first != second:
                    exchange_bilinear += cp.einsum(
                        "li,pij,jk,pkl->", transition[second],
                        reverse_derivative, dm[first], value, optimize=True,
                    )
                    exchange_bilinear += cp.einsum(
                        "li,pij,jk,pkl->", transition[second], reverse_value,
                        dm[first], derivative, optimize=True,
                    )
                del derivative, reverse_derivative

            target[component] = (
                cp.einsum(
                    "p,p->", reference_aux, reference_derivative_aux,
                    optimize=True,
                ).real*scale
            )
            coulomb_bilinear = (
                cp.einsum(
                    "p,p->", reference_aux, transition_derivative_aux,
                    optimize=True,
                )
                + cp.einsum(
                    "p,p->", reference_derivative_aux, transition_aux,
                    optimize=True,
                )
            )
            adjoint[component] = (
                2.0*coulomb_bilinear.real-exchange_bilinear.real
            )*scale

        return DeviceDFContractedVoigtStrain(
            target_coulomb=target,
            adjoint_restricted_hf=adjoint,
            derivative_evaluations=derivative_evaluations,
            metadata={
                "owner": "gpu_analytic_df_direct_contracted",
                "derivative_kind": "analytic_source_metric_factor_response",
                "backend": self.backend,
                "component_count": 6,
                "canonical_pair_count": len(pairs),
                "derivative_evaluation_count": derivative_evaluations,
                "materialized_jk_response": False,
                "retained_ao_response_bytes": 0,
                "fixed_density_count": 2,
            },
        )


def build_device_df_strain_response(
    provider, *, host_workers=None, diagonal_cache_bytes=256*(1 << 20),
    device_memory_fraction=0.85, reciprocal_block_size=None,
):
    """Generate analytic host primitives and factor every pair on the GPU."""

    (
        backend, cell, auxcell, kpts, parameters, source_omega,
        source_mesh, metric_mesh, rsdf_parameters,
    ) = _response_context(provider)

    pairs = canonical_pairs(len(kpts))
    worker_count = _host_worker_count(host_workers, len(pairs))
    cache_limit = int(diagonal_cache_bytes)
    if cache_limit < 0:
        raise ValueError("diagonal_cache_bytes must be non-negative")
    memory_fraction = float(device_memory_fraction)
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("device_memory_fraction must be in (0, 1]")
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    preflight = _preflight_device_memory(
        pair_count=len(pairs), diagonal_pair_count=len(kpts), naux=naux,
        nkpts=len(kpts), nao=nao, diagonal_cache_bytes=cache_limit,
        device_memory_fraction=memory_fraction,
    )
    diagonal_response_bytes = preflight["diagonal_response_bytes"]
    diagonal_cache_enabled = False

    authoritative = _authoritative_blocks(provider)
    primitives = {}
    parity = 0.0 if authoritative is not None else None
    pair_payload_bytes = []
    central_input_bytes = 0
    device_seconds = 0.0
    pipeline_start = perf_counter()
    reciprocal_metadata = None
    source_host_seconds = 0.0
    metric_host_seconds = 0.0
    source_timing_totals = {}
    stream = _host_pair_stream(
        pairs, worker_count, backend, cell, auxcell, kpts, source_omega,
        source_mesh, metric_mesh, rsdf_parameters, reciprocal_block_size,
    )
    for pair, source, metric, source_seconds, metric_seconds in stream:
        source_host_seconds += source_seconds
        metric_host_seconds += metric_seconds
        for name, value in getattr(source, "timings_s", {}).items():
            source_timing_totals[name] = (
                source_timing_totals.get(name, 0.0)+float(value)
            )
        if reciprocal_metadata is None:
            reciprocal_metadata = {
                "stress_reciprocal_block_size": int(
                    source.reciprocal_block_size
                ),
                "stress_reciprocal_block_size_requested": (
                    None if reciprocal_block_size is None
                    else int(reciprocal_block_size)
                ),
                "stress_reciprocal_grid_points": int(
                    source.reciprocal_grid_points
                ),
                "stress_estimated_block_work_bytes": int(
                    source.estimated_block_work_bytes
                ),
            }
        pair_payload_bytes.append(_host_pair_bytes(source, metric))
        central_input_bytes += int(
            np.asarray(source.value).nbytes+np.asarray(metric.value).nbytes
        )
        device_start = perf_counter()
        factorization = CDERIFactorization.build(
            cp.asarray(source.value), cp.asarray(metric.value), nao=nao
        )
        if authoritative is not None:
            central = cp.asarray(authoritative[pair])
            gap = float(cp.max(cp.abs(factorization.value-central)).get())
            parity = max(parity, gap)
            if gap > 1e-7:
                raise RuntimeError(
                    f"GPU central factor parity failed for {pair}: {gap:.3e}"
                )
            factorization = CDERIFactorization.from_value_and_metric(
                central, cp.asarray(metric.value), nao=nao
            )
        primitives[pair] = PairStrainPrimitive(
            source_derivative=matrix_tensor_to_voigt(source.raw_derivative),
            metric_derivative=matrix_tensor_to_voigt(metric.raw_derivative),
            factorization=factorization,
        )
        cp.cuda.Stream.null.synchronize()
        device_seconds += perf_counter()-device_start
        del source, metric
    host_seconds = max(0.0, perf_counter()-pipeline_start-device_seconds)
    if reciprocal_metadata is None:
        raise RuntimeError("DF stress response did not build any k-point pair")
    parameters = dict(parameters)
    parameters.update(reciprocal_metadata)
    derivative_input_bytes = sum(
        int(
            np.asarray(primitive.source_derivative).nbytes
            + np.asarray(primitive.metric_derivative).nbytes
        )
        for primitive in primitives.values()
    )
    parameters.update({
        "stress_df_source_host_seconds_sum": source_host_seconds,
        "stress_df_metric_host_seconds_sum": metric_host_seconds,
        "stress_compact_source_components": True,
        "stress_df_derivative_layout": "symmetric_voigt_6",
        "stress_df_pair_contraction": "single_pass_per_voigt_component",
        "stress_df_diagonal_cache_policy": "unused_single_pass",
        "stress_ft_ao_pair_kernel": "fused_overlap_center_i_center_j",
        "stress_ft_image_context_reused": True,
        "stress_host_pair_pipeline": "bounded_producer_consumer",
        "stress_df_canonical_pair_count": int(len(pairs)),
        "stress_df_reverse_pair_reuse_count": int(len(kpts)**2-len(pairs)),
        "stress_df_source_build_count": int(len(pairs)),
        "stress_df_metric_build_count": int(len(pairs)),
        "stress_df_central_factorization_count": int(len(pairs)),
        "stress_df_directional_factorization_count": int(6*len(pairs)),
        "stress_df_explicit_host_to_device_central_bytes": int(
            central_input_bytes
        ),
        "stress_df_explicit_host_to_device_derivative_bytes": int(
            derivative_input_bytes
        ),
        "stress_df_source_timing_profile": source_timing_totals,
    })
    host_build_memory = _host_build_memory(
        pair_payload_bytes, worker_count
    )
    memory_estimate = _memory_estimate(
        primitives, host_build_memory=host_build_memory,
        reciprocal_grid_block_bytes=int(
            parameters["stress_estimated_block_work_bytes"]
        ), nkpts=len(kpts), nao=nao,
        diagonal_cache_bytes=0,
    )
    return DeviceDFStrainResponse(
        backend=backend,
        primitives=MappingProxyType(primitives),
        nkpts=len(kpts),
        nao=nao,
        parameters=parameters,
        memory_estimate=memory_estimate,
        diagonal_response_bytes=diagonal_response_bytes,
        diagonal_cache_enabled=diagonal_cache_enabled,
        preflight_free_device_bytes=preflight["free_device_bytes"],
        device_memory_fraction=memory_fraction,
        host_primitive_seconds=host_seconds,
        host_worker_count=worker_count,
        device_factorization_seconds=device_seconds,
        central_parity_max_abs=parity,
    )


def build_and_contract_device_df_strain_direct(
    provider, density, transition_density, *, host_workers=None,
    device_memory_fraction=0.85, reciprocal_block_size=None,
    pair_response_matmul=None,
    pair_response_mode="auto",
    factorized_projection=None,
):
    """Stream analytic pair derivatives directly into DC scalar responses."""

    (
        backend, cell, auxcell, kpts, parameters, source_omega,
        source_mesh, metric_mesh, rsdf_parameters,
    ) = _response_context(provider)
    dm = cp.asarray(density)
    transition = cp.asarray(transition_density)
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    expected = (len(kpts), nao, nao)
    if dm.shape != expected or transition.shape != expected:
        raise ValueError(
            "density and transition_density must both have shape "
            f"{expected}; got {dm.shape} and {transition.shape}"
        )
    pairs = canonical_pairs(len(kpts))
    worker_count = _host_worker_count(host_workers, len(pairs))
    memory_fraction = float(device_memory_fraction)
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("device_memory_fraction must be in (0, 1]")
    preflight = _preflight_device_memory(
        pair_count=1, diagonal_pair_count=1, naux=naux,
        nkpts=len(kpts), nao=nao, diagonal_cache_bytes=0,
        device_memory_fraction=memory_fraction,
    )

    reference_aux = cp.zeros(naux, dtype=cp.complex128)
    transition_aux = cp.zeros_like(reference_aux)
    reference_derivative_aux = cp.zeros((6, naux), dtype=cp.complex128)
    transition_derivative_aux = cp.zeros_like(reference_derivative_aux)
    exchange_bilinear = cp.zeros(6, dtype=cp.complex128)
    authoritative = _authoritative_blocks(provider)
    parity = 0.0 if authoritative is not None else None
    pair_payload_bytes = []
    peak_pair_device_bytes = 0
    source_host_seconds = 0.0
    metric_host_seconds = 0.0
    source_timing_totals = {}
    resolved_pair_response_modes = set()
    pair_response_image_bytes = 0
    device_seconds = 0.0
    derivative_evaluations = 0
    reciprocal_metadata = None
    total_started = perf_counter()
    stream = _host_pair_stream(
        pairs, worker_count, backend, cell, auxcell, kpts, source_omega,
        source_mesh, metric_mesh, rsdf_parameters, reciprocal_block_size,
        pair_response_matmul=pair_response_matmul,
        pair_response_mode=pair_response_mode,
        factorized_projection=factorized_projection,
    )
    for pair, source, metric, source_seconds, metric_seconds in stream:
        source_host_seconds += source_seconds
        metric_host_seconds += metric_seconds
        for name, value in getattr(source, "timings_s", {}).items():
            source_timing_totals[name] = (
                source_timing_totals.get(name, 0.0)+float(value)
            )
        resolved_pair_response_modes.add(str(source.pair_response_mode))
        pair_response_image_bytes = max(
            pair_response_image_bytes,
            int(source.pair_response_image_bytes),
        )
        if reciprocal_metadata is None:
            reciprocal_metadata = {
                "stress_reciprocal_block_size": int(
                    source.reciprocal_block_size
                ),
                "stress_reciprocal_block_size_requested": (
                    None if reciprocal_block_size is None
                    else int(reciprocal_block_size)
                ),
                "stress_reciprocal_grid_points": int(
                    source.reciprocal_grid_points
                ),
                "stress_estimated_block_work_bytes": int(
                    source.estimated_block_work_bytes
                ),
            }
        pair_payload_bytes.append(_host_pair_bytes(source, metric))
        device_started = perf_counter()
        factorization = CDERIFactorization.build(
            cp.asarray(source.value), cp.asarray(metric.value), nao=nao
        )
        if authoritative is not None:
            central = cp.asarray(authoritative[pair])
            gap = float(cp.max(cp.abs(
                factorization.value-central
            )).get())
            parity = max(parity, gap)
            if gap > 1e-7:
                raise RuntimeError(
                    f"GPU central factor parity failed for {pair}: {gap:.3e}"
                )
            factorization = CDERIFactorization.from_value_and_metric(
                central, cp.asarray(metric.value), nao=nao
            )
        value = factorization.value
        first, second = pair
        reverse_value = value.conj().transpose(0, 2, 1)
        if first == second:
            reference_aux += cp.einsum(
                "pij,ji->p", value, dm[first], optimize=True
            )
            transition_aux += cp.einsum(
                "pij,ji->p", value, transition[first], optimize=True
            )

        source_raw = np.asarray(source.raw_derivative)
        metric_raw = np.asarray(metric.raw_derivative)
        for component in range(6):
            derivative = factorization.differentiate(
                cp.asarray(_voigt_component(source_raw, component)),
                cp.asarray(_voigt_component(metric_raw, component)),
            ).total
            derivative_evaluations += 1
            reverse_derivative = derivative.conj().transpose(0, 2, 1)
            if first == second:
                reference_derivative_aux[component] += cp.einsum(
                    "pij,ji->p", derivative, dm[first], optimize=True
                )
                transition_derivative_aux[component] += cp.einsum(
                    "pij,ji->p", derivative, transition[first],
                    optimize=True,
                )
            exchange_bilinear[component] += cp.einsum(
                "li,pij,jk,pkl->", transition[first], derivative,
                dm[second], reverse_value, optimize=True,
            )
            exchange_bilinear[component] += cp.einsum(
                "li,pij,jk,pkl->", transition[first], value,
                dm[second], reverse_derivative, optimize=True,
            )
            if first != second:
                exchange_bilinear[component] += cp.einsum(
                    "li,pij,jk,pkl->", transition[second],
                    reverse_derivative, dm[first], value, optimize=True,
                )
                exchange_bilinear[component] += cp.einsum(
                    "li,pij,jk,pkl->", transition[second], reverse_value,
                    dm[first], derivative, optimize=True,
                )
            peak_pair_device_bytes = max(
                peak_pair_device_bytes,
                int(
                    value.nbytes+factorization.factor.nbytes
                    + derivative.nbytes
                    + reference_derivative_aux.nbytes
                    + transition_derivative_aux.nbytes
                ),
            )
            del derivative, reverse_derivative
        cp.cuda.Stream.null.synchronize()
        device_seconds += perf_counter()-device_started
        del factorization, value, reverse_value, source, metric

    scale = 1.0/(len(kpts)*len(kpts))
    target = cp.empty(6, dtype=cp.float64)
    adjoint = cp.empty(6, dtype=cp.float64)
    for component in range(6):
        target[component] = cp.einsum(
            "p,p->", reference_aux, reference_derivative_aux[component],
            optimize=True,
        ).real*scale
        coulomb_bilinear = (
            cp.einsum(
                "p,p->", reference_aux,
                transition_derivative_aux[component], optimize=True,
            )
            + cp.einsum(
                "p,p->", reference_derivative_aux[component],
                transition_aux, optimize=True,
            )
        )
        adjoint[component] = (
            2.0*coulomb_bilinear.real-exchange_bilinear[component].real
        )*scale
    cp.cuda.Stream.null.synchronize()
    total_seconds = perf_counter()-total_started
    host_pipeline_wall = max(0.0, total_seconds-device_seconds)
    host_memory = _host_build_memory(pair_payload_bytes, worker_count)
    memory_estimate = {
        "host_primitive_build_payload_bytes": host_memory["total"],
        "host_parallel_build_upper_bound_bytes": host_memory["concurrent"],
        "host_retained_pair_derivative_bytes": 0,
        "central_device_factor_retained_bytes": 0,
        "retained_ao_response_bytes": 0,
        "auxiliary_accumulator_bytes": int(
            reference_aux.nbytes+transition_aux.nbytes
            + reference_derivative_aux.nbytes
            + transition_derivative_aux.nbytes
            + exchange_bilinear.nbytes
        ),
        "estimated_peak_pair_device_bytes": int(peak_pair_device_bytes),
        "preflight_allowed_device_bytes": preflight["allowed_device_bytes"],
    }
    resolved_parameters = dict(parameters)
    resolved_parameters.update(reciprocal_metadata or {})
    resolved_mode = (
        next(iter(resolved_pair_response_modes))
        if len(resolved_pair_response_modes) == 1
        else sorted(resolved_pair_response_modes)
    )
    factorized_only = resolved_mode == "factorized"
    resolved_parameters.update({
        "stress_compact_source_components": True,
        "stress_df_source_derivative_layout": "symmetric_voigt_6",
        "stress_ao_pair_response_matmuls_per_block": (
            15 if factorized_only else 6
        ),
        "stress_long_range_source_matmuls_per_block": (
            27 if factorized_only else 18
        ),
        "stress_cartesian_reference_matmuls_per_block": 27,
        "stress_df_pair_response_contraction_backend": (
            "numpy_blas"
            if pair_response_matmul is None
            else getattr(pair_response_matmul, "__name__", "custom")
        ),
        "stress_df_factorized_projection_backend": (
            "numpy_einsum"
            if factorized_projection is None
            else getattr(factorized_projection, "__name__", "custom")
        ),
        "stress_df_pair_response_mode_requested": str(pair_response_mode),
        "stress_df_pair_response_mode_resolved": resolved_mode,
        "stress_df_pair_response_auto_threshold_bytes": 256*(1 << 20),
        "stress_df_pair_response_image_bytes": int(
            pair_response_image_bytes
        ),
    })
    return DeviceDFContractedVoigtStrain(
        target_coulomb=target,
        adjoint_restricted_hf=adjoint,
        derivative_evaluations=derivative_evaluations,
        metadata={
            "owner": "gpu_analytic_df_streaming_direct_contracted",
            "derivative_kind": "analytic_source_metric_factor_response",
            "backend": backend,
            "component_count": 6,
            "canonical_pair_count": len(pairs),
            "derivative_evaluation_count": derivative_evaluations,
            "materialized_jk_response": False,
            "retained_ao_response_bytes": 0,
            "retained_pair_derivative_bytes": 0,
            "fixed_density_count": 2,
            "host_worker_count": worker_count,
            "central_parity_max_abs": parity,
            "parameters": resolved_parameters,
            "memory_estimate": memory_estimate,
            "timings_s": {
                "host_source_sum": float(source_host_seconds),
                "host_metric_sum": float(metric_host_seconds),
                "host_pipeline_wall": float(host_pipeline_wall),
                "device_pair_contraction": float(device_seconds),
                "total": float(total_seconds),
            },
            "source_timings_s": source_timing_totals,
        },
    )


def build_and_contract_device_df_strain_adjoint_reference(
    provider, density, transition_density, *, host_workers=None,
    device_memory_fraction=0.85, reciprocal_block_size=None,
):
    """Contract host analytic primitives through reverse Cholesky adjoints.

    This is the algebraic bridge to a native source-contracted stress builder.
    It deliberately retains the established host ``dB``/``dM`` primitives so
    that reverse-mode parity can be validated independently.  Unlike the
    forward direct route, it never forms a directional ``dCDERI`` tensor.
    """

    (
        backend, cell, auxcell, kpts, parameters, source_omega,
        source_mesh, metric_mesh, rsdf_parameters,
    ) = _response_context(provider)
    dm = cp.asarray(density)
    transition = cp.asarray(transition_density)
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    expected = (len(kpts), nao, nao)
    if dm.shape != expected or transition.shape != expected:
        raise ValueError(
            "density and transition_density must both have shape "
            f"{expected}; got {dm.shape} and {transition.shape}"
        )
    central = _authoritative_blocks(provider)
    if central is None:
        raise RuntimeError(
            "reverse-adjoint DF stress requires provider-owned central "
            "CDERI blocks"
        )
    pairs = canonical_pairs(len(kpts))
    worker_count = _host_worker_count(host_workers, len(pairs))
    memory_fraction = float(device_memory_fraction)
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("device_memory_fraction must be in (0, 1]")
    preflight = _preflight_device_memory(
        pair_count=1, diagonal_pair_count=1, naux=naux,
        nkpts=len(kpts), nao=nao, diagonal_cache_bytes=0,
        device_memory_fraction=memory_fraction,
    )
    auxiliary_reference, auxiliary_transition = (
        auxiliary_density_contractions(central, (dm, transition))
    )

    target = cp.zeros(6, dtype=cp.float64)
    adjoint = cp.zeros(6, dtype=cp.float64)
    pair_payload_bytes = []
    source_host_seconds = 0.0
    metric_host_seconds = 0.0
    device_seconds = 0.0
    derivative_evaluations = 0
    reciprocal_metadata = None
    parity = 0.0
    peak_pair_device_bytes = 0
    total_started = perf_counter()
    stream = _host_pair_stream(
        pairs, worker_count, backend, cell, auxcell, kpts, source_omega,
        source_mesh, metric_mesh, rsdf_parameters, reciprocal_block_size,
    )
    for pair, source, metric, source_seconds, metric_seconds in stream:
        source_host_seconds += source_seconds
        metric_host_seconds += metric_seconds
        if reciprocal_metadata is None:
            reciprocal_metadata = {
                "stress_reciprocal_block_size": int(
                    source.reciprocal_block_size
                ),
                "stress_reciprocal_grid_points": int(
                    source.reciprocal_grid_points
                ),
                "stress_estimated_block_work_bytes": int(
                    source.estimated_block_work_bytes
                ),
            }
        pair_payload_bytes.append(_host_pair_bytes(source, metric))
        device_started = perf_counter()
        factorization = CDERIFactorization.from_value_and_metric(
            central[pair], cp.asarray(metric.value), nao=nao
        )
        source_factorization = CDERIFactorization.build(
            cp.asarray(source.value), cp.asarray(metric.value), nao=nao
        )
        gap = float(cp.max(cp.abs(
            source_factorization.value-cp.asarray(central[pair])
        )).get())
        parity = max(parity, gap)
        if gap > 1e-7:
            raise RuntimeError(
                f"GPU central factor parity failed for {pair}: {gap:.3e}"
            )
        cderi_target, cderi_adjoint = build_pair_cderi_adjoints(
            central, pair, dm, transition,
            auxiliary_reference, auxiliary_transition,
        )
        source_adjoints = (
            push_cderi_adjoint_through_source(
                factorization.factor, cderi_target
            ),
            push_cderi_adjoint_through_source(
                factorization.factor, cderi_adjoint
            ),
        )
        metric_adjoints = (
            push_cderi_adjoint_through_metric(
                factorization, cderi_target
            ),
            push_cderi_adjoint_through_metric(
                factorization, cderi_adjoint
            ),
        )
        source_raw = np.asarray(source.raw_derivative)
        metric_raw = np.asarray(metric.raw_derivative)
        for component in range(6):
            source_component = cp.asarray(
                _voigt_component(source_raw, component)
            )
            metric_component = cp.asarray(
                _voigt_component(metric_raw, component)
            )
            target[component] += cp.real(cp.einsum(
                "qp,qp->", source_adjoints[0], source_component,
                optimize=True,
            ))
            target[component] += cp.real(cp.einsum(
                "pq,pq->", metric_adjoints[0], metric_component,
                optimize=True,
            ))
            adjoint[component] += cp.real(cp.einsum(
                "qp,qp->", source_adjoints[1], source_component,
                optimize=True,
            ))
            adjoint[component] += cp.real(cp.einsum(
                "pq,pq->", metric_adjoints[1], metric_component,
                optimize=True,
            ))
            derivative_evaluations += 1
            peak_pair_device_bytes = max(
                peak_pair_device_bytes,
                int(
                    sum(value.nbytes for value in source_adjoints)
                    + sum(value.nbytes for value in metric_adjoints)
                    + source_component.nbytes+metric_component.nbytes
                ),
            )
            del source_component, metric_component
        cp.cuda.Stream.null.synchronize()
        device_seconds += perf_counter()-device_started
        del (
            factorization, source_factorization, cderi_target,
            cderi_adjoint, source_adjoints, metric_adjoints, source, metric,
        )

    total_seconds = perf_counter()-total_started
    host_pipeline_wall = max(0.0, total_seconds-device_seconds)
    host_memory = _host_build_memory(pair_payload_bytes, worker_count)
    resolved_parameters = dict(parameters)
    resolved_parameters.update(reciprocal_metadata or {})
    resolved_parameters.update({
        "stress_compact_source_components": True,
        "stress_df_source_derivative_layout": "symmetric_voigt_6",
        "stress_df_reverse_cholesky_adjoint": True,
        "stress_df_materialized_directional_cderi": False,
        "stress_df_native_source_contraction": False,
    })
    memory_estimate = {
        "host_primitive_build_payload_bytes": host_memory["total"],
        "host_parallel_build_upper_bound_bytes": host_memory["concurrent"],
        "host_retained_pair_derivative_bytes": 0,
        "central_device_factor_retained_bytes": 0,
        "retained_ao_response_bytes": 0,
        "estimated_peak_pair_device_bytes": int(peak_pair_device_bytes),
        "preflight_allowed_device_bytes": preflight["allowed_device_bytes"],
    }
    return DeviceDFContractedVoigtStrain(
        target_coulomb=target,
        adjoint_restricted_hf=adjoint,
        derivative_evaluations=derivative_evaluations,
        metadata={
            "owner": "gpu_analytic_df_reverse_adjoint_host_primitives",
            "derivative_kind": (
                "analytic_source_metric_reverse_cholesky_contraction"
            ),
            "backend": backend,
            "component_count": 6,
            "canonical_pair_count": len(pairs),
            "derivative_evaluation_count": derivative_evaluations,
            "materialized_jk_response": False,
            "materialized_directional_cderi": False,
            "native_source_contraction": False,
            "retained_ao_response_bytes": 0,
            "retained_pair_derivative_bytes": 0,
            "fixed_density_count": 2,
            "host_worker_count": worker_count,
            "central_parity_max_abs": parity,
            "parameters": resolved_parameters,
            "memory_estimate": memory_estimate,
            "timings_s": {
                "host_source_sum": float(source_host_seconds),
                "host_metric_sum": float(metric_host_seconds),
                "host_pipeline_wall": float(host_pipeline_wall),
                "device_pair_contraction": float(device_seconds),
                "total": float(total_seconds),
            },
        },
    )


def _resolve_gpu_ft_dispatch(kpts, *, gamma_gpu_ft, gpu_ft):
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    at_gamma = np.max(np.abs(kpts)) <= 1e-10
    if gamma_gpu_ft and not at_gamma:
        raise NotImplementedError(
            "GPU FT source-adjoint stress is currently restricted to Gamma"
        )
    if gamma_gpu_ft or gpu_ft:
        return "gamma" if at_gamma else "sampled_k"
    return "host"


def build_and_contract_device_df_strain_source_adjoint(
    provider, density, transition_density, *, host_workers=None,
    device_memory_fraction=0.85, reciprocal_block_size=None,
    gamma_gpu_ft=False, gpu_ft=False,
):
    """Contract source strain natively through reverse Cholesky adjoints.

    The provider-owned central CDERI supplies the observable adjoints.  The
    host source builder consumes those adjoints directly and never returns a
    complete long-range source derivative.  Metric derivatives remain exact
    host primitives contracted with their small reverse adjoints.
    """

    (
        backend, cell, auxcell, kpts, parameters, source_omega,
        source_mesh, metric_mesh, rsdf_parameters,
    ) = _response_context(provider)
    dm = cp.asarray(density)
    transition = cp.asarray(transition_density)
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    expected = (len(kpts), nao, nao)
    if dm.shape != expected or transition.shape != expected:
        raise ValueError(
            "density and transition_density must both have shape "
            f"{expected}; got {dm.shape} and {transition.shape}"
        )
    gpu_ft_dispatch = _resolve_gpu_ft_dispatch(
        kpts, gamma_gpu_ft=gamma_gpu_ft, gpu_ft=gpu_ft
    )
    use_gamma_gpu_ft = gpu_ft_dispatch == "gamma"
    requested_workers = _host_worker_count(
        host_workers, len(canonical_pairs(len(kpts)))
    )
    worker_count = 1
    memory_fraction = float(device_memory_fraction)
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("device_memory_fraction must be in (0, 1]")
    preflight = _preflight_device_memory(
        pair_count=1, diagonal_pair_count=1, naux=naux,
        nkpts=len(kpts), nao=nao, diagonal_cache_bytes=0,
        device_memory_fraction=memory_fraction,
    )
    central = _authoritative_blocks(
        provider, allow_native_gamma=bool(gamma_gpu_ft or gpu_ft)
    )
    if central is None:
        raise RuntimeError(
            "source-adjoint DF stress requires provider-owned central CDERI "
            "blocks; native compressed GDF is currently supported only at "
            "Gamma through its explicit factorization metadata"
        )
    auxiliary_reference, auxiliary_transition = (
        auxiliary_density_contractions(central, (dm, transition))
    )

    target = cp.zeros(6, dtype=cp.float64)
    adjoint = cp.zeros(6, dtype=cp.float64)
    derivative_evaluations = 0
    source_host_seconds = 0.0
    metric_host_seconds = 0.0
    device_seconds = 0.0
    peak_pair_device_bytes = 0
    peak_host_bytes = 0
    source_timing_totals = {
        "short_range_full_tensor": 0.0,
        "reciprocal_direct_adjoint": 0.0,
        "vbar_direct_adjoint": 0.0,
    }
    reciprocal_metadata = None
    pairs = canonical_pairs(len(kpts))
    total_started = perf_counter()
    for pair in pairs:
        first, second = pair
        working_cell = cell.copy(deep=True)
        working_aux = auxcell.copy(deep=True)
        kpti, kptj = kpts[[first, second]]
        metric_started = perf_counter()
        if backend == "gdf":
            metric = kpoint_auxiliary_metric_strain(
                working_cell, working_aux, source_omega, metric_mesh,
                kptj-kpti, reciprocal_block_size=reciprocal_block_size,
            )
        else:
            metric = kpoint_rsdf_metric_strain(
                working_cell, working_aux, rsdf_parameters, kptj-kpti,
                reciprocal_block_size=reciprocal_block_size,
            )
        metric_host_seconds += perf_counter()-metric_started

        device_started = perf_counter()
        factorization = CDERIFactorization.from_value_and_metric(
            central[pair], cp.asarray(metric.value), nao=nao
        )
        cderi_target, cderi_adjoint = build_pair_cderi_adjoints(
            central, pair, dm, transition,
            auxiliary_reference, auxiliary_transition,
        )
        source_adjoints = (
            push_cderi_adjoint_through_source(
                factorization.factor, cderi_target
            ),
            push_cderi_adjoint_through_source(
                factorization.factor, cderi_adjoint
            ),
        )
        metric_adjoints = (
            push_cderi_adjoint_through_metric(
                factorization, cderi_target
            ),
            push_cderi_adjoint_through_metric(
                factorization, cderi_adjoint
            ),
        )
        metric_raw = np.asarray(metric.raw_derivative)
        for component in range(6):
            metric_component = cp.asarray(
                _voigt_component(metric_raw, component)
            )
            target[component] += cp.real(cp.einsum(
                "pq,pq->", metric_adjoints[0], metric_component,
                optimize=True,
            ))
            adjoint[component] += cp.real(cp.einsum(
                "pq,pq->", metric_adjoints[1], metric_component,
                optimize=True,
            ))
            derivative_evaluations += 1
            del metric_component
        source_adjoints_host = cp.asnumpy(cp.stack(source_adjoints))
        peak_pair_device_bytes = max(
            peak_pair_device_bytes,
            int(
                sum(value.nbytes for value in source_adjoints)
                + sum(value.nbytes for value in metric_adjoints)
                + cderi_target.nbytes+cderi_adjoint.nbytes
            ),
        )
        cp.cuda.Stream.null.synchronize()
        device_seconds += perf_counter()-device_started
        del (
            factorization, cderi_target, cderi_adjoint,
            source_adjoints, metric_adjoints,
        )

        source_started = perf_counter()
        source_precision = (
            None if rsdf_parameters is None else rsdf_parameters.precision_r
        )
        if use_gamma_gpu_ft:
            source = gamma_three_center_source_adjoint_strain_gpu(
                working_cell, working_aux, source_omega, source_mesh,
                source_adjoints_host, precision=source_precision,
                block_size=reciprocal_block_size,
            )
        elif gpu_ft:
            source = kpoint_three_center_source_adjoint_strain_gpu(
                working_cell, working_aux, source_omega, source_mesh,
                kpts, kpti, kptj, source_adjoints_host,
                precision=source_precision,
                block_size=reciprocal_block_size,
            )
        else:
            source = kpoint_three_center_source_adjoint_strain(
                working_cell, working_aux, source_omega, source_mesh,
                kpti, kptj, source_adjoints_host,
                precision=source_precision,
                reciprocal_block_size=reciprocal_block_size,
            )
        source_host_seconds += perf_counter()-source_started
        target += cp.asarray(source.values[0])
        adjoint += cp.asarray(source.values[1])
        for name in source_timing_totals:
            source_timing_totals[name] += source.timings_s[name]
        peak_host_bytes = max(
            peak_host_bytes,
            int(
                source_adjoints_host.nbytes
                + source.memory_estimate["short_range_tensor_bytes"]
            ),
        )
        if reciprocal_metadata is None:
            reciprocal_metadata = {
                "stress_reciprocal_block_size": int(
                    source.reciprocal_block_size
                ),
                "stress_reciprocal_block_size_requested": (
                    None if reciprocal_block_size is None
                    else int(reciprocal_block_size)
                ),
                "stress_reciprocal_auto_ao_pair_work_budget_bytes": (
                    _AUTO_AO_PAIR_WORK_BUDGET_BYTES
                    if reciprocal_block_size is None
                    else None
                ),
                "stress_reciprocal_grid_points": int(
                    source.reciprocal_grid_points
                ),
                "stress_estimated_block_work_bytes": int(
                    source.estimated_block_work_bytes
                ),
                "stress_reciprocal_full_response_avoided_bytes": int(
                    source.memory_estimate[
                        "reciprocal_full_response_avoided_bytes"
                    ]
                ),
            }
        del source_adjoints_host, source, metric

    cp.cuda.Stream.null.synchronize()
    total_seconds = perf_counter()-total_started
    host_pipeline_wall = max(0.0, total_seconds-device_seconds)
    resolved_parameters = dict(parameters)
    resolved_parameters.update(reciprocal_metadata or {})
    resolved_parameters.update({
        "stress_compact_source_components": True,
        "stress_df_source_derivative_layout": "source_adjoint_scalar_6",
        "stress_df_reverse_cholesky_adjoint": True,
        "stress_df_materialized_directional_cderi": False,
        "stress_df_native_source_contraction": True,
        "stress_df_source_short_range_mode": "exact_full_tensor_then_contract",
        "stress_df_source_reciprocal_mode": "direct_reverse_adjoint",
        "stress_df_source_reciprocal_owner": (
            "gpu4pyscf_ft_strain_kernel" if gamma_gpu_ft or gpu_ft
            else "host_analytic_fourier_integral"
        ),
        "stress_df_gpu_ft_dispatch": (
            None if gpu_ft_dispatch == "host" else gpu_ft_dispatch
        ),
        "stress_df_requested_host_workers": int(requested_workers),
        "stress_df_effective_host_workers": int(worker_count),
    })
    memory_estimate = {
        "host_primitive_build_payload_bytes": int(peak_host_bytes),
        "host_parallel_build_upper_bound_bytes": int(peak_host_bytes),
        "host_retained_pair_derivative_bytes": 0,
        "central_device_factor_retained_bytes": 0,
        "retained_ao_response_bytes": 0,
        "estimated_peak_pair_device_bytes": int(peak_pair_device_bytes),
        "preflight_allowed_device_bytes": preflight["allowed_device_bytes"],
    }
    return DeviceDFContractedVoigtStrain(
        target_coulomb=target,
        adjoint_restricted_hf=adjoint,
        derivative_evaluations=derivative_evaluations,
        metadata={
            "owner": (
                "gpu_analytic_df_source_adjoint_gamma_ft"
                if use_gamma_gpu_ft
                else "gpu_analytic_df_source_adjoint_ft"
                if gpu_ft
                else "gpu_analytic_df_source_adjoint_contracted"
            ),
            "derivative_kind": (
                "analytic_source_reverse_adjoint_metric_reverse_adjoint"
            ),
            "backend": backend,
            "component_count": 6,
            "canonical_pair_count": len(pairs),
            "derivative_evaluation_count": derivative_evaluations,
            "materialized_jk_response": False,
            "materialized_directional_cderi": False,
            "native_source_contraction": True,
            "retained_ao_response_bytes": 0,
            "retained_pair_derivative_bytes": 0,
            "fixed_density_count": 2,
            "host_worker_count": worker_count,
            "central_parity_max_abs": None,
            "central_source_validation": False,
            "parameters": resolved_parameters,
            "memory_estimate": memory_estimate,
            "source_timings_s": {
                key: float(value) for key, value in source_timing_totals.items()
            },
            "timings_s": {
                "host_source_sum": float(source_host_seconds),
                "host_metric_sum": float(metric_host_seconds),
                "host_pipeline_wall": float(host_pipeline_wall),
                "device_pair_contraction": float(device_seconds),
                "total": float(total_seconds),
            },
        },
    )


def _response_context(provider):
    backend = getattr(provider, "backend_name", None)
    if backend not in {"gdf", "rsdf"}:
        raise TypeError(f"unsupported GPU stress DF backend {backend!r}")
    if getattr(provider, "auxcell", None) is None:
        provider.build(j_only=False)
    cell = provider.cell
    auxcell = provider.auxcell
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    if backend == "gdf":
        parameters = resolved_gdf_parameters(provider)
        source_omega = parameters["omega"]
        source_mesh = parameters["source_mesh"]
        metric_mesh = parameters["metric_mesh"]
        rsdf_parameters = None
    else:
        rsdf_parameters = parameters_from_provider(provider)
        source_omega = rsdf_parameters.source_omega
        source_mesh = rsdf_parameters.source_mesh
        metric_mesh = rsdf_parameters.metric_mesh
        parameters = {
            "source_omega": rsdf_parameters.source_omega,
            "source_mesh": list(rsdf_parameters.source_mesh),
            "metric_omega": rsdf_parameters.metric_omega,
            "metric_mesh": list(rsdf_parameters.metric_mesh),
            "precision_R": rsdf_parameters.precision_r,
            "precision_G": rsdf_parameters.precision_g,
            "precision_j2c": rsdf_parameters.precision_j2c,
            "use_bvk": rsdf_parameters.use_bvk,
        }
    return (
        backend, cell, auxcell, kpts, parameters, source_omega,
        source_mesh, metric_mesh, rsdf_parameters,
    )


def _voigt_component(tensor, component):
    value = np.asarray(tensor)
    if value.ndim == 3 and value.shape[0] == 6:
        return value[int(component)]
    indices = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    row, column = indices[int(component)]
    if row == column:
        return value[row, column]
    return 0.5*(value[row, column]+value[column, row])


def _build_host_pair(
    backend, cell, auxcell, kpts, pair, source_omega, source_mesh,
    metric_mesh, rsdf_parameters, reciprocal_block_size,
    pair_response_matmul=None,
    pair_response_mode="assembled",
    factorized_projection=None,
):
    """Build one independent pair with private mutable PySCF cell state."""

    working_cell = cell.copy(deep=True)
    working_aux = auxcell.copy(deep=True)
    kpti, kptj = kpts[list(pair)]
    started = perf_counter()
    source = kpoint_three_center_source_strain(
        working_cell, working_aux, source_omega, source_mesh, kpti, kptj,
        precision=(
            None if rsdf_parameters is None else rsdf_parameters.precision_r
        ),
        reciprocal_block_size=reciprocal_block_size,
        retain_components=False,
        compact_voigt=True,
        pair_response_matmul=pair_response_matmul,
        pair_response_mode=pair_response_mode,
        factorized_projection=factorized_projection,
    )
    source_seconds = perf_counter()-started
    started = perf_counter()
    if backend == "gdf":
        metric = kpoint_auxiliary_metric_strain(
            working_cell, working_aux, source_omega, metric_mesh, kptj-kpti,
            reciprocal_block_size=reciprocal_block_size,
        )
    else:
        metric = kpoint_rsdf_metric_strain(
            working_cell, working_aux, rsdf_parameters, kptj-kpti,
            reciprocal_block_size=reciprocal_block_size,
        )
    metric_seconds = perf_counter()-started
    return source, metric, source_seconds, metric_seconds


def _host_pair_stream(
    pairs, worker_count, backend, cell, auxcell, kpts, source_omega,
    source_mesh, metric_mesh, rsdf_parameters, reciprocal_block_size,
    pair_response_matmul=None,
    pair_response_mode="assembled",
    factorized_projection=None,
):
    """Yield at most ``worker_count`` completed raw host pairs at a time."""

    def build(pair):
        source, metric, source_seconds, metric_seconds = _build_host_pair(
            backend, cell, auxcell, kpts, pair, source_omega, source_mesh,
            metric_mesh, rsdf_parameters, reciprocal_block_size,
            pair_response_matmul=pair_response_matmul,
            pair_response_mode=pair_response_mode,
            factorized_projection=factorized_projection,
        )
        return pair, source, metric, source_seconds, metric_seconds

    if worker_count == 1:
        for pair in pairs:
            yield build(pair)
        return

    pair_iterator = iter(pairs)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending = {}
        for _ in range(worker_count):
            try:
                pair = next(pair_iterator)
            except StopIteration:
                break
            pending[executor.submit(build, pair)] = pair

        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                yield future.result()
                try:
                    pair = next(pair_iterator)
                except StopIteration:
                    continue
                pending[executor.submit(build, pair)] = pair


def _host_worker_count(requested, pair_count):
    if requested is None:
        available = os.cpu_count() or 1
        return max(1, min(int(pair_count), int(available), 4))
    count = int(requested)
    if count < 1:
        raise ValueError("host_workers must be positive")
    return min(count, int(pair_count))


def _memory_estimate(
    primitives, *, host_build_memory, reciprocal_grid_block_bytes,
    nkpts, nao, diagonal_cache_bytes,
):
    """Return conservative host/device bytes for the current response route."""

    retained_host_bytes = 0
    central_device = 0
    peak_pair_device = 0
    for primitive in primitives.values():
        source_derivative = np.asarray(primitive.source_derivative)
        metric_derivative = np.asarray(primitive.metric_derivative)
        retained_host_bytes += int(
            source_derivative.nbytes+metric_derivative.nbytes
        )
        factorization = primitive.factorization
        central_device += int(
            factorization.factor.nbytes+factorization.value.nbytes
        )
        source_component = int(source_derivative[0].nbytes)
        metric_component = int(metric_derivative[0].nbytes)
        # Source response, metric response, their sum, and Cholesky workspace.
        pair_workspace = (
            source_component+metric_component
            + 4*int(factorization.value.nbytes)
            + 4*int(factorization.factor.nbytes)
        )
        peak_pair_device = max(peak_pair_device, pair_workspace)
    complex_bytes = np.dtype(np.complex128).itemsize
    jk_workspace = int(12*nkpts*nao*nao*complex_bytes)
    return {
        "host_derivative_payload_bytes": retained_host_bytes,
        "host_retained_voigt_derivative_bytes": retained_host_bytes,
        "host_primitive_build_payload_bytes": host_build_memory["total"],
        "host_parallel_build_upper_bound_bytes": host_build_memory[
            "concurrent"
        ],
        "central_device_factor_bytes": central_device,
        "diagonal_response_cache_bytes": int(diagonal_cache_bytes),
        "peak_component_device_workspace_bytes": peak_pair_device,
        "jk_device_workspace_bytes": jk_workspace,
        "estimated_peak_df_device_bytes": int(
            central_device+diagonal_cache_bytes+peak_pair_device+jk_workspace
        ),
        "reciprocal_grid_block_bytes": int(reciprocal_grid_block_bytes),
    }


def _host_pair_bytes(source, metric):
    return sum((
        int(np.asarray(source.value).nbytes),
        int(np.asarray(metric.value).nbytes),
        int(np.asarray(source.raw_derivative).nbytes),
        int(np.asarray(metric.raw_derivative).nbytes),
    ))


def _host_build_memory(pair_bytes, worker_count):
    """Measure the transient raw host primitives before Voigt compaction."""

    payloads = [int(value) for value in pair_bytes]
    return {
        "total": int(sum(payloads)),
        "concurrent": int(sum(sorted(payloads, reverse=True)[:worker_count])),
    }


def _preflight_device_memory(
    *, pair_count, diagonal_pair_count, naux, nkpts, nao,
    diagonal_cache_bytes, device_memory_fraction,
):
    """Reject impossible single-pass Voigt device allocations early."""

    del diagonal_cache_bytes
    complex_bytes = np.dtype(np.complex128).itemsize
    value_bytes = int(naux)*nao*nao*complex_bytes
    factor_bytes = int(naux)*int(naux)*complex_bytes
    central = int(pair_count)*(value_bytes+factor_bytes)
    pair_workspace = 5*value_bytes+5*factor_bytes
    diagonal_response = int(diagonal_pair_count)*value_bytes
    jk_workspace = 12*nkpts*nao*nao*complex_bytes
    free_device, _ = cp.cuda.runtime.memGetInfo()
    allowed = int(int(free_device)*device_memory_fraction)
    requested = central+pair_workspace+jk_workspace
    if requested > allowed:
        raise MemoryError(
            "GPU DF strain response preflight requires approximately "
            f"{requested/(1 << 30):.3f} GiB but only "
            f"{allowed/(1 << 30):.3f} GiB is allowed from the current free "
            "device memory. Reduce k-points/basis/auxbasis or use a device "
            "with more free memory."
        )
    return {
        "free_device_bytes": int(free_device),
        "allowed_device_bytes": allowed,
        "estimated_device_bytes": int(requested),
        "diagonal_response_bytes": int(diagonal_response),
        "diagonal_cache_enabled": False,
    }


def _differentiate_voigt(primitive, component):
    response = primitive.factorization.differentiate(
        cp.asarray(primitive.source_derivative[component]),
        cp.asarray(primitive.metric_derivative[component]),
    )
    return response.total


def _authoritative_blocks(provider, *, allow_native_gamma=False):
    try:
        return provider.central_cderi_blocks
    except RuntimeError:
        if not allow_native_gamma:
            return None
    if getattr(provider, "backend_name", None) != "gdf":
        return None
    points = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    if len(points) > 1 or np.max(np.abs(points)) > 1e-12:
        provider._ensure_built()
        if getattr(provider, "_cderip", None) is not None:
            raise NotImplementedError(
                "native sampled-k GDF negative-metric factors are not "
                "supported by the source-adjoint stress route"
            )
        return _canonical_pair_mapping(
            provider._cderi,
            provider._cderi_idx,
            cell=provider.cell,
            kpts=points,
            kmesh=np.asarray(provider.kmesh, dtype=int),
        )
    try:
        metadata = provider.gamma_factorization_metadata
    except (RuntimeError, NotImplementedError):
        return None
    compressed = metadata.compressed_cderi
    if len(compressed) != 1 or 0 not in compressed:
        raise RuntimeError(
            "native Gamma GDF expected one compressed CDERI block"
        )
    packed = cp.asarray(compressed[0])
    nao = int(metadata.nao)
    naux = int(metadata.naux)
    if packed.shape != (naux, len(metadata.pair_address)):
        raise NotImplementedError(
            "native Gamma GDF auxiliary rank reduction requires a packed "
            "source-adjoint implementation"
        )
    dense = cp.empty((naux, nao, nao), dtype=packed.dtype)
    addresses = cp.asarray(metadata.pair_address)
    first = addresses//nao
    second = addresses%nao
    dense[:, first, second] = packed
    dense[:, second, first] = packed
    return MappingProxyType({(0, 0): dense})


def device_host_pair_response_matmul(left, right):
    """Execute one host-generated AO response contraction with CuPy GEMM."""

    left_device = cp.asarray(left)
    right_device = cp.asarray(right)
    result_device = left_device@right_device
    result = cp.asnumpy(result_device)
    del left_device, right_device, result_device
    return result


def device_host_factorized_projection(
    center_i, center_j, moment, momentum, weighted_conjugate,
):
    """Project factorized host Fourier derivatives with bounded CuPy work."""

    weight_device = cp.asarray(weighted_conjugate)
    center_i_device = cp.asarray(center_i)
    center_i_projected = cp.asnumpy(cp.einsum(
        "xgij,gp->xijp", center_i_device, weight_device, optimize=True
    ))
    del center_i_device
    center_j_device = cp.asarray(center_j)
    center_j_projected = cp.asnumpy(cp.einsum(
        "xgij,gp->xijp", center_j_device, weight_device, optimize=True
    ))
    del center_j_device
    moment_device = cp.asarray(moment)
    momentum_device = cp.asarray(momentum)
    nao = int(moment_device.shape[2])
    naux = int(weight_device.shape[1])
    moment_projected_device = cp.empty(
        (3, 3, nao, nao, naux), dtype=cp.complex128
    )
    for x in range(3):
        momentum_weighted = momentum_device[:, x, None]*weight_device
        moment_projected_device[x] = cp.einsum(
            "ygij,gp->yijp", moment_device, momentum_weighted,
            optimize=True,
        )
        del momentum_weighted
    moment_projected = cp.asnumpy(moment_projected_device)
    del (
        weight_device, moment_device, momentum_device,
        moment_projected_device,
    )
    return center_i_projected, center_j_projected, moment_projected


def device_host_factorized_projection_bounded(
    center_i, center_j, moment, momentum, weighted_conjugate,
    *, transfer_block_bytes=512*(1 << 20), device_memory_fraction=0.75,
):
    """Project host Fourier factors with a bounded transfer working set.

    The full factor arrays can be several GiB even though their projected
    ``(nao, nao, naux)`` results are modest.  Accumulating over G-vector
    blocks avoids a full-size host-to-device staging allocation.  Each raw
    center/moment slice is uploaded once per block.
    """

    center_i = np.asarray(center_i)
    center_j = np.asarray(center_j)
    moment = np.asarray(moment)
    momentum = np.asarray(momentum)
    weighted_conjugate = np.asarray(weighted_conjugate)
    ngrid = int(weighted_conjugate.shape[0])
    nao = int(center_i.shape[2])
    naux = int(weighted_conjugate.shape[1])
    complex_bytes = np.dtype(np.complex128).itemsize
    real_bytes = np.dtype(np.float64).itemsize
    bytes_per_grid = (
        nao*nao*complex_bytes
        + 4*naux*complex_bytes
        + 3*real_bytes
    )
    block_size = max(
        1, min(ngrid, int(transfer_block_bytes)//bytes_per_grid)
    )
    projected_part_bytes = nao*nao*naux*complex_bytes
    projected_output_bytes = 15*projected_part_bytes
    transfer_work_bytes = block_size*bytes_per_grid
    estimated_device_bytes = (
        projected_output_bytes+transfer_work_bytes+2*projected_part_bytes
    )
    free_device_bytes, _ = cp.cuda.runtime.memGetInfo()
    allowed_device_bytes = int(
        float(device_memory_fraction)*int(free_device_bytes)
    )
    if estimated_device_bytes > allowed_device_bytes:
        raise MemoryError(
            "bounded factorized projection requires an estimated "
            f"{estimated_device_bytes/(1 << 30):.3f} GiB, but only "
            f"{allowed_device_bytes/(1 << 30):.3f} GiB is allowed from "
            "current free VRAM; use the host factorized backend or lower "
            "the transfer block size"
        )

    center_i_projected_device = cp.zeros(
        (3, nao, nao, naux), dtype=cp.complex128
    )
    center_j_projected_device = cp.zeros_like(center_i_projected_device)
    moment_projected_device = cp.zeros(
        (3, 3, nao, nao, naux), dtype=cp.complex128
    )

    for start in range(0, ngrid, block_size):
        stop = min(start+block_size, ngrid)
        weight_device = cp.asarray(weighted_conjugate[start:stop])
        for axis in range(3):
            factor_device = cp.asarray(center_i[axis, start:stop])
            center_i_projected_device[axis] += cp.einsum(
                "gij,gp->ijp", factor_device, weight_device, optimize=True
            )
            del factor_device
        for axis in range(3):
            factor_device = cp.asarray(center_j[axis, start:stop])
            center_j_projected_device[axis] += cp.einsum(
                "gij,gp->ijp", factor_device, weight_device, optimize=True
            )
            del factor_device

        momentum_device = cp.asarray(momentum[start:stop])
        momentum_weights = [
            momentum_device[:, axis, None]*weight_device for axis in range(3)
        ]
        for moment_axis in range(3):
            factor_device = cp.asarray(moment[moment_axis, start:stop])
            for momentum_axis, momentum_weight in enumerate(momentum_weights):
                moment_projected_device[momentum_axis, moment_axis] += cp.einsum(
                    "gij,gp->ijp", factor_device, momentum_weight,
                    optimize=True,
                )
            del factor_device
        del momentum_weights, momentum_device, weight_device

    center_i_projected = cp.asnumpy(center_i_projected_device)
    center_j_projected = cp.asnumpy(center_j_projected_device)
    moment_projected = cp.asnumpy(moment_projected_device)
    del (
        center_i_projected_device, center_j_projected_device,
        moment_projected_device,
    )
    return center_i_projected, center_j_projected, moment_projected
