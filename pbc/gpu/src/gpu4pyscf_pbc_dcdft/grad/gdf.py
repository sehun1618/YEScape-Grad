"""Restricted semilocal DC-DFT analytic nuclear gradient with GPU GDF.

Native mode uses GPU4PySCF CDERI generation and source-contracted periodic GDF
derivative kernels.  The DC Z-response is evaluated with a polarization
identity, so production does not materialize full ``dCDERI``, ``dJ`` or ``dK``
tensors.  Reference mode retains those tensors for equation-level diagnostics.
Remaining host integral boundaries are explicit in result metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cupy as cp
import numpy as np
from pyscf.pbc.grad import krhf as cpu_krhf_grad
from pyscf_pbc_dcdft.grad.assembly import GradientContributions

from ..df.gdf.contracted_derivative import (
    coulomb_energy_derivative,
    restricted_hf_bilinear_derivative,
    restricted_hf_bilinear_derivative_one_pass,
    restricted_hf_bilinear_derivative_one_pass_gamma,
    target_j_plus_restricted_hf_bilinear_derivative_one_pass,
)
from ..df.gdf.jk_derivative import (
    GDFJKDerivatives,
    contract_coulomb_energy_derivative,
    ewald_exchange_derivative,
    gdf_jk_derivative,
)
from ..df.gdf.signed_density import (
    ewald_restricted_hf_bilinear_derivative,
)
from ..df.gdf.response import build_gdf_response_tensors
from ..lib import to_host
from .context import RestrictedGradientContext
from .gdf_one_electron import build_gdf_one_electron_derivatives
from .memory import DeviceMemorySnapshot, DeviceMemoryTracker
from .nuclear_response import (
    DEFAULT_BATCH_MEMORY_FRACTION,
    NuclearResponseBatchExecution,
    NuclearResponseBatchPlan,
    build_adjoint_nuclear_response_state,
    contract_adjoint_z_nuclear_sources,
    contract_z_vector,
    contract_z_vector_sets,
    execute_adaptive_batches,
    nuclear_response_source_vo,
    nuclear_response_source_vo_sets,
    plan_nuclear_response_batches,
)
from .orbitals import build_mo_partitions, target_fock_vo_blocks
from .pulay import contract_target_overlap
from .target import (
    target_energy_weighted_density,
    target_fock_kpts,
    target_xc_atom_gradient,
)
from .zvector import ZVectorResult, solve_z_vector


@dataclass(frozen=True)
class GDFGradientResult:
    """Host-visible Gaussian-DF gradient with device diagnostics."""

    backend: str
    contributions: GradientContributions
    one_electron: np.ndarray
    target_coulomb: np.ndarray | None
    fused_target_j_z_bilinear: np.ndarray | None
    target_xc: np.ndarray
    target_overlap_pulay: np.ndarray
    nuclear: np.ndarray
    z_vector: ZVectorResult
    gdf_setup_mode: str
    rsdf_central_setup: str | None
    cderi_value_parity_max_abs: float | None
    omega: float | None
    metric_omega: float | None
    source_mesh: tuple[int, int, int] | None
    metric_mesh: tuple[int, int, int] | None
    nkpts: int
    z_nuclear_response_batch_size: int | None
    z_nuclear_response_batch_plan: NuclearResponseBatchPlan | None
    z_nuclear_response_batch_execution: NuclearResponseBatchExecution | None
    gdf_bilinear_mode: str
    gdf_reciprocal_memory: dict[str, object] | None
    df_response_profile: dict[str, object] | None
    native_one_electron: bool
    target_fock_workspace_reused: bool
    timings_s: dict[str, float]

    @property
    def gradient(self) -> np.ndarray:
        """Return ``dE/dR`` in ``Ha/Bohr`` as a host NumPy array."""

        return self.contributions.total

    def metadata(self) -> dict[str, object]:
        """Report numerical settings, device ownership, and host boundaries."""

        label = self.backend.upper()
        native = self.gdf_setup_mode == "native"
        exact_rsdf_central = (
            self.backend == "rsdf"
            and self.rsdf_central_setup == "exact_gpu_source"
        )
        native_one_electron = self.native_one_electron
        host_boundaries = [
            "PySCF Cell/AuxCell and integral-optimizer metadata",
            "Ewald ion-ion nuclear gradient",
            "final NumPy result reporting",
        ]
        if native:
            host_boundaries.extend(
                [
                    "GPU4PySCF mapped-host CDERI storage/bookkeeping",
                    f"{label} local-part-2 pseudopotential value setup",
                ]
            )
        elif self.backend == "rsdf" and self.df_response_profile is not None:
            host_boundaries.append(
                "RSDF parameter resolution and screened two-center metric value"
                if exact_rsdf_central
                else "RSDF central 2-center/3-center integral and CDERI setup"
            )
            host_boundaries.extend(
                self.df_response_profile.get("host_owned_operations", ())
            )
        else:
            host_boundaries.extend(
                [
                    f"{label} central 2-center/3-center integral and CDERI setup",
                    f"{label} 2-center/3-center derivative primitive generation",
                ]
            )
        if native_one_electron:
            host_boundaries.append(
                f"{label} local-part-2 analytic H1 primitive generation"
            )
        else:
            host_boundaries.append(
                f"{label} central local/nonlocal GTH hcore matrix setup"
            )
        device_operations = [
            f"source HF {label} SCF J/K iterations",
            "target-DFA energy/Fock/Veff contractions",
            "HF AX and adjoint Z-vector",
            "target XC, Pulay, response, and final electronic assembly",
        ]
        contracted_rsdf = bool(
            self.backend == "rsdf"
            and self.df_response_profile
            and self.df_response_profile.get("response_mode")
            != "full_tensor_reference"
        )
        if native:
            device_operations.extend(
                [
                    f"native {label} CDERI generation",
                    f"source-contracted {label} J/K nuclear derivatives",
                    "bilinear Z-vector two-electron response",
                ]
            )
        elif contracted_rsdf:
            device_operations.extend(
                self.df_response_profile.get("device_owned_operations", ())
            )
            device_operations.append(
                "no retained full dCDERI, dJ, or dK derivative tensors"
            )
        else:
            device_operations.extend(
                [
                    f"{label} metric Cholesky and metric-square-root response",
                    "dCDERI source/metric assembly",
                    "full dJ/dR and dK/dR contraction",
                ]
            )
        if native_one_electron:
            device_operations.extend(
                [
                    "bounded kinetic/local-part-2 H1 streaming",
                    "source-contracted local-part-1 SR/LR GTH derivative",
                    "density-contracted nonlocal GTH derivative",
                ]
            )
        return {
            "backend": self.backend,
            "device": "gpu",
            "gdf_setup_mode": self.gdf_setup_mode,
            "rsdf_central_setup": self.rsdf_central_setup,
            "gdf_bilinear_mode": self.gdf_bilinear_mode,
            "gdf_reciprocal_memory": self.gdf_reciprocal_memory,
            "df_response_profile": self.df_response_profile,
            "df_observable_mode": (
                None
                if self.df_response_profile is None
                else self.df_response_profile.get("observable_mode")
            ),
            "native_one_electron": self.native_one_electron,
            "gradient_partition": (
                "explicit_response_coupled"
                if self.fused_target_j_z_bilinear is not None
                else "explicit_response"
            ),
            "target_coulomb_component_resolved": (
                self.target_coulomb is not None
            ),
            "units": "Ha/Bohr",
            "nkpts": self.nkpts,
            "omega": self.omega,
            "source_omega": self.omega,
            "metric_omega": self.metric_omega,
            "source_mesh": (
                None if self.source_mesh is None else list(self.source_mesh)
            ),
            "metric_mesh": (
                None if self.metric_mesh is None else list(self.metric_mesh)
            ),
            "target_numint_backend": "numint",
            "target_fock_workspace_reused": (
                self.target_fock_workspace_reused
            ),
            "cderi_value_parity_max_abs": self.cderi_value_parity_max_abs,
            "z_vector_dimension": self.z_vector.dimension,
            "z_vector_solver": self.z_vector.solver,
            "z_vector_solver_reason": self.z_vector.solver_reason,
            "z_vector_iterations": self.z_vector.iterations,
            "z_vector_solver_wall_time_s": self.z_vector.solver_wall_time_s,
            "z_vector_mean_iteration_wall_time_s": (
                self.z_vector.mean_iteration_wall_time_s
            ),
            "z_vector_dense_matrix_gib": self.z_vector.dense_matrix_gib,
            "z_vector_residual_max_abs": self.z_vector.residual_max_abs,
            "z_vector_condition_number": self.z_vector.condition_number,
            "z_vector_condition_number_status": (
                self.z_vector.condition_number_status
            ),
            "z_vector_matrix_asymmetry_max_abs": (
                self.z_vector.matrix_asymmetry_max_abs
            ),
            "z_vector_equation": self.z_vector.equation,
            "z_nuclear_response_batch_size": (
                self.z_nuclear_response_batch_size
            ),
            "z_nuclear_response_batch_plan": (
                None
                if self.z_nuclear_response_batch_plan is None
                else self.z_nuclear_response_batch_plan.metadata()
            ),
            "z_nuclear_response_batch_execution": (
                None
                if self.z_nuclear_response_batch_execution is None
                else self.z_nuclear_response_batch_execution.metadata()
            ),
            "silent_cpu_fallback": False,
            "host_boundaries": host_boundaries,
            "device_owned_operations": device_operations,
            "timings_s": dict(self.timings_s),
        }


def gdf_gpu_gradient(
    dc_mf,
    *,
    z_equation="adjoint",
    z_solver="auto",
    z_tolerance=1.0e-12,
    z_residual_tolerance=1.0e-8,
    z_max_cycle=80,
    z_dense_max_dimension=256,
    z_nuclear_batch_memory_fraction=DEFAULT_BATCH_MEMORY_FRACTION,
    z_nuclear_batch_max_components=None,
    gdf_bilinear_mode="two_pass",
    progress_callback=None,
) -> GDFGradientResult:
    """Evaluate the restricted Gamma/k-point GDF analytic gradient."""

    return _gaussian_df_gpu_gradient(
        dc_mf,
        backend_name="gdf",
        response_builder=build_gdf_response_tensors,
        z_equation=z_equation,
        z_solver=z_solver,
        z_tolerance=z_tolerance,
        z_residual_tolerance=z_residual_tolerance,
        z_max_cycle=z_max_cycle,
        z_dense_max_dimension=z_dense_max_dimension,
        z_nuclear_batch_memory_fraction=z_nuclear_batch_memory_fraction,
        z_nuclear_batch_max_components=z_nuclear_batch_max_components,
        gdf_bilinear_mode=gdf_bilinear_mode,
        progress_callback=progress_callback,
    )


def _gaussian_df_gpu_gradient(
    dc_mf,
    *,
    backend_name,
    response_builder,
    z_equation,
    z_solver,
    z_tolerance,
    z_residual_tolerance,
    z_max_cycle,
    z_dense_max_dimension,
    z_nuclear_batch_memory_fraction=DEFAULT_BATCH_MEMORY_FRACTION,
    z_nuclear_batch_max_components=None,
    gdf_bilinear_mode="two_pass",
    progress_callback=None,
) -> GDFGradientResult:
    """Shared device assembly for GDF-family CDERI providers."""

    if getattr(dc_mf.with_df, "backend_name", None) != backend_name:
        raise NotImplementedError(
            f"{backend_name} GPU gradient requires matching provider"
        )
    total_start = perf_counter()
    timings = {}
    native_gdf = bool(
        getattr(dc_mf.with_df, "uses_native_integrals", False)
    )
    native_one_electron = native_gdf or backend_name == "rsdf"
    gdf_bilinear_mode = _validate_gdf_bilinear_mode(
        gdf_bilinear_mode,
        backend_name=backend_name,
        native_gdf=native_gdf,
        nkpts=len(np.asarray(dc_mf.with_df.kpts).reshape(-1, 3)),
    )
    fused_two_electron = gdf_bilinear_mode == "one_pass_fused"
    reciprocal_memory_diagnostics = {}
    contracted_two_electron_response = None
    fused_target_j_z_bilinear = None

    if progress_callback is not None:
        progress_callback("gradient_target_fock_rhs", state="started")
    start = perf_counter()
    context = RestrictedGradientContext.from_dc_method(dc_mf)
    partitions = build_mo_partitions(context)
    target_fock, target_fock_workspace_reused = target_fock_kpts(
        context, return_reuse=True
    )
    target_vo = target_fock_vo_blocks(target_fock, partitions)
    cp.cuda.Stream.null.synchronize()
    timings["target_fock_and_rhs"] = perf_counter() - start
    if progress_callback is not None:
        progress_callback(
            "gradient_target_fock_rhs",
            state="complete",
            wall_time_s=timings["target_fock_and_rhs"],
        )

    start = perf_counter()
    z_result = solve_z_vector(
        context.source_mf,
        partitions,
        target_vo,
        equation=z_equation,
        solver=z_solver,
        tolerance=z_tolerance,
        residual_tolerance=z_residual_tolerance,
        max_cycle=z_max_cycle,
        dense_max_dimension=z_dense_max_dimension,
        progress_callback=progress_callback,
    )
    cp.cuda.Stream.null.synchronize()
    timings["hf_ax_and_zvector"] = perf_counter() - start

    if progress_callback is not None:
        progress_callback("gradient_one_electron", state="started")
    start = perf_counter()
    one_sources = build_gdf_one_electron_derivatives(
        context.cell,
        context.kpts,
        native_local_part1=native_one_electron,
    )
    hcore_by_atom = one_sources.hcore_by_atom
    native_response_core = None
    z_nuclear_batch_plan = None
    z_nuclear_batch_execution = None
    if native_one_electron:
        fused_start = perf_counter()
        (
            one_electron,
            native_response_core,
            z_nuclear_batch_plan,
            z_nuclear_batch_execution,
        ) = _batched_native_one_electron_and_z_response(
            context,
            partitions,
            z_result,
            one_sources,
            memory_fraction=z_nuclear_batch_memory_fraction,
            max_batch_size=z_nuclear_batch_max_components,
            progress_callback=progress_callback,
        )
        cp.cuda.Stream.null.synchronize()
        timings["one_electron_z_source_fused"] = (
            perf_counter() - fused_start
        )
        timings["z_nuclear_cphf_source"] = timings[
            "one_electron_z_source_fused"
        ]
    else:
        one_electron = cp.einsum(
            "axkij,kji->ax", hcore_by_atom, context.density
        ).real / context.nkpts
    if native_one_electron:
        local_part1_start = perf_counter()
        one_electron += one_sources.contract_local_part1(context.density)
        one_electron += one_sources.contract_nonlocal(context.density)
        cp.cuda.Stream.null.synchronize()
        timings[f"{backend_name}_native_contracted_pseudo_explicit"] = (
            perf_counter() - local_part1_start
        )
    cp.cuda.Stream.null.synchronize()
    timings["one_electron_host_integrals"] = (
        one_sources.host_integral_seconds
    )
    timings["one_electron_total"] = perf_counter() - start
    if progress_callback is not None:
        progress_callback(
            "gradient_one_electron",
            state="complete",
            wall_time_s=timings["one_electron_total"],
        )

    if native_gdf:
        cderi_response = None
        jk_derivative = None
        target_coulomb = None
        if not fused_two_electron:
            if progress_callback is not None:
                progress_callback("gdf_target_j_derivative", state="started")
            start = perf_counter()
            target_coulomb = coulomb_energy_derivative(
                dc_mf.with_df,
                context.density,
            )
            cp.cuda.Stream.null.synchronize()
            timings[f"{backend_name}_native_target_j_derivative"] = (
                perf_counter() - start
            )
            if progress_callback is not None:
                progress_callback(
                    "gdf_target_j_derivative",
                    state="complete",
                    wall_time_s=timings[
                        f"{backend_name}_native_target_j_derivative"
                    ],
                )
    else:
        if progress_callback is not None:
            progress_callback("gdf_response_tensors", state="started")
        start = perf_counter()
        if backend_name == "rsdf":
            transition_density = _z_transition_density(
                z_result, partitions
            )
            cderi_response = response_builder(
                dc_mf.with_df,
                context.density,
                transition_density,
                observable_mode="fused",
                validate_central_source=False,
            )
            fused_target_j_z_bilinear = (
                cderi_response.coupled_target_j_bilinear
            )
            if fused_target_j_z_bilinear is None:
                target_coulomb = cderi_response.target_coulomb
                contracted_two_electron_response = (
                    cderi_response.restricted_hf_bilinear
                )
            else:
                target_coulomb = None
                contracted_two_electron_response = None
            if context.source_mf.exxdiv is not None:
                if context.source_mf.exxdiv != "ewald":
                    raise NotImplementedError(
                        "GPU RSDF supports exxdiv='ewald' or None, got "
                        f"{context.source_mf.exxdiv!r}"
                    )
                ewald_response = ewald_restricted_hf_bilinear_derivative(
                    context.cell,
                    context.kpts,
                    context.density,
                    transition_density,
                    _full_overlap_derivatives(context, one_sources),
                )
                if fused_target_j_z_bilinear is None:
                    contracted_two_electron_response += ewald_response
                else:
                    fused_target_j_z_bilinear += ewald_response
        else:
            cderi_response = response_builder(dc_mf.with_df)
        timings[f"{backend_name}_host_primitive_integrals"] = (
            cderi_response.host_primitive_seconds
        )
        timings[f"{backend_name}_metric_cderi_device_response"] = (
            cderi_response.device_response_seconds
        )
        timings[f"{backend_name}_response_tensor_total"] = (
            perf_counter() - start
        )
        if progress_callback is not None:
            progress_callback(
                "gdf_response_tensors",
                state="complete",
                wall_time_s=timings[
                    f"{backend_name}_response_tensor_total"
                ],
            )

        if backend_name != "rsdf":
            if progress_callback is not None:
                progress_callback("gdf_jk_derivative", state="started")
            start = perf_counter()
            jk_derivative = gdf_jk_derivative(
                cderi_response.cderi,
                cderi_response.dcderi,
                context.density,
            )
            overlap_derivative = _full_overlap_derivatives(
                context, one_sources
            )
            jk_derivative = _apply_exchange_divergence_response(
                context, jk_derivative, overlap_derivative
            )
            cp.cuda.Stream.null.synchronize()
            timings[f"{backend_name}_jk_device_contraction"] = (
                perf_counter() - start
            )
            if progress_callback is not None:
                progress_callback(
                    "gdf_jk_derivative",
                    state="complete",
                    wall_time_s=timings[
                        f"{backend_name}_jk_device_contraction"
                    ],
                )
            target_coulomb = contract_coulomb_energy_derivative(
                context.density, jk_derivative.coulomb
            )

    if progress_callback is not None:
        progress_callback("gradient_target_explicit", state="started")
    start = perf_counter()
    target_xc = target_xc_atom_gradient(context)
    weighted_density = target_energy_weighted_density(
        context, target_fock
    )
    target_overlap = contract_target_overlap(
        context, one_sources.overlap_rows, weighted_density
    )
    nuclear = np.asarray(
        cpu_krhf_grad.grad_nuc(
            context.cell, range(context.cell.natm)
        ),
        dtype=float,
    )
    explicit_device = (
        one_electron
        + target_xc
        + target_overlap
        + cp.asarray(nuclear)
    )
    if target_coulomb is not None:
        explicit_device += target_coulomb
    cp.cuda.Stream.null.synchronize()
    timings["target_explicit_device_assembly"] = perf_counter() - start
    if progress_callback is not None:
        progress_callback(
            "gradient_target_explicit",
            state="complete",
            wall_time_s=timings["target_explicit_device_assembly"],
        )

    start = perf_counter()
    z_nuclear_batch_size = None
    source_hf = (
        None
        if (
            native_gdf
            or contracted_two_electron_response is not None
            or fused_target_j_z_bilinear is not None
        )
        else jk_derivative.restricted_hf
    )
    if native_one_electron:
        response = native_response_core
        z_nuclear_batch_size = z_nuclear_batch_execution.final_batch_size
        transition_density = _z_transition_density(z_result, partitions)
        local_part1_start = perf_counter()
        response += 2.0 * one_sources.contract_local_part1(
            transition_density
        )
        response += 2.0 * one_sources.contract_nonlocal(
            transition_density
        )
        cp.cuda.Stream.null.synchronize()
        timings[f"{backend_name}_native_contracted_pseudo_z_response"] = (
            perf_counter() - local_part1_start
        )
        if native_gdf:
            if progress_callback is not None:
                progress_callback(
                    "gdf_bilinear_z_response", state="started"
                )
            bilinear_start = perf_counter()
            if fused_two_electron:
                fused_target_j_z_bilinear = (
                    target_j_plus_restricted_hf_bilinear_derivative_one_pass(
                        dc_mf.with_df,
                        context.density,
                        transition_density,
                        exxdiv=context.source_mf.exxdiv,
                        memory_diagnostics=reciprocal_memory_diagnostics,
                    )
                )
            else:
                response += _native_bilinear_derivative(
                    dc_mf.with_df,
                    context.density,
                    transition_density,
                    exxdiv=context.source_mf.exxdiv,
                    mode=gdf_bilinear_mode,
                    nkpts=context.nkpts,
                    memory_diagnostics=reciprocal_memory_diagnostics,
                )
            cp.cuda.Stream.null.synchronize()
            timings[
                "gdf_native_target_j_z_bilinear_fused"
                if fused_two_electron
                else "gdf_native_bilinear_z_response"
            ] = perf_counter() - bilinear_start
            if progress_callback is not None:
                progress_callback(
                    "gdf_bilinear_z_response", state="complete"
                )
        else:
            rsdf_response_start = perf_counter()
            if fused_target_j_z_bilinear is None:
                if contracted_two_electron_response is None:
                    response += _componentwise_z_two_electron_response(
                        context,
                        partitions,
                        z_result,
                        source_hf,
                    )
                else:
                    response += contracted_two_electron_response
            cp.cuda.Stream.null.synchronize()
            timings["rsdf_z_two_electron_response"] = (
                perf_counter() - rsdf_response_start
            )
    else:
        cphf_source_start = perf_counter()
        response = _componentwise_z_nuclear_response(
            context,
            partitions,
            z_result,
            hcore_by_atom,
            one_sources,
            source_hf,
            progress_callback=progress_callback,
        )
        cp.cuda.Stream.null.synchronize()
        timings["z_nuclear_cphf_source"] = (
            perf_counter() - cphf_source_start
        )
    cp.cuda.Stream.null.synchronize()
    timings["z_nuclear_device_contraction"] = (
        perf_counter() - start
        + (
            timings.get("z_nuclear_cphf_source", 0.0)
            if native_one_electron
            else 0.0
        )
    )
    if progress_callback is not None:
        progress_callback(
            "z_nuclear_response_complete",
            state="complete",
            wall_time_s=timings["z_nuclear_device_contraction"],
        )

    if cderi_response is None:
        parity = None
        omega = float(getattr(dc_mf.with_df, "_omega", 0.0))
        metric_omega = omega
        source_mesh = None
        metric_mesh = None
        df_response_profile = None
    else:
        parity = cderi_response.value_parity_max_abs
        omega = float(
            getattr(
                cderi_response.parameters,
                "source_omega",
                cderi_response.parameters.omega,
            )
        )
        metric_omega = float(
            getattr(
                cderi_response.parameters,
                "metric_omega",
                cderi_response.parameters.omega,
            )
        )
        source_mesh = cderi_response.parameters.source_mesh
        metric_mesh = cderi_response.parameters.metric_mesh
        profile = getattr(cderi_response, "profile", None)
        df_response_profile = (
            None if profile is None else profile.metadata()
        )
        if profile is not None:
            timings.update(
                {
                    "rsdf_host_source_value": (
                        profile.host_source_value_seconds
                    ),
                    "rsdf_host_metric_value": (
                        profile.host_metric_value_seconds
                    ),
                    "rsdf_host_source_derivative": (
                        profile.host_source_derivative_seconds
                    ),
                    "rsdf_host_metric_derivative": (
                        profile.host_metric_derivative_seconds
                    ),
                    "rsdf_device_central_upload_factorization": (
                        profile.device_central_upload_factorization_seconds
                    ),
                    "rsdf_device_derivative_upload_response": (
                        profile.device_derivative_upload_response_seconds
                    ),
                    "rsdf_device_native_source_derivative": (
                        profile.device_native_source_derivative_seconds
                    ),
                    "rsdf_device_native_metric_derivative": (
                        profile.device_native_metric_derivative_seconds
                    ),
                    "rsdf_device_adjoint_component": (
                        profile.device_adjoint_component_seconds
                    ),
                    "rsdf_device_native_source_setup": (
                        profile.device_native_source_setup_seconds
                    ),
                    "rsdf_device_native_source_short_range": (
                        profile.device_native_source_short_range_seconds
                    ),
                    "rsdf_device_native_source_reciprocal": (
                        profile.device_native_source_reciprocal_seconds
                    ),
                }
            )

    start = perf_counter()
    result = GDFGradientResult(
        backend=backend_name,
        contributions=GradientContributions.assemble(
            to_host(explicit_device),
            to_host(response),
            coupled=(
                None
                if fused_target_j_z_bilinear is None
                else to_host(fused_target_j_z_bilinear)
            ),
        ),
        one_electron=to_host(one_electron),
        target_coulomb=(
            None if target_coulomb is None else to_host(target_coulomb)
        ),
        fused_target_j_z_bilinear=(
            None
            if fused_target_j_z_bilinear is None
            else to_host(fused_target_j_z_bilinear)
        ),
        target_xc=to_host(target_xc),
        target_overlap_pulay=to_host(target_overlap),
        nuclear=nuclear,
        z_vector=z_result,
        gdf_setup_mode=("native" if native_gdf else "reference"),
        rsdf_central_setup=getattr(
            dc_mf.with_df,
            "central_setup",
            None,
        ),
        cderi_value_parity_max_abs=parity,
        omega=omega,
        metric_omega=metric_omega,
        source_mesh=source_mesh,
        metric_mesh=metric_mesh,
        nkpts=context.nkpts,
        z_nuclear_response_batch_size=z_nuclear_batch_size,
        z_nuclear_response_batch_plan=z_nuclear_batch_plan,
        z_nuclear_response_batch_execution=z_nuclear_batch_execution,
        gdf_bilinear_mode=gdf_bilinear_mode,
        gdf_reciprocal_memory=(
            reciprocal_memory_diagnostics or None
        ),
        df_response_profile=df_response_profile,
        native_one_electron=native_one_electron,
        target_fock_workspace_reused=target_fock_workspace_reused,
        timings_s=timings,
    )
    timings["final_host_reporting"] = perf_counter() - start
    timings["gradient_total"] = perf_counter() - total_start
    if progress_callback is not None:
        progress_callback(
            "gradient_backend_complete",
            backend=backend_name,
            wall_time_s=timings["gradient_total"],
        )
    return result


def _validate_gdf_bilinear_mode(
    mode,
    *,
    backend_name,
    native_gdf,
    nkpts,
):
    normalized = str(mode).strip().lower()
    allowed = {
        "two_pass",
        "one_pass",
        "one_pass_gamma",
        "one_pass_fused",
    }
    if normalized not in allowed:
        raise ValueError(
            f"gdf_bilinear_mode must be one of {sorted(allowed)}, got {mode!r}"
        )
    if normalized in {
        "one_pass",
        "one_pass_gamma",
        "one_pass_fused",
    } and (
        backend_name != "gdf" or not native_gdf
    ):
        raise NotImplementedError(
            f"{normalized} requires native GPU GDF"
        )
    if normalized == "one_pass_gamma" and int(nkpts) != 1:
        raise NotImplementedError(
            "one_pass_gamma cannot be used with sampled k-points"
        )
    return normalized


def _native_bilinear_derivative(
    provider,
    reference_density,
    response_density,
    *,
    exxdiv,
    mode,
    nkpts,
    memory_diagnostics,
):
    if mode == "two_pass":
        return restricted_hf_bilinear_derivative(
            provider,
            reference_density,
            response_density,
            exxdiv=exxdiv,
        )
    if mode == "one_pass":
        return restricted_hf_bilinear_derivative_one_pass(
            provider,
            reference_density,
            response_density,
            exxdiv=exxdiv,
            memory_diagnostics=memory_diagnostics,
        )
    if nkpts != 1:
        raise NotImplementedError(
            "one_pass_gamma cannot be used with sampled k-points"
        )
    return restricted_hf_bilinear_derivative_one_pass_gamma(
        provider,
        reference_density,
        response_density,
        exxdiv=exxdiv,
        memory_diagnostics=memory_diagnostics,
    )


def _full_overlap_derivatives(context, one_sources):
    return cp.stack(
        [
            cp.stack(
                [one_sources.overlap(atom_id, axis) for axis in range(3)],
                axis=1,
            )
            for atom_id in range(context.cell.natm)
        ]
    )


def _componentwise_z_nuclear_response(
    context,
    partitions,
    z_result,
    hcore_by_atom,
    one_sources,
    source_hf,
    *,
    progress_callback,
):
    """Reference component loop retained for non-native GDF/RSDF tensors."""

    natoms = context.cell.natm
    response = cp.zeros((natoms, 3), dtype=cp.float64)
    if progress_callback is not None:
        progress_callback("z_nuclear_response", completed=0, total=natoms)
    for atom_id in range(natoms):
        for axis in range(3):
            h1_kpts = hcore_by_atom[atom_id, axis]
            if source_hf is not None:
                h1_kpts = h1_kpts + source_hf[atom_id, :, axis]
            source_vo = nuclear_response_source_vo(
                context,
                partitions,
                h1_kpts,
                one_sources.overlap(atom_id, axis),
            )
            response[atom_id, axis] = contract_z_vector(
                z_result.blocks,
                source_vo,
                nkpts=context.nkpts,
            )
        if progress_callback is not None:
            progress_callback(
                "z_nuclear_response",
                completed=atom_id + 1,
                total=natoms,
            )
    return response


def _componentwise_z_two_electron_response(
    context,
    partitions,
    z_result,
    source_hf,
):
    """Contract only the RSDF ``dJ-dK/2`` source with the Z-vector.

    The streamed one-electron route already accounts for H1, overlap Pulay,
    and the overlap-induced HF response.  Linearity leaves this independent
    two-electron source term, which is available from the canonical RSDF
    ``dCDERI`` response.
    """

    if source_hf is None:
        raise ValueError("RSDF two-electron response source is required")
    response = cp.zeros((context.cell.natm, 3), dtype=cp.float64)
    for atom_id in range(context.cell.natm):
        for axis in range(3):
            source_blocks = []
            for k_index, partition in enumerate(partitions):
                source_blocks.append(
                    partition.virtual_coeff.conj().T
                    @ source_hf[atom_id, k_index, axis]
                    @ partition.occupied_coeff
                )
            response[atom_id, axis] = contract_z_vector(
                z_result.blocks,
                source_blocks,
                nkpts=context.nkpts,
            )
    return response


def _batched_native_one_electron_and_z_response(
    context,
    partitions,
    z_result,
    one_sources,
    *,
    memory_fraction,
    max_batch_size,
    progress_callback,
):
    """Stream native H1 batches into density and adjoint contractions."""

    natoms = context.cell.natm
    component_count = 3 * natoms
    memory_start = DeviceMemorySnapshot.capture()
    memory_tracker = DeviceMemoryTracker(memory_start)
    plan = plan_nuclear_response_batches(
        component_count,
        context.nkpts,
        context.cell.nao_nr(),
        free_device_bytes=memory_start.free_device_bytes,
        total_device_bytes=memory_start.total_device_bytes,
        memory_fraction=memory_fraction,
        max_batch_size=max_batch_size,
    )
    one_electron = cp.empty(component_count, dtype=cp.float64)
    response = cp.empty(component_count, dtype=cp.float64)
    adjoint_state = build_adjoint_nuclear_response_state(
        context, partitions, z_result.blocks
    )
    if progress_callback is not None:
        progress_callback(
            "z_nuclear_response",
            completed=0,
            total=component_count,
            batch_size=plan.batch_size,
            batch_count=plan.batch_count,
            estimated_batch_working_bytes=(
                plan.estimated_batch_working_bytes
            ),
            free_device_bytes_at_plan=plan.free_device_bytes,
        )

    def worker(start, stop):
        h1_batch = one_sources.hcore_batch(start, stop)
        overlap_batch = cp.stack(
            [
                one_sources.overlap(component // 3, component % 3)
                for component in range(start, stop)
            ]
        )
        explicit = cp.einsum(
            "xkij,kji->x", h1_batch, context.density, optimize=True
        ).real / context.nkpts
        adjoint = contract_adjoint_z_nuclear_sources(
            adjoint_state, partitions, h1_batch, overlap_batch,
            nkpts=context.nkpts
        )
        return explicit, adjoint

    def accept_result(start, stop, value):
        explicit, adjoint = value
        one_electron[start:stop] = explicit
        response[start:stop] = adjoint

    def on_oom(_start, _stop, _error):
        memory_tracker.observe()
        cp.get_default_memory_pool().free_all_blocks()

    def on_rebatch(scheduler, start, reduced_size):
        if progress_callback is not None:
            progress_callback(
                "z_nuclear_response_rebatch",
                completed=scheduler.completed_count,
                total=component_count,
                failed_start=start,
                reduced_batch_size=reduced_size,
                rebatch_count=scheduler.rebatch_count,
            )

    def on_success(scheduler, _start, stop):
        memory_tracker.observe()
        if progress_callback is not None:
            progress_callback(
                "z_nuclear_response",
                completed=stop,
                total=component_count,
                batch_size=scheduler.batch_size,
                rebatch_count=scheduler.rebatch_count,
            )

    scheduler = execute_adaptive_batches(
        component_count,
        plan.batch_size,
        worker,
        accept_result,
        on_oom=on_oom,
        on_rebatch=on_rebatch,
        on_success=on_success,
    )
    execution = NuclearResponseBatchExecution(
        initial_batch_size=plan.batch_size,
        final_batch_size=scheduler.batch_size,
        minimum_batch_size=int(scheduler.minimum_batch_size),
        completed_count=scheduler.completed_count,
        rebatch_count=scheduler.rebatch_count,
        device_memory=memory_tracker.result(),
    )
    return (
        one_electron.reshape(natoms, 3),
        response.reshape(natoms, 3),
        plan,
        execution,
    )


def _z_transition_density(z_result, partitions):
    """Return Hermitian AO transition densities for bilinear GDF response."""

    blocks = []
    for z_block, partition in zip(
        z_result.blocks, partitions, strict=True
    ):
        one_sided = (
            partition.virtual_coeff
            @ z_block
            @ partition.occupied_coeff.conj().T
        )
        blocks.append(one_sided + one_sided.conj().T)
    return cp.stack(blocks)


def _apply_exchange_divergence_response(
    context,
    derivative,
    overlap_derivative,
):
    exxdiv = context.source_mf.exxdiv
    if exxdiv is None:
        return derivative
    if exxdiv != "ewald":
        raise NotImplementedError(
            f"GPU GDF supports exxdiv='ewald' or None, got {exxdiv!r}"
        )
    correction = ewald_exchange_derivative(
        context.cell,
        context.kpts,
        context.density,
        overlap_derivative,
    )
    return GDFJKDerivatives(
        coulomb=derivative.coulomb,
        exchange=derivative.exchange + correction,
    )
