"""Restricted GPU DC-DFT analytic stress with explicit host-integral boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import cupy as cp
import numpy as np

from gpu4pyscf_pbc_dcdft.grad.context import RestrictedGradientContext
from gpu4pyscf_pbc_dcdft.grad.memory import DeviceMemoryTracker
from gpu4pyscf_pbc_dcdft.grad.nuclear_response import (
    build_adjoint_nuclear_response_state,
    contract_adjoint_z_nuclear_sources,
)
from gpu4pyscf_pbc_dcdft.grad.orbitals import (
    build_mo_partitions,
    target_fock_vo_blocks,
)
from gpu4pyscf_pbc_dcdft.grad.target import (
    target_energy_weighted_density,
    target_fock_kpts,
)
from gpu4pyscf_pbc_dcdft.grad.zvector import solve_z_vector
from pyscf_pbc_dcdft_stress.conventions import stress_voigt_to_tensor
from pyscf_pbc_dcdft_stress.ewald_strain import analytic_ewald_strain
from pyscf_pbc_dcdft_stress.gdf_cderi_strain import kpoint_madelung_strain
from pyscf_pbc_dcdft_stress.gdf_pp_strain import (
    kpoint_gdf_pseudopotential_strain,
)
from pyscf_pbc_dcdft_stress.one_electron_strain import (
    kpoint_one_electron_strain,
)

from .df_response import (
    build_and_contract_device_df_strain_direct,
    build_and_contract_device_df_strain_adjoint_reference,
    build_and_contract_device_df_strain_source_adjoint,
    build_device_df_strain_response,
    device_host_pair_response_matmul,
    device_host_factorized_projection,
    device_host_factorized_projection_bounded,
)
from .contracted_df_response import ContractedDFStrain, contracted_gpu_df_strain
from .pp_response import contracted_gpu4pyscf_pp_strain


_VOIGT_COMPONENTS = (
    ((0, 0, 1.0),),
    ((1, 1, 1.0),),
    ((2, 2, 1.0),),
    ((1, 2, 0.5), (2, 1, 0.5)),
    ((0, 2, 0.5), (2, 0, 0.5)),
    ((0, 1, 0.5), (1, 0, 0.5)),
)


@dataclass(frozen=True)
class GPUAnalyticStressResult:
    """Host-visible GPU stress with device ownership diagnostics."""

    component_derivatives_ha: Mapping[str, np.ndarray]
    stress_voigt_ha_per_bohr3: np.ndarray
    stress_tensor_ha_per_bohr3: np.ndarray
    volume_bohr3: float
    z_vector: object
    timings_s: Mapping[str, float]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "component_derivatives_ha",
            MappingProxyType({
                key: np.asarray(value, dtype=float)
                for key, value in self.component_derivatives_ha.items()
            }),
        )
        object.__setattr__(
            self, "stress_voigt_ha_per_bohr3",
            np.asarray(self.stress_voigt_ha_per_bohr3, dtype=float),
        )
        object.__setattr__(
            self, "stress_tensor_ha_per_bohr3",
            np.asarray(self.stress_tensor_ha_per_bohr3, dtype=float),
        )
        object.__setattr__(self, "timings_s", MappingProxyType(dict(self.timings_s)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def as_dict(self):
        return {
            "component_derivatives_ha": {
                key: value.tolist()
                for key, value in self.component_derivatives_ha.items()
            },
            "stress_voigt_ha_per_bohr3": self.stress_voigt_ha_per_bohr3.tolist(),
            "stress_tensor_ha_per_bohr3": self.stress_tensor_ha_per_bohr3.tolist(),
            "volume_bohr3": self.volume_bohr3,
            "z_vector": {
                "dimension": int(self.z_vector.dimension),
                "residual_max_abs": float(self.z_vector.residual_max_abs),
                "solver": self.z_vector.solver,
                "equation": self.z_vector.equation,
                "iterations": int(self.z_vector.iterations),
                "solver_wall_time_s": float(
                    self.z_vector.solver_wall_time_s
                ),
                "mean_iteration_wall_time_s": (
                    None if self.z_vector.mean_iteration_wall_time_s is None
                    else float(self.z_vector.mean_iteration_wall_time_s)
                ),
                "dense_matrix_gib": float(self.z_vector.dense_matrix_gib),
                "condition_number": self.z_vector.condition_number,
                "condition_number_status": (
                    self.z_vector.condition_number_status
                ),
            },
            "timings_s": dict(self.timings_s),
            "metadata": dict(self.metadata),
        }


def gpu_analytic_stress(
    method, *, z_equation="adjoint", z_solver="auto",
    primitive_backend="host_integral_device_response", host_workers=None,
    df_diagonal_cache_bytes=256*(1 << 20), df_device_memory_fraction=0.85,
    host_reciprocal_block_size=None, df_host_reciprocal_block_size=None,
    shared_z_vector=None,
    progress_callback=None, pp_response_backend="host_analytic_matrix",
    pp_grid_block_size=None, pp_nonlocal_host_block_size=8192,
    df_response_backend="host_analytic_device_factorization",
    df_contracted_fd_step=1.0e-5, detailed_profile=False,
):
    """Evaluate analytic stress with GPU state, response, and contractions.

    PySCF/GPU4PySCF do not expose GPU 2c/3c homogeneous-strain integrals.
    ``primitive_backend`` therefore makes the current host integral boundary
    mandatory and visible. No CPU SCF, CPU target energy, CPU Z-vector, or
    full CPU stress driver is called.
    """

    if primitive_backend != "host_integral_device_response":
        raise ValueError(
            "primitive_backend must be 'host_integral_device_response'"
        )
    backend = getattr(getattr(method, "with_df", None), "backend_name", None)
    if backend not in {"gdf", "rsdf"}:
        raise NotImplementedError("GPU analytic stress supports GDF/RSDF only")
    if getattr(method, "target_integrator", "uniform") != "uniform":
        raise NotImplementedError("GPU analytic stress currently requires uniform XC")
    if getattr(method, "dispersion", None) is not None:
        raise NotImplementedError("dispersion stress is outside S7")
    xctype = _xctype(method)
    resolved_df_reciprocal_block_size = (
        host_reciprocal_block_size
        if df_host_reciprocal_block_size is None
        else df_host_reciprocal_block_size
    )
    if pp_response_backend not in {
        "host_analytic_matrix", "gpu4pyscf_native_contracted",
        "gpu4pyscf_native_contracted_analytic_nonlocal",
        "gpu4pyscf_native_contracted_analytic_nonlocal_device",
    }:
        raise ValueError(
            "pp_response_backend must be 'host_analytic_matrix' or "
            "one of the gpu4pyscf native contracted routes"
        )
    if df_response_backend not in {
        "host_analytic_device_factorization",
        "host_analytic_direct_contracted",
        "host_analytic_adjoint_contracted",
        "host_analytic_source_adjoint_contracted",
        "host_analytic_source_adjoint_gamma_gpu_ft",
        "host_analytic_source_adjoint_gpu_ft",
        "host_analytic_direct_contracted_device_gemm",
        "host_analytic_direct_contracted_factorized",
        "host_analytic_direct_contracted_factorized_device",
        "host_analytic_direct_contracted_factorized_device_bounded",
        "gpu_fixed_density_fd_contracted",
    }:
        raise ValueError(
            "df_response_backend must be 'host_analytic_device_factorization' "
            "'host_analytic_direct_contracted', "
            "'host_analytic_adjoint_contracted', "
            "'host_analytic_source_adjoint_contracted', "
            "'host_analytic_source_adjoint_gamma_gpu_ft', "
            "'host_analytic_source_adjoint_gpu_ft', "
            "'host_analytic_direct_contracted_device_gemm', "
            "'host_analytic_direct_contracted_factorized', or "
            "'host_analytic_direct_contracted_factorized_device', "
            "'host_analytic_direct_contracted_factorized_device_bounded', "
            "or "
            "'gpu_fixed_density_fd_contracted'"
        )

    total_start = perf_counter()
    timings = {}
    memory = DeviceMemoryTracker()
    context = RestrictedGradientContext.from_dc_method(method)
    kpts = np.asarray(context.kpts, dtype=float).reshape(-1, 3)
    nkpts = context.nkpts
    cell = context.cell

    _progress(progress_callback, "stress_target_fock_and_zvector", state="started")
    start = perf_counter()
    partitions = build_mo_partitions(context)
    target_fock, reused = target_fock_kpts(context, return_reuse=True)
    target_vo = target_fock_vo_blocks(target_fock, partitions)
    if detailed_profile:
        cp.cuda.Stream.null.synchronize()
    timings["target_fock_and_rhs_device"] = perf_counter()-start
    z_vector = shared_z_vector
    z_started = perf_counter()
    if z_vector is None:
        z_vector = solve_z_vector(
            context.source_mf, partitions, target_vo,
            equation=z_equation, solver=z_solver,
        )
    elif len(z_vector.blocks) != nkpts:
        raise ValueError("shared Z-vector k-point count does not match the method")
    if detailed_profile:
        cp.cuda.Stream.null.synchronize()
    timings["z_vector_solve_observed_device"] = (
        0.0 if shared_z_vector is not None else perf_counter()-z_started
    )
    timings["z_vector_solver_internal"] = (
        0.0 if shared_z_vector is not None
        else float(z_vector.solver_wall_time_s)
    )
    weighted_density = target_energy_weighted_density(context, target_fock)
    adjoint_start = perf_counter()
    adjoint_response = build_adjoint_nuclear_response_state(
        context, partitions, z_vector.blocks
    )
    cp.cuda.Stream.null.synchronize()
    timings["adjoint_hf_response_device"] = perf_counter()-adjoint_start
    timings["target_fock_and_zvector_device"] = perf_counter()-start
    memory.observe()
    _progress(
        progress_callback, "stress_target_fock_and_zvector", state="complete",
        wall_s=timings["target_fock_and_zvector_device"],
    )

    _progress(progress_callback, "stress_host_one_electron_pp", state="started")
    start = perf_counter()
    kinetic_host = kpoint_one_electron_strain(
        cell, kpts, "int1e_kin"
    ).voigt_derivative
    overlap_host = kpoint_one_electron_strain(
        cell, kpts, "int1e_ovlp"
    ).voigt_derivative
    pp_host = None
    if pp_response_backend == "host_analytic_matrix":
        pp_host = kpoint_gdf_pseudopotential_strain(
            cell, kpts, source_block_size=host_reciprocal_block_size
        ).voigt_derivative
    nuclear_host = _tensor_to_voigt(analytic_ewald_strain(cell).raw_derivative)
    timings["host_one_electron_pp_ewald_primitives"] = perf_counter()-start
    kinetic = cp.asarray(kinetic_host)
    overlap = cp.asarray(overlap_host)
    pseudopotential_matrices = None if pp_host is None else cp.asarray(pp_host)
    memory.observe()
    _progress(
        progress_callback, "stress_host_one_electron_pp", state="complete",
        wall_s=timings["host_one_electron_pp_ewald_primitives"],
    )

    contracted_pp = None
    if pp_response_backend in {
        "gpu4pyscf_native_contracted",
        "gpu4pyscf_native_contracted_analytic_nonlocal",
        "gpu4pyscf_native_contracted_analytic_nonlocal_device",
    }:
        _progress(progress_callback, "stress_device_pp_response", state="started")
        start = perf_counter()
        contracted_pp = contracted_gpu4pyscf_pp_strain(
            context, adjoint_response, grid_block_size=pp_grid_block_size,
            detailed_profile=detailed_profile,
            nonlocal_response_backend=(
                "gpu_analytic"
                if pp_response_backend
                == "gpu4pyscf_native_contracted_analytic_nonlocal_device"
                else "host_analytic_matrix"
                if pp_response_backend
                == "gpu4pyscf_native_contracted_analytic_nonlocal"
                else "centered_cell_displacement"
            ),
            nonlocal_host_block_size=pp_nonlocal_host_block_size,
        )
        timings["device_pp_response"] = perf_counter()-start
        memory.observe()
        _progress(
            progress_callback, "stress_device_pp_response", state="complete",
            wall_s=timings["device_pp_response"],
        )

    density = cp.asarray(context.density)
    contracted_df = None
    streaming_direct_df = None
    df_response = jk_response = None
    j_matrices = k_matrices = None
    _progress(progress_callback, "stress_df_host_primitives", state="started")
    start = perf_counter()
    df_derivative_evaluations = 0
    if df_response_backend == "host_analytic_device_factorization":
        df_response = build_device_df_strain_response(
            method.with_df, host_workers=host_workers,
            diagonal_cache_bytes=df_diagonal_cache_bytes,
            device_memory_fraction=df_device_memory_fraction,
            reciprocal_block_size=resolved_df_reciprocal_block_size,
        )
        contraction_start = perf_counter()
        jk_response = df_response.contract_jk_voigt(density)
        j_matrices = jk_response.coulomb
        k_matrices = jk_response.exchange
        df_derivative_evaluations = jk_response.derivative_evaluations
        memory.observe()
        if method.exxdiv == "ewald":
            k_matrices += _device_ewald_exchange_strain(
                cell, kpts, density, overlap
            )
        elif method.exxdiv is not None:
            raise NotImplementedError(
                f"unsupported exxdiv={method.exxdiv!r}"
            )
        cp.cuda.Stream.null.synchronize()
        timings["df_host_primitives"] = df_response.host_primitive_seconds
        timings["df_device_factorization"] = (
            df_response.device_factorization_seconds
        )
        timings["df_device_jk_components"] = perf_counter()-contraction_start
    elif df_response_backend in {
        "host_analytic_direct_contracted",
        "host_analytic_adjoint_contracted",
        "host_analytic_source_adjoint_contracted",
        "host_analytic_source_adjoint_gamma_gpu_ft",
        "host_analytic_source_adjoint_gpu_ft",
        "host_analytic_direct_contracted_device_gemm",
        "host_analytic_direct_contracted_factorized",
        "host_analytic_direct_contracted_factorized_device",
        "host_analytic_direct_contracted_factorized_device_bounded",
    }:
        builder = (
            build_and_contract_device_df_strain_source_adjoint
            if df_response_backend
            in {
                "host_analytic_source_adjoint_contracted",
                "host_analytic_source_adjoint_gamma_gpu_ft",
                "host_analytic_source_adjoint_gpu_ft",
            }
            else
            build_and_contract_device_df_strain_adjoint_reference
            if df_response_backend == "host_analytic_adjoint_contracted"
            else build_and_contract_device_df_strain_direct
        )
        streaming_direct_df = builder(
            method.with_df, density,
            cp.stack(adjoint_response.transition_density),
            host_workers=host_workers,
            device_memory_fraction=df_device_memory_fraction,
            reciprocal_block_size=resolved_df_reciprocal_block_size,
            **(
                {"gamma_gpu_ft": True}
                if df_response_backend
                == "host_analytic_source_adjoint_gamma_gpu_ft"
                else {}
            ),
            **(
                {"gpu_ft": True}
                if df_response_backend
                == "host_analytic_source_adjoint_gpu_ft"
                else {}
            ),
            **(
                {"pair_response_matmul": device_host_pair_response_matmul}
                if df_response_backend
                == "host_analytic_direct_contracted_device_gemm"
                else {}
            ),
            **(
                {"pair_response_mode": "factorized"}
                if df_response_backend
                in {
                    "host_analytic_direct_contracted_factorized",
                    "host_analytic_direct_contracted_factorized_device",
                    "host_analytic_direct_contracted_factorized_device_bounded",
                }
                else {}
            ),
            **(
                {"factorized_projection": device_host_factorized_projection}
                if df_response_backend
                == "host_analytic_direct_contracted_factorized_device"
                else {
                    "factorized_projection":
                    device_host_factorized_projection_bounded
                }
                if df_response_backend
                == "host_analytic_direct_contracted_factorized_device_bounded"
                else {}
            ),
        )
        adjoint_value = streaming_direct_df.adjoint_restricted_hf
        if method.exxdiv == "ewald":
            ewald_exchange = _device_ewald_exchange_strain(
                cell, kpts, density, overlap
            )
            adjoint_value -= cp.einsum(
                "xkij,kji->x", ewald_exchange,
                cp.stack(adjoint_response.transition_density),
                optimize=True,
            ).real/nkpts
        elif method.exxdiv is not None:
            raise NotImplementedError(f"unsupported exxdiv={method.exxdiv!r}")
        direct_timings = streaming_direct_df.metadata["timings_s"]
        contracted_df = ContractedDFStrain(
            target_coulomb=streaming_direct_df.target_coulomb,
            adjoint_restricted_hf=adjoint_value,
            timings_s=direct_timings,
            metadata=streaming_direct_df.metadata,
        )
        df_derivative_evaluations = (
            streaming_direct_df.derivative_evaluations
        )
        timings["df_host_primitives"] = (
            direct_timings["host_pipeline_wall"]
        )
        timings["df_device_factorization"] = 0.0
        timings["df_device_jk_components"] = direct_timings[
            "device_pair_contraction"
        ]
    else:
        contracted_df = contracted_gpu_df_strain(
            method.with_df, density,
            cp.stack(adjoint_response.transition_density),
            exxdiv=method.exxdiv, step=df_contracted_fd_step,
        )
        timings["df_host_primitives"] = 0.0
        timings["df_device_factorization"] = 0.0
        timings["df_device_jk_components"] = contracted_df.timings_s["total"]
    timings["df_response_total"] = perf_counter()-start
    _progress(
        progress_callback, "stress_df_device_response", state="complete",
        host_wall_s=timings["df_host_primitives"],
        device_wall_s=timings["df_device_jk_components"],
    )

    _progress(progress_callback, "stress_xc_lagrangian_assembly", state="started")
    start = perf_counter()
    if contracted_pp is None:
        target_xc_started = perf_counter()
        target_xc = _device_tensor_to_voigt(_gpu_target_xc_strain(context))
        cp.cuda.Stream.null.synchronize()
        timings["target_xc_device"] = perf_counter()-target_xc_started
        pseudopotential = cp.einsum(
            "xkij,kji->x", pseudopotential_matrices, density, optimize=True
        ).real/nkpts
    else:
        target_xc = contracted_pp.target_xc_derivative
        pseudopotential = contracted_pp.density_derivative
        timings["target_xc_device"] = contracted_pp.timings_s["source_xc"]
    kinetic_energy = cp.einsum(
        "xkij,kji->x", kinetic, density, optimize=True
    ).real/nkpts
    if contracted_df is None:
        target_coulomb = 0.5*cp.einsum(
            "xkij,kji->x", j_matrices, density, optimize=True
        ).real/nkpts
    else:
        target_coulomb = contracted_df.target_coulomb
    target_overlap = -cp.einsum(
        "xkij,kji->x", overlap, weighted_density, optimize=True
    ).real/nkpts
    source_fock = kinetic
    if contracted_df is None:
        source_fock = source_fock+j_matrices-0.5*k_matrices
    if pseudopotential_matrices is not None:
        source_fock = source_fock+pseudopotential_matrices
    adjoint_contraction_start = perf_counter()
    orbital_response = contract_adjoint_z_nuclear_sources(
        adjoint_response, partitions, source_fock, overlap, nkpts=nkpts
    )
    if contracted_pp is not None:
        orbital_response += contracted_pp.adjoint_derivative
    if contracted_df is not None:
        orbital_response += contracted_df.adjoint_restricted_hf
    cp.cuda.Stream.null.synchronize()
    timings["adjoint_source_contraction_device"] = (
        perf_counter()-adjoint_contraction_start
    )
    nuclear = cp.asarray(nuclear_host)
    one_electron = kinetic_energy+pseudopotential
    explicit = (
        one_electron+target_coulomb+target_xc+target_overlap+nuclear
    )
    total = explicit+orbital_response
    cp.cuda.Stream.null.synchronize()
    timings["device_xc_lagrangian_assembly"] = perf_counter()-start
    timings["total"] = perf_counter()-total_start
    memory.observe()
    _progress(
        progress_callback, "stress_xc_lagrangian_assembly", state="complete",
        wall_s=timings["device_xc_lagrangian_assembly"],
    )

    components_device = {
        "kinetic": kinetic_energy,
        "pseudopotential": pseudopotential,
        "one_electron": one_electron,
        "target_coulomb": target_coulomb,
        "target_xc": target_xc,
        "target_overlap": target_overlap,
        "nuclear": nuclear,
        "explicit": explicit,
        "orbital_response": orbital_response,
        "total": total,
    }
    components = {
        key: cp.asnumpy(value) for key, value in components_device.items()
    }
    numerical_components = []
    if (
        contracted_pp is not None
        and contracted_pp.metadata["nonlocal_derivative_numerical"]
    ):
        numerical_components.append("gpu4pyscf_pp_factor_cell_displacement")
    if df_response_backend == "gpu_fixed_density_fd_contracted":
        numerical_components.append("fixed_density_df_scalar_cell_displacement")
    volume = float(cell.vol)
    stress_voigt = components["total"]/volume
    return GPUAnalyticStressResult(
        component_derivatives_ha=components,
        stress_voigt_ha_per_bohr3=stress_voigt,
        stress_tensor_ha_per_bohr3=stress_voigt_to_tensor(stress_voigt),
        volume_bohr3=volume,
        z_vector=z_vector,
        timings_s=timings,
        metadata={
            "stage": f"S7_{backend}_{xctype.lower()}_analytic",
            "production": not numerical_components,
            "device": "gpu",
            "df_backend": backend,
            "target_xctype": xctype,
            "nkpts": nkpts,
            "fixed_fractional_kpoints": True,
            "primitive_backend": primitive_backend,
            "silent_cpu_fallback": False,
            "source_hf_owner": "gpu",
            "target_fock_owner": "gpu",
            "z_vector_owner": "gpu",
            "z_vector_reused": shared_z_vector is not None,
            "z_vector_iterations": int(z_vector.iterations),
            "z_vector_solver_wall_time_s": float(
                z_vector.solver_wall_time_s
            ),
            "z_vector_mean_iteration_wall_time_s": (
                None if z_vector.mean_iteration_wall_time_s is None
                else float(z_vector.mean_iteration_wall_time_s)
            ),
            "z_vector_residual_max_abs": float(z_vector.residual_max_abs),
            "target_xc_stress_owner": "gpu4pyscf",
            "df_cholesky_response_owner": (
                "gpu_streaming_pair"
                if streaming_direct_df is not None
                else ("gpu" if df_response is not None else "not_materialized")
            ),
            "df_jk_stress_contraction_owner": (
                "gpu_single_pass_voigt" if contracted_df is None
                else contracted_df.metadata["owner"]
            ),
            "df_response_backend": df_response_backend,
            "df_cderi_derivative_evaluations": (
                df_derivative_evaluations
            ),
            "df_contracted_response_metadata": (
                None if contracted_df is None else dict(contracted_df.metadata)
            ),
            "df_contracted_response_timings_s": (
                None if contracted_df is None else dict(contracted_df.timings_s)
            ),
            "df_contracted_target_coulomb_ha": (
                None if contracted_df is None
                else cp.asnumpy(contracted_df.target_coulomb).tolist()
            ),
            "df_contracted_adjoint_restricted_hf_ha": (
                None if contracted_df is None
                else cp.asnumpy(contracted_df.adjoint_restricted_hf).tolist()
            ),
            "z_strain_response_route": "single_adjoint_hf_response",
            "z_strain_response_hf_density_count": 1,
            "z_strain_source_component_count": 6,
            "one_electron_pp_strain_primitive_owner": (
                "host_analytic_matrix" if contracted_pp is None
                else contracted_pp.metadata["owner"]
            ),
            "pp_response_backend": pp_response_backend,
            "detailed_profile": bool(detailed_profile),
            "pp_response_metadata": (
                None if contracted_pp is None else dict(contracted_pp.metadata)
            ),
            "pp_response_timings_s": (
                None if contracted_pp is None else dict(contracted_pp.timings_s)
            ),
            "work_profile": {
                "pp": (
                    {"owner": "host_analytic_matrix"}
                    if contracted_pp is None
                    else dict(contracted_pp.metadata["work_profile"])
                ),
                "df": (
                    {
                        key: value for key, value in df_response.parameters.items()
                        if key.startswith("stress_df_")
                    }
                    if contracted_df is None
                    else dict(contracted_df.metadata)
                ),
            },
            "pp_g0_overlap_reused_with_vbar": True,
            "df_2c3c_strain_primitive_owner": (
                "not_materialized"
                if df_response_backend == "gpu_fixed_density_fd_contracted"
                else "host_analytic"
            ),
            "target_fock_workspace_reused": bool(reused),
            "df_parameters": (
                streaming_direct_df.metadata["parameters"]
                if streaming_direct_df is not None
                else ({} if df_response is None else df_response.parameters)
            ),
            "df_memory_estimate": (
                streaming_direct_df.metadata["memory_estimate"]
                if streaming_direct_df is not None
                else ({} if df_response is None else df_response.memory_estimate)
            ),
            "df_diagonal_response_bytes": (
                0 if df_response is None else df_response.diagonal_response_bytes
            ),
            "df_diagonal_cache_enabled": (
                False if df_response is None else df_response.diagonal_cache_enabled
            ),
            "df_preflight_free_device_bytes": (
                None if df_response is None
                else df_response.preflight_free_device_bytes
            ),
            "df_device_memory_fraction": (
                float(df_device_memory_fraction) if df_response is None
                else df_response.device_memory_fraction
            ),
            "host_reciprocal_block_size_requested": (
                None if host_reciprocal_block_size is None
                else int(host_reciprocal_block_size)
            ),
            "df_host_reciprocal_block_size_requested": (
                None if resolved_df_reciprocal_block_size is None
                else int(resolved_df_reciprocal_block_size)
            ),
            "one_electron_pp_host_workers": 1,
            "df_host_primitive_workers": (
                streaming_direct_df.metadata["host_worker_count"]
                if streaming_direct_df is not None
                else (0 if df_response is None else df_response.host_worker_count)
            ),
            "central_cderi_parity_max_abs": (
                streaming_direct_df.metadata["central_parity_max_abs"]
                if streaming_direct_df is not None
                else (
                    None if df_response is None
                    else df_response.central_parity_max_abs
                )
            ),
            "device_memory": memory.result().metadata(),
            "low_level_numerical_components": numerical_components,
        },
    )


def _progress(callback, stage, **details):
    if callback is not None:
        callback(stage, **details)


def _device_ewald_exchange_strain(cell, kpts, density, overlap_derivative):
    overlap = cp.asarray(cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts))
    madelung = kpoint_madelung_strain(cell, kpts)
    output = cp.empty_like(overlap_derivative, dtype=cp.complex128)
    for index, terms in enumerate(_VOIGT_COMPONENTS):
        scalar_derivative = sum(
            weight*madelung.raw_derivative[x, y]
            for x, y, weight in terms
        )
        for kpoint_id in range(len(kpts)):
            central = overlap[kpoint_id]@density[kpoint_id]@overlap[kpoint_id]
            ds = overlap_derivative[index, kpoint_id]
            output[index, kpoint_id] = (
                scalar_derivative*central
                + madelung.value*(
                    ds@density[kpoint_id]@overlap[kpoint_id]
                    + overlap[kpoint_id]@density[kpoint_id]@ds
                )
            )
    return output


def _gpu_target_xc_strain(context):
    if context.nkpts == 1 and not bool(getattr(context.dc_mf, "kpoint_mode", False)):
        from gpu4pyscf.pbc.grad import rks as gpu_rks_grad
        from gpu4pyscf.pbc.grad import rks_stress

        gradient = gpu_rks_grad.Gradients(context.target_ks)
        return rks_stress.get_vxc(
            gradient, context.cell, context.density[0],
            with_j=False, with_nuc=False,
        )
    from gpu4pyscf.pbc.grad import krks as gpu_krks_grad
    from gpu4pyscf.pbc.grad import krks_stress

    gradient = gpu_krks_grad.Gradients(context.target_ks)
    return krks_stress.get_vxc(
        gradient, context.cell, context.density, context.kpts,
        with_j=False, with_nuc=False,
    )


def _xctype(method):
    normalized = str(method.xc).upper().replace(" ", "")
    mapping = {
        "PBE": "GGA",
        "GGA_X_PBE,GGA_C_PBE": "GGA",
        "R2SCAN": "MGGA",
        "MGGA_X_R2SCAN,MGGA_C_R2SCAN": "MGGA",
    }
    try:
        return mapping[normalized]
    except KeyError as error:
        raise NotImplementedError(
            f"GPU analytic stress supports PBE/r2SCAN, got {method.xc!r}"
        ) from error


def _tensor_to_voigt(value):
    tensor = np.asarray(value, dtype=float)
    return np.asarray((
        tensor[0, 0], tensor[1, 1], tensor[2, 2],
        0.5*(tensor[1, 2]+tensor[2, 1]),
        0.5*(tensor[0, 2]+tensor[2, 0]),
        0.5*(tensor[0, 1]+tensor[1, 0]),
    ))


def _device_tensor_to_voigt(value):
    tensor = cp.asarray(value)
    return cp.stack((
        tensor[0, 0], tensor[1, 1], tensor[2, 2],
        0.5*(tensor[1, 2]+tensor[2, 1]),
        0.5*(tensor[0, 2]+tensor[2, 0]),
        0.5*(tensor[0, 1]+tensor[1, 0]),
    ))
