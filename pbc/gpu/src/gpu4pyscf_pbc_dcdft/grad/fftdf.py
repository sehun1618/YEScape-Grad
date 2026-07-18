"""Restricted semilocal DC-DFT analytic gradient with GPU FFTDF kernels."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cupy as cp
import numpy as np
from pyscf.pbc.grad import krhf as cpu_krhf_grad
from pyscf_pbc_dcdft.dft.target_integrator import target_owns_coulomb
from pyscf_pbc_dcdft.grad.assembly import GradientContributions

from ..df.fft import fft_jk_derivative
from ..lib import to_host
from .context import RestrictedGradientContext
from .nuclear_response import contract_z_vector, nuclear_response_source_vo
from .one_electron import build_one_electron_derivatives
from .orbitals import build_mo_partitions, target_fock_vo_blocks
from .pulay import contract_target_overlap
from .target import (
    contract_coulomb_full_derivative,
    target_energy_weighted_density,
    target_fock_kpts,
    target_xc_atom_gradient,
)
from .zvector import ZVectorResult, solve_z_vector


@dataclass(frozen=True)
class FFTDFGradientResult:
    """Host-visible gradient decomposition with GPU Z-vector diagnostics."""

    contributions: GradientContributions
    one_electron: np.ndarray
    target_coulomb_xc: np.ndarray
    target_overlap_pulay: np.ndarray
    nuclear: np.ndarray
    z_vector: ZVectorResult
    mesh: tuple[int, int, int]
    ke_cutoff_ha: float
    nkpts: int
    target_fock_workspace_reused: bool
    target_integrator: str
    target_integration_metadata: dict[str, object]
    timings_s: dict[str, float]

    @property
    def gradient(self):
        return self.contributions.total

    def metadata(self):
        return {
            "backend": "fftdf",
            "device": "gpu",
            "units": "Ha/Bohr",
            "nkpts": self.nkpts,
            "mesh": list(self.mesh),
            "ke_cutoff_ha": self.ke_cutoff_ha,
            "target_integrator": self.target_integrator,
            "target_integration": dict(self.target_integration_metadata),
            "target_numint_backend": (
                "numint"
                if self.target_integrator == "uniform"
                else "multigrid_v2"
            ),
            "target_fock_workspace_reused": (
                self.target_fock_workspace_reused
            ),
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
            "silent_cpu_fallback": False,
            "host_boundaries": [
                "Cell/grid/projector metadata setup",
                "Ewald nuclear-repulsion gradient",
                "final result reporting",
            ],
            "timings_s": dict(self.timings_s),
        }


def _contract_one_electron(context, hcore_by_atom):
    return cp.einsum(
        "axkij,kji->ax", hcore_by_atom, context.density
    ).real / context.nkpts


def fftdf_gpu_gradient(
    dc_mf,
    *,
    z_equation="adjoint",
    z_solver="auto",
    z_tolerance=1.0e-12,
    z_residual_tolerance=1.0e-8,
    z_max_cycle=80,
    z_dense_max_dimension=256,
    progress_callback=None,
):
    """Evaluate the restricted FFTDF GPU analytic nuclear gradient."""

    if getattr(dc_mf.with_df, "backend_name", None) != "fftdf":
        raise NotImplementedError("fftdf_gpu_gradient requires GPU FFTDF")
    total_start = perf_counter()
    timings = {}
    if progress_callback is not None:
        progress_callback("gradient_target_fock_rhs", state="started")
    context = RestrictedGradientContext.from_dc_method(dc_mf)
    partitions = build_mo_partitions(context)
    target_fock, target_fock_workspace_reused = target_fock_kpts(
        context, return_reuse=True
    )
    target_vo = target_fock_vo_blocks(target_fock, partitions)
    cp.cuda.Stream.null.synchronize()
    timings["target_fock_and_rhs"] = perf_counter() - total_start
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
    one_sources = build_one_electron_derivatives(context.cell, context.kpts)
    hcore_by_atom = cp.stack(
        [one_sources.hcore(atom_id) for atom_id in range(context.cell.natm)]
    )
    one_electron = _contract_one_electron(context, hcore_by_atom)
    cp.cuda.Stream.null.synchronize()
    timings["one_electron_and_pseudopotential"] = perf_counter() - start
    if progress_callback is not None:
        progress_callback(
            "gradient_one_electron",
            state="complete",
            wall_time_s=timings["one_electron_and_pseudopotential"],
        )
        progress_callback("gradient_fft_jk_derivative", state="started")

    start = perf_counter()
    source_jk = fft_jk_derivative(
        context.cell,
        context.kpts,
        context.density,
        exxdiv=context.source_mf.exxdiv,
        with_df=dc_mf.with_df,
    )
    cp.cuda.Stream.null.synchronize()
    timings["full_jk_nuclear_derivative"] = perf_counter() - start

    if progress_callback is not None:
        progress_callback(
            "gradient_fft_jk_derivative",
            state="complete",
            wall_time_s=timings["full_jk_nuclear_derivative"],
        )
        progress_callback("gradient_target_explicit", state="started")
    start = perf_counter()
    target_coulomb = (
        cp.zeros((context.cell.natm, 3), dtype=cp.float64)
        if target_owns_coulomb(context.target_evaluation.target_integrator)
        else contract_coulomb_full_derivative(context, source_jk.coulomb)
    )
    target_xc = target_xc_atom_gradient(context)
    target_coulomb_xc = target_coulomb + target_xc
    cp.cuda.Stream.null.synchronize()
    timings["target_coulomb_xc"] = perf_counter() - start

    start = perf_counter()
    weighted_density = target_energy_weighted_density(context, target_fock)
    target_overlap = contract_target_overlap(
        context, one_sources.overlap_rows, weighted_density
    )
    # Ewald geometry metadata and the final O(Natom^2) ion-ion derivative are
    # host-side in PySCF/GPU4PySCF. This boundary is explicit in metadata.
    nuclear = np.asarray(
        cpu_krhf_grad.grad_nuc(
            context.cell, range(context.cell.natm)
        ),
        dtype=float,
    )
    explicit_device = (
        one_electron + target_coulomb_xc + target_overlap + cp.asarray(nuclear)
    )
    cp.cuda.Stream.null.synchronize()
    timings["pulay_and_nuclear"] = perf_counter() - start
    if progress_callback is not None:
        progress_callback(
            "gradient_target_explicit",
            state="complete",
            wall_time_s=(
                timings["target_coulomb_xc"]
                + timings["pulay_and_nuclear"]
            ),
        )

    start = perf_counter()
    if progress_callback is not None:
        progress_callback(
            "z_nuclear_response", completed=0, total=context.cell.natm
        )
    response = cp.zeros((context.cell.natm, 3), dtype=cp.float64)
    restricted_hf_derivative = source_jk.restricted_hf
    for atom_id in range(context.cell.natm):
        for axis in range(3):
            h1_kpts = (
                hcore_by_atom[atom_id, axis]
                + restricted_hf_derivative[atom_id, :, axis]
            )
            s1_kpts = one_sources.overlap(atom_id, axis)
            source_vo = nuclear_response_source_vo(
                context, partitions, h1_kpts, s1_kpts
            )
            response[atom_id, axis] = contract_z_vector(
                z_result.blocks, source_vo, nkpts=context.nkpts
            )
        if progress_callback is not None:
            progress_callback(
                "z_nuclear_response",
                completed=atom_id + 1,
                total=context.cell.natm,
            )
    cp.cuda.Stream.null.synchronize()
    timings["nuclear_response_contraction"] = perf_counter() - start
    if progress_callback is not None:
        progress_callback(
            "z_nuclear_response_complete",
            state="complete",
            wall_time_s=timings["nuclear_response_contraction"],
        )

    start = perf_counter()
    explicit = to_host(explicit_device)
    response_host = to_host(response)
    result = FFTDFGradientResult(
        contributions=GradientContributions.assemble(explicit, response_host),
        one_electron=to_host(one_electron),
        target_coulomb_xc=to_host(target_coulomb_xc),
        target_overlap_pulay=to_host(target_overlap),
        nuclear=nuclear,
        z_vector=z_result,
        mesh=tuple(int(value) for value in np.asarray(context.cell.mesh)),
        ke_cutoff_ha=float(context.cell.ke_cutoff),
        nkpts=context.nkpts,
        target_fock_workspace_reused=target_fock_workspace_reused,
        target_integrator=context.target_evaluation.target_integrator,
        target_integration_metadata=dict(
            context.target_evaluation.ownership
        ),
        timings_s=timings,
    )
    timings["final_host_reporting"] = perf_counter() - start
    timings["gradient_total"] = perf_counter() - total_start
    if progress_callback is not None:
        progress_callback(
            "gradient_backend_complete",
            backend="fftdf",
            wall_time_s=timings["gradient_total"],
        )
    return result
