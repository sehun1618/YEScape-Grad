"""Restricted semilocal DC-DFT analytic nuclear gradient with CPU GDF."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np
from pyscf.pbc.grad import krhf as pyscf_krhf_grad

from ..df.gdf import build_gdf_response_tensors
from ..df.gdf.jk_derivative import (
    GDFJKDerivatives,
    contract_coulomb_energy_derivative,
    ewald_exchange_derivative,
    gdf_jk_derivative,
)
from .assembly import (
    GradientContributions,
    contract_one_electron,
    full_overlap_derivatives,
)
from .context import RestrictedGradientContext
from .nuclear_response import contract_z_vector, nuclear_response_source_vo
from .gdf_one_electron import build_gdf_one_electron_derivatives
from .orbitals import build_mo_partitions, target_fock_vo_blocks
from .pulay import contract_target_overlap
from .target import (
    contract_target_potential,
    target_energy_weighted_density,
    target_fock_kpts,
    target_xc_row_derivative,
)
from .zvector import ZVectorResult, solve_z_vector


@dataclass(frozen=True)
class GDFGradientResult:
    """Complete CPU Gaussian-DF gradient and auxiliary response audit."""

    backend: str
    contributions: GradientContributions
    one_electron: np.ndarray
    target_coulomb: np.ndarray
    target_xc: np.ndarray
    target_overlap_pulay: np.ndarray
    nuclear: np.ndarray
    z_vector: ZVectorResult
    cderi_value_parity_max_abs: float
    omega: float
    metric_omega: float
    source_mesh: tuple[int, int, int]
    metric_mesh: tuple[int, int, int]
    nkpts: int
    timings_s: Mapping[str, float]

    @property
    def gradient(self) -> np.ndarray:
        """Return the total energy derivative in Hartree/Bohr."""

        return self.contributions.total

    @property
    def target_coulomb_xc(self) -> np.ndarray:
        """Return the combined fixed-density target two-electron term."""

        return self.target_coulomb + self.target_xc

    def metadata(self) -> dict[str, object]:
        """Return numerical settings, timings, and response diagnostics."""

        return {
            "backend": self.backend,
            "device": "cpu",
            "units": "Ha/Bohr",
            "nkpts": self.nkpts,
            "omega": self.omega,
            "source_omega": self.omega,
            "metric_omega": self.metric_omega,
            "source_mesh": list(self.source_mesh),
            "metric_mesh": list(self.metric_mesh),
            "gradient_shape": list(self.gradient.shape),
            "cderi_value_parity_max_abs": self.cderi_value_parity_max_abs,
            "z_vector_dimension": self.z_vector.dimension,
            "z_vector_residual_max_abs": self.z_vector.residual_max_abs,
            "z_vector_condition_number": self.z_vector.condition_number,
            "z_vector_equation": self.z_vector.equation,
            "timings_s": dict(self.timings_s),
            "component_max_abs": {
                "one_electron": _max_abs(self.one_electron),
                "target_coulomb": _max_abs(self.target_coulomb),
                "target_xc": _max_abs(self.target_xc),
                "target_overlap_pulay": _max_abs(
                    self.target_overlap_pulay
                ),
                "nuclear": _max_abs(self.nuclear),
                "orbital_response": _max_abs(
                    self.contributions.response
                ),
            },
        }


def gdf_cpu_gradient(dc_mf, *, z_equation: str = "adjoint") -> GDFGradientResult:
    """Evaluate the restricted CPU GDF analytic gradient.

    Both the source-HF response and the target Hartree derivative are generated
    from analytic ``dCDERI``.  The CDERI source and metric terms are retained
    separately inside the GDF response provider and its central value is
    checked against the exact factors used by the converged PySCF calculation.
    """

    return _gaussian_df_cpu_gradient(
        dc_mf,
        backend_name="gdf",
        response_builder=build_gdf_response_tensors,
        z_equation=z_equation,
    )


def _gaussian_df_cpu_gradient(
    dc_mf,
    *,
    backend_name,
    response_builder,
    z_equation,
) -> GDFGradientResult:
    """Shared restricted gradient assembly for GDF-family providers.

    Backend-specific code owns only the central/source/metric CDERI response.
    Once ``C`` and ``dC`` have the canonical pair layout, Coulomb, exchange,
    target-DFA, one-electron, and Z-vector contractions are identical.
    """

    if getattr(dc_mf.with_df, "backend_name", None) != backend_name:
        raise NotImplementedError(
            f"{backend_name} gradient requires df.{backend_name.upper()}"
        )
    timings = {}
    total_start = perf_counter()

    start = perf_counter()
    context = RestrictedGradientContext.from_dc_method(dc_mf)
    partitions = build_mo_partitions(context.source_mf)
    target_fock = target_fock_kpts(context)
    target_vo = target_fock_vo_blocks(list(target_fock), partitions)
    z_result = solve_z_vector(
        context.source_mf,
        partitions,
        target_vo,
        equation=z_equation,
    )
    timings["z_vector"] = perf_counter() - start

    start = perf_counter()
    one_electron_sources = build_gdf_one_electron_derivatives(
        context.cell, context.kpts
    )
    hcore_by_atom = np.asarray(
        [
            one_electron_sources.hcore(atom_id)
            for atom_id in range(context.cell.natm)
        ]
    )
    one_electron = contract_one_electron(context, hcore_by_atom)
    overlap_derivative = full_overlap_derivatives(one_electron_sources)
    timings["one_electron_and_overlap"] = perf_counter() - start

    start = perf_counter()
    cderi_response = response_builder(dc_mf.with_df)
    jk_derivative = gdf_jk_derivative(
        cderi_response.cderi,
        cderi_response.dcderi,
        context.density,
    )
    jk_derivative = _apply_exchange_divergence_response(
        context,
        jk_derivative,
        overlap_derivative,
    )
    timings[f"{backend_name}_cderi_and_jk_response"] = (
        perf_counter() - start
    )

    start = perf_counter()
    target_coulomb = contract_coulomb_energy_derivative(
        context.density, jk_derivative.coulomb
    )
    target_xc = contract_target_potential(
        context, target_xc_row_derivative(context)
    )
    target_weighted_density = target_energy_weighted_density(
        context, target_fock
    )
    target_overlap = contract_target_overlap(
        context,
        one_electron_sources.overlap_rows,
        target_weighted_density,
    )
    nuclear = np.asarray(
        pyscf_krhf_grad.grad_nuc(
            context.cell, range(context.cell.natm)
        ),
        dtype=float,
    )
    explicit = (
        one_electron
        + target_coulomb
        + target_xc
        + target_overlap
        + nuclear
    )
    timings["target_explicit_assembly"] = perf_counter() - start

    start = perf_counter()
    response = _contract_orbital_response(
        context,
        partitions,
        z_result,
        hcore_by_atom,
        one_electron_sources,
        jk_derivative.restricted_hf,
    )
    timings["z_nuclear_contraction"] = perf_counter() - start
    timings["gradient_total"] = perf_counter() - total_start

    settings = cderi_response.parameters
    source_omega = float(getattr(settings, "source_omega", settings.omega))
    metric_omega = float(getattr(settings, "metric_omega", settings.omega))
    return GDFGradientResult(
        backend=backend_name,
        contributions=GradientContributions.assemble(explicit, response),
        one_electron=one_electron,
        target_coulomb=target_coulomb,
        target_xc=target_xc,
        target_overlap_pulay=target_overlap,
        nuclear=nuclear,
        z_vector=z_result,
        cderi_value_parity_max_abs=(
            cderi_response.value_parity_max_abs
        ),
        omega=source_omega,
        metric_omega=metric_omega,
        source_mesh=settings.source_mesh,
        metric_mesh=settings.metric_mesh,
        nkpts=context.nkpts,
        timings_s=MappingProxyType(timings),
    )


def _apply_exchange_divergence_response(
    context,
    derivative: GDFJKDerivatives,
    overlap_derivative: np.ndarray,
) -> GDFJKDerivatives:
    exxdiv = context.source_mf.exxdiv
    if exxdiv is None:
        return derivative
    if exxdiv != "ewald":
        raise NotImplementedError(
            f"GDF analytic response supports exxdiv='ewald' or None, got {exxdiv!r}"
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


def _contract_orbital_response(
    context,
    partitions,
    z_result,
    hcore_by_atom,
    one_electron_sources,
    source_jk,
):
    response = np.zeros((context.cell.natm, 3), dtype=float)
    for atom_id in range(context.cell.natm):
        for axis in range(3):
            h1_kpts = (
                hcore_by_atom[atom_id, axis]
                + source_jk[atom_id, :, axis]
            )
            source_vo = nuclear_response_source_vo(
                context,
                partitions,
                h1_kpts,
                one_electron_sources.overlap(atom_id, axis),
            )
            response[atom_id, axis] = contract_z_vector(
                z_result.blocks,
                source_vo,
                nkpts=context.nkpts,
            )
    return response


def _max_abs(value):
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0
