"""Restricted Gamma/GDF semilocal DC-DFT analytic stress assembly.

All cell derivatives in this route are analytic: one-electron and overlap,
GTH pseudopotential, Ewald ion-ion and Madelung, GDF metric/source/CDERI/J/K,
uniform-grid PBE/r2SCAN XC, and the adjoint DC response.  The historical
``semi_analytic`` API remains as a compatibility alias only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .conventions import stress_voigt_to_tensor
from .frozen_density import (
    frozen_density_analytic_components,
    kpoint_frozen_density_analytic_components,
)
from .gdf_cderi_strain import (
    gamma_cderi_strain,
    gamma_ewald_exchange_strain,
    gamma_gdf_jk_strain,
    kpoint_cderi_strain,
    kpoint_ewald_exchange_strain,
    kpoint_gdf_jk_strain,
)
from .gdf_pp_strain import (
    gamma_gdf_pseudopotential_strain,
    kpoint_gdf_pseudopotential_strain,
)
from .gradient_adapter import (
    build_dc_response_state,
    contract_strain_response,
    resolved_gdf_parameters,
)
from .one_electron_strain import (
    gamma_kinetic_strain,
    gamma_overlap_strain,
    kpoint_one_electron_strain,
)
from .rsdf_cderi_strain import (
    gamma_rsdf_cderi_strain,
    kpoint_rsdf_cderi_strain,
)


@dataclass(frozen=True)
class GammaGaussianDFAnalyticStressResult:
    """Complete restricted Gamma Gaussian-DF semilocal stress result."""

    component_derivatives_ha: Mapping[str, np.ndarray]
    component_stress_ha_per_bohr3: Mapping[str, np.ndarray]
    stress_voigt_ha_per_bohr3: np.ndarray
    stress_tensor_ha_per_bohr3: np.ndarray
    volume_bohr3: float
    z_vector: object
    timings_s: Mapping[str, float]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("component_derivatives_ha", "component_stress_ha_per_bohr3"):
            values = getattr(self, name)
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {key: np.asarray(value, dtype=float) for key, value in values.items()}
                ),
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

    def as_dict(self) -> dict[str, object]:
        return {
            "component_derivatives_ha": {
                key: value.tolist() for key, value in self.component_derivatives_ha.items()
            },
            "component_stress_ha_per_bohr3": {
                key: value.tolist()
                for key, value in self.component_stress_ha_per_bohr3.items()
            },
            "stress_voigt_ha_per_bohr3": self.stress_voigt_ha_per_bohr3.tolist(),
            "stress_tensor_ha_per_bohr3": self.stress_tensor_ha_per_bohr3.tolist(),
            "volume_bohr3": float(self.volume_bohr3),
            "z_vector": {
                "dimension": int(self.z_vector.dimension),
                "residual_max_abs": float(self.z_vector.residual_max_abs),
                "equation": self.z_vector.equation,
            },
            "timings_s": dict(self.timings_s),
            "metadata": dict(self.metadata),
        }


def gamma_gdf_analytic_stress(
    method, *, z_equation="adjoint", shared_z_vector=None,
) -> GammaGaussianDFAnalyticStressResult:
    """Assemble full DC-PBE/r2SCAN stress without strained-cell calculations."""

    return _gamma_gaussian_df_analytic_stress(
        method, backend="gdf", z_equation=z_equation,
        shared_z_vector=shared_z_vector,
    )


def gamma_rsdf_analytic_stress(
    method, *, z_equation="adjoint", shared_z_vector=None,
) -> GammaGaussianDFAnalyticStressResult:
    """Assemble restricted Gamma/RSDF DC-PBE/r2SCAN analytic stress."""

    return _gamma_gaussian_df_analytic_stress(
        method, backend="rsdf", z_equation=z_equation,
        shared_z_vector=shared_z_vector,
    )


def kpoint_gdf_analytic_stress(
    method, *, z_equation="adjoint", shared_z_vector=None,
) -> GammaGaussianDFAnalyticStressResult:
    """Assemble restricted sampled-k GDF DC-PBE/r2SCAN analytic stress."""

    return _kpoint_gaussian_df_analytic_stress(
        method, backend="gdf", z_equation=z_equation,
        shared_z_vector=shared_z_vector,
    )


def kpoint_rsdf_analytic_stress(
    method, *, z_equation="adjoint", shared_z_vector=None,
) -> GammaGaussianDFAnalyticStressResult:
    """Assemble restricted sampled-k RSDF DC-PBE/r2SCAN analytic stress."""

    return _kpoint_gaussian_df_analytic_stress(
        method, backend="rsdf", z_equation=z_equation,
        shared_z_vector=shared_z_vector,
    )


def _gamma_gaussian_df_analytic_stress(
    method, *, backend, z_equation, shared_z_vector,
):
    xctype = _validate_semilocal_scope(method, backend=backend)

    total_start = perf_counter()
    timings = {}

    start = perf_counter()
    response_state = build_dc_response_state(
        method, z_equation=z_equation, shared_z_vector=shared_z_vector
    )
    timings["z_vector_and_target_state"] = perf_counter() - start

    start = perf_counter()
    partial = frozen_density_analytic_components(method)
    kinetic_raw = gamma_kinetic_strain(method.cell).raw_derivative
    overlap_raw = gamma_overlap_strain(method.cell).raw_derivative
    kinetic_matrices = _matrix_tensor_to_voigt(kinetic_raw)
    overlap_matrices = _matrix_tensor_to_voigt(overlap_raw)
    timings["analytic_one_electron_xc_ewald_primitives"] = perf_counter() - start

    start = perf_counter()
    pseudopotential_response = gamma_gdf_pseudopotential_strain(method.cell)
    pp_matrices = pseudopotential_response.voigt_derivative
    timings[f"analytic_{backend}_pseudopotential"] = perf_counter() - start

    start = perf_counter()
    density = np.asarray(method.source_scf.make_rdm1())
    if backend == "gdf":
        parameters = resolved_gdf_parameters(method.with_df)
        cderi_response = gamma_cderi_strain(
            method.cell, method.with_df.auxcell, parameters["omega"],
            parameters["source_mesh"], parameters["metric_mesh"],
        )
        cderi_parity = None
    else:
        cderi_response = gamma_rsdf_cderi_strain(method.with_df)
        resolved = cderi_response.parameters
        parameters = {
            "source_omega": resolved.source_omega,
            "source_mesh": list(resolved.source_mesh),
            "metric_omega": resolved.metric_omega,
            "metric_mesh": list(resolved.metric_mesh),
            "precision_R": resolved.precision_r,
            "precision_G": resolved.precision_g,
            "precision_j2c": resolved.precision_j2c,
            "use_bvk": resolved.use_bvk,
        }
        cderi_parity = cderi_response.reconstructed_value_parity_max_abs
    jk_response = gamma_gdf_jk_strain(cderi_response, density)
    j_matrices = _matrix_tensor_to_voigt(jk_response.coulomb)
    k_matrices = _matrix_tensor_to_voigt(jk_response.exchange)
    if method.exxdiv == "ewald":
        k_matrices += _matrix_tensor_to_voigt(
            gamma_ewald_exchange_strain(method.cell, density)
        )
    elif method.exxdiv is not None:
        raise NotImplementedError(
            f"Gamma {backend.upper()} analytic stress does not support "
            f"exxdiv={method.exxdiv!r}"
        )
    timings[f"analytic_{backend}_cderi_jk"] = perf_counter() - start

    start = perf_counter()
    kinetic = _tensor_to_voigt(partial.derivatives_ha["kinetic"])
    pseudopotential = np.einsum(
        "xij,ji->x", pp_matrices, density, optimize=True
    ).real
    target_coulomb = 0.5 * np.einsum(
        "xij,ji->x", j_matrices, density, optimize=True
    ).real
    target_xc = _tensor_to_voigt(partial.derivatives_ha["xc"])
    nuclear = _tensor_to_voigt(partial.derivatives_ha["nuclear"])
    one_electron = kinetic + pseudopotential

    weighted_density = response_state.target_energy_weighted_density
    target_overlap = np.empty(6)
    orbital_response = np.empty(6)
    for index in range(6):
        target_overlap[index] = (
            -np.einsum(
                "kij,kji->", overlap_matrices[index][None], weighted_density,
                optimize=True,
            ).real
            / response_state.context.nkpts
        )
        source_fock = (
            kinetic_matrices[index]
            + pp_matrices[index]
            + j_matrices[index]
            - 0.5 * k_matrices[index]
        )
        orbital_response[index] = contract_strain_response(
            response_state, source_fock[None], overlap_matrices[index][None]
        )

    explicit = (
        one_electron + target_coulomb + target_xc + target_overlap + nuclear
    )
    total = explicit + orbital_response
    components = {
        "kinetic": kinetic,
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
    timings["lagrangian_assembly"] = perf_counter() - start
    timings["total"] = perf_counter() - total_start
    volume = float(method.cell.vol)
    stresses = {key: value / volume for key, value in components.items()}
    stress_voigt = stresses["total"]
    stage_number = "S5" if backend == "rsdf" else ("S4" if xctype == "MGGA" else "S3")
    return GammaGaussianDFAnalyticStressResult(
        component_derivatives_ha=components,
        component_stress_ha_per_bohr3=stresses,
        stress_voigt_ha_per_bohr3=stress_voigt,
        stress_tensor_ha_per_bohr3=stress_voigt_to_tensor(stress_voigt),
        volume_bohr3=volume,
        z_vector=response_state.z_vector,
        timings_s=timings,
        metadata={
            "stage": (
                f"{stage_number}_gamma_{backend}_{xctype.lower()}_analytic"
            ),
            "production": True,
            "source_hf_rerun": False,
            "z_vector_reused": shared_z_vector is not None,
            "df_backend": backend,
            "target_integrator": "uniform",
            "target_xc": str(method.xc),
            "target_xctype": xctype,
            "tau_channel": xctype == "MGGA",
            "low_level_numerical_components": [],
            "analytic_components": [
                "kinetic_matrix",
                "overlap_matrix",
                "ewald_ion_ion",
                "target_xc",
                f"{backend}_metric",
                f"{backend}_three_center_source",
                f"{backend}_coulomb_matrix",
                f"{backend}_exchange_matrix",
                "gaussian_df_gth_local_part1_matrix",
                "gaussian_df_gth_local_part2_matrix",
                "gaussian_df_gth_nonlocal_matrix",
                "ewald_exchange_madelung",
                "dc_lagrangian_assembly",
            ],
            f"{backend}_parameters": parameters,
            "cderi_reconstructed_value_parity_max_abs": cderi_parity,
        },
    )


def _kpoint_gaussian_df_analytic_stress(
    method, *, backend, z_equation, shared_z_vector,
):
    xctype = _validate_semilocal_scope(method, backend=backend)
    if not bool(getattr(method, "kpoint_mode", False)):
        raise TypeError("sampled-k analytic stress requires DCKRKS")
    if getattr(method, "target_integrator", "uniform") != "uniform":
        raise NotImplementedError("sampled-k analytic stress requires uniform XC")
    if getattr(method, "dispersion", None) is not None:
        raise NotImplementedError("dispersion stress is outside S6")

    total_start = perf_counter()
    timings = {}
    cell = method.cell
    kpts = np.asarray(method.kpts, dtype=float).reshape(-1, 3)
    nkpts = len(kpts)

    start = perf_counter()
    response_state = build_dc_response_state(
        method, z_equation=z_equation, shared_z_vector=shared_z_vector
    )
    timings["z_vector_and_target_state"] = perf_counter()-start

    start = perf_counter()
    partial = kpoint_frozen_density_analytic_components(method)
    kinetic_raw = kpoint_one_electron_strain(
        cell, kpts, "int1e_kin"
    ).raw_derivative
    overlap_raw = kpoint_one_electron_strain(
        cell, kpts, "int1e_ovlp"
    ).raw_derivative
    kinetic_matrices = _matrix_tensor_to_voigt(kinetic_raw)
    overlap_matrices = _matrix_tensor_to_voigt(overlap_raw)
    timings["analytic_one_electron_xc_ewald_primitives"] = perf_counter()-start

    start = perf_counter()
    pp_response = kpoint_gdf_pseudopotential_strain(cell, kpts)
    pp_matrices = pp_response.voigt_derivative
    timings[f"analytic_{backend}_pseudopotential"] = perf_counter()-start

    start = perf_counter()
    density = np.asarray(method.source_scf.make_rdm1())
    pairs = tuple(
        (ki, kj) for ki in range(nkpts) for kj in range(ki+1)
    )
    cderi_responses = {}
    parity = {}
    if backend == "gdf":
        from pyscf_pbc_dcdft.df.gdf.storage import load_cderi_storage

        parameters = resolved_gdf_parameters(method.with_df)
        storage = load_cderi_storage(method.with_df, pairs=pairs)
        for pair in pairs:
            response = kpoint_cderi_strain(
                cell, method.with_df.auxcell, parameters["omega"],
                parameters["source_mesh"], parameters["metric_mesh"],
                kpts[pair[0]], kpts[pair[1]],
            )
            stored = np.asarray(storage.blocks[pair])
            pair_error = float(np.max(np.abs(response.value-stored)))
            parity[str(pair)] = pair_error
            if pair_error > 1e-7:
                raise RuntimeError(
                    f"reconstructed GDF pair {pair} differs from storage by "
                    f"{pair_error:.3e}"
                )
            cderi_responses[pair] = replace(response, value=stored)
    else:
        for pair in pairs:
            response = kpoint_rsdf_cderi_strain(
                method.with_df, pair[0], pair[1]
            )
            cderi_responses[pair] = response
            parity[str(pair)] = response.reconstructed_value_parity_max_abs
        resolved = next(iter(cderi_responses.values())).parameters
        parameters = {
            "source_omega": resolved.source_omega,
            "source_mesh": list(resolved.source_mesh),
            "metric_omega": resolved.metric_omega,
            "metric_mesh": list(resolved.metric_mesh),
            "precision_R": resolved.precision_r,
            "precision_G": resolved.precision_g,
            "precision_j2c": resolved.precision_j2c,
            "use_bvk": resolved.use_bvk,
        }
    jk_response = kpoint_gdf_jk_strain(cderi_responses, density)
    j_matrices = _matrix_tensor_to_voigt(jk_response.coulomb)
    k_matrices = _matrix_tensor_to_voigt(jk_response.exchange)
    if method.exxdiv == "ewald":
        k_matrices += _matrix_tensor_to_voigt(
            kpoint_ewald_exchange_strain(cell, kpts, density)
        )
    elif method.exxdiv is not None:
        raise NotImplementedError(
            f"sampled-k {backend.upper()} stress does not support "
            f"exxdiv={method.exxdiv!r}"
        )
    timings[f"analytic_{backend}_cderi_jk"] = perf_counter()-start

    start = perf_counter()
    kinetic = _tensor_to_voigt(partial.derivatives_ha["kinetic"])
    pseudopotential = np.einsum(
        "xkij,kji->x", pp_matrices, density, optimize=True
    ).real/nkpts
    target_coulomb = 0.5*np.einsum(
        "xkij,kji->x", j_matrices, density, optimize=True
    ).real/nkpts
    target_xc = _tensor_to_voigt(partial.derivatives_ha["xc"])
    nuclear = _tensor_to_voigt(partial.derivatives_ha["nuclear"])
    one_electron = kinetic+pseudopotential

    weighted_density = response_state.target_energy_weighted_density
    target_overlap = np.empty(6)
    orbital_response = np.empty(6)
    for index in range(6):
        target_overlap[index] = -np.einsum(
            "kij,kji->", overlap_matrices[index], weighted_density,
            optimize=True,
        ).real/nkpts
        source_fock = (
            kinetic_matrices[index]+pp_matrices[index]+j_matrices[index]
            - 0.5*k_matrices[index]
        )
        orbital_response[index] = contract_strain_response(
            response_state, source_fock, overlap_matrices[index]
        )
    explicit = (
        one_electron+target_coulomb+target_xc+target_overlap+nuclear
    )
    total = explicit+orbital_response
    components = {
        "kinetic": kinetic,
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
    timings["lagrangian_assembly"] = perf_counter()-start
    timings["total"] = perf_counter()-total_start
    volume = float(cell.vol)
    stresses = {key: value/volume for key, value in components.items()}
    stress_voigt = stresses["total"]
    return GammaGaussianDFAnalyticStressResult(
        component_derivatives_ha=components,
        component_stress_ha_per_bohr3=stresses,
        stress_voigt_ha_per_bohr3=stress_voigt,
        stress_tensor_ha_per_bohr3=stress_voigt_to_tensor(stress_voigt),
        volume_bohr3=volume,
        z_vector=response_state.z_vector,
        timings_s=timings,
        metadata={
            "stage": f"S6_kpoint_{backend}_{xctype.lower()}_analytic",
            "production": True,
            "source_hf_rerun": False,
            "z_vector_reused": shared_z_vector is not None,
            "df_backend": backend,
            "target_integrator": "uniform",
            "target_xc": str(method.xc),
            "target_xctype": xctype,
            "tau_channel": xctype == "MGGA",
            "nkpts": nkpts,
            "kpoint_weight": 1.0/nkpts,
            "fixed_fractional_kpoints": True,
            "low_level_numerical_components": [],
            "analytic_components": [
                "kpoint_kinetic_matrix",
                "kpoint_overlap_matrix",
                "ewald_ion_ion",
                "target_xc",
                f"{backend}_complex_metric",
                f"{backend}_complex_three_center_source",
                f"{backend}_kpoint_coulomb_matrix",
                f"{backend}_kpoint_exchange_matrix",
                "kpoint_gth_pseudopotential_matrix",
                "kmesh_ewald_exchange_madelung",
                "dc_lagrangian_assembly",
            ],
            f"{backend}_parameters": parameters,
            "cderi_pair_parity_max_abs": parity,
        },
    )


def gamma_gdf_semi_analytic_stress(
    method, *, matrix_fd_step=2.5e-4, z_equation="adjoint"
) -> GammaGaussianDFAnalyticStressResult:
    """Compatibility alias for :func:`gamma_gdf_analytic_stress`.

    ``matrix_fd_step`` is accepted so existing development scripts keep
    working, but no finite difference is performed.
    """

    del matrix_fd_step
    return gamma_gdf_analytic_stress(method, z_equation=z_equation)


GammaGDFAnalyticStressResult = GammaGaussianDFAnalyticStressResult
GammaRSDFAnalyticStressResult = GammaGaussianDFAnalyticStressResult
GammaGDFSemiAnalyticStressResult = GammaGaussianDFAnalyticStressResult
KPointGaussianDFAnalyticStressResult = GammaGaussianDFAnalyticStressResult


def _validate_semilocal_scope(method, *, backend):
    observed_backend = getattr(getattr(method, "with_df", None), "backend_name", None)
    if observed_backend != backend:
        raise TypeError(
            f"requested {backend.upper()} stress for provider {observed_backend!r}"
        )
    xc = str(getattr(method, "xc", "")).upper().replace(" ", "")
    supported = {
        "PBE": "GGA",
        "GGA_X_PBE,GGA_C_PBE": "GGA",
        "R2SCAN": "MGGA",
        "MGGA_X_R2SCAN,MGGA_C_R2SCAN": "MGGA",
    }
    if xc not in supported:
        raise NotImplementedError(
            "Gamma Gaussian-DF analytic stress is validated for DC-PBE and "
            "DC-r2SCAN only; "
            f"got xc={getattr(method, 'xc', None)!r}"
        )
    return supported[xc]


def _matrix_tensor_to_voigt(raw):
    value = np.asarray(raw)
    if value.shape[:2] != (3, 3):
        raise ValueError(f"matrix strain tensor must start with (3,3), got {value.shape}")
    return np.asarray(
        [
            value[0, 0],
            value[1, 1],
            value[2, 2],
            0.5 * (value[1, 2] + value[2, 1]),
            0.5 * (value[0, 2] + value[2, 0]),
            0.5 * (value[0, 1] + value[1, 0]),
        ]
    )


def _tensor_to_voigt(value):
    tensor = np.asarray(value, dtype=float)
    return np.asarray(
        [
            tensor[0, 0], tensor[1, 1], tensor[2, 2],
            tensor[1, 2], tensor[0, 2], tensor[0, 1],
        ]
    )
