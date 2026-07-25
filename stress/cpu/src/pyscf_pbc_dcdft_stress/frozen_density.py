"""Partial analytic explicit stress at the frozen source-HF density."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .ewald_strain import analytic_ewald_strain
from .one_electron_strain import gamma_kinetic_strain, kpoint_one_electron_strain
from .upstream import PySCFKPointStressPrimitives, PySCFStressPrimitives


AVAILABLE_COMPONENTS = ("kinetic", "xc", "nuclear")
MISSING_COMPONENTS = (
    "gdf_pseudopotential",
    "gdf_coulomb",
    "overlap_lagrangian",
    "orbital_response",
)


@dataclass(frozen=True)
class FrozenDensityAnalyticResult:
    """Analytic S2 terms, deliberately excluding unimplemented GDF Coulomb."""

    raw_derivatives_ha: Mapping[str, np.ndarray]
    derivatives_ha: Mapping[str, np.ndarray]
    stress_ha_per_bohr3: Mapping[str, np.ndarray]
    diagnostic_derivatives_ha: Mapping[str, np.ndarray]
    volume_bohr3: float
    missing_components: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        for field_name in (
            "raw_derivatives_ha",
            "derivatives_ha",
            "stress_ha_per_bohr3",
            "diagnostic_derivatives_ha",
        ):
            values = getattr(self, field_name)
            frozen = MappingProxyType(
                {key: np.asarray(value, dtype=float) for key, value in values.items()}
            )
            object.__setattr__(self, field_name, frozen)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def complete(self):
        return not self.missing_components

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_derivatives_ha": {
                key: value.tolist() for key, value in self.raw_derivatives_ha.items()
            },
            "derivatives_ha": {
                key: value.tolist() for key, value in self.derivatives_ha.items()
            },
            "stress_ha_per_bohr3": {
                key: value.tolist() for key, value in self.stress_ha_per_bohr3.items()
            },
            "diagnostic_derivatives_ha": {
                key: value.tolist()
                for key, value in self.diagnostic_derivatives_ha.items()
            },
            "volume_bohr3": float(self.volume_bohr3),
            "missing_components": list(self.missing_components),
            "complete": bool(self.complete),
            "metadata": dict(self.metadata),
        }


def frozen_density_analytic_components(method) -> FrozenDensityAnalyticResult:
    """Return implemented Gamma/GDF explicit terms without claiming total stress.

    The returned derivative tensors are derivatives with respect to symmetric
    strain and have units of Hartree.  Division by the central cell volume
    gives tensile-positive stress.  GDF Coulomb, overlap/Lagrangian, and
    orbital-response terms remain explicit missing capabilities.
    """

    from pyscf.pbc.grad import rks as pyscf_rks_grad
    from pyscf_pbc_dcdft.dft.state import require_current_state

    require_current_state(method)
    _validate_method(method)
    cell = method.cell
    backend = getattr(method.with_df, "backend_name", None)
    density = np.asarray(method.source_scf.make_rdm1())
    # PySCF's public RKS.nuc_grad_method() intentionally rejects uniform-grid
    # RKS in 2.13.  The stress primitive itself requires the concrete periodic
    # RKS Gradients container only for its base method and grid settings.
    target_gradient = pyscf_rks_grad.Gradients(method.target_ks)
    primitives = PySCFStressPrimitives.load()

    kinetic = np.einsum(
        "xyij,ji->xy", gamma_kinetic_strain(cell).raw_derivative,
        density, optimize=True,
    ).real
    xc = primitives.xc_derivative(target_gradient, cell, density)
    xc_pp = primitives.xc_and_pseudopotential_derivative(
        target_gradient, cell, density
    )
    fft_pseudopotential = xc_pp - xc
    ewald = analytic_ewald_strain(cell)
    nuclear = ewald.raw_derivative
    raw = {
        "kinetic": kinetic,
        "xc": xc,
        "nuclear": nuclear,
    }
    derivatives = {key: _symmetric_projection(value) for key, value in raw.items()}
    volume = float(cell.vol)
    stress = {key: value / volume for key, value in derivatives.items()}
    residuals = {
        key: float(np.max(np.abs(value - value.T))) for key, value in raw.items()
    }
    return FrozenDensityAnalyticResult(
        raw_derivatives_ha=raw,
        derivatives_ha=derivatives,
        stress_ha_per_bohr3=stress,
        diagnostic_derivatives_ha={
            "fft_pseudopotential_not_gdf": _symmetric_projection(
                fft_pseudopotential
            )
        },
        volume_bohr3=volume,
        missing_components=MISSING_COMPONENTS,
        metadata={
            "stage": "S2_partial",
            "production": False,
            "density_relaxed": False,
            "df_backend": backend,
            "target_integrator": "uniform",
            "pyscf_version": primitives.version,
            "pseudopotential_diagnostic": (
                "PySCF rks_stress uses an FFT pseudopotential representation; "
                "the DC target hcore uses GDF.get_pp, so this value is not an "
                "implemented GDF stress component"
            ),
            "symmetric_projection": "(dE/de_xy + dE/de_yx)/2",
            "raw_antisymmetric_residual_max_abs_ha": residuals,
            "ewald_partition": {
                "eta": ewald.eta,
                "cutoff_bohr": ewald.cutoff,
                "mesh": list(ewald.mesh),
                "lattice_image_count": ewald.lattice_image_count,
                "fixed_during_derivative": True,
            },
        },
    )


def kpoint_frozen_density_analytic_components(method) -> FrozenDensityAnalyticResult:
    """Return sampled-k kinetic/XC/Ewald strain at fixed source density."""

    from pyscf.pbc.grad import krks as pyscf_krks_grad
    from pyscf_pbc_dcdft.dft.state import require_current_state

    require_current_state(method)
    if not bool(getattr(method, "kpoint_mode", False)):
        raise NotImplementedError("sampled-k components require DCKRKS")
    backend = getattr(getattr(method, "with_df", None), "backend_name", None)
    if backend not in {"gdf", "rsdf"}:
        raise NotImplementedError("sampled-k components support GDF/RSDF")
    if getattr(method, "target_integrator", "uniform") != "uniform":
        raise NotImplementedError("sampled-k components require uniform integration")
    if getattr(method, "dispersion", None) is not None:
        raise NotImplementedError("dispersion stress is outside S6")

    cell = method.cell
    kpts = np.asarray(method.kpts, dtype=float).reshape(-1, 3)
    density = np.asarray(method.source_scf.make_rdm1())
    kinetic_matrices = kpoint_one_electron_strain(
        cell, kpts, "int1e_kin"
    ).raw_derivative
    kinetic = np.einsum(
        "xykij,kji->xy", kinetic_matrices, density, optimize=True
    ).real / len(kpts)
    target_gradient = pyscf_krks_grad.Gradients(method.target_ks)
    primitives = PySCFKPointStressPrimitives.load()
    xc = primitives.xc_derivative(
        target_gradient, cell, density, kpts
    )
    ewald = analytic_ewald_strain(cell)
    raw = {"kinetic": kinetic, "xc": xc, "nuclear": ewald.raw_derivative}
    derivatives = {key: _symmetric_projection(value) for key, value in raw.items()}
    volume = float(cell.vol)
    return FrozenDensityAnalyticResult(
        raw_derivatives_ha=raw,
        derivatives_ha=derivatives,
        stress_ha_per_bohr3={
            key: value / volume for key, value in derivatives.items()
        },
        diagnostic_derivatives_ha={},
        volume_bohr3=volume,
        missing_components=MISSING_COMPONENTS,
        metadata={
            "stage": "S6_partial",
            "production": False,
            "density_relaxed": False,
            "df_backend": backend,
            "target_integrator": "uniform",
            "nkpts": len(kpts),
            "kpoint_weight": 1.0 / len(kpts),
            "pyscf_version": primitives.version,
            "fixed_fractional_kpoints": True,
            "ewald_partition": {
                "eta": ewald.eta,
                "cutoff_bohr": ewald.cutoff,
                "mesh": list(ewald.mesh),
            },
        },
    )


def _validate_method(method):
    if bool(getattr(method, "kpoint_mode", False)):
        raise NotImplementedError("S2 analytic components support Gamma only")
    backend = getattr(getattr(method, "with_df", None), "backend_name", None)
    if backend not in {"gdf", "rsdf"}:
        raise NotImplementedError("analytic components support GDF/RSDF only")
    if getattr(method, "target_integrator", "uniform") != "uniform":
        raise NotImplementedError("S2 analytic components support uniform target integration")
    if getattr(method, "dispersion", None) is not None:
        raise NotImplementedError("dispersion stress is outside S2")


def _symmetric_projection(value):
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"strain derivative must have shape (3, 3), got {array.shape}")
    return 0.5 * (array + array.T)
