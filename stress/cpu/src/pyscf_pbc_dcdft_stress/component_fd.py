"""Component-resolved finite differences at one fixed AO density matrix.

This module is deliberately separate from :mod:`fd`.  The S1 oracle reruns
the source HF calculation after every strain, whereas this S2 oracle keeps the
central spin-summed AO density matrix fixed.  It therefore tests explicit cell
derivatives before orbital response and overlap/Lagrangian terms are added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .conventions import VOIGT_LABELS, component_index, stress_voigt_to_tensor
from .gdf_settings import GDFProviderSettings
from .strain import deform_cell


COMPONENT_NAMES = (
    "kinetic",
    "pseudopotential",
    "one_electron",
    "coulomb",
    "xc",
    "nuclear",
    "total",
)


@dataclass(frozen=True)
class FrozenDensityEnergy:
    """One target-energy decomposition evaluated at a supplied AO density."""

    kinetic: float
    pseudopotential: float
    one_electron: float
    coulomb: float
    xc: float
    nuclear: float
    total: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        expected_one_electron = self.kinetic + self.pseudopotential
        expected_total = self.one_electron + self.coulomb + self.xc + self.nuclear
        if not np.isclose(self.one_electron, expected_one_electron, atol=1e-9, rtol=0):
            raise ValueError("one-electron decomposition is inconsistent")
        if not np.isclose(self.total, expected_total, atol=1e-9, rtol=0):
            raise ValueError("frozen-density energy decomposition is inconsistent")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def components(self) -> Mapping[str, float]:
        return MappingProxyType({name: float(getattr(self, name)) for name in COMPONENT_NAMES})

    def as_dict(self) -> dict[str, object]:
        return {
            "components_ha": dict(self.components()),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FrozenDensityComponentFDResult:
    """Central-FD explicit stress for each fixed-density energy component."""

    derivatives_ha: Mapping[str, np.ndarray]
    stress_ha_per_bohr3: Mapping[str, np.ndarray]
    volume_bohr3: float
    step: float
    components: tuple[int, ...]
    complete: bool
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "derivatives_ha",
            MappingProxyType({key: np.asarray(value, dtype=float) for key, value in self.derivatives_ha.items()}),
        )
        object.__setattr__(
            self,
            "stress_ha_per_bohr3",
            MappingProxyType({key: np.asarray(value, dtype=float) for key, value in self.stress_ha_per_bohr3.items()}),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def as_dict(self) -> dict[str, object]:
        return {
            "derivatives_ha": {key: value.tolist() for key, value in self.derivatives_ha.items()},
            "stress_ha_per_bohr3": {
                key: value.tolist() for key, value in self.stress_ha_per_bohr3.items()
            },
            "volume_bohr3": float(self.volume_bohr3),
            "step": float(self.step),
            "components": [VOIGT_LABELS[index] for index in self.components],
            "complete": bool(self.complete),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GammaGDFFrozenDensityEvaluator:
    """Evaluate Gamma/GDF target components without rerunning source HF."""

    density: np.ndarray = field(repr=False, compare=False)
    provider_settings: GDFProviderSettings
    xc: str = "PBE"
    verbose: int = 0

    @classmethod
    def from_method(cls, method):
        from pyscf_pbc_dcdft.dft.state import require_current_state

        require_current_state(method)
        if bool(getattr(method, "kpoint_mode", False)):
            raise NotImplementedError("S2 frozen-density evaluator supports Gamma only")
        provider = getattr(method, "with_df", None)
        if getattr(provider, "backend_name", None) != "gdf":
            raise NotImplementedError("S2 frozen-density evaluator supports GDF only")
        if getattr(method, "target_integrator", "uniform") != "uniform":
            raise NotImplementedError("S2 frozen-density evaluator supports uniform target integration")
        if getattr(method, "dispersion", None) is not None:
            raise NotImplementedError("dispersion stress is outside S2")
        return cls(
            density=np.array(method.source_scf.make_rdm1(), copy=True),
            provider_settings=GDFProviderSettings.from_provider(provider),
            xc=str(method.xc),
            verbose=int(method.verbose),
        )

    def __call__(self, cell) -> FrozenDensityEnergy:
        from pyscf.pbc import dft as pbc_dft
        density = np.asarray(self.density)
        if density.shape != (cell.nao_nr(), cell.nao_nr()):
            raise ValueError(
                "strained cell AO dimension changed; the central fixed density is incompatible"
            )
        target = pbc_dft.RKS(cell)
        target.xc = self.xc
        target.with_df = self.provider_settings.build(cell)
        target.verbose = self.verbose
        target.build()

        kinetic_matrix = np.asarray(cell.pbc_intor("int1e_kin"))
        hcore = np.asarray(target.get_hcore(cell))
        veff = target.get_veff(cell, density)
        kinetic = _real_scalar(np.einsum("ij,ji->", kinetic_matrix, density), "kinetic")
        one_electron = _real_scalar(np.einsum("ij,ji->", hcore, density), "one-electron")
        pseudopotential = one_electron - kinetic
        coulomb = _real_scalar(getattr(veff, "ecoul", None), "Coulomb")
        xc = _real_scalar(getattr(veff, "exc", None), "XC")
        nuclear = _real_scalar(cell.energy_nuc(), "nuclear")
        total = one_electron + coulomb + xc + nuclear
        return FrozenDensityEnergy(
            kinetic=kinetic,
            pseudopotential=pseudopotential,
            one_electron=one_electron,
            coulomb=coulomb,
            xc=xc,
            nuclear=nuclear,
            total=total,
            metadata={
                "density_source": "central_source_hf_ao_density",
                "density_relaxed": False,
                "df_backend": "gdf",
                "gdf_settings": self.provider_settings.as_dict(),
                "target_integrator": "uniform",
                "mesh": [int(value) for value in cell.mesh],
            },
        )


def component_finite_difference(evaluator, cell, *, step=5e-4, components=tuple(range(6))):
    """Differentiate every energy component using the same fixed AO density."""

    step = float(step)
    if step <= 0:
        raise ValueError("finite-difference strain step must be positive")
    indices = tuple(component_index(value) for value in components)
    if len(indices) != len(set(indices)):
        raise ValueError("finite-difference components must be unique")

    center_mesh = tuple(int(value) for value in cell.mesh)
    volume = float(cell.vol)
    voigt = {name: np.full(6, np.nan) for name in COMPONENT_NAMES}
    for index in indices:
        plus_cell = deform_cell(cell, index, step, mesh=center_mesh, preserve_mesh=True)
        minus_cell = deform_cell(cell, index, -step, mesh=center_mesh, preserve_mesh=True)
        plus = evaluator(plus_cell).components()
        minus = evaluator(minus_cell).components()
        for name in COMPONENT_NAMES:
            voigt[name][index] = (plus[name] - minus[name]) / (2.0 * step)

    complete = set(indices) == set(range(6))
    derivatives = {}
    stresses = {}
    for name, values in voigt.items():
        if complete:
            derivatives[name] = stress_voigt_to_tensor(values)
            stresses[name] = derivatives[name] / volume
        else:
            derivatives[name] = values.copy()
            stresses[name] = values / volume
    return FrozenDensityComponentFDResult(
        derivatives_ha=derivatives,
        stress_ha_per_bohr3=stresses,
        volume_bohr3=volume,
        step=step,
        components=indices,
        complete=complete,
        metadata={
            "method": "central_finite_difference",
            "production": False,
            "density_relaxed": False,
            "fixed_fractional_coordinates": True,
            "fixed_integer_mesh": True,
        },
    )


def _real_scalar(value, label, *, imag_tol=1e-9):
    if value is None:
        raise RuntimeError(f"target calculation did not expose {label} energy")
    scalar = complex(np.asarray(value).item())
    if abs(scalar.imag) > imag_tol:
        raise ValueError(f"{label} energy has an imaginary residual {scalar.imag:.3e}")
    return float(scalar.real)
