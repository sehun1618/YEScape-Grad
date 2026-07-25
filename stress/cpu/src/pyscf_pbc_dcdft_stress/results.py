"""Structured finite-difference stress results and JSON-safe metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .conventions import (
    DEFAULT_CONVENTION,
    pressure_from_stress,
    stress_to_gpa,
    virial_from_stress,
)


@dataclass(frozen=True)
class EnergyEvaluation:
    energy_ha: float
    wall_seconds: float
    converged: bool = True
    density: Any = field(default=None, repr=False, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "energy_ha": float(self.energy_ha),
            "wall_seconds": float(self.wall_seconds),
            "converged": bool(self.converged),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class FDPairResult:
    component: str
    step: float
    plus: EnergyEvaluation
    minus: EnergyEvaluation
    derivative_ha: float
    plus_volume_bohr3: float
    minus_volume_bohr3: float
    plus_mesh: tuple[int, int, int]
    minus_mesh: tuple[int, int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "step": float(self.step),
            "plus": self.plus.as_dict(),
            "minus": self.minus.as_dict(),
            "derivative_ha": float(self.derivative_ha),
            "plus_volume_bohr3": float(self.plus_volume_bohr3),
            "minus_volume_bohr3": float(self.minus_volume_bohr3),
            "plus_mesh": list(self.plus_mesh),
            "minus_mesh": list(self.minus_mesh),
        }


@dataclass(frozen=True)
class FDStressResult:
    stress_tensor_ha_per_bohr3: np.ndarray
    stress_voigt_ha_per_bohr3: np.ndarray
    volume_bohr3: float
    center: EnergyEvaluation
    pairs: tuple[FDPairResult, ...]
    complete: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pressure_ha_per_bohr3(self) -> float:
        if not self.complete:
            return float("nan")
        return pressure_from_stress(self.stress_tensor_ha_per_bohr3)

    @property
    def stress_tensor_gpa(self) -> np.ndarray:
        return stress_to_gpa(self.stress_tensor_ha_per_bohr3)

    @property
    def virial_ha(self) -> np.ndarray:
        if not self.complete:
            return np.full((3, 3), np.nan)
        return virial_from_stress(self.stress_tensor_ha_per_bohr3, self.volume_bohr3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stress_tensor_ha_per_bohr3": self.stress_tensor_ha_per_bohr3.tolist(),
            "stress_voigt_ha_per_bohr3": self.stress_voigt_ha_per_bohr3.tolist(),
            "stress_tensor_gpa": self.stress_tensor_gpa.tolist(),
            "pressure_ha_per_bohr3": float(self.pressure_ha_per_bohr3),
            "virial_ha": self.virial_ha.tolist(),
            "volume_bohr3": float(self.volume_bohr3),
            "center": self.center.as_dict(),
            "pairs": [pair.as_dict() for pair in self.pairs],
            "complete": bool(self.complete),
            "convention": DEFAULT_CONVENTION.as_dict(),
            "metadata": _json_safe(self.metadata),
        }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
