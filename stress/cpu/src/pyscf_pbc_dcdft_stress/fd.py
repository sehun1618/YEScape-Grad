"""Central finite-difference stress oracle for independent validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .conventions import (
    DEFAULT_CONVENTION,
    VOIGT_LABELS,
    component_index,
    stress_voigt_to_tensor,
)
from .results import EnergyEvaluation, FDPairResult, FDStressResult
from .strain import component_label, deform_cell


class EnergyEvaluator(Protocol):
    def __call__(self, cell, *, dm0=None, kpts=None) -> EnergyEvaluation: ...


@dataclass(frozen=True)
class FDStressConfig:
    step: float = 5.0e-4
    components: tuple[int | str, ...] = tuple(range(6))
    fixed_mesh: bool = True
    warm_start: bool = False

    def __post_init__(self):
        if float(self.step) <= 0:
            raise ValueError("finite-difference strain step must be positive")
        normalized = tuple(component_index(value) for value in self.components)
        if len(set(normalized)) != len(normalized):
            raise ValueError("finite-difference components must be unique")
        object.__setattr__(self, "components", normalized)


class FDStress:
    """Evaluate central-FD stress without entering the analytic code path."""

    def __init__(self, evaluator: EnergyEvaluator, config=None):
        self.evaluator = evaluator
        self.config = FDStressConfig() if config is None else config

    def kernel(self, cell) -> FDStressResult:
        if not bool(getattr(cell, "_built", False)):
            raise RuntimeError("build the PySCF cell before requesting FD stress")
        if int(getattr(cell, "dimension", 3)) != 3:
            raise NotImplementedError("FD stress currently supports 3D cells only")

        mesh = tuple(int(value) for value in np.asarray(cell.mesh).reshape(3))
        center = self.evaluator(cell, dm0=None, kpts=None)
        if not center.converged:
            raise RuntimeError("central DC-DFT energy did not converge")
        initial_density = center.density if self.config.warm_start else None
        volume = float(cell.vol)
        stress_voigt = np.full(6, np.nan)
        pairs = []

        for index in self.config.components:
            pair_mesh = mesh if self.config.fixed_mesh else None
            plus_cell = deform_cell(
                cell, index, self.config.step, mesh=pair_mesh,
                preserve_mesh=self.config.fixed_mesh,
            )
            minus_cell = deform_cell(
                cell, index, -self.config.step, mesh=pair_mesh,
                preserve_mesh=self.config.fixed_mesh,
            )
            plus = self.evaluator(plus_cell, dm0=initial_density, kpts=None)
            minus = self.evaluator(minus_cell, dm0=initial_density, kpts=None)
            if not plus.converged or not minus.converged:
                raise RuntimeError(f"FD pair {component_label(index)} did not converge")
            plus_mesh = tuple(int(value) for value in plus_cell.mesh)
            minus_mesh = tuple(int(value) for value in minus_cell.mesh)
            if self.config.fixed_mesh and (plus_mesh != mesh or minus_mesh != mesh):
                raise RuntimeError("fixed-mesh FD pair changed its integer mesh")
            derivative = (plus.energy_ha - minus.energy_ha) / (2.0 * self.config.step)
            stress_voigt[index] = derivative / volume
            pairs.append(
                FDPairResult(
                    component=VOIGT_LABELS[index], step=self.config.step,
                    plus=plus, minus=minus, derivative_ha=derivative,
                    plus_volume_bohr3=float(plus_cell.vol),
                    minus_volume_bohr3=float(minus_cell.vol),
                    plus_mesh=plus_mesh, minus_mesh=minus_mesh,
                )
            )

        complete = set(self.config.components) == set(range(6))
        stress_tensor = (
            stress_voigt_to_tensor(stress_voigt)
            if complete else np.full((3, 3), np.nan)
        )
        return FDStressResult(
            stress_tensor_ha_per_bohr3=stress_tensor,
            stress_voigt_ha_per_bohr3=stress_voigt,
            volume_bohr3=volume, center=center, pairs=tuple(pairs),
            complete=complete,
            metadata={
                "method": "central_finite_difference",
                "production": False,
                "fixed_fractional_coordinates": True,
                "fixed_integer_mesh": bool(self.config.fixed_mesh),
                "warm_start": bool(self.config.warm_start),
                "components": [VOIGT_LABELS[i] for i in self.config.components],
                "convention": DEFAULT_CONVENTION.as_dict(),
            },
        )


def step_convergence(evaluator, cell, steps, *, components=tuple(range(6))):
    """Return independent FD results for a sequence of positive strain steps."""

    results = []
    for step in steps:
        config = FDStressConfig(step=float(step), components=tuple(components))
        results.append(FDStress(evaluator, config).kernel(cell))
    return tuple(results)
