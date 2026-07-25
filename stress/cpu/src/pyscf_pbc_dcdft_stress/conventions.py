"""Tensor, Voigt, pressure, virial, and unit conventions for PBC stress."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
HARTREE_PER_BOHR3_TO_GPA = 29421.0156965221


@dataclass(frozen=True)
class StressConvention:
    """Machine-readable convention shared by FD and future analytic routes."""

    stress_sign: str = "tensile_positive"
    pressure_definition: str = "-trace(stress)/3"
    virial_definition: str = "-volume*stress"
    voigt_order: tuple[str, ...] = VOIGT_LABELS
    shear_coordinate: str = "engineering_shear_gamma=2*epsilon_offdiag"
    internal_unit: str = "Eh/Bohr^3"

    def as_dict(self) -> dict[str, object]:
        return {
            "stress_sign": self.stress_sign,
            "pressure_definition": self.pressure_definition,
            "virial_definition": self.virial_definition,
            "voigt_order": list(self.voigt_order),
            "shear_coordinate": self.shear_coordinate,
            "internal_unit": self.internal_unit,
        }


DEFAULT_CONVENTION = StressConvention()


def component_index(component: int | str) -> int:
    """Return the canonical Voigt index for an integer or label."""

    if isinstance(component, str):
        label = component.lower()
        if label not in VOIGT_LABELS:
            raise ValueError(f"unknown stress component {component!r}")
        return VOIGT_LABELS.index(label)
    index = int(component)
    if index < 0 or index >= len(VOIGT_LABELS):
        raise ValueError(f"stress component index must be in [0, 5], got {index}")
    return index


def strain_basis(component: int | str) -> np.ndarray:
    """Return ``d epsilon / d q`` for one engineering-Voigt coordinate."""

    index = component_index(component)
    basis = np.zeros((3, 3), dtype=float)
    if index < 3:
        basis[index, index] = 1.0
    else:
        row, col = ((1, 2), (0, 2), (0, 1))[index - 3]
        basis[row, col] = basis[col, row] = 0.5
    return basis


def voigt_strain_to_tensor(values) -> np.ndarray:
    """Convert engineering-Voigt strain to a symmetric strain tensor."""

    array = np.asarray(values, dtype=float)
    if array.shape != (6,):
        raise ValueError(f"Voigt strain must have shape (6,), got {array.shape}")
    return sum((array[i] * strain_basis(i) for i in range(6)), np.zeros((3, 3)))


def tensor_strain_to_voigt(tensor, *, tolerance=1.0e-12) -> np.ndarray:
    """Convert a symmetric strain tensor to engineering-Voigt coordinates."""

    value = _symmetric_tensor(tensor, tolerance=tolerance, label="strain")
    return np.asarray(
        [value[0, 0], value[1, 1], value[2, 2],
         2.0 * value[1, 2], 2.0 * value[0, 2], 2.0 * value[0, 1]]
    )


def stress_voigt_to_tensor(values) -> np.ndarray:
    """Convert physical Voigt stress values to a symmetric tensor."""

    value = np.asarray(values, dtype=float)
    if value.shape != (6,):
        raise ValueError(f"Voigt stress must have shape (6,), got {value.shape}")
    xx, yy, zz, yz, xz, xy = value
    return np.asarray(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))


def stress_tensor_to_voigt(tensor, *, tolerance=1.0e-12) -> np.ndarray:
    """Convert a symmetric physical stress tensor to canonical Voigt order."""

    value = _symmetric_tensor(tensor, tolerance=tolerance, label="stress")
    return np.asarray(
        [value[0, 0], value[1, 1], value[2, 2],
         value[1, 2], value[0, 2], value[0, 1]]
    )


def pressure_from_stress(tensor) -> float:
    """Return hydrostatic pressure under the tensile-positive convention."""

    return -float(np.trace(_symmetric_tensor(tensor, label="stress"))) / 3.0


def virial_from_stress(tensor, volume: float) -> np.ndarray:
    """Return the MLIP virial ``-Omega * sigma`` in Hartree."""

    if float(volume) <= 0:
        raise ValueError("cell volume must be positive")
    return -float(volume) * _symmetric_tensor(tensor, label="stress")


def stress_to_gpa(tensor) -> np.ndarray:
    return np.asarray(tensor, dtype=float) * HARTREE_PER_BOHR3_TO_GPA


def _symmetric_tensor(tensor, *, tolerance=1.0e-12, label: str) -> np.ndarray:
    value = np.asarray(tensor, dtype=float)
    if value.shape != (3, 3):
        raise ValueError(f"{label} tensor must have shape (3, 3), got {value.shape}")
    residual = float(np.max(np.abs(value - value.T)))
    if residual > tolerance:
        raise ValueError(f"{label} tensor is not symmetric; residual={residual:.3e}")
    return 0.5 * (value + value.T)
