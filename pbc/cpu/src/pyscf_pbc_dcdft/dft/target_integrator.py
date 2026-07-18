"""Names and validation shared by target-DFA integration policies."""

from __future__ import annotations

import numpy as np


UNIFORM = "uniform"
MULTIGRID_XC = "multigrid_xc"
MULTIGRID_J_XC = "multigrid_j_xc"
SUPPORTED_TARGET_INTEGRATORS = frozenset(
    {UNIFORM, MULTIGRID_XC, MULTIGRID_J_XC}
)


def normalize_target_integrator(value) -> str:
    """Return a canonical target-integration mode or raise clearly."""

    mode = str(value).strip().lower()
    if mode not in SUPPORTED_TARGET_INTEGRATORS:
        raise ValueError(
            "target_integrator must be one of "
            f"{sorted(SUPPORTED_TARGET_INTEGRATORS)}, got {value!r}"
        )
    return mode


def normalize_target_multigrid_mesh(value):
    """Return a validated three-integer mesh or ``None``."""

    if value is None:
        return None
    mesh = np.asarray(value, dtype=int).reshape(3)
    if np.any(mesh <= 0):
        raise ValueError("target_multigrid_mesh dimensions must be positive")
    return mesh.copy()


def target_owns_coulomb(mode) -> bool:
    """Whether target J belongs to the target numerical integrator."""

    return normalize_target_integrator(mode) == MULTIGRID_J_XC
