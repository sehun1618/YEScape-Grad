"""Canonical k-point representations used throughout the package."""

from __future__ import annotations

import numpy as np


def gamma_point() -> np.ndarray:
    """Return the Gamma point as a fresh ``(1, 3)`` array in reciprocal Bohr."""

    return np.zeros((1, 3), dtype=float)


def normalize_kpts(kpts) -> np.ndarray:
    """Return k-points as a contiguous real ``(nkpts, 3)`` array.

    PySCF accepts both a single three-vector and a two-dimensional k-point
    array.  Normalizing at the package boundary prevents shape-dependent
    branches from spreading into energy and response contractions.
    """

    if kpts is None:
        return gamma_point()
    array = np.asarray(kpts, dtype=float)
    if array.size == 0 or array.size % 3:
        raise ValueError("kpts must contain one or more three-component vectors")
    return np.ascontiguousarray(array.reshape(-1, 3))


def is_gamma(kpts, *, atol: float = 1.0e-12) -> bool:
    """Return whether ``kpts`` is exactly one Gamma point within ``atol``."""

    array = normalize_kpts(kpts)
    return bool(array.shape == (1, 3) and np.max(np.abs(array)) <= atol)
