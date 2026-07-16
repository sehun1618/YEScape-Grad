"""Small, explicit CuPy host/device conversion helpers."""

from __future__ import annotations

import cupy as cp
import numpy as np


def to_device(value, *, dtype=None):
    """Return a contiguous CuPy array without an intermediate NumPy copy."""

    return cp.asarray(value, dtype=dtype, order="C")


def to_host(value) -> np.ndarray:
    """Copy one numerical result to host memory for public reporting."""

    return cp.asnumpy(value)


def real_scalar(value, *, label: str, imag_tol: float = 1.0e-9) -> float:
    """Transfer and validate a scalar that should be real by construction."""

    scalar = complex(cp.asnumpy(cp.asarray(value)).item())
    if abs(scalar.imag) > imag_tol:
        raise ValueError(f"{label} has an imaginary residual {scalar.imag:.3e}")
    return float(scalar.real)
