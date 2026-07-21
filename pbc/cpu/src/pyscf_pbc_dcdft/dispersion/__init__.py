"""Geometry-only dispersion corrections for periodic DC-DFT."""

from .d2c import (
    D2CCorrection,
    DispersionCorrectedGradientResult,
    apply_d2c_energy,
    apply_d2c_gradient,
    evaluate_d2c,
    get_d2c_parameters,
    normalize_dispersion,
)

__all__ = [
    "D2CCorrection",
    "DispersionCorrectedGradientResult",
    "apply_d2c_energy",
    "apply_d2c_gradient",
    "evaluate_d2c",
    "get_d2c_parameters",
    "normalize_dispersion",
]
