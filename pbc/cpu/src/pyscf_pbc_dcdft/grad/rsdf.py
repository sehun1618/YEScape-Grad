"""Restricted semilocal DC-DFT analytic nuclear gradient with CPU RSDF."""

from __future__ import annotations

from ..df.rsdf import build_rsdf_response_tensors
from .gdf import GDFGradientResult, _gaussian_df_cpu_gradient


def rsdf_cpu_gradient(
    dc_mf, *, z_equation: str = "adjoint"
) -> GDFGradientResult:
    """Evaluate the full restricted CPU RSDF analytic gradient.

    RSDF-specific source and metric derivatives are constructed with the
    resolved public PySCF RSDF parameters.  The canonical ``C``/``dC`` blocks
    then enter the same closed-shell J/K, target-DFA, and adjoint Z-vector
    algebra as GDF.
    """

    return _gaussian_df_cpu_gradient(
        dc_mf,
        backend_name="rsdf",
        response_builder=build_rsdf_response_tensors,
        z_equation=z_equation,
    )
