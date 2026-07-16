"""Restricted semilocal DC-DFT analytic gradient with GPU RSDF."""

from __future__ import annotations

from ..df.rsdf import build_rsdf_response_tensors
from .gdf import GDFGradientResult, _gaussian_df_gpu_gradient


def rsdf_gpu_gradient(
    dc_mf,
    *,
    z_equation: str = "adjoint",
    z_solver="auto",
    z_tolerance=1.0e-12,
    z_residual_tolerance=1.0e-8,
    z_max_cycle=80,
    z_dense_max_dimension=256,
    progress_callback=None,
) -> GDFGradientResult:
    """Evaluate Gamma/k-point RSDF response with CuPy contractions."""

    return _gaussian_df_gpu_gradient(
        dc_mf,
        backend_name="rsdf",
        response_builder=build_rsdf_response_tensors,
        z_equation=z_equation,
        z_solver=z_solver,
        z_tolerance=z_tolerance,
        z_residual_tolerance=z_residual_tolerance,
        z_max_cycle=z_max_cycle,
        z_dense_max_dimension=z_dense_max_dimension,
        progress_callback=progress_callback,
    )
