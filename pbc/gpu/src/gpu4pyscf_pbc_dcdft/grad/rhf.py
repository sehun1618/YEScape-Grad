"""PySCF-style public API for Gamma-point GPU DC-DFT gradients."""

from __future__ import annotations

import numpy as np
from pyscf import lib

from .fftdf import fftdf_gpu_gradient
from .gdf import gdf_gpu_gradient
from .nuclear_response import DEFAULT_BATCH_MEMORY_FRACTION
from .rsdf import rsdf_gpu_gradient


class DCGradients(lib.StreamObject):
    """Analytic nuclear gradient of restricted semilocal GPU DC-DFT."""

    _keys = {
        "z_equation",
        "z_solver",
        "z_tolerance",
        "z_residual_tolerance",
        "z_max_cycle",
        "z_dense_max_dimension",
        "z_nuclear_batch_memory_fraction",
        "z_nuclear_batch_max_components",
        "gdf_bilinear_mode",
        "verbose",
        "stdout",
        "max_memory",
        "progress_callback",
    }

    def __init__(self, base):
        self.base = base
        self.cell = base.cell
        self.mol = base.cell
        self.verbose = base.verbose
        self.stdout = base.stdout
        self.max_memory = base.max_memory
        self.z_equation = "adjoint"
        self.z_solver = "auto"
        self.z_tolerance = 1.0e-12
        self.z_residual_tolerance = 1.0e-8
        self.z_max_cycle = 80
        self.z_dense_max_dimension = 256
        self.z_nuclear_batch_memory_fraction = (
            DEFAULT_BATCH_MEMORY_FRACTION
        )
        self.z_nuclear_batch_max_components = None
        self.gdf_bilinear_mode = _default_gdf_bilinear_mode(base)
        self.de = None
        self.result = None
        self.atmlst = None
        self.progress_callback = None

    def kernel(self, atmlst=None, **kwargs):
        """Return a host NumPy gradient after GPU analytic assembly."""

        if kwargs:
            self.set(**kwargs)
        backend = getattr(self.base.with_df, "backend_name", None)
        if backend == "fftdf":
            self.result = fftdf_gpu_gradient(
                self.base,
                z_equation=self.z_equation,
                z_solver=self.z_solver,
                z_tolerance=self.z_tolerance,
                z_residual_tolerance=self.z_residual_tolerance,
                z_max_cycle=self.z_max_cycle,
                z_dense_max_dimension=self.z_dense_max_dimension,
                progress_callback=self.progress_callback,
            )
        elif backend == "gdf":
            self.result = gdf_gpu_gradient(
                self.base,
                z_equation=self.z_equation,
                z_solver=self.z_solver,
                z_tolerance=self.z_tolerance,
                z_residual_tolerance=self.z_residual_tolerance,
                z_max_cycle=self.z_max_cycle,
                z_dense_max_dimension=self.z_dense_max_dimension,
                z_nuclear_batch_memory_fraction=(
                    self.z_nuclear_batch_memory_fraction
                ),
                z_nuclear_batch_max_components=(
                    self.z_nuclear_batch_max_components
                ),
                gdf_bilinear_mode=self.gdf_bilinear_mode,
                progress_callback=self.progress_callback,
            )
        elif backend == "rsdf":
            self.result = rsdf_gpu_gradient(
                self.base,
                z_equation=self.z_equation,
                z_solver=self.z_solver,
                z_tolerance=self.z_tolerance,
                z_residual_tolerance=self.z_residual_tolerance,
                z_max_cycle=self.z_max_cycle,
                z_dense_max_dimension=self.z_dense_max_dimension,
                progress_callback=self.progress_callback,
            )
        else:
            raise NotImplementedError(
                f"GPU analytic gradient does not support {backend!r}"
            )
        full_gradient = np.asarray(self.result.gradient)
        if atmlst is None:
            self.atmlst = list(range(self.cell.natm))
            self.de = full_gradient
        else:
            self.atmlst = [int(atom_id) for atom_id in atmlst]
            self.de = full_gradient[self.atmlst]
        return self.de

    grad = kernel

    def as_scanner(self):
        raise NotImplementedError("GPU scanner support is not yet implemented")


def _default_gdf_bilinear_mode(base):
    provider = getattr(base, "with_df", None)
    if getattr(provider, "backend_name", None) != "gdf":
        return "one_pass"
    return (
        "one_pass"
        if bool(getattr(provider, "uses_native_integrals", False))
        else "two_pass"
    )
