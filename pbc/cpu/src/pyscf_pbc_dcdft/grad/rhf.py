"""Gamma-point DC-DFT analytic-gradient public API."""

from __future__ import annotations

import numpy as np
from pyscf import lib

from .fftdf import fftdf_cpu_gradient
from .gdf import gdf_cpu_gradient
from .rsdf import rsdf_cpu_gradient


_GRADIENT_KERNELS = {
    "fftdf": fftdf_cpu_gradient,
    "gdf": gdf_cpu_gradient,
    "rsdf": rsdf_cpu_gradient,
}


class DCGradients(lib.StreamObject):
    """Nuclear gradient of ``E_DFA[n_RHF]``.

    ``kernel`` follows PySCF convention and returns a ``(natm, 3)`` NumPy
    array in Hartree/Bohr.  The full physical decomposition and Z-vector
    diagnostics remain available as :attr:`result`.
    """

    _keys = {
        "z_equation",
        "verbose",
        "stdout",
        "max_memory",
    }

    def __init__(self, base):
        self.base = base
        self.cell = base.cell
        self.mol = base.cell
        self.verbose = base.verbose
        self.stdout = base.stdout
        self.max_memory = base.max_memory
        self.z_equation = "adjoint"
        self.de = None
        self.result = None
        self.atmlst = None

    def kernel(self, atmlst=None, **kwargs):
        """Compute the selected CPU DF backend's analytic gradient.

        Parameters
        ----------
        atmlst
            Optional atom indices.  The full gradient is assembled before
            slicing so response and translational diagnostics stay complete.
        """

        if kwargs:
            self.set(**kwargs)
        backend = getattr(self.base.with_df, "backend_name", None)
        try:
            gradient_kernel = _GRADIENT_KERNELS[backend]
        except KeyError as exc:
            raise NotImplementedError(
                f"no restricted analytic gradient for DF backend {backend!r}"
            ) from exc
        self.result = gradient_kernel(
            self.base,
            z_equation=self.z_equation,
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
        """Scanner support is deferred until backend promotion gates close."""

        raise NotImplementedError(
            "geometry scanner support follows the backend completion gates"
        )
