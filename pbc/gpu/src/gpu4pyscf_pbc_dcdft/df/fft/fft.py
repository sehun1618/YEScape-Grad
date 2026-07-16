"""GPU4PySCF FFTDF provider with an explicit DC-DFT contract."""

from __future__ import annotations

import numpy as np
from gpu4pyscf.pbc.df.fft import FFTDF as GPUFFTDF
from pyscf_pbc_dcdft.lib.kpoints import normalize_kpts


class FFTDF(GPUFFTDF):
    """Plane-wave density fitting used by the GPU DC-DFT route.

    Parameters
    ----------
    cell
        Built periodic PySCF cell. Geometry and lattice metadata remain host
        objects, while AO, FFT, J/K and target-XC numerical arrays live on the
        active CuPy device.
    kpts
        Cartesian reciprocal vectors in ``1/Bohr``.
    mesh
        Optional three-integer FFT mesh. It must match the energy and all
        derivative providers; no gradient-specific mesh is generated.
    """

    backend_name = "fftdf"
    device_name = "gpu"

    def __init__(self, cell, kpts=None, *, mesh=None):
        super().__init__(cell, kpts=normalize_kpts(kpts))
        if mesh is not None:
            mesh_array = np.asarray(mesh, dtype=int).reshape(3)
            if np.any(mesh_array <= 0):
                raise ValueError("FFT mesh dimensions must be positive")
            self.mesh = mesh_array

    def capability(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "device": self.device_name,
            "mesh": [int(value) for value in np.asarray(self.mesh)],
            "energy": True,
            "analytic_gradient": True,
            "gradient_status": "stage_3_gamma_kpoint_fd_parity_gate_passed",
            "target_xc": ["PBE", "SCAN", "R2SCAN"],
            "silent_cpu_fallback": False,
        }
