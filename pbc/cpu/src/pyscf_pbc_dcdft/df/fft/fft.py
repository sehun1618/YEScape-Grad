"""PySCF FFTDF provider with explicit DC-DFT metadata."""

from __future__ import annotations

import numpy as np
from pyscf.pbc import df as pbc_df

from ...lib.kpoints import normalize_kpts


class FFTDF(pbc_df.FFTDF):
    """FFTDF backend used by source HF and target DFA calculations.

    Parameters
    ----------
    cell
        Built :class:`pyscf.pbc.gto.Cell`.
    kpts
        Sampling k-points in reciprocal Bohr. ``None`` means Gamma only.
    mesh
        Optional explicit FFT mesh.  When omitted, the already-built
        ``cell.mesh`` is used.  This keeps PySCF's cutoff-to-mesh conversion as
        the single source of truth.
    """

    backend_name = "fftdf"
    device = "cpu"

    def __init__(self, cell, kpts=None, *, mesh=None):
        super().__init__(cell, kpts=normalize_kpts(kpts))
        if mesh is not None:
            mesh_array = np.asarray(mesh, dtype=int).reshape(3)
            if np.any(mesh_array <= 0):
                raise ValueError("FFT mesh dimensions must be positive")
            self.mesh = mesh_array

    def capability(self) -> dict[str, object]:
        """Return stable, machine-readable backend metadata."""

        return {
            "backend": self.backend_name,
            "device": self.device,
            "mesh": [int(value) for value in np.asarray(self.mesh)],
            "energy": True,
            "analytic_gradient": True,
            "gradient_status": "stage_2_cpu_gamma_kpoint_fd_gate_passed",
            "target_xc": ["PBE", "SCAN", "R2SCAN"],
            "reference": ["RHF", "KRHF"],
        }
