"""PySCF-style RSDF provider for restricted periodic DC-DFT.

PySCF owns the central range-separated CDERI construction.  The project
package owns the matching fixed-cell nuclear response in :mod:`.response`.
The provider is deliberately opt-in because RSDF has independent source and
metric range-separation parameters that must not be confused with GDF.
"""

from __future__ import annotations

import numpy as np
from pyscf.pbc.df import rsdf as pyscf_rsdf

from ...lib.kpoints import normalize_kpts


class RSDF(pyscf_rsdf.RSGDF):
    """Range-separated Gaussian density fitting for periodic DC-DFT.

    Parameters
    ----------
    cell
        Three-dimensional PySCF periodic cell.
    kpts
        Gamma or sampled k-points in reciprocal Bohr.
    auxbasis
        Optional Gaussian auxiliary basis accepted by PySCF.

    Notes
    -----
    ``omega``/``mesh_compact`` define the three-center source while
    ``omega_j2c``/``mesh_j2c`` define the two-center metric.  Keeping those
    two numerical representations distinct is required for an analytic
    derivative of the actual RSDF energy.
    """

    backend_name = "rsdf"
    device = "cpu"

    def __init__(self, cell, kpts=None, *, auxbasis=None):
        super().__init__(cell, kpts=normalize_kpts(kpts))
        if auxbasis is not None:
            self.auxbasis = auxbasis

    def capability(self) -> dict[str, object]:
        """Return the restricted CPU RSDF public contract."""

        return {
            "backend": self.backend_name,
            "device": self.device,
            "auxbasis": self.auxbasis,
            "energy": True,
            "analytic_gradient": True,
            "gradient_status": "stage_6_2_cpu_completion_gate_passed",
            "target_xc": ["PBE", "SCAN", "R2SCAN"],
            "reference": ["RHF", "KRHF"],
            "pseudopotential": ["GTH"],
            "auxiliary_response": True,
            "dimension": [3],
            "host_integral_generation": True,
        }

    def response_settings(self) -> dict[str, object]:
        """Report resolved source and metric settings after ``build``."""

        required = ("omega", "mesh_compact", "omega_j2c", "mesh_j2c")
        missing = [name for name in required if getattr(self, name, None) is None]
        if missing:
            raise RuntimeError(
                "RSDF numerical settings are unresolved; run build/kernel first: "
                + ", ".join(missing)
            )
        return {
            "source_omega": float(self.omega),
            "source_mesh": [int(value) for value in np.asarray(self.mesh_compact)],
            "metric_omega": float(self.omega_j2c),
            "metric_mesh": [int(value) for value in np.asarray(self.mesh_j2c)],
            "precision_R": float(self.precision_R),
            "precision_G": float(self.precision_G),
            "precision_j2c": float(self.precision_j2c),
            "use_bvk": bool(self.use_bvk),
        }
