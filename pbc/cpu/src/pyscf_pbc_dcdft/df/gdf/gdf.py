"""PySCF GDF provider with an explicit auxiliary-basis contract."""

from __future__ import annotations

from pyscf.pbc import df as pbc_df

from ...lib.kpoints import normalize_kpts


class GDF(pbc_df.GDF):
    """Gaussian density-fitting backend for periodic DC-DFT.

    PySCF owns the central CDERI value construction.  This package owns the
    matching analytic three-center source, two-center metric, Cholesky,
    auxiliary-center, and GTH one-electron responses used by DC-PBE.
    """

    backend_name = "gdf"
    device = "cpu"

    def __init__(self, cell, kpts=None, *, auxbasis=None):
        super().__init__(cell, kpts=normalize_kpts(kpts))
        if auxbasis is not None:
            self.auxbasis = auxbasis

    def capability(self) -> dict[str, object]:
        """Return the Stage-4 CPU completion contract."""

        return {
            "backend": self.backend_name,
            "device": self.device,
            "auxbasis": self.auxbasis,
            "energy": True,
            "analytic_gradient": True,
            "gradient_status": "stage_6_1_mgga_gamma_kpoint_fd_gate_passed",
            "target_xc": ["PBE", "SCAN", "R2SCAN"],
            "reference": ["RHF", "KRHF"],
            "pseudopotential": ["GTH"],
            "auxiliary_response": True,
        }
