"""Project-local GPU RSDF with CPU-identical factors and CuPy J/K."""

from __future__ import annotations

import numpy as np
from pyscf_pbc_dcdft import df as cpu_df

from ..gdf.gdf import GDF


class RSDF(GDF):
    """Range-separated GDF with explicit host integral setup.

    PySCF constructs the central RSDF two-/three-center integrals because
    GPU4PySCF has no public RSDF provider with the required derivative
    contract.  Canonical CDERI factors are uploaded once; source SCF, target
    DFA, J/K, AX/Z-vector, and final gradient contractions use CuPy kernels.
    """

    backend_name = "rsdf"

    def __init__(self, cell, kpts=None, *, auxbasis=None):
        # RSDF owns an independent CPU-identical source/metric contract.  Do
        # not inherit GDF's native central-builder default implicitly.
        super().__init__(
            cell,
            kpts=kpts,
            auxbasis=auxbasis,
            setup_mode="reference",
        )
        template = cpu_df.RSDF(
            cell, kpts=np.asarray(self.kpts), auxbasis=auxbasis
        )
        for name in (
            "use_bvk",
            "precision_R",
            "precision_G",
            "npw_max",
            "_omega_min",
            "omega",
            "ke_cutoff",
            "mesh_compact",
            "omega_j2c",
            "mesh_j2c",
            "precision_j2c",
            "j2c_eig_always",
        ):
            setattr(self, name, getattr(template, name))

    def _build_cpu_witness(self, points):
        witness = cpu_df.RSDF(
            self.cell,
            kpts=points,
            auxbasis=self.auxbasis,
        )
        for name in (
            "use_bvk",
            "precision_R",
            "precision_G",
            "npw_max",
            "_omega_min",
            "omega",
            "ke_cutoff",
            "mesh_compact",
            "omega_j2c",
            "mesh_j2c",
            "precision_j2c",
            "j2c_eig_always",
        ):
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            setattr(witness, name, value)
        witness.linear_dep_threshold = float(self.linear_dep_threshold)
        witness.build(j_only=False)
        self._adopt_resolved_settings(witness)
        return witness

    def _adopt_resolved_settings(self, witness):
        for name in (
            "omega",
            "ke_cutoff",
            "mesh_compact",
            "omega_j2c",
            "mesh_j2c",
        ):
            value = getattr(witness, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            setattr(self, name, value)

    def capability(self) -> dict[str, object]:
        """Return the current GPU RSDF ownership contract."""

        return {
            "backend": self.backend_name,
            "device": self.device_name,
            "auxbasis": self.auxbasis,
            "energy": True,
            "analytic_gradient": True,
            "gradient_status": "stage_6_2_gpu_completion_gate_passed",
            "target_xc": ["PBE", "SCAN", "R2SCAN"],
            "central_cderi": "PySCF RSDF host setup, canonical CuPy storage",
            "jk_contraction": "project-local CuPy",
            "silent_cpu_fallback": False,
            "host_boundaries": [
                "RSDF central and derivative periodic integral generation",
                "GTH hcore and H1 integral generation",
                "PySCF Cell/AuxCell metadata",
            ],
        }

    def to_cpu(self):
        """Create a matching project-local CPU RSDF provider."""

        provider = cpu_df.RSDF(
            self.cell,
            kpts=np.asarray(self.kpts),
            auxbasis=self.auxbasis,
        )
        for name in (
            "use_bvk",
            "precision_R",
            "precision_G",
            "npw_max",
            "_omega_min",
            "omega",
            "mesh_compact",
            "omega_j2c",
            "mesh_j2c",
            "precision_j2c",
            "j2c_eig_always",
        ):
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            setattr(provider, name, value)
        return provider
