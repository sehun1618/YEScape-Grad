"""Target-DFA numerical integration policy for the GPU DC driver.

The active package uses GPU4PySCF ``NumInt``/``KNumInt`` for the target
semilocal DFA.  Source HF and target Coulomb terms remain owned by the
selected FFTDF, GDF, or RSDF provider.
"""

from __future__ import annotations

import cupy as cp


def evaluate_target_hcore(target):
    """Return target hcore from the selected FFTDF/GDF/RSDF provider."""

    return target.get_hcore()


def evaluate_target_veff(target, density, *, with_df, kpts):
    """Return target ``J + Vxc`` using ordinary GPU numerical integration."""

    del with_df, kpts
    return target.get_veff(target.cell, cp.asarray(density))


def target_numint_metadata(target) -> dict[str, object]:
    """Return a serializable target-integration ownership report."""

    return {
        "backend": "numint",
        "scope": "target_dfa",
        "device": "gpu",
        "target_hcore_owner": "selected_df_backend",
        "target_j_owner": "selected_df_backend",
        "target_xc": ["PBE", "SCAN", "R2SCAN"],
        "silent_cpu_fallback": False,
    }
