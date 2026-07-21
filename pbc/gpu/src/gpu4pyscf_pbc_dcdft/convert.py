"""CPU-to-GPU conversion entry point.

Conversion creates a new GPU driver and GPU density-fitting provider.  It does
not return the CPU object with a decorative device label.  Converged orbitals
are intentionally not reused by default because :meth:`kernel` must prove that
the HF density itself can be generated through the selected GPU backend.
"""

from __future__ import annotations

import numpy as np

from . import df


def _convert_density_fitting(cpu_df):
    """Create the matching project-local GPU DF provider."""

    backend = getattr(cpu_df, "backend_name", None)
    if backend == "fftdf":
        return df.FFTDF(
            cpu_df.cell,
            kpts=np.asarray(cpu_df.kpts),
            mesh=np.asarray(cpu_df.mesh, dtype=int),
        )
    if backend == "gdf":
        return df.GDF(
            cpu_df.cell,
            kpts=np.asarray(cpu_df.kpts),
            auxbasis=cpu_df.auxbasis,
            setup_mode="native",
        )
    if backend == "rsdf":
        provider = df.RSDF(
            cpu_df.cell,
            kpts=np.asarray(cpu_df.kpts),
            auxbasis=cpu_df.auxbasis,
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
            value = getattr(cpu_df, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            setattr(provider, name, value)
        return provider
    raise NotImplementedError(
        f"CPU-to-GPU conversion does not support DF backend {backend!r}"
    )


def to_gpu(mf):
    """Create a GPU DC-DFT driver with the same physical input settings."""

    from .dft import DCKRKS, DCRKS

    kpoint_mode = bool(getattr(mf, "kpoint_mode", False))
    cls = DCKRKS if kpoint_mode else DCRKS
    gpu_mf = cls(mf.cell, kpts=np.asarray(mf.kpts), xc=mf.xc)
    gpu_mf.with_df = _convert_density_fitting(mf.with_df)
    for name in (
        "conv_tol",
        "conv_tol_grad",
        "max_cycle",
        "exxdiv",
        "verbose",
        "stdout",
        "max_memory",
        "dispersion",
    ):
        if hasattr(mf, name):
            setattr(gpu_mf, name, getattr(mf, name))
    return gpu_mf
