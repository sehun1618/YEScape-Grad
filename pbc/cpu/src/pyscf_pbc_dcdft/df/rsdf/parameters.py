"""Resolved numerical contract of one built PySCF RSDF provider."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf.pbc.tools import k2gamma


@dataclass(frozen=True)
class RSDFParameters:
    """Numerical parameters defining the source and metric factorization.

    ``source_omega`` and ``source_mesh`` belong to the three-center source
    ``B``.  ``metric_omega`` and ``metric_mesh`` belong to the two-center
    metric ``M``.  PySCF RSDF intentionally permits these values to differ.
    """

    source_omega: float
    source_mesh: tuple[int, int, int]
    metric_omega: float
    metric_mesh: tuple[int, int, int]
    precision_r: float
    precision_g: float
    precision_j2c: float
    use_bvk: bool
    bvk_kmesh: tuple[int, int, int] | None
    linear_dep_threshold: float

    @property
    def omega(self) -> float:
        """Compatibility alias for the three-center range separator."""

        return self.source_omega


def parameters_from_provider(provider) -> RSDFParameters:
    """Return immutable settings from a built project RSDF provider.

    Raises
    ------
    TypeError
        If ``provider`` is not the explicit project RSDF backend.
    RuntimeError
        If central CDERI construction has not resolved the RSDF meshes.
    """

    if getattr(provider, "backend_name", None) != "rsdf":
        raise TypeError("parameters_from_provider requires df.RSDF")
    required = ("omega", "mesh_compact", "omega_j2c", "mesh_j2c", "auxcell")
    missing = [name for name in required if getattr(provider, name, None) is None]
    if missing:
        raise RuntimeError(
            "build the RSDF provider before requesting response parameters: "
            + ", ".join(missing)
        )
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    use_bvk = bool(provider.use_bvk and provider.kpts_band is None)
    bvk_kmesh = None
    if use_bvk:
        bvk_kmesh = tuple(
            int(value) for value in k2gamma.kpts_to_kmesh(provider.cell, kpts)
        )
    return RSDFParameters(
        source_omega=float(provider.omega),
        source_mesh=_mesh(provider.mesh_compact, "mesh_compact"),
        metric_omega=float(provider.omega_j2c),
        metric_mesh=_mesh(provider.mesh_j2c, "mesh_j2c"),
        precision_r=float(provider.precision_R),
        precision_g=float(provider.precision_G),
        precision_j2c=float(provider.precision_j2c),
        use_bvk=use_bvk,
        bvk_kmesh=bvk_kmesh,
        linear_dep_threshold=float(provider.linear_dep_threshold),
    )


def _mesh(value, label):
    mesh = np.asarray(value, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError(f"{label} must contain three positive integers: {mesh}")
    return tuple(int(item) for item in mesh)
