"""Stable ownership key for cached periodic GDF factors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GDFBuildSignature:
    """Inputs that determine a reusable central GDF factorization."""

    geometry: str
    lattice: str
    kpoints: str
    basis: str
    pseudo: str
    mesh: tuple[int, int, int]
    omega: float
    auxbasis: str
    setup_mode: str
    linear_dep_threshold: float
    exp_to_discard: float | None
    cell_controls: str


def make_gdf_build_signature(provider) -> GDFBuildSignature:
    """Return the complete cache key for a project-local GDF provider."""

    cell = provider.cell
    mesh = getattr(provider, "mesh", None)
    if mesh is None:
        mesh = cell.mesh
    geometry = (
        tuple(cell.atom_symbol(index) for index in range(cell.natm)),
        np.asarray(cell.atom_coords(unit="Bohr"), dtype=np.float64),
    )
    controls = (
        int(cell.dimension),
        str(cell.low_dim_ft_type),
        float(cell.precision),
        float(getattr(cell, "rcut", 0.0) or 0.0),
    )
    exp_to_discard = getattr(provider, "exp_to_discard", None)
    return GDFBuildSignature(
        geometry=_digest(geometry),
        lattice=_digest(np.asarray(cell.lattice_vectors(), dtype=np.float64)),
        kpoints=_digest(np.asarray(provider.kpts, dtype=np.float64)),
        basis=_digest(cell._basis),
        pseudo=_digest(cell._pseudo),
        mesh=tuple(int(value) for value in np.asarray(mesh).reshape(3)),
        omega=float(cell.omega),
        auxbasis=_digest(provider.auxbasis),
        setup_mode=_setup_mode(provider),
        linear_dep_threshold=float(provider.linear_dep_threshold),
        exp_to_discard=(
            None if exp_to_discard is None else float(exp_to_discard)
        ),
        cell_controls=_digest(controls),
    )


def _digest(value: Any) -> str:
    payload = repr(_freeze(value)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _setup_mode(provider) -> str:
    """Include backend-specific central ownership in the cache key."""

    mode = str(provider.setup_mode)
    central = getattr(provider, "central_setup", None)
    return mode if central is None else f"{mode}:{central}"


def _freeze(value: Any):
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return (
            "ndarray",
            contiguous.dtype.str,
            contiguous.shape,
            contiguous.tobytes().hex(),
        )
    if isinstance(value, dict):
        return tuple(
            (repr(key), _freeze(item))
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)
