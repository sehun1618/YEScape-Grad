"""Lifecycle and scope checks shared by CPU and GPU public drivers."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


_COMPUTED_FIELDS = (
    "e_tot",
    "e_hf",
    "mo_coeff",
    "mo_occ",
    "mo_energy",
    "source_scf",
    "target_ks",
    "target_evaluation",
    "dc_energy_result",
    "dispersion_result",
)
_PROVIDER_SETTINGS = (
    "backend_name",
    "mesh",
    "auxbasis",
    "setup_mode",
    "central_setup",
)
_TARGET_SETTINGS = (
    "target_integrator",
    "target_multigrid_mesh",
    "dispersion",
)


def invalidate_computed_state(method: Any) -> None:
    """Clear all values derived from a previous energy calculation."""

    method.converged = False
    for name in _COMPUTED_FIELDS:
        if hasattr(method, name):
            setattr(method, name, None)
    if hasattr(method, "timings_s"):
        method.timings_s = {}
    method._dcdft_input_signature = None


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def validate_restricted_input(method: Any) -> None:
    """Reject inputs outside the implemented closed-shell fixed-cell scope."""

    cell = method.cell
    spin = int(getattr(cell, "spin", 0))
    if spin != 0:
        raise NotImplementedError(
            "periodic DC-DFT currently supports restricted spin=0 only"
        )
    electrons = int(cell.nelectron)
    if electrons % 2:
        raise ValueError(
            "restricted periodic DC-DFT requires an even electron count; "
            f"got {electrons}"
        )

    provider = method.with_df
    if getattr(provider, "backend_name", None) == "fftdf":
        provider_mesh = np.asarray(provider.mesh, dtype=int)
        cell_mesh = np.asarray(cell.mesh, dtype=int)
        if not np.array_equal(provider_mesh, cell_mesh):
            raise ValueError(
                "FFTDF analytic gradients require with_df.mesh to match "
                "cell.mesh; set cell.mesh before cell.build() or omit the "
                "custom fft_density_fit(mesh=...) argument"
            )


def validate_restricted_occupations(
    occupations: Any,
    *,
    nkpts: int,
    nelectron: int,
    tolerance: float = 1.0e-8,
) -> None:
    """Require integer restricted occupations and the expected electron sum."""

    values = _as_numpy(occupations)
    if values.ndim == 1:
        values = values[None]
    if len(values) != int(nkpts):
        raise ValueError(
            f"MO occupations have {len(values)} k-point blocks; "
            f"expected {nkpts}"
        )
    integer_mask = np.isclose(values, 0.0, atol=tolerance, rtol=0.0)
    integer_mask |= np.isclose(values, 2.0, atol=tolerance, rtol=0.0)
    if not np.all(integer_mask):
        raise NotImplementedError(
            "periodic DC-DFT gradients currently require occupations "
            "equal to 0 or 2; fractional/open-shell occupations are unsupported"
        )
    observed = float(np.sum(values) / int(nkpts))
    if not np.isclose(observed, nelectron, atol=tolerance, rtol=0.0):
        raise ValueError(
            "restricted occupation electron count mismatch: "
            f"observed {observed:.12g}, expected {nelectron}"
        )


def _update_array(digest, label: str, value: Any) -> None:
    array = np.ascontiguousarray(_as_numpy(value))
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def input_signature(method: Any) -> str:
    """Hash mutable inputs whose change requires rerunning ``kernel()``."""

    digest = hashlib.sha256()
    cell = method.cell
    digest.update(f"cell-id:{id(cell)}".encode("ascii"))
    _update_array(digest, "coordinates", cell.atom_coords())
    _update_array(digest, "lattice", cell.lattice_vectors())
    _update_array(digest, "cell-mesh", cell.mesh)
    _update_array(digest, "kpts", method.kpts)
    digest.update(f"spin:{int(cell.spin)}".encode("ascii"))
    digest.update(f"charge:{int(cell.charge)}".encode("ascii"))
    digest.update(f"nelectron:{int(cell.nelectron)}".encode("ascii"))
    digest.update(f"xc:{method.xc}".encode("utf-8"))
    digest.update(f"exxdiv:{method.exxdiv}".encode("utf-8"))

    provider = method.with_df
    digest.update(f"provider-id:{id(provider)}".encode("ascii"))
    digest.update(
        f"provider-class:{provider.__class__.__module__}."
        f"{provider.__class__.__name__}".encode("utf-8")
    )
    for name in _PROVIDER_SETTINGS:
        if not hasattr(provider, name):
            continue
        value = getattr(provider, name)
        if isinstance(value, np.ndarray) or hasattr(value, "get"):
            _update_array(digest, f"provider-{name}", value)
        else:
            digest.update(f"{name}:{value!r}".encode("utf-8"))
    for name in _TARGET_SETTINGS:
        value = getattr(method, name, None)
        if isinstance(value, np.ndarray) or hasattr(value, "get"):
            _update_array(digest, f"target-{name}", value)
        else:
            digest.update(f"target-{name}:{value!r}".encode("utf-8"))
    return digest.hexdigest()


def mark_current_state(method: Any) -> None:
    method._dcdft_input_signature = input_signature(method)


def require_current_state(method: Any) -> None:
    """Fail before a stale energy state can enter gradient construction."""

    if not bool(getattr(method, "converged", False)):
        raise RuntimeError("run the DC-DFT energy kernel before its gradient")
    recorded = getattr(method, "_dcdft_input_signature", None)
    if recorded is None or recorded != input_signature(method):
        raise RuntimeError(
            "DC-DFT inputs changed after kernel(); rerun kernel() before "
            "requesting the density or analytic gradient"
        )


def state_is_current(method: Any) -> bool:
    if not bool(getattr(method, "converged", False)):
        return False
    recorded = getattr(method, "_dcdft_input_signature", None)
    return recorded is not None and recorded == input_signature(method)
