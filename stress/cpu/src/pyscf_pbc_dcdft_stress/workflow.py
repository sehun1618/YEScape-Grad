"""Same-state CPU energy, force, and analytic-stress workflow."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from time import perf_counter
from types import MappingProxyType

import numpy as np

from .mlip import mlip_record_from_results, stress_metadata_for_mlip
from .semi_analytic import (
    gamma_gdf_analytic_stress,
    gamma_rsdf_analytic_stress,
    kpoint_gdf_analytic_stress,
    kpoint_rsdf_analytic_stress,
)


@dataclass(frozen=True)
class EnergyForceStressResult:
    """One converged electronic state and all MLIP observables derived from it."""

    record: object
    nuclear_gradient_ha_per_bohr: np.ndarray
    stress_result: object
    timings_s: dict[str, float]

    def __post_init__(self):
        object.__setattr__(
            self, "nuclear_gradient_ha_per_bohr",
            np.asarray(self.nuclear_gradient_ha_per_bohr, dtype=float),
        )
        object.__setattr__(self, "timings_s", MappingProxyType(dict(self.timings_s)))

    def as_dict(self):
        return {
            "record": self.record.as_dict(),
            "nuclear_gradient_ha_per_bohr": (
                self.nuclear_gradient_ha_per_bohr.tolist()
            ),
            "stress": self.stress_result.as_dict(),
            "timings_s": dict(self.timings_s),
        }


def cpu_energy_force_stress(method):
    """Derive CPU energy, gradient, and stress without another source-HF SCF."""

    if not bool(getattr(method, "converged", False)):
        raise RuntimeError("run method.kernel() to convergence before this workflow")
    stress_driver = _cpu_stress_driver(method)
    fingerprint = calculation_state_fingerprint(method)
    started = perf_counter()
    gradient_object = method.nuc_grad_method()
    gradient = np.asarray(gradient_object.kernel(), dtype=float)
    gradient_seconds = perf_counter()-started
    started = perf_counter()
    shared_z_vector = getattr(gradient_object.result, "z_vector", None)
    if shared_z_vector is None:
        raise RuntimeError("gradient result did not expose its converged Z-vector")
    stress = stress_driver(method, shared_z_vector=shared_z_vector)
    stress_seconds = perf_counter()-started
    record = mlip_record_from_results(
        method.cell, method.e_tot, gradient, stress,
        metadata={
            "state_fingerprint": fingerprint,
            "device": "cpu",
            "df_backend": getattr(method.with_df, "backend_name", None),
            "xc": method.xc,
            **stress_metadata_for_mlip(stress),
        },
    )
    return EnergyForceStressResult(
        record=record, nuclear_gradient_ha_per_bohr=gradient,
        stress_result=stress,
        timings_s={
            "gradient": gradient_seconds,
            "stress": stress_seconds,
            "gradient_and_stress": gradient_seconds+stress_seconds,
        },
    )


def calculation_state_fingerprint(method, *, density=None):
    """Hash the cell, k-points, density, backend, and target functional."""

    if density is None:
        density = method.make_rdm1()
    digest = hashlib.sha256()
    for value in (
        np.asarray(method.cell.lattice_vectors(), dtype=np.float64),
        np.asarray(method.cell.atom_coords(), dtype=np.float64),
        np.asarray(getattr(method, "kpts", np.zeros((1, 3))), dtype=np.float64),
        np.asarray(density),
    ):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    digest.update(str(method.xc).encode("utf-8"))
    digest.update(str(getattr(method.with_df, "backend_name", None)).encode("utf-8"))
    return digest.hexdigest()


def _cpu_stress_driver(method):
    backend = getattr(getattr(method, "with_df", None), "backend_name", None)
    if backend not in {"gdf", "rsdf"}:
        raise NotImplementedError("CPU stress workflow supports GDF/RSDF only")
    kpts = np.asarray(
        getattr(method, "kpts", np.zeros((1, 3)))
    ).reshape(-1, 3)
    sampled = len(kpts) > 1 or bool(getattr(method, "kpoint_mode", False))
    if backend == "gdf":
        return kpoint_gdf_analytic_stress if sampled else gamma_gdf_analytic_stress
    return kpoint_rsdf_analytic_stress if sampled else gamma_rsdf_analytic_stress
