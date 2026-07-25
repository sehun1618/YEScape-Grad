"""Same-state GPU energy, force, and analytic-stress workflow."""

from __future__ import annotations

from time import perf_counter

import cupy as cp

from pyscf_pbc_dcdft_stress import (
    EnergyForceStressResult,
    calculation_state_fingerprint,
    mlip_record_from_results,
    stress_metadata_for_mlip,
)

from .analytic import gpu_analytic_stress


def gpu_energy_force_stress(method, **stress_options):
    """Derive GPU energy, gradient, and stress without another source-HF SCF."""

    if not bool(getattr(method, "converged", False)):
        raise RuntimeError("run method.kernel() to convergence before this workflow")
    density = cp.asnumpy(cp.asarray(method.make_rdm1()))
    fingerprint = calculation_state_fingerprint(method, density=density)
    started = perf_counter()
    gradient_object = method.nuc_grad_method()
    gradient = cp.asnumpy(cp.asarray(gradient_object.kernel()))
    cp.cuda.Stream.null.synchronize()
    gradient_seconds = perf_counter()-started
    started = perf_counter()
    shared_z_vector = getattr(gradient_object.result, "z_vector", None)
    if shared_z_vector is None:
        raise RuntimeError("gradient result did not expose its converged Z-vector")
    stress = gpu_analytic_stress(
        method, shared_z_vector=shared_z_vector, **stress_options
    )
    cp.cuda.Stream.null.synchronize()
    stress_seconds = perf_counter()-started
    record = mlip_record_from_results(
        method.cell, method.e_tot, gradient, stress,
        metadata={
            "state_fingerprint": fingerprint,
            "device": "gpu",
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
