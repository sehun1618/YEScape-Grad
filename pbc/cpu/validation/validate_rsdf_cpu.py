#!/usr/bin/env python
"""Validate the restricted CPU RSDF analytic gradient against energy FD.

``--full`` covers H2 and distorted H2O, Gamma and a ``(2,1,1)`` k-mesh,
plus GTH-SZV/GTH-DZVP primary-basis transfer.  Every displaced point repeats
the RHF/KRHF source calculation and evaluates the complete target PBE energy.
Coordinates and lattice vectors are Bohr; energies and gradients are Hartree
and Hartree/Bohr.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import df, dft


@dataclass(frozen=True)
class ValidationCase:
    """One reproducible RSDF analytic-vs-FD calculation."""

    name: str
    atoms: tuple[str, ...]
    coordinates_bohr: tuple[tuple[float, float, float], ...]
    cell_length_bohr: float
    basis: str
    auxbasis: str
    cutoff_ha: float
    kmesh: tuple[int, int, int]
    fd_step_bohr: float


WATER = (
    (0.1, -0.2, 0.3),
    (1.55, 0.15, 0.05),
    (-0.35, 1.25, -0.25),
)
CASES = (
    ValidationCase(
        "h2_gamma_szv",
        ("H", "H"),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.4)),
        6.0,
        "gth-szv",
        "gth-szv",
        20.0,
        (1, 1, 1),
        5.0e-4,
    ),
    ValidationCase(
        "h2_k211_dzvp",
        ("H", "H"),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.4)),
        6.0,
        "gth-dzvp",
        "gth-szv",
        15.0,
        (2, 1, 1),
        5.0e-4,
    ),
    ValidationCase(
        "distorted_h2o_gamma_szv",
        ("O", "H", "H"),
        WATER,
        7.0,
        "gth-szv",
        "gth-szv",
        25.0,
        (1, 1, 1),
        2.0e-4,
    ),
    ValidationCase(
        "distorted_h2o_k211_szv",
        ("O", "H", "H"),
        WATER,
        7.0,
        "gth-szv",
        "gth-szv",
        15.0,
        (2, 1, 1),
        2.0e-4,
    ),
)


def build_cell(case, coordinates):
    """Build the finite-mesh cell used by both analytic and FD points."""

    return gto.Cell(
        atom=list(zip(case.atoms, coordinates, strict=True)),
        a=np.eye(3) * case.cell_length_bohr,
        unit="Bohr",
        basis=case.basis,
        pseudo="gth-hf",
        ke_cutoff=case.cutoff_ha,
        verbose=0,
    ).build()


def build_method(case, coordinates):
    """Build one Gamma or sampled-k-point DC-PBE RSDF method."""

    cell = build_cell(case, coordinates)
    kpts = cell.make_kpts(case.kmesh, wrap_around=True)
    if case.kmesh == (1, 1, 1):
        method = dft.DCRKS(cell, xc="PBE")
    else:
        method = dft.DCKRKS(cell, kpts=kpts, xc="PBE")
    method.with_df = df.RSDF(
        cell, kpts=kpts, auxbasis=case.auxbasis
    )
    method.conv_tol = 1.0e-11
    method.max_cycle = 100
    return method


def energy(case, coordinates):
    """Return ``E_PBE[n_HF]`` for one displaced geometry."""

    return float(build_method(case, coordinates).kernel())


def central_fd_gradient(case, coordinates):
    """Differentiate all atomic Cartesian coordinates by central FD."""

    gradient = np.zeros_like(coordinates)
    for atom_id in range(len(coordinates)):
        for axis in range(3):
            plus = np.array(coordinates, copy=True)
            minus = np.array(coordinates, copy=True)
            plus[atom_id, axis] += case.fd_step_bohr
            minus[atom_id, axis] -= case.fd_step_bohr
            gradient[atom_id, axis] = (
                energy(case, plus) - energy(case, minus)
            ) / (2.0 * case.fd_step_bohr)
    return gradient


def run_case(case):
    """Run one case and return a JSON-safe evidence record."""

    coordinates = np.asarray(case.coordinates_bohr, dtype=float)
    method = build_method(case, coordinates)
    start = perf_counter()
    dc_energy = method.kernel()
    energy_seconds = perf_counter() - start

    gradient_object = method.nuc_grad_method()
    start = perf_counter()
    analytic = gradient_object.kernel()
    analytic_seconds = perf_counter() - start

    start = perf_counter()
    finite_difference = central_fd_gradient(case, coordinates)
    fd_seconds = perf_counter() - start
    difference = analytic - finite_difference
    maximum_error = float(np.max(np.abs(difference)))
    return {
        "case": case.name,
        "status": "pass" if maximum_error <= 1.0e-5 else "fail",
        "threshold_ha_per_bohr": 1.0e-5,
        "energy_ha": float(dc_energy),
        "analytic_gradient_ha_per_bohr": analytic.tolist(),
        "fd_gradient_ha_per_bohr": finite_difference.tolist(),
        "analytic_minus_fd_ha_per_bohr": difference.tolist(),
        "max_abs_error_ha_per_bohr": maximum_error,
        "net_analytic_gradient_ha_per_bohr": analytic.sum(axis=0).tolist(),
        "net_analytic_gradient_norm_ha_per_bohr": float(
            np.linalg.norm(analytic.sum(axis=0))
        ),
        "input": {
            "atoms": list(case.atoms),
            "coordinates_bohr": coordinates.tolist(),
            "cell_length_bohr": case.cell_length_bohr,
            "basis": case.basis,
            "auxbasis": case.auxbasis,
            "pseudo": "gth-hf",
            "cutoff_ha": case.cutoff_ha,
            "mesh": [int(value) for value in method.cell.mesh],
            "kmesh": list(case.kmesh),
            "fd_step_bohr": case.fd_step_bohr,
        },
        "timing_seconds": {
            "dc_energy": energy_seconds,
            "analytic_gradient": analytic_seconds,
            "full_fd_gradient": fd_seconds,
        },
        "gradient_metadata": gradient_object.result.metadata(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    records = [run_case(case) for case in (CASES if args.full else CASES[:1])]
    payload = {
        "schema": "pyscf_pbc_dcdft_rsdf_cpu_validation_v1",
        "overall_status": (
            "pass"
            if all(record["status"] == "pass" for record in records)
            else "fail"
        ),
        "requested_cpu_threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "notes": {
            "analytic_fd_gate": (
                "Analytic and FD points use the same RSDF source/metric and "
                "finite target-XC mesh."
            ),
            "source_metric_contract": (
                "Three-center omega/mesh_compact and two-center "
                "omega_j2c/mesh_j2c are recorded independently."
            ),
        },
        "cases": records,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
