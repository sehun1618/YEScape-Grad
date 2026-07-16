#!/usr/bin/env python
"""Reproduce the Stage-2 CPU FFTDF analytic-vs-FD completion matrix.

The default ``quick`` case validates H2 at Gamma.  ``--full`` additionally
checks GTH-DZVP basis transfer and all nine Cartesian components of a distorted
H2O cell at Gamma and a ``(2,1,1)`` k-mesh.  Coordinates and lattice vectors
are in Bohr; energy and gradient units are Hartree and Hartree/Bohr.

Examples
--------
Run the fast gate and print JSON::

    python validation/validate_fftdf_cpu.py

Run the complete matrix and save reproducible evidence::

    python validation/validate_fftdf_cpu.py --full --output fftdf_cpu.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


@dataclass(frozen=True)
class ValidationCase:
    """One fully specified analytic-vs-FD calculation."""

    name: str
    atoms: tuple[str, ...]
    coordinates_bohr: tuple[tuple[float, float, float], ...]
    cell_length_bohr: float
    basis: str
    cutoff_ha: float
    kmesh: tuple[int, int, int]
    fd_step_bohr: float


CASES = (
    ValidationCase(
        name="h2_gamma_szv",
        atoms=("H", "H"),
        coordinates_bohr=((0.0, 0.0, 0.0), (0.0, 0.0, 1.4)),
        cell_length_bohr=6.0,
        basis="gth-szv",
        cutoff_ha=20.0,
        kmesh=(1, 1, 1),
        fd_step_bohr=5.0e-4,
    ),
    ValidationCase(
        name="h2_k211_dzvp",
        atoms=("H", "H"),
        coordinates_bohr=((0.0, 0.0, 0.0), (0.0, 0.0, 1.4)),
        cell_length_bohr=6.0,
        basis="gth-dzvp",
        cutoff_ha=15.0,
        kmesh=(2, 1, 1),
        fd_step_bohr=5.0e-4,
    ),
    ValidationCase(
        name="distorted_h2o_gamma_szv",
        atoms=("O", "H", "H"),
        coordinates_bohr=(
            (0.0, 0.0, 0.0),
            (1.43, 0.0, 1.1),
            (-1.2, 0.2, 1.0),
        ),
        cell_length_bohr=7.0,
        basis="gth-szv",
        cutoff_ha=25.0,
        kmesh=(1, 1, 1),
        fd_step_bohr=2.0e-4,
    ),
    ValidationCase(
        name="distorted_h2o_k211_szv",
        atoms=("O", "H", "H"),
        coordinates_bohr=(
            (0.0, 0.0, 0.0),
            (1.43, 0.0, 1.1),
            (-1.2, 0.2, 1.0),
        ),
        cell_length_bohr=7.0,
        basis="gth-szv",
        cutoff_ha=15.0,
        kmesh=(2, 1, 1),
        fd_step_bohr=2.0e-4,
    ),
)


def build_cell(case: ValidationCase, coordinates: np.ndarray):
    """Build one cell while preserving the case's cutoff-to-mesh mapping."""

    cell = gto.Cell()
    cell.atom = list(zip(case.atoms, coordinates, strict=True))
    cell.a = np.eye(3) * case.cell_length_bohr
    cell.unit = "Bohr"
    cell.basis = case.basis
    cell.pseudo = "gth-hf"
    cell.ke_cutoff = case.cutoff_ha
    cell.verbose = 0
    return cell.build()


def build_method(case: ValidationCase, coordinates: np.ndarray):
    """Return a Gamma or sampled-k-point DC-PBE method for one geometry."""

    cell = build_cell(case, coordinates)
    if case.kmesh == (1, 1, 1):
        return dft.DCRKS(cell, xc="PBE")
    return dft.DCKRKS(
        cell,
        kpts=cell.make_kpts(case.kmesh),
        xc="PBE",
    )


def energy(case: ValidationCase, coordinates: np.ndarray) -> float:
    method = build_method(case, coordinates)
    return float(method.kernel())


def central_fd_gradient(case: ValidationCase, coordinates: np.ndarray) -> np.ndarray:
    """Differentiate the complete DC-PBE energy, including a new HF SCF."""

    gradient = np.zeros_like(coordinates)
    step = case.fd_step_bohr
    for atom_id in range(len(coordinates)):
        for axis in range(3):
            plus = np.array(coordinates, copy=True)
            minus = np.array(coordinates, copy=True)
            plus[atom_id, axis] += step
            minus[atom_id, axis] -= step
            gradient[atom_id, axis] = (
                energy(case, plus) - energy(case, minus)
            ) / (2.0 * step)
    return gradient


def run_case(case: ValidationCase) -> dict[str, object]:
    """Run analytic and FD gradients and return JSON-safe evidence."""

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
        "input": {
            "atoms": list(case.atoms),
            "coordinates_bohr": coordinates.tolist(),
            "cell_length_bohr": case.cell_length_bohr,
            "basis": case.basis,
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
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all Gamma/k-point and SZV/DZVP completion cases",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = CASES if args.full else CASES[:1]
    records = [run_case(case) for case in selected]
    payload = {
        "schema": "pyscf_pbc_dcdft_fftdf_cpu_validation_v1",
        "overall_status": (
            "pass" if all(record["status"] == "pass" for record in records) else "fail"
        ),
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
