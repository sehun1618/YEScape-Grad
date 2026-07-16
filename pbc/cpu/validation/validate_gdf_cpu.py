#!/usr/bin/env python
"""Reproduce the Stage-4 CPU GDF analytic-vs-FD completion matrix.

The default quick gate validates one H2 Gamma case.  ``--full`` evaluates all
Cartesian components for H2 and distorted H2O, Gamma and a ``(2,1,1)``
k-mesh, and GTH-SZV/GTH-DZVP primary-basis transfer.  Every finite-difference
point repeats the HF calculation and evaluates the complete
``E_PBE[n_HF]`` energy.

Coordinates and lattice vectors are in Bohr.  Energies are in Hartree and
gradients are in Hartree/Bohr.  The auxiliary basis is recorded independently
from the primary AO basis.

Examples
--------
Run the quick gate::

    python validation/validate_gdf_cpu.py

Create the full completion record::

    python validation/validate_gdf_cpu.py --full --output gdf_cpu.json
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
    """One reproducible GDF analytic-vs-FD calculation."""

    name: str
    atoms: tuple[str, ...]
    coordinates_bohr: tuple[tuple[float, float, float], ...]
    cell_length_bohr: float
    basis: str
    auxbasis: str
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
        auxbasis="gth-szv",
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
        auxbasis="gth-szv",
        cutoff_ha=15.0,
        kmesh=(2, 1, 1),
        fd_step_bohr=5.0e-4,
    ),
    ValidationCase(
        name="distorted_h2o_gamma_szv",
        atoms=("O", "H", "H"),
        coordinates_bohr=(
            (0.1, -0.2, 0.3),
            (1.55, 0.15, 0.05),
            (-0.35, 1.25, -0.25),
        ),
        cell_length_bohr=7.0,
        basis="gth-szv",
        auxbasis="gth-szv",
        cutoff_ha=25.0,
        kmesh=(1, 1, 1),
        fd_step_bohr=2.0e-4,
    ),
    ValidationCase(
        name="distorted_h2o_k211_szv",
        atoms=("O", "H", "H"),
        coordinates_bohr=(
            (0.1, -0.2, 0.3),
            (1.55, 0.15, 0.05),
            (-0.35, 1.25, -0.25),
        ),
        cell_length_bohr=7.0,
        basis="gth-szv",
        auxbasis="gth-szv",
        cutoff_ha=15.0,
        kmesh=(2, 1, 1),
        fd_step_bohr=2.0e-4,
    ),
)


def build_cell(case: ValidationCase, coordinates: np.ndarray):
    """Build one periodic cell with an explicit cutoff-to-mesh mapping."""

    return gto.Cell(
        atom=list(zip(case.atoms, coordinates, strict=True)),
        a=np.eye(3) * case.cell_length_bohr,
        unit="Bohr",
        basis=case.basis,
        pseudo="gth-hf",
        ke_cutoff=case.cutoff_ha,
        verbose=0,
    ).build()


def build_method(case: ValidationCase, coordinates: np.ndarray):
    """Build the Gamma or sampled-k-point DC-PBE GDF method."""

    cell = build_cell(case, coordinates)
    if case.kmesh == (1, 1, 1):
        method = dft.DCRKS(cell, xc="PBE")
    else:
        method = dft.DCKRKS(
            cell,
            kpts=cell.make_kpts(case.kmesh),
            xc="PBE",
        )
    method.density_fit(auxbasis=case.auxbasis)
    method.conv_tol = 1.0e-11
    method.max_cycle = 100
    return method


def energy(case: ValidationCase, coordinates: np.ndarray) -> float:
    """Return the complete frozen-HF-density target energy."""

    return float(build_method(case, coordinates).kernel())


def central_fd_gradient(
    case: ValidationCase,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Differentiate all nuclear Cartesian coordinates by central FD."""

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
    """Run one case and return a JSON-safe, auditable evidence record."""

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
    net_gradient = np.asarray(analytic).sum(axis=0)
    return {
        "case": case.name,
        "status": "pass" if maximum_error <= 1.0e-5 else "fail",
        "threshold_ha_per_bohr": 1.0e-5,
        "energy_ha": float(dc_energy),
        "analytic_gradient_ha_per_bohr": analytic.tolist(),
        "fd_gradient_ha_per_bohr": finite_difference.tolist(),
        "analytic_minus_fd_ha_per_bohr": difference.tolist(),
        "max_abs_error_ha_per_bohr": maximum_error,
        "net_analytic_gradient_ha_per_bohr": net_gradient.tolist(),
        "net_analytic_gradient_norm_ha_per_bohr": float(
            np.linalg.norm(net_gradient)
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
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all H2/H2O, Gamma/k-point, and basis-transfer cases",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = CASES if args.full else CASES[:1]
    records = [run_case(case) for case in selected]
    payload = {
        "schema": "pyscf_pbc_dcdft_gdf_cpu_validation_v1",
        "overall_status": (
            "pass"
            if all(record["status"] == "pass" for record in records)
            else "fail"
        ),
        "notes": {
            "analytic_fd_gate": (
                "Compares derivatives at the same finite mesh and GDF "
                "auxiliary representation."
            ),
            "translation_gate": (
                "Reported separately because finite uniform XC grids exhibit "
                "an egg-box error that must be assessed by cutoff/mesh "
                "convergence rather than by changing the analytic formula."
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
