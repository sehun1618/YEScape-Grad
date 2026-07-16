#!/usr/bin/env python
"""Validate GPU RSDF DC-PBE energy and analytic nuclear gradients.

Every finite-difference point repeats GPU RHF/KRHF and ``E_PBE[n_HF]`` with
the project-local RSDF provider.  The completed CPU RSDF route evaluates the
central geometry independently to separate derivative accuracy from device
parity.
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
from pyscf_pbc_dcdft import df as cpu_df
from pyscf_pbc_dcdft import dft as cpu_dft

from gpu4pyscf_pbc_dcdft import df as gpu_df
from gpu4pyscf_pbc_dcdft import dft as gpu_dft


@dataclass(frozen=True)
class ValidationCase:
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
    return gto.Cell(
        atom=list(zip(case.atoms, coordinates, strict=True)),
        a=np.eye(3) * case.cell_length_bohr,
        unit="Bohr",
        basis=case.basis,
        pseudo="gth-hf",
        ke_cutoff=case.cutoff_ha,
        verbose=0,
    ).build()


def build_method(case, coordinates, *, device):
    cell = build_cell(case, coordinates)
    kpts = cell.make_kpts(case.kmesh, wrap_around=True)
    dft_module = gpu_dft if device == "gpu" else cpu_dft
    df_module = gpu_df if device == "gpu" else cpu_df
    if case.kmesh == (1, 1, 1):
        method = dft_module.DCRKS(cell, xc="PBE")
    else:
        method = dft_module.DCKRKS(cell, kpts=kpts, xc="PBE")
    method.with_df = df_module.RSDF(
        cell, kpts=kpts, auxbasis=case.auxbasis
    )
    method.conv_tol = 1.0e-11
    method.conv_tol_grad = 1.0e-9
    method.max_cycle = 100
    return method


def gpu_energy(case, coordinates):
    return float(build_method(case, coordinates, device="gpu").kernel())


def central_fd(case, coordinates):
    gradient = np.zeros_like(coordinates)
    for atom_id in range(len(coordinates)):
        for axis in range(3):
            plus = np.array(coordinates, copy=True)
            minus = np.array(coordinates, copy=True)
            plus[atom_id, axis] += case.fd_step_bohr
            minus[atom_id, axis] -= case.fd_step_bohr
            gradient[atom_id, axis] = (
                gpu_energy(case, plus) - gpu_energy(case, minus)
            ) / (2.0 * case.fd_step_bohr)
    return gradient


def run_case(case):
    coordinates = np.asarray(case.coordinates_bohr, dtype=float)
    gpu_method = build_method(case, coordinates, device="gpu")
    start = perf_counter()
    gpu_energy_value = gpu_method.kernel()
    gpu_energy_seconds = perf_counter() - start
    gpu_gradient_object = gpu_method.nuc_grad_method()
    start = perf_counter()
    gpu_gradient = gpu_gradient_object.kernel()
    gpu_gradient_seconds = perf_counter() - start

    cpu_method = build_method(case, coordinates, device="cpu")
    cpu_energy_value = cpu_method.kernel()
    cpu_gradient = cpu_method.nuc_grad_method().kernel()

    start = perf_counter()
    fd_gradient = central_fd(case, coordinates)
    fd_seconds = perf_counter() - start
    analytic_fd_error = float(np.max(np.abs(gpu_gradient - fd_gradient)))
    energy_parity = abs(float(cpu_energy_value) - float(gpu_energy_value))
    gradient_parity = float(np.max(np.abs(cpu_gradient - gpu_gradient)))
    passed = (
        analytic_fd_error <= 1.0e-5
        and energy_parity <= 1.0e-8
        and gradient_parity <= 1.0e-7
    )
    return {
        "case": case.name,
        "status": "pass" if passed else "fail",
        "gpu_energy_ha": float(gpu_energy_value),
        "gpu_analytic_gradient_ha_per_bohr": gpu_gradient.tolist(),
        "gpu_fd_gradient_ha_per_bohr": fd_gradient.tolist(),
        "gpu_analytic_minus_fd_ha_per_bohr": (
            gpu_gradient - fd_gradient
        ).tolist(),
        "max_abs_analytic_fd_error_ha_per_bohr": analytic_fd_error,
        "cpu_gpu_energy_abs_error_ha": energy_parity,
        "cpu_gpu_gradient_max_abs_error_ha_per_bohr": gradient_parity,
        "thresholds": {
            "analytic_fd_ha_per_bohr": 1.0e-5,
            "cpu_gpu_energy_ha": 1.0e-8,
            "cpu_gpu_gradient_ha_per_bohr": 1.0e-7,
        },
        "input": {
            "coordinates_bohr": coordinates.tolist(),
            "cell_length_bohr": case.cell_length_bohr,
            "basis": case.basis,
            "auxbasis": case.auxbasis,
            "pseudo": "gth-hf",
            "cutoff_ha": case.cutoff_ha,
            "mesh": [int(value) for value in gpu_method.cell.mesh],
            "kmesh": list(case.kmesh),
            "fd_step_bohr": case.fd_step_bohr,
        },
        "timing_seconds": {
            "gpu_dc_energy": gpu_energy_seconds,
            "gpu_analytic_gradient": gpu_gradient_seconds,
            "gpu_full_fd_gradient": fd_seconds,
        },
        "gradient_metadata": gpu_gradient_object.result.metadata(),
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
        "schema": "gpu4pyscf_pbc_dcdft_rsdf_validation_v1",
        "overall_status": (
            "pass"
            if all(record["status"] == "pass" for record in records)
            else "fail"
        ),
        "requested_cpu_threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "cases": records,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
