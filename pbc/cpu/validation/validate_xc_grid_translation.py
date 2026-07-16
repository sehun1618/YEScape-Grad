#!/usr/bin/env python
"""Validate uniform-grid translation convergence for periodic DC-DFT.

The fixed uniform grid is part of the discretized PBE energy.  At finite mesh
the energy has an egg-box dependence on a rigid translation of all nuclei.
The analytic gradient must first match finite differences of that same energy;
physical translation invariance is then recovered by mesh convergence.

This script isolates the target-XC contribution, which owns the observed net
gradient after the one-electron, GDF J/K, overlap, nuclear, and Z-vector terms
have cancelled.  The first mesh also evaluates the complete analytic gradient
and reports the residual net contribution excluding target XC.

The default meshes are a compact reproduction of the convergence trend.  The
largest ``225^3`` grid is intentionally expensive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft
from pyscf_pbc_dcdft.grad.context import RestrictedGradientContext
from pyscf_pbc_dcdft.grad.target import (
    contract_target_potential,
    target_xc_row_derivative,
)


ATOMS = ("O", "H", "H")
COORDINATES_BOHR = np.asarray(
    ((0.1, -0.2, 0.3), (1.55, 0.15, 0.05), (-0.35, 1.25, -0.25))
)
THRESHOLD = 1.0e-6


def build_method(mesh_size: int, *, xc: str):
    """Return the distorted-H2O Gamma GDF calculation at one cubic mesh."""

    cell = gto.Cell(
        atom=list(zip(ATOMS, COORDINATES_BOHR, strict=True)),
        a=np.eye(3) * 7.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-hf",
        mesh=np.asarray((mesh_size,) * 3),
        verbose=0,
    ).build()
    method = dft.DCRKS(cell, xc=xc).density_fit(auxbasis="gth-szv")
    method.conv_tol = 1.0e-11
    method.max_cycle = 100
    return method


def run_mesh(
    mesh_size: int,
    *,
    xc: str,
    complete_gradient: bool,
) -> dict[str, object]:
    """Return target-XC and optional complete-gradient translation evidence."""

    method = build_method(mesh_size, xc=xc)
    start = perf_counter()
    method.kernel()
    context = RestrictedGradientContext.from_dc_method(method)
    target_xc = contract_target_potential(
        context,
        target_xc_row_derivative(context),
    )
    target_xc_net = target_xc.sum(axis=0)
    record = {
        "mesh": [mesh_size] * 3,
        "target_xc_net_ha_per_bohr": target_xc_net.tolist(),
        "target_xc_net_norm_ha_per_bohr": float(
            np.linalg.norm(target_xc_net)
        ),
    }
    if complete_gradient:
        gradient = method.nuc_grad_method().kernel()
        complete_net = np.asarray(gradient).sum(axis=0)
        non_xc_net = complete_net - target_xc_net
        record.update(
            {
                "complete_net_ha_per_bohr": complete_net.tolist(),
                "non_xc_net_ha_per_bohr": non_xc_net.tolist(),
                "non_xc_net_norm_ha_per_bohr": float(
                    np.linalg.norm(non_xc_net)
                ),
            }
        )
    record["wall_time_s"] = perf_counter() - start
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meshes",
        type=int,
        nargs="+",
        default=(17, 81, 225),
        help="odd cubic mesh dimensions in increasing convergence order",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--xc",
        default="PBE",
        help="pure semilocal target DFA, for example PBE, SCAN, or R2SCAN",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    meshes = tuple(int(value) for value in args.meshes)
    if not meshes or any(value <= 0 or value % 2 == 0 for value in meshes):
        raise ValueError("meshes must contain positive odd integers")
    records = [
        run_mesh(mesh, xc=args.xc, complete_gradient=(index == 0))
        for index, mesh in enumerate(meshes)
    ]
    first_non_xc = records[0]["non_xc_net_norm_ha_per_bohr"]
    final_xc = records[-1]["target_xc_net_norm_ha_per_bohr"]
    passed = first_non_xc <= THRESHOLD and final_xc <= THRESHOLD
    payload = {
        "schema": "pyscf_pbc_dcdft_xc_grid_translation_v2",
        "status": "pass" if passed else "fail",
        "threshold_ha_per_bohr": THRESHOLD,
        "system": "distorted_h2o_gamma_gdf_gth_szv",
        "xc": args.xc,
        "coordinates_bohr": COORDINATES_BOHR.tolist(),
        "cell_bohr": (np.eye(3) * 7.0).tolist(),
        "interpretation": (
            "Non-XC terms are translationally closed at the first mesh; "
            "the fixed uniform-grid XC egg-box term converges separately."
        ),
        "records": records,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
