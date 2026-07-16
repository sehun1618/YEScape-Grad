#!/usr/bin/env python
"""Validate MGGA uniform-grid translation convergence on the GPU.

The complete gradient is evaluated at the first mesh to establish that all
non-XC terms are translationally closed.  Higher meshes then evaluate only the
target-XC row derivative, which isolates the finite-grid egg-box error and
avoids repeating GDF derivative-integral work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
from pyscf.pbc import gto

from gpu4pyscf_pbc_dcdft import dft
from gpu4pyscf_pbc_dcdft.grad.context import RestrictedGradientContext
from gpu4pyscf_pbc_dcdft.grad.target import (
    contract_xc_row_derivative,
    target_xc_row_derivative,
)


ATOMS = ("O", "H", "H")
COORDINATES_BOHR = np.asarray(
    ((0.1, -0.2, 0.3), (1.55, 0.15, 0.05), (-0.35, 1.25, -0.25))
)
THRESHOLD = 1.0e-6


def build_method(mesh_size: int, *, xc: str):
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
    method.conv_tol_grad = 1.0e-9
    method.max_cycle = 100
    return method


def run_mesh(mesh_size: int, *, xc: str, complete_gradient: bool):
    method = build_method(mesh_size, xc=xc)
    start = perf_counter()
    method.kernel()
    context = RestrictedGradientContext.from_dc_method(method)
    target_xc = contract_xc_row_derivative(
        context,
        target_xc_row_derivative(context),
    )
    cp.cuda.Stream.null.synchronize()
    target_xc_host = cp.asnumpy(target_xc)
    target_xc_net = target_xc_host.sum(axis=0)
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
    cp.cuda.Stream.null.synchronize()
    record["wall_time_s"] = perf_counter() - start
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xc", default="SCAN")
    parser.add_argument(
        "--meshes",
        type=int,
        nargs="+",
        default=(17, 225, 301, 401),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    meshes = tuple(int(value) for value in args.meshes)
    if not meshes or any(value <= 0 or value % 2 == 0 for value in meshes):
        raise ValueError("meshes must contain positive odd integers")
    records = [
        run_mesh(
            mesh,
            xc=args.xc,
            complete_gradient=(index == 0),
        )
        for index, mesh in enumerate(meshes)
    ]
    passed = (
        records[0]["non_xc_net_norm_ha_per_bohr"] <= THRESHOLD
        and records[-1]["target_xc_net_norm_ha_per_bohr"] <= THRESHOLD
    )
    payload = {
        "schema": "gpu4pyscf_pbc_dcdft_mgga_grid_translation_v1",
        "status": "pass" if passed else "fail",
        "threshold_ha_per_bohr": THRESHOLD,
        "xc": args.xc,
        "system": "distorted_h2o_gamma_gdf_gth_szv",
        "coordinates_bohr": COORDINATES_BOHR.tolist(),
        "cell_bohr": (np.eye(3) * 7.0).tolist(),
        "records": records,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
