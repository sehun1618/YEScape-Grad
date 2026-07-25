#!/usr/bin/env python
"""Measure rigid-translation phase sensitivity of analytic DC-r2SCAN stress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from pyscf.pbc import gto


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "stress" / "cpu" / "src"))
sys.path.insert(0, str(ROOT / "pbc" / "cpu" / "src"))

from pyscf_pbc_dcdft import dft  # noqa: E402
from pyscf_pbc_dcdft_stress import gamma_gdf_analytic_stress  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", type=int, nargs="+", default=(9, 15, 21))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    base = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.4)))
    shift = np.asarray((0.23, 0.17, 0.11))
    records = []
    for mesh_value in args.meshes:
        pair = []
        for label, coordinates in (("base", base), ("shifted", base + shift)):
            cell = gto.Cell(
                atom=[("H", coordinates[0]), ("H", coordinates[1])],
                a=np.eye(3) * 5.0, unit="Bohr", basis="gth-szv",
                pseudo="gth-hf",
                mesh=np.asarray((mesh_value,) * 3), verbose=0,
            ).build()
            method = dft.DCRKS(cell, xc="r2SCAN").density_fit(
                auxbasis="gth-szv"
            )
            method.conv_tol = 1e-10
            method.kernel()
            started = time.perf_counter()
            result = gamma_gdf_analytic_stress(method)
            pair.append({
                "geometry": label,
                "stress_voigt_ha_per_bohr3": (
                    result.stress_voigt_ha_per_bohr3.tolist()
                ),
                "stress_wall_seconds": time.perf_counter() - started,
                "z_residual_max_abs": float(result.z_vector.residual_max_abs),
            })
        difference = np.asarray(pair[1]["stress_voigt_ha_per_bohr3"]) - np.asarray(
            pair[0]["stress_voigt_ha_per_bohr3"]
        )
        records.append({
            "mesh": [mesh_value] * 3,
            "pair": pair,
            "translation_difference_voigt_ha_per_bohr3": difference.tolist(),
            "translation_difference_max_abs_ha_per_bohr3": float(
                np.max(np.abs(difference))
            ),
        })
    report = {
        "xc": "r2SCAN",
        "df_backend": "gdf",
        "shift_bohr": shift.tolist(),
        "records": records,
        "interpretation": (
            "Rigid-translation differences are uniform-grid phase error, "
            "not a fitted correction or analytic-kernel residual."
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
