#!/usr/bin/env python
"""Validate sampled-k analytic DC-DFT stress against selected energy FD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from pyscf.pbc import gto


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "stress" / "cpu" / "src"))
sys.path.insert(0, str(ROOT / "pbc" / "cpu" / "src"))

from pyscf_pbc_dcdft import dft  # noqa: E402
from pyscf_pbc_dcdft_stress import (  # noqa: E402
    FDStress,
    FDStressConfig,
    KPointGaussianDFEnergyEvaluator,
    kpoint_gdf_analytic_stress,
    kpoint_rsdf_analytic_stress,
)
from pyscf_pbc_dcdft_stress.conventions import component_index  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--df-backend", choices=("gdf", "rsdf"), required=True)
    parser.add_argument("--xc", choices=("PBE", "R2SCAN"), default="PBE")
    parser.add_argument("--kmesh", type=int, nargs=3, default=(2, 1, 1))
    parser.add_argument("--mesh", type=int, nargs=3, default=(11, 11, 11))
    parser.add_argument("--fd-components", nargs="+", default=("xx", "xy"))
    parser.add_argument("--step", type=float, default=2.5e-4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cell = gto.Cell(
        atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
        a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3), (0.1, 0.0, 6.0))),
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray(args.mesh), verbose=0,
    ).build()
    kpts = cell.make_kpts(tuple(args.kmesh), wrap_around=True)
    method = dft.DCKRKS(cell, kpts=kpts, xc=args.xc)
    if args.df_backend == "gdf":
        method = method.density_fit(auxbasis="gth-szv")
        stress_driver = kpoint_gdf_analytic_stress
    else:
        method = method.rs_density_fit(auxbasis="gth-szv")
        stress_driver = kpoint_rsdf_analytic_stress
    method.conv_tol = 1e-11

    started = perf_counter()
    energy = method.kernel()
    scf_seconds = perf_counter()-started
    started = perf_counter()
    analytic = stress_driver(method)
    analytic_seconds = perf_counter()-started
    started = perf_counter()
    fd = FDStress(
        KPointGaussianDFEnergyEvaluator.from_method(method),
        FDStressConfig(step=args.step, components=tuple(args.fd_components)),
    ).kernel(cell)
    fd_seconds = perf_counter()-started
    selected = [component_index(label) for label in args.fd_components]
    difference = (
        analytic.stress_voigt_ha_per_bohr3[selected]
        - fd.stress_voigt_ha_per_bohr3[selected]
    )
    report = {
        "input": {
            "df_backend": args.df_backend,
            "xc": args.xc,
            "kmesh": list(args.kmesh),
            "mesh": list(args.mesh),
            "fd_components": list(args.fd_components),
            "fd_step": args.step,
        },
        "energy_ha": float(energy),
        "timings_s": {
            "source_hf_and_target_state": scf_seconds,
            "analytic_stress": analytic_seconds,
            "selected_energy_fd": fd_seconds,
        },
        "max_abs_selected_difference_ha_per_bohr3": float(
            np.max(np.abs(difference))
        ),
        "selected_difference_ha_per_bohr3": difference.tolist(),
        "analytic": analytic.as_dict(),
        "fd": fd.as_dict(),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "max_abs_selected_difference_ha_per_bohr3": report[
            "max_abs_selected_difference_ha_per_bohr3"
        ],
        "timings_s": report["timings_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
