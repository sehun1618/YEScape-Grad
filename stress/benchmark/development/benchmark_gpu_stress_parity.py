#!/usr/bin/env python
"""Compare the S7 GPU stress route with an existing S6 CPU evidence file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from pyscf.lib import param
from pyscf.pbc import gto


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "stress" / "gpu" / "src",
    ROOT / "stress" / "cpu" / "src",
    ROOT / "pbc" / "gpu" / "src",
    ROOT / "pbc" / "cpu" / "src",
):
    sys.path.insert(0, str(path))

from gpu4pyscf_pbc_dcdft import dft  # noqa: E402
from gpu4pyscf_pbc_dcdft_stress import gpu_analytic_stress  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--df-backend", choices=("gdf", "rsdf"), required=True)
    parser.add_argument("--xc", choices=("PBE", "R2SCAN"), default="PBE")
    parser.add_argument("--cpu-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-workers", type=int)
    parser.add_argument("--df-diagonal-cache-bytes", type=int, default=256*(1 << 20))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cpu_report = json.loads(args.cpu_evidence.read_text(encoding="utf-8"))
    settings = cpu_report["input"]
    sampled_k = "kmesh" in settings
    if sampled_k:
        cell = gto.Cell(
            atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
            a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3), (0.1, 0.0, 6.0))),
            unit="Bohr", basis="gth-szv", pseudo="gth-hf",
            mesh=np.asarray(settings["mesh"]), verbose=0,
        ).build()
        kpts = cell.make_kpts(tuple(settings["kmesh"]), wrap_around=True)
        method = dft.DCKRKS(cell, kpts=kpts, xc=args.xc)
    else:
        cell = gto.Cell(
            atom=_read_xyz_bohr(Path(settings["xyz"])),
            a=np.asarray(settings["cell_bohr"]), unit="Bohr",
            basis=settings["basis"], pseudo=settings["pseudo"],
            mesh=np.asarray(settings["mesh"]), verbose=0,
        ).build()
        method = dft.DCRKS(cell, xc=args.xc)
    if args.df_backend == "gdf":
        method = method.density_fit(auxbasis="gth-szv")
    else:
        method = method.rs_density_fit(auxbasis="gth-szv")
    method.conv_tol = 1e-11
    started = perf_counter()
    energy = method.kernel()
    energy_seconds = perf_counter()-started
    started = perf_counter()
    result = gpu_analytic_stress(
        method, host_workers=args.host_workers,
        df_diagonal_cache_bytes=args.df_diagonal_cache_bytes,
    )
    stress_seconds = perf_counter()-started
    result_key = "analytic" if sampled_k else "result"
    cpu_stress = np.asarray(
        cpu_report[result_key]["stress_voigt_ha_per_bohr3"], dtype=float
    )
    difference = result.stress_voigt_ha_per_bohr3-cpu_stress
    report = {
        "input": {
            "df_backend": args.df_backend,
            "xc": args.xc,
            "sampling": "sampled_k" if sampled_k else "gamma",
            "kmesh": settings.get("kmesh", [1, 1, 1]),
            "mesh": settings["mesh"],
            "host_workers": args.host_workers,
            "df_diagonal_cache_bytes": args.df_diagonal_cache_bytes,
            "cpu_evidence": str(args.cpu_evidence.resolve()),
        },
        "energy_ha": float(energy),
        "timings_s": {
            "gpu_source_hf_and_target_state": energy_seconds,
            "gpu_analytic_stress": stress_seconds,
        },
        "cpu_gpu_max_abs_difference_ha_per_bohr3": float(
            np.max(np.abs(difference))
        ),
        "cpu_gpu_difference_ha_per_bohr3": difference.tolist(),
        "gpu": result.as_dict(),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "cpu_gpu_max_abs_difference_ha_per_bohr3": report[
            "cpu_gpu_max_abs_difference_ha_per_bohr3"
        ],
        "timings_s": report["timings_s"],
    }, indent=2))
    return 0


def _read_xyz_bohr(path):
    lines = path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
    natm = int(lines[0])
    atoms = []
    for line in lines[2:2+natm]:
        symbol, x, y, z = line.split()[:4]
        coordinate = np.asarray((float(x), float(y), float(z)))/param.BOHR
        atoms.append((symbol, coordinate))
    if len(atoms) != natm:
        raise ValueError(f"{path} contains {len(atoms)} atoms; expected {natm}")
    return atoms


if __name__ == "__main__":
    raise SystemExit(main())
