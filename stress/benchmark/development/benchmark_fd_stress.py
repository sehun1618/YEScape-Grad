#!/usr/bin/env python
"""Benchmark the CPU Gamma/GDF central-FD stress validation route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from pyscf.lib import param
from pyscf.pbc import gto


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "stress" / "cpu" / "src"))
sys.path.insert(0, str(ROOT / "pbc" / "cpu" / "src"))

from pyscf_pbc_dcdft_stress import (  # noqa: E402
    FDStress,
    FDStressConfig,
    GammaGDFEnergyEvaluator,
)


def read_xyz(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    natm = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + natm]:
        symbol, x, y, z = line.split()[:4]
        coordinate = np.asarray((float(x), float(y), float(z))) / param.BOHR
        atoms.append((symbol, coordinate))
    if len(atoms) != natm:
        raise ValueError(f"{path} contains {len(atoms)} atoms; expected {natm}")
    return atoms


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz", type=Path)
    parser.add_argument("--cubic-cell-bohr", type=float, required=True)
    parser.add_argument("--mesh", type=int, nargs=3, required=True)
    parser.add_argument("--basis", default="gth-szv")
    parser.add_argument("--pseudo", default="gth-hf")
    parser.add_argument("--auxbasis", default="gth-szv")
    parser.add_argument("--xc", default="PBE")
    parser.add_argument("--step", type=float, default=5e-4)
    parser.add_argument("--components", nargs="+", default=None)
    parser.add_argument("--conv-tol", type=float, default=1e-10)
    parser.add_argument("--max-cycle", type=int, default=80)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cell = gto.Cell(
        atom=read_xyz(args.xyz), a=np.eye(3) * args.cubic_cell_bohr,
        unit="Bohr", basis=args.basis, pseudo=args.pseudo,
        mesh=np.asarray(args.mesh), verbose=0,
    ).build()
    evaluator = GammaGDFEnergyEvaluator(
        xc=args.xc, auxbasis=args.auxbasis, conv_tol=args.conv_tol,
        max_cycle=args.max_cycle, verbose=0,
    )
    components = tuple(range(6)) if args.components is None else tuple(args.components)
    config = FDStressConfig(
        step=args.step, components=components, warm_start=args.warm_start
    )
    started = time.perf_counter()
    result = FDStress(evaluator, config).kernel(cell)
    total_seconds = time.perf_counter() - started
    report = {
        "input": {
            "xyz": str(args.xyz.expanduser().resolve()),
            "cell_bohr": np.asarray(cell.a).tolist(),
            "mesh": [int(value) for value in cell.mesh],
            "basis": args.basis,
            "pseudo": args.pseudo,
            "auxbasis": args.auxbasis,
            "xc": args.xc,
            "step": args.step,
            "components": list(config.components),
            "warm_start": args.warm_start,
        },
        "total_wall_seconds": total_seconds,
        "result": result.as_dict(),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
