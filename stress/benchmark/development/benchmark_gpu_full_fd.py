#!/usr/bin/env python
"""Compare GPU analytic stress with a full 6-component GPU energy FD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import cupy as cp
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
from pyscf_pbc_dcdft_stress import (  # noqa: E402
    EnergyEvaluation,
    FDStress,
    FDStressConfig,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--df-backend", choices=("gdf", "rsdf"), required=True)
    parser.add_argument("--xc", choices=("PBE", "R2SCAN"), default="PBE")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--step", type=float, default=2.5e-4)
    parser.add_argument("--host-workers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


class GPUGammaEnergyEvaluator:
    """Fresh GPU source-HF/target-DFA energy for every strained cell."""

    def __init__(self, *, df_backend, xc, auxbasis, conv_tol):
        self.df_backend = df_backend
        self.xc = xc
        self.auxbasis = auxbasis
        self.conv_tol = conv_tol

    def method(self, cell):
        method = dft.DCRKS(cell, xc=self.xc)
        if self.df_backend == "gdf":
            method = method.density_fit(auxbasis=self.auxbasis)
        else:
            method = method.rs_density_fit(auxbasis=self.auxbasis)
        method.conv_tol = self.conv_tol
        return method

    def __call__(self, cell, *, dm0=None, kpts=None):
        if kpts is not None:
            raise ValueError("Gamma evaluator does not accept explicit k-points")
        method = self.method(cell)
        started = perf_counter()
        energy = method.kernel(dm0=dm0)
        cp.cuda.Stream.null.synchronize()
        elapsed = perf_counter()-started
        density = cp.asnumpy(cp.asarray(method.make_rdm1()))
        return EnergyEvaluation(
            energy_ha=float(energy), wall_seconds=elapsed,
            converged=bool(method.converged), density=density,
            metadata={"device": "gpu", "df_backend": self.df_backend},
        )


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    cell = gto.Cell(
        atom=_read_xyz_bohr(ROOT / "pbc" / "benchmark" / "structures" / "h2.xyz"),
        a=np.eye(3)*8.0, unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((15, 15, 15)), verbose=0,
    ).build()
    evaluator = GPUGammaEnergyEvaluator(
        df_backend=args.df_backend, xc=args.xc,
        auxbasis="gth-szv", conv_tol=1e-11,
    )
    repeats = []
    for repeat in range(args.repeats):
        print(f"repeat {repeat+1}/{args.repeats}: source state", flush=True)
        method = evaluator.method(cell)
        started = perf_counter()
        energy = method.kernel()
        cp.cuda.Stream.null.synchronize()
        source_seconds = perf_counter()-started
        print(f"repeat {repeat+1}/{args.repeats}: analytic stress", flush=True)
        started = perf_counter()
        analytic = gpu_analytic_stress(method, host_workers=args.host_workers)
        cp.cuda.Stream.null.synchronize()
        analytic_seconds = perf_counter()-started

        cp.get_default_memory_pool().free_all_blocks()
        print(f"repeat {repeat+1}/{args.repeats}: full GPU FD", flush=True)
        started = perf_counter()
        fd = FDStress(
            evaluator, FDStressConfig(step=args.step)
        ).kernel(cell)
        cp.cuda.Stream.null.synchronize()
        fd_seconds = perf_counter()-started
        difference = (
            analytic.stress_voigt_ha_per_bohr3
            - fd.stress_voigt_ha_per_bohr3
        )
        repeats.append({
            "repeat": repeat+1,
            "energy_ha": float(energy),
            "source_hf_and_target_state_s": source_seconds,
            "analytic_stress_s": analytic_seconds,
            "analytic_end_to_end_s": source_seconds+analytic_seconds,
            "full_gpu_fd_s": fd_seconds,
            "fd_over_analytic_stress_ratio": fd_seconds/analytic_seconds,
            "max_abs_analytic_fd_ha_per_bohr3": float(
                np.max(np.abs(difference))
            ),
            "analytic": analytic.as_dict(),
            "fd": fd.as_dict(),
        })
        print(f"repeat {repeat+1}/{args.repeats}: complete", flush=True)
        cp.get_default_memory_pool().free_all_blocks()

    report = {
        "input": {
            "df_backend": args.df_backend,
            "xc": args.xc,
            "sampling": "gamma",
            "mesh": [15, 15, 15],
            "repeats": args.repeats,
            "fd_step": args.step,
            "host_workers": args.host_workers,
        },
        "median_s": {
            "source_hf_and_target_state": statistics.median(
                item["source_hf_and_target_state_s"] for item in repeats
            ),
            "analytic_stress": statistics.median(
                item["analytic_stress_s"] for item in repeats
            ),
            "analytic_end_to_end": statistics.median(
                item["analytic_end_to_end_s"] for item in repeats
            ),
            "full_gpu_fd": statistics.median(
                item["full_gpu_fd_s"] for item in repeats
            ),
        },
        "repeats": repeats,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "median_s": report["median_s"],
        "max_abs_analytic_fd_ha_per_bohr3": max(
            item["max_abs_analytic_fd_ha_per_bohr3"] for item in repeats
        ),
    }, indent=2))
    return 0


def _read_xyz_bohr(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    natm = int(lines[0])
    atoms = []
    for line in lines[2:2+natm]:
        symbol, x, y, z = line.split()[:4]
        atoms.append((
            symbol,
            np.asarray((float(x), float(y), float(z)))/param.BOHR,
        ))
    return atoms


if __name__ == "__main__":
    raise SystemExit(main())
