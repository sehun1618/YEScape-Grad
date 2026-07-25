#!/usr/bin/env python
"""Compare optimized and reference molecular GPU DC-gradient routes."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

import cupy as cp
import numpy as np
from pyscf import gto

from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf_dcdft_grad import RDCGradient


MODES = {
    "optimized": {
        "response_backend": "mo_direct",
        "share_df": True,
        "factorized_z_f1": True,
    },
    "reference": {
        "response_backend": "ao",
        "share_df": False,
        "factorized_z_f1": False,
    },
}


def read_xyz(path):
    lines = Path(path).expanduser().resolve().read_text().splitlines()
    natm = int(lines[0].strip())
    atoms = [line for line in lines[2:2 + natm] if line.strip()]
    if len(atoms) != natm:
        raise ValueError(f"Expected {natm} atoms in {path}, found {len(atoms)}")
    return "\n".join(atoms)


def process_vram_mib(pid):
    command = [
        "nvidia-smi", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    if completed.returncode:
        return None
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            return float(fields[1])
    return 0.0


class VRAMSampler:
    def __init__(self, interval=0.1):
        self.interval = float(interval)
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        pid = os.getpid()
        while not self._stop.is_set():
            value = process_vram_mib(pid)
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()

    @property
    def peak_mib(self):
        return max(self.samples, default=None)


def clean_device():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()


def run_case(mol, args, mode):
    options = MODES[mode]
    mf = gpu_hf.RHF(mol).density_fit(auxbasis=args.auxbasis)
    mf.conv_tol = args.scf_tol
    mf.max_cycle = args.scf_max_cycle
    scf_start = time.perf_counter()
    energy = mf.kernel()
    scf_seconds = time.perf_counter() - scf_start
    if not mf.converged:
        raise RuntimeError(f"{mode} RHF did not converge")

    driver = RDCGradient(
        mf, xc=args.xc, cphf_tol=args.cphf_tol,
        cphf_max_cycle=args.cphf_max_cycle,
        atom_grid=tuple(args.atom_grid), grid_response=True,
        df_deriv_memory_gb=args.df_deriv_memory_gb, **options,
    )
    with VRAMSampler(args.sample_interval) as sampler:
        start = time.perf_counter()
        gradient = driver.kernel(verbose=0)
        gradient_seconds = time.perf_counter() - start

    record = {
        "mode": mode,
        "energy_eh": float(energy),
        "scf_seconds": scf_seconds,
        "gradient_seconds": gradient_seconds,
        "gradient_eh_per_bohr": np.asarray(gradient).tolist(),
        "peak_process_vram_mib": sampler.peak_mib,
        "vram_sample_count": len(sampler.samples),
        "memory_diagnostics": driver.memory_diagnostics,
        "optimization_diagnostics": driver.optimization_diagnostics,
    }
    del driver, mf
    clean_device()
    return record


def summarize(records):
    summary = {}
    for mode in MODES:
        selected = [record for record in records if record["mode"] == mode]
        summary[mode] = {
            "gradient_median_seconds": statistics.median(
                record["gradient_seconds"] for record in selected
            ),
            "peak_process_vram_mib": max(
                record["peak_process_vram_mib"] or 0.0 for record in selected
            ),
        }

    differences = []
    for repeat in sorted({record["repeat"] for record in records}):
        pair = {record["mode"]: record for record in records
                if record["repeat"] == repeat}
        optimized = np.asarray(pair["optimized"]["gradient_eh_per_bohr"])
        reference = np.asarray(pair["reference"]["gradient_eh_per_bohr"])
        differences.append(float(np.max(np.abs(optimized - reference))))
    summary["max_abs_gradient_difference"] = max(differences)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz", type=Path)
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--auxbasis", default=None)
    parser.add_argument("--xc", default="PBE")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--atom-grid", type=int, nargs=2, default=(50, 194))
    parser.add_argument("--scf-tol", type=float, default=1e-10)
    parser.add_argument("--scf-max-cycle", type=int, default=100)
    parser.add_argument("--cphf-tol", type=float, default=1e-10)
    parser.add_argument("--cphf-max-cycle", type=int, default=500)
    parser.add_argument("--df-deriv-memory-gb", type=float)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    mol = gto.M(
        atom=read_xyz(args.xyz), basis=args.basis, charge=args.charge,
        spin=args.spin, verbose=0, max_memory=128000,
    )
    records = []
    for repeat in range(1, args.repeats + 1):
        order = tuple(MODES) if repeat % 2 else tuple(reversed(MODES))
        for mode in order:
            record = run_case(mol, args, mode)
            record["repeat"] = repeat
            records.append(record)
            print(
                f"repeat={repeat} mode={mode} "
                f"gradient={record['gradient_seconds']:.3f}s "
                f"peak={record['peak_process_vram_mib']} MiB",
                flush=True,
            )

    report = {
        "input": {
            "xyz": str(args.xyz.expanduser().resolve()),
            "basis": args.basis,
            "auxbasis": args.auxbasis,
            "xc": args.xc,
            "atom_grid": list(args.atom_grid),
            "scf_tol": args.scf_tol,
            "cphf_tol": args.cphf_tol,
            "df_deriv_memory_gb": args.df_deriv_memory_gb,
        },
        "records": records,
        "summary": summarize(records),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
