#!/usr/bin/env python
"""Validate user-run GDF/RSDF stress scaling evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_PP = "gpu4pyscf_native_contracted_analytic_nonlocal_device"
EXPECTED_DF = "host_analytic_source_adjoint_gpu_ft"


def validate_report(path, *, minimum_atoms=60):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    backend = report.get("df_backend")
    require(report.get("status") == "complete", "status is not complete")
    require(
        report.get("execution_status") == "complete",
        "execution_status is not complete",
    )
    require(report.get("device") == "gpu", "device is not gpu")
    require(
        report.get("calculation_mode") == "stress_only",
        "calculation_mode is not stress_only",
    )
    require(backend in {"gdf", "rsdf"}, "df_backend is not gdf/rsdf")
    require(
        int((report.get("system") or {}).get("natoms", 0)) >= minimum_atoms,
        f"system has fewer than {minimum_atoms} atoms",
    )
    require(bool(report.get("input_fingerprint")), "input fingerprint missing")
    require(bool(report.get("state_fingerprint")), "state fingerprint missing")

    stress = report.get("stress") or {}
    metadata = stress.get("metadata") or {}
    require(metadata.get("production") is True, "stress is not production")
    require(
        metadata.get("pp_response_backend") == EXPECTED_PP,
        "unexpected PP response backend",
    )
    require(
        metadata.get("df_response_backend") == EXPECTED_DF,
        "unexpected DF response backend",
    )
    require(
        metadata.get("low_level_numerical_components") == [],
        "low-level numerical stress component is present",
    )
    require(
        metadata.get("silent_cpu_fallback") is False,
        "silent_cpu_fallback is not explicitly false",
    )

    timing = report.get("timing_s") or {}
    stress_wall = timing.get("analytic_stress_wall_s")
    require(
        isinstance(stress_wall, (int, float)) and stress_wall > 0.0,
        "analytic stress wall time missing",
    )
    memory = report.get("memory") or {}
    host_peak = memory.get("host_peak_rss_bytes")
    require(
        isinstance(host_peak, int) and host_peak > 0,
        "host peak RSS missing",
    )
    vram = memory.get("vram_peak_tracking") or {}
    require(vram.get("status") == "complete", "VRAM tracking is not complete")
    vram_peak = vram.get("peak_observed_cupy_pool_reserved_bytes")
    require(
        isinstance(vram_peak, int) and vram_peak >= 0,
        "process-local VRAM peak missing",
    )
    return {
        "path": str(Path(path).resolve()),
        "backend": backend,
        "pass": not errors,
        "errors": errors,
        "analytic_stress_wall_s": stress_wall,
        "host_peak_rss_bytes": host_peak,
        "peak_cupy_pool_reserved_bytes": vram_peak,
    }


def validate_result_root(root, *, minimum_atoms=60):
    root = Path(root).expanduser().resolve()
    cases = [
        validate_report(path, minimum_atoms=minimum_atoms)
        for path in sorted(root.glob("*/report.json"))
    ]
    by_backend = {case["backend"]: case for case in cases}
    missing = sorted({"gdf", "rsdf"}-set(by_backend))
    passed = not missing and all(by_backend[name]["pass"] for name in ("gdf", "rsdf"))
    return {
        "schema": "pbc_dcdft_stress_scaling_gate_v1",
        "result_root": str(root),
        "minimum_atoms": int(minimum_atoms),
        "required_backends": ["gdf", "rsdf"],
        "missing_backends": missing,
        "cases": cases,
        "pass": passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--minimum-atoms", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_result_root(
        args.result_root, minimum_atoms=args.minimum_atoms
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)+"\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
