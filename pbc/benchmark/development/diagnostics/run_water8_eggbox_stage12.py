#!/usr/bin/env python
"""Run the Water8 egg-box mesh/axis study and summarize convergence."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


BENCHMARK = Path(__file__).resolve().parents[2]
DIAGNOSTIC = (
    BENCHMARK
    / "development"
    / "diagnostics"
    / "diagnose_rigid_translation_eggbox.py"
)
DEFAULT_QUEUE = BENCHMARK / "user" / "queues" / "current_correctness.json"
DEFAULT_OUTPUT = (
    BENCHMARK / "development" / "results" / "current_water8_eggbox_stage12"
)
R2SCAN_MESHES = (123, 173, 211, 240, 271)
AXES = ("x", "y", "z")


def build_study_cases():
    cases = [
        {"xc": "r2SCAN", "mesh": mesh, "axis": axis}
        for mesh in R2SCAN_MESHES
        for axis in AXES
    ]
    cases.extend(
        {"xc": "PBE", "mesh": 173, "axis": axis} for axis in AXES
    )
    return cases


def _effective_input(queue_path: Path):
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    defaults = dict(payload["defaults"])
    selected = next(
        job
        for job in payload["jobs"]
        if job.get("device") == "gpu"
        and job.get("df_backend") == "gdf"
        and job.get("target_integrator", "uniform") == "uniform"
    )
    defaults.update(selected)
    atom_path = Path(defaults["atom_file"])
    if not atom_path.is_absolute():
        atom_path = (queue_path.parent / atom_path).resolve()
    defaults["atom_file"] = str(atom_path)
    defaults["device"] = "gpu"
    defaults["df_backend"] = "gdf"
    defaults["target_integrator"] = "uniform"
    defaults.pop("target_multigrid_mesh", None)
    defaults["run_fd"] = False
    return defaults


def _case_stem(case):
    return f"{case['xc'].lower()}_mesh{case['mesh']}_{case['axis']}"


def _run_case(input_path, output_path, case):
    command = [
        sys.executable,
        str(DIAGNOSTIC),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--axis",
        case["axis"],
        "--xc",
        case["xc"],
        "--mesh",
        *([str(case["mesh"])] * 3),
        "--center-phase",
        "0.25",
        "--slope-step-bohr",
        "0.0001",
    ]
    subprocess.run(command, check=True)


def _case_summary(result):
    translation = result["analytic"][
        "translation_sum_vector_ha_per_bohr"
    ]
    translation_norm = sum(float(value) ** 2 for value in translation) ** 0.5
    return {
        "xc": result["xc"],
        "mesh": int(result["mesh"][0]),
        "axis": result["axis"],
        "grid_step_norm_bohr": float(result["grid_step_norm_bohr"]),
        "energy_peak_to_peak_ha": float(result["energy_peak_to_peak_ha"]),
        "worst_sampled_slope_ha_per_bohr": float(
            result["worst_sampled_rigid_translation_slope_ha_per_bohr"]
        ),
        "rms_sampled_slope_ha_per_bohr": float(
            result["rms_sampled_rigid_translation_slope_ha_per_bohr"]
        ),
        "center_translation_sum_norm_ha_per_bohr": translation_norm,
        "center_directional_analytic_ha_per_bohr": float(
            result["analytic_directional_translation_sum_ha_per_bohr"]
        ),
        "center_directional_fd_ha_per_bohr": float(
            result["small_step_rigid_translation_slope_ha_per_bohr"]
        ),
        "analytic_minus_fd_abs_ha_per_bohr": abs(
            float(result["analytic_minus_small_step_fd_ha_per_bohr"])
        ),
        "wall_time_s": float(result["wall_time_s"]),
    }


def _aggregate(rows):
    groups = {}
    for row in rows:
        key = (row["xc"], row["mesh"])
        groups.setdefault(key, []).append(row)
    aggregate = []
    for (xc, mesh), values in sorted(groups.items()):
        aggregate.append(
            {
                "xc": xc,
                "mesh": mesh,
                "maximum_axis_energy_peak_to_peak_ha": max(
                    value["energy_peak_to_peak_ha"] for value in values
                ),
                "maximum_axis_worst_sampled_slope_ha_per_bohr": max(
                    value["worst_sampled_slope_ha_per_bohr"]
                    for value in values
                ),
                "maximum_center_translation_sum_norm_ha_per_bohr": max(
                    value["center_translation_sum_norm_ha_per_bohr"]
                    for value in values
                ),
                "maximum_analytic_minus_fd_abs_ha_per_bohr": max(
                    value["analytic_minus_fd_abs_ha_per_bohr"]
                    for value in values
                ),
                "wall_time_s": sum(value["wall_time_s"] for value in values),
            }
        )
    return aggregate


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    queue_path = args.queue.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = output_root / "effective_input.json"
    input_path.write_text(
        json.dumps(_effective_input(queue_path), indent=2) + "\n",
        encoding="utf-8",
    )

    rows = []
    for case in build_study_cases():
        output_path = output_root / f"{_case_stem(case)}.json"
        if args.overwrite or not output_path.is_file():
            print(f"running {_case_stem(case)}", flush=True)
            _run_case(input_path, output_path, case)
        else:
            print(f"reusing {_case_stem(case)}", flush=True)
        rows.append(
            _case_summary(json.loads(output_path.read_text(encoding="utf-8")))
        )

    aggregate = _aggregate(rows)
    summary = {
        "schema": "pbc_dcdft_water8_eggbox_stage12_v1",
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "queue": str(queue_path),
        "case_count": len(rows),
        "cases": rows,
        "mesh_convergence": aggregate,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_root / "cases.csv", rows)
    _write_csv(output_root / "mesh_convergence.csv", aggregate)

    print("xc      mesh   max dE(Ha)   max slope(Ha/Bohr)   max |A-FD|")
    for row in aggregate:
        print(
            f"{row['xc']:<7} {row['mesh']:>4d} "
            f"{row['maximum_axis_energy_peak_to_peak_ha']:>12.5e} "
            f"{row['maximum_axis_worst_sampled_slope_ha_per_bohr']:>19.5e} "
            f"{row['maximum_analytic_minus_fd_abs_ha_per_bohr']:>12.5e}"
        )
    print(f"output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
