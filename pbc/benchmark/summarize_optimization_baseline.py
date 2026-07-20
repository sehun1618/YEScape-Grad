#!/usr/bin/env python
"""Summarize repeated GDF/RSDF optimization baseline reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median, pstdev

import numpy as np


DEFAULT_ROOT = (
    Path(__file__).resolve().parent
    / "results"
    / "current_gdf_rsdf_optimization_baseline"
)


def _system_name(report):
    path = report["evidence"]["atom_input"]["path"]
    return Path(path).stem.lower()


def _response_time(report):
    timings = report["timing_s"]["gradient_internal_s"]
    if report["df_backend"] == "gdf":
        return float(timings["gdf_native_target_j_z_bilinear_fused"])
    return float(timings["rsdf_response_tensor_total"])


def _row(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise RuntimeError(f"incomplete report: {report_path}")
    timing = report["timing_s"]
    gradient_timing = timing["gradient_internal_s"]
    z_iterations = int(report["gradient_metadata"]["z_vector_iterations"])
    return {
        "system": _system_name(report),
        "df_backend": report["df_backend"],
        "run_name": report["run_name"],
        "energy_ha": float(report["energy_ha"]),
        "gradient_path": report["arrays"]["analytic_gradient"],
        "total_s": float(timing["workflow_without_fd_wall_s"]),
        "hf_scf_s": float(timing["method_internal_s"]["hf_scf"]),
        "gradient_s": float(timing["analytic_gradient_wall_s"]),
        "df_response_s": _response_time(report),
        "one_electron_s": float(gradient_timing["one_electron_total"]),
        "ax_z_s": float(gradient_timing["hf_ax_and_zvector"]),
        "target_explicit_s": float(gradient_timing["target_explicit_total"]),
        "z_iterations": z_iterations,
        "z_residual": float(
            report["gradient_metadata"]["z_vector_residual_max_abs"]
        ),
        "host_peak_rss_bytes": int(report["memory"]["host_peak_rss_bytes"]),
        "sampled_gpu_peak_bytes": int(
            report["memory"]["vram_peak_tracking"][
                "peak_observed_gpu_device_used_delta_from_baseline_bytes"
            ]
        ),
    }


def _stats(values):
    numbers = [float(value) for value in values]
    return {
        "median": median(numbers),
        "minimum": min(numbers),
        "maximum": max(numbers),
        "population_stddev": pstdev(numbers),
        "relative_spread": (max(numbers) - min(numbers)) / median(numbers),
    }


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["system"], row["df_backend"]), []).append(row)
    output = []
    timing_fields = (
        "total_s",
        "hf_scf_s",
        "gradient_s",
        "df_response_s",
        "one_electron_s",
        "ax_z_s",
        "target_explicit_s",
    )
    for (system, backend), values in sorted(groups.items()):
        reference_gradient = np.load(values[0]["gradient_path"])
        maximum_gradient_drift = 0.0
        for value in values[1:]:
            gradient = np.load(value["gradient_path"])
            maximum_gradient_drift = max(
                maximum_gradient_drift,
                float(np.max(np.abs(gradient - reference_gradient))),
            )
        output.append(
            {
                "system": system,
                "df_backend": backend,
                "repeat_count": len(values),
                "timing": {
                    field: _stats([value[field] for value in values])
                    for field in timing_fields
                },
                "maximum_energy_drift_ha": max(
                    value["energy_ha"] for value in values
                )
                - min(value["energy_ha"] for value in values),
                "maximum_gradient_drift_ha_per_bohr": maximum_gradient_drift,
                "z_iterations": sorted(
                    {value["z_iterations"] for value in values}
                ),
                "maximum_z_residual": max(
                    value["z_residual"] for value in values
                ),
                "median_host_peak_rss_bytes": int(
                    median([value["host_peak_rss_bytes"] for value in values])
                ),
                "median_sampled_gpu_peak_bytes": int(
                    median(
                        [value["sampled_gpu_peak_bytes"] for value in values]
                    )
                ),
            }
        )
    return output


def _write_csv(path, groups):
    rows = []
    for group in groups:
        rows.append(
            {
                "system": group["system"],
                "df_backend": group["df_backend"],
                "repeat_count": group["repeat_count"],
                "median_total_s": group["timing"]["total_s"]["median"],
                "median_hf_scf_s": group["timing"]["hf_scf_s"]["median"],
                "median_gradient_s": group["timing"]["gradient_s"]["median"],
                "median_df_response_s": group["timing"]["df_response_s"][
                    "median"
                ],
                "median_one_electron_s": group["timing"]["one_electron_s"][
                    "median"
                ],
                "median_ax_z_s": group["timing"]["ax_z_s"]["median"],
                "median_target_explicit_s": group["timing"][
                    "target_explicit_s"
                ]["median"],
                "total_relative_spread": group["timing"]["total_s"][
                    "relative_spread"
                ],
                "maximum_energy_drift_ha": group["maximum_energy_drift_ha"],
                "maximum_gradient_drift_ha_per_bohr": group[
                    "maximum_gradient_drift_ha_per_bohr"
                ],
                "z_iterations": ";".join(
                    str(value) for value in group["z_iterations"]
                ),
                "maximum_z_residual": group["maximum_z_residual"],
                "median_host_peak_rss_bytes": group[
                    "median_host_peak_rss_bytes"
                ],
                "median_sampled_gpu_peak_bytes": group[
                    "median_sampled_gpu_peak_bytes"
                ],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", nargs="?", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.results_root.expanduser().resolve()
    report_paths = sorted(root.glob("*/report.json"))
    if not report_paths:
        raise FileNotFoundError(f"no report.json files below {root}")
    rows = [_row(path) for path in report_paths]
    groups = summarize(rows)
    payload = {
        "schema": "pbc_dcdft_gdf_rsdf_optimization_baseline_v1",
        "results_root": str(root),
        "groups": groups,
    }
    (root / "optimization_baseline.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(root / "optimization_baseline.csv", groups)
    print("system   df    total(s) response(s) spread   max|dG|")
    for group in groups:
        print(
            f"{group['system']:<8} {group['df_backend']:<5} "
            f"{group['timing']['total_s']['median']:>8.3f} "
            f"{group['timing']['df_response_s']['median']:>11.3f} "
            f"{group['timing']['total_s']['relative_spread']:>7.2%} "
            f"{group['maximum_gradient_drift_ha_per_bohr']:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
