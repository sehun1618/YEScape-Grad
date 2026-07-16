#!/usr/bin/env python
"""Expand and run CPU/GPU x FFTDF/GDF/RSDF benchmark queues."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
CASE_RUNNER = HERE / "dcdft_benchmark.py"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_.")
    return result or "benchmark"


def _structure_slug(run: Mapping[str, Any]) -> str:
    """Return the XYZ filename stem used in a result-directory name."""

    atom_file = run.get("atom_file")
    if atom_file is not None:
        return _slug(Path(atom_file).stem)
    return "inline_atoms"


def _as_choices(job: Mapping[str, Any], plural: str, singular: str, default: str):
    if plural in job:
        values = job[plural]
    elif singular in job:
        values = [job[singular]]
    else:
        values = [default]
    if isinstance(values, str):
        values = [values]
    return [str(value).lower() for value in values]


def expand_jobs(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand each job's device/backend Cartesian product."""

    defaults = dict(queue.get("defaults", {}))
    runs: list[dict[str, Any]] = []
    for job_index, raw_job in enumerate(queue.get("jobs", [])):
        if not bool(raw_job.get("enabled", True)):
            continue
        merged = {**defaults, **raw_job}
        devices = _as_choices(merged, "devices", "device", "cpu")
        backends = _as_choices(merged, "df_backends", "df_backend", "fftdf")
        base_name = _slug(str(merged.get("name", f"job_{job_index:03d}")))
        for device in devices:
            for backend in backends:
                run = dict(merged)
                for key in ("devices", "df_backends", "enabled"):
                    run.pop(key, None)
                run["device"] = device
                run["df_backend"] = backend
                run["name"] = f"{base_name}_{device}_{backend}"
                runs.append(run)
    return runs


def _resolve_atom_file(run: dict[str, Any], queue_dir: Path) -> None:
    if run.get("atom_file") is None:
        return
    path = Path(run["atom_file"]).expanduser()
    if not path.is_absolute():
        path = (queue_dir / path).resolve()
    run["atom_file"] = str(path)


def _completed_report(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return _read_json(path).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def _case_environment(run: Mapping[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    threads = str(int(run.get("num_threads", 1)))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = threads
    if run.get("gpu_id") is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(run["gpu_id"])
    return environment


def _run_case(run: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / "run_input.json"
    _write_json(input_path, run)
    command = [
        sys.executable,
        str(CASE_RUNNER),
        "--input",
        str(input_path),
        "--output-dir",
        str(case_dir),
    ]
    (case_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    with (case_dir / "stdout.txt").open("w", encoding="utf-8") as stdout, (
        case_dir / "stderr.txt"
    ).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=HERE,
            env=_case_environment(run),
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    report_path = case_dir / "report.json"
    report = _read_json(report_path) if report_path.is_file() else {}
    return {
        "returncode": completed.returncode,
        "status": report.get("status", "failed"),
        "report": report,
    }


def _summary_row(run_id: str, case_dir: Path, report: Mapping[str, Any]):
    timing = report.get("timing_s") or {}
    accuracy = report.get("accuracy") or {}
    system = report.get("system") or {}
    settings = report.get("numerical_settings") or {}
    memory = report.get("memory") or {}
    return {
        "run_id": run_id,
        "status": report.get("status"),
        "device": report.get("device"),
        "df_backend": report.get("df_backend"),
        "xc": report.get("xc"),
        "natoms": system.get("natoms"),
        "nkpts": settings.get("nkpts"),
        "energy_ha": report.get("energy_ha"),
        "setup_wall_s": timing.get("setup_wall_s"),
        "energy_wall_s": timing.get("energy_wall_s"),
        "analytic_gradient_wall_s": timing.get("analytic_gradient_wall_s"),
        "analytic_total_wall_s": timing.get("analytic_total_wall_s"),
        "fd_wall_s": timing.get("fd_wall_s"),
        "fd_components": accuracy.get("component_count"),
        "max_abs_error_ha_per_bohr": accuracy.get(
            "max_abs_error_ha_per_bohr"
        ),
        "rms_error_ha_per_bohr": accuracy.get("rms_error_ha_per_bohr"),
        "host_peak_rss_bytes": memory.get("host_peak_rss_bytes"),
        "peak_observed_cupy_pool_reserved_bytes": memory.get(
            "peak_observed_cupy_pool_reserved_bytes"
        ),
        "minimum_observed_gpu_free_device_bytes": memory.get(
            "minimum_observed_gpu_free_device_bytes"
        ),
        "report_path": str(case_dir / "report.json"),
    }


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="override queue output_root; relative paths use the queue directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun cases whose report already has status=complete",
    )
    parser.add_argument(
        "--only",
        help="run only expanded case names containing this text",
    )
    args = parser.parse_args()

    queue_path = args.queue.expanduser().resolve()
    queue = _read_json(queue_path)
    output_value = args.output_root or queue.get(
        "output_root", f"results/{queue_path.stem}"
    )
    output_root = Path(output_value).expanduser()
    if not output_root.is_absolute():
        output_root = (queue_path.parent / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    runs = expand_jobs(queue)
    for index, run in enumerate(runs):
        _resolve_atom_file(run, queue_path.parent)
        run_id = (
            f"{index:03d}_{_structure_slug(run)}_{_slug(run['name'])}"
        )
        if args.only and args.only not in run_id:
            continue
        case_dir = output_root / run_id
        report_path = case_dir / "report.json"
        if _completed_report(report_path) and not args.overwrite:
            report = _read_json(report_path)
            state = "skipped_complete"
            print(f"[{run_id}] skip: complete")
        else:
            print(f"[{run_id}] run")
            outcome = _run_case(run, case_dir)
            report = outcome["report"]
            state = outcome["status"]
            print(f"[{run_id}] {state}")
        manifest.append(
            {
                "run_id": run_id,
                "state": state,
                "device": run.get("device"),
                "df_backend": run.get("df_backend"),
                "report_path": str(report_path),
                "progress_path": str(case_dir / "progress.json"),
            }
        )
        summary_rows.append(_summary_row(run_id, case_dir, report))
        _write_json(
            output_root / "queue_manifest.json",
            {
                "queue": str(queue_path),
                "output_root": str(output_root),
                "runs": manifest,
            },
        )
        _write_summary(output_root / "summary.csv", summary_rows)

    failures = sum(row.get("status") != "complete" for row in summary_rows)
    print(f"output: {output_root}")
    print(f"cases: {len(summary_rows)}, failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
