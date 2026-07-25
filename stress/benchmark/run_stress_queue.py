#!/usr/bin/env python
"""Expand and run CPU/GPU GDF/RSDF analytic-stress benchmark queues."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

from extension_build import ensure_stress_extension


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASE_RUNNER = HERE / "stress_benchmark.py"
KST = ZoneInfo("Asia/Seoul")
SCOPES = {"user", "development"}
CASE_ARTIFACTS = (
    "run_input.json", "command.txt", "stdout.txt", "stderr.txt",
    "progress.json", "report.json", "attempt.json", "arrays",
    "mlip_frame.json", "mlip_frame.xyz",
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _slug(value):
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_.")
    return result or "benchmark"


def _choices(mapping, plural, singular, default):
    value = mapping.get(plural, mapping.get(singular, default))
    if isinstance(value, str):
        value = [value]
    return [str(item).lower() for item in value]


def expand_jobs(queue):
    defaults = dict(queue.get("defaults", {}))
    expanded = []
    for index, raw in enumerate(queue.get("jobs", [])):
        if not bool(raw.get("enabled", True)):
            continue
        merged = {**defaults, **raw}
        devices = _choices(merged, "devices", "device", ["cpu"])
        backends = _choices(merged, "df_backends", "df_backend", ["gdf"])
        modes = _choices(
            merged, "calculation_modes", "calculation_mode", ["stress_only"]
        )
        base = _slug(merged.get("name", f"job_{index:03d}"))
        for device in devices:
            for backend in backends:
                for mode in modes:
                    run = dict(merged)
                    for key in (
                        "devices", "df_backends", "calculation_modes", "enabled",
                    ):
                        run.pop(key, None)
                    run.update({
                        "device": device,
                        "df_backend": backend,
                        "calculation_mode": mode,
                        "name": f"{base}_{device}_{backend}_{mode}",
                    })
                    expanded.append(run)
    return expanded


def _managed_scope(queue_path):
    try:
        relative = queue_path.resolve().relative_to(HERE)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) >= 3 and parts[0] in SCOPES and parts[1] == "queues":
        return parts[0]
    return None


def _resolve_output_root(queue_path, payload, override):
    value = override if override is not None else payload.get("output_root")
    if value is None:
        value = f"../results/{queue_path.stem}"
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (queue_path.parent/path).resolve()
    scope = _managed_scope(queue_path)
    if scope is not None:
        declared = str(payload.get("benchmark_scope", "")).lower()
        if declared != scope:
            raise ValueError(
                f"{queue_path} must declare benchmark_scope={scope!r}"
            )
        expected = (HERE/scope/"results").resolve()
        if not path.is_relative_to(expected):
            raise ValueError(f"{scope} output must stay under {expected}")
    return path, scope


def _resolve_atom_file(run, queue_dir):
    value = run.get("atom_file")
    if value is None:
        return
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (queue_dir/path).resolve()
    run["atom_file"] = str(path)


def _source_digest():
    digest = hashlib.sha256()
    roots = (
        ROOT/"stress"/"cpu"/"src", ROOT/"stress"/"gpu"/"src",
        ROOT/"pbc"/"cpu"/"src", ROOT/"pbc"/"gpu"/"src",
    )
    files = [
        CASE_RUNNER,
        HERE/"extension_build.py",
        HERE/"run_stress_queue.py",
        ROOT/"pbc"/"benchmark"/"benchmark_checkpoint.py",
        ROOT/"pbc"/"benchmark"/"dcdft_benchmark.py",
        ROOT/"pbc"/"benchmark"/"evidence.py",
    ]
    for root in roots:
        files.extend(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix in {".py", ".c", ".h"}
        )
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _input_fingerprint(run, source_digest):
    normalized = {
        key: value for key, value in run.items()
        if key not in {"run_identity", "input_fingerprint"}
    }
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    atom_file = normalized.get("atom_file")
    if atom_file is not None:
        digest.update(Path(atom_file).read_bytes())
    digest.update(source_digest.encode("ascii"))
    return digest.hexdigest()


def _read_optional(path):
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _attempt_is_active(case_dir):
    attempt = _read_optional(case_dir/"attempt.json")
    if attempt is None or attempt.get("status") != "running":
        return False
    identity = attempt.get("run_identity") or {}
    if identity.get("hostname") != socket.gethostname():
        return True
    try:
        os.kill(int(attempt["child_pid"]), 0)
    except (KeyError, TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _archive(case_dir):
    existing = [case_dir/name for name in CASE_ARTIFACTS]
    existing = [path for path in existing if path.exists()]
    if not existing:
        return None
    stamp = datetime.now(KST).strftime("%Y%m%dT%H%M%S.%f%z")
    target = case_dir/"attempts"/f"{stamp}_{uuid.uuid4().hex}"
    target.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), str(target/path.name))
    return target


def _summary_row(index, run, case_dir, status, returncode=None):
    report = _read_optional(case_dir/"report.json") or {}
    timing = report.get("timing_s") or {}
    accuracy = report.get("accuracy") or {}
    checkpoint = report.get("checkpoint") or {}
    memory = report.get("memory") or {}
    vram = memory.get("vram_peak_tracking") or {}
    return {
        "index": index,
        "name": run["name"],
        "status": status,
        "returncode": returncode,
        "device": run["device"],
        "df_backend": run["df_backend"],
        "calculation_mode": run["calculation_mode"],
        "energy_ha": report.get("energy_ha"),
        "gradient_wall_s": timing.get("analytic_gradient_wall_s"),
        "stress_wall_s": timing.get("analytic_stress_wall_s"),
        "analytic_total_wall_s": timing.get("analytic_total_wall_s"),
        "stress_fd_wall_s": timing.get("stress_fd_wall_s"),
        "current_attempt_analytic_total_wall_s": timing.get(
            "current_attempt_analytic_total_wall_s"
        ),
        "stress_fd_max_abs_error": accuracy.get(
            "max_abs_analytic_minus_fd_ha_per_bohr3"
        ),
        "checkpoint_gradient_reused": checkpoint.get("gradient_reused"),
        "checkpoint_stress_reused": checkpoint.get("stress_reused"),
        "host_peak_rss_bytes": memory.get("host_peak_rss_bytes"),
        "peak_gpu_device_used_bytes": vram.get(
            "peak_observed_gpu_device_used_bytes"
        ),
        "peak_cupy_pool_reserved_bytes": vram.get(
            "peak_observed_cupy_pool_reserved_bytes"
        ),
        "case_dir": str(case_dir),
    }


def _write_summary(path, rows):
    fields = list(rows[0]) if rows else ["index", "name", "status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    queue_path = args.queue.expanduser().resolve()
    payload = _read_json(queue_path)
    output_root, scope = _resolve_output_root(
        queue_path, payload, args.output_root
    )
    output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_root/".queue.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            f"another stress queue runner is using {output_root}"
        ) from error
    source_digest = _source_digest()
    runs = list(enumerate(expand_jobs(payload)))
    if args.only:
        runs = [
            (index, run) for index, run in runs
            if args.only.lower() in run["name"].lower()
        ]
    rows = []
    manifest = {
        "queue": str(queue_path),
        "output_root": str(output_root),
        "benchmark_scope": scope,
        "source_digest": source_digest,
        "started_at_kst": datetime.now(KST).isoformat(),
        "cases": [],
    }
    if not args.dry_run and runs:
        manifest["stress_extension"] = str(ensure_stress_extension())

    for index, run in runs:
        run = dict(run)
        _resolve_atom_file(run, queue_path.parent)
        run["benchmark_scope"] = scope or payload.get("benchmark_scope")
        fingerprint = _input_fingerprint(run, source_digest)
        run["input_fingerprint"] = fingerprint
        structure = _slug(Path(run.get("atom_file", "inline_atoms")).stem)
        case_dir = output_root/f"{index:03d}_{structure}_{run['name']}"
        if _attempt_is_active(case_dir):
            print(f"SKIP {run['name']}: recorded child process is still active")
            rows.append(_summary_row(index, run, case_dir, "skipped_running"))
            manifest["cases"].append(rows[-1])
            continue
        existing = _read_optional(case_dir/"report.json")
        same_complete = (
            existing is not None
            and existing.get("execution_status") == "complete"
            and existing.get("input_fingerprint") == fingerprint
        )
        if same_complete and not args.overwrite:
            print(f"SKIP {run['name']}: matching completed case")
            rows.append(_summary_row(index, run, case_dir, "skipped_complete", 0))
            manifest["cases"].append(rows[-1])
            continue

        if args.dry_run:
            print(f"DRY  {run['name']} -> {case_dir}")
            rows.append(_summary_row(index, run, case_dir, "dry_run"))
            manifest["cases"].append(rows[-1])
            continue

        case_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            shutil.rmtree(case_dir/"checkpoints", ignore_errors=True)
        _archive(case_dir)
        identity = {
            "attempt_id": uuid.uuid4().hex,
            "started_at_kst": datetime.now(KST).isoformat(),
            "hostname": socket.gethostname(),
        }
        run["run_identity"] = identity
        input_path = case_dir/"run_input.json"
        _write_json(input_path, run)
        command = [
            sys.executable, str(CASE_RUNNER), "--input", str(input_path),
            "--output-dir", str(case_dir),
        ]
        (case_dir/"command.txt").write_text(
            shlex.join(command)+"\n", encoding="utf-8"
        )
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(run.get("num_threads", 1))
        environment["MKL_NUM_THREADS"] = str(run.get("num_threads", 1))
        if run["device"] == "gpu":
            environment["CUDA_VISIBLE_DEVICES"] = str(run.get("gpu_id", 0))
        print(f"RUN  {run['name']} -> {case_dir}")
        with (case_dir/"stdout.txt").open("w", encoding="utf-8") as stdout, (
            case_dir/"stderr.txt"
        ).open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command, cwd=str(HERE), env=environment,
                stdout=stdout, stderr=stderr, text=True,
            )
            _write_json(case_dir/"attempt.json", {
                "status": "running", "child_pid": process.pid,
                "run_identity": identity,
            })
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        report = _read_optional(case_dir/"report.json") or {}
        execution = report.get("execution_status")
        status = "complete" if returncode == 0 and execution == "complete" else "failed"
        _write_json(case_dir/"attempt.json", {
            "status": status,
            "returncode": returncode,
            "finished_at_kst": datetime.now(KST).isoformat(),
            "run_identity": identity,
        })
        row = _summary_row(index, run, case_dir, status, returncode)
        rows.append(row)
        manifest["cases"].append(row)
        _write_summary(output_root/"summary.csv", rows)
        _write_json(output_root/"queue_manifest.json", manifest)

    manifest["finished_at_kst"] = datetime.now(KST).isoformat()
    _write_summary(output_root/"summary.csv", rows)
    _write_json(output_root/"queue_manifest.json", manifest)
    failures = [row for row in rows if row["status"] == "failed"]
    print(f"Completed queue: {len(rows)-len(failures)}/{len(rows)} cases without execution failure")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
