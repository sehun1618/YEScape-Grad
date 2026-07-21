#!/usr/bin/env python
"""Expand and run CPU/GPU x FFTDF/GDF/RSDF benchmark queues."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import evidence


HERE = Path(__file__).resolve().parent
CASE_RUNNER = HERE / "dcdft_benchmark.py"
KST = ZoneInfo("Asia/Seoul")
MANAGED_QUEUE_SCOPES = ("user", "development")
CASE_ARTIFACTS = (
    "run_input.json",
    "command.txt",
    "stdout.txt",
    "stderr.txt",
    "progress.json",
    "report.json",
    "attempt.json",
    "arrays",
)


def _managed_queue_scope(queue_path: Path) -> str | None:
    """Return the repository-managed scope for a queue path."""

    try:
        relative = queue_path.resolve().relative_to(HERE)
    except ValueError:
        return None
    parts = relative.parts
    if (
        len(parts) >= 3
        and parts[0] in MANAGED_QUEUE_SCOPES
        and parts[1] == "queues"
    ):
        return parts[0]
    return None


def _validate_managed_queue(
    queue_path: Path,
    payload: Mapping[str, Any],
    output_root: Path,
) -> str | None:
    """Keep user and development queues and outputs physically separate."""

    scope = _managed_queue_scope(queue_path)
    if scope is None:
        return None
    declared = str(payload.get("benchmark_scope", "")).strip().lower()
    if declared != scope:
        raise ValueError(
            f"{queue_path} must declare benchmark_scope={scope!r}, "
            f"got {declared!r}"
        )
    expected_root = (HERE / scope / "results").resolve()
    if not output_root.resolve().is_relative_to(expected_root):
        raise ValueError(
            f"{scope} queue output must stay under {expected_root}, "
            f"got {output_root}"
        )
    return scope


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def _new_run_identity(run_id: str) -> dict[str, str]:
    return {
        "case_run_id": str(run_id),
        "attempt_id": uuid.uuid4().hex,
        "started_at_kst": _now_kst(),
        "hostname": socket.gethostname(),
    }


def _process_start_ticks(pid: int) -> int | None:
    """Return Linux process start ticks, robust to spaces in ``comm``."""

    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    close = stat.rfind(")")
    if close < 0:
        return None
    fields_after_comm = stat[close + 1 :].split()
    try:
        return int(fields_after_comm[19])
    except (IndexError, ValueError):
        return None


def _attempt_is_active(attempt: Mapping[str, Any]) -> bool | None:
    identity = attempt.get("run_identity") or {}
    if identity.get("hostname") != socket.gethostname():
        return None
    try:
        pid = int(attempt["child_pid"])
    except (KeyError, TypeError, ValueError):
        return False
    observed_start = _process_start_ticks(pid)
    if observed_start is None:
        return False
    recorded_start = attempt.get("child_process_start_ticks")
    if recorded_start is None:
        return True
    try:
        return int(recorded_start) == observed_start
    except (TypeError, ValueError):
        return False


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    """Stop a child owned by this queue runner after local interruption."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _mark_progress_interrupted(
    case_dir: Path,
    run_identity: Mapping[str, Any],
    *,
    reason: str,
) -> None:
    path = case_dir / "progress.json"
    progress = _read_optional_json(path)
    if progress is None or progress.get("status") != "running":
        return
    context = progress.get("context") or {}
    if context.get("run_identity") != dict(run_identity):
        return
    previous_stage = progress.get("stage")
    progress["status"] = "interrupted"
    progress["stage"] = "interrupted"
    progress["stage_fraction"] = None
    progress["updated_at_kst"] = _now_kst()
    progress["details"] = {
        "previous_stage": previous_stage,
        "reason": str(reason),
    }
    progress.setdefault("history", []).append(
        {
            "stage": "interrupted",
            "elapsed_s": progress.get("elapsed_s"),
        }
    )
    _write_json(path, progress)


def _reconcile_previous_attempt(case_dir: Path) -> str | None:
    """Resolve a stale running attempt before queue skip/run decisions."""

    path = case_dir / "attempt.json"
    attempt = _read_optional_json(path)
    if attempt is None:
        return None
    status = str(attempt.get("status", "unknown"))
    if status != "running":
        return status
    active = _attempt_is_active(attempt)
    if active is not False:
        return "running"
    identity = dict(attempt.get("run_identity") or {})
    attempt["status"] = "interrupted"
    attempt["finished_at_kst"] = _now_kst()
    attempt["interruption_reason"] = "recorded child process is no longer active"
    _write_json(path, attempt)
    _mark_progress_interrupted(
        case_dir,
        identity,
        reason=attempt["interruption_reason"],
    )
    return "interrupted"


def _archive_case_artifacts(case_dir: Path) -> Path | None:
    existing = [case_dir / name for name in CASE_ARTIFACTS]
    existing = [path for path in existing if path.exists()]
    if not existing:
        return None
    attempt = _read_optional_json(case_dir / "attempt.json") or {}
    identity = attempt.get("run_identity") or {}
    attempt_id = _slug(str(identity.get("attempt_id", "legacy")))
    timestamp = datetime.now(KST).strftime("%Y%m%dT%H%M%S.%f%z")
    archive = case_dir / "attempts" / f"{timestamp}_{attempt_id}"
    archive.mkdir(parents=True, exist_ok=False)
    for source in existing:
        source.replace(archive / source.name)
    return archive


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
    """Expand each job's device/backend/target-integrator product."""

    defaults = dict(queue.get("defaults", {}))
    runs: list[dict[str, Any]] = []
    for job_index, raw_job in enumerate(queue.get("jobs", [])):
        if not bool(raw_job.get("enabled", True)):
            continue
        merged = {**defaults, **raw_job}
        devices = _as_choices(merged, "devices", "device", "cpu")
        backends = _as_choices(merged, "df_backends", "df_backend", "fftdf")
        integrators = _as_choices(
            merged,
            "target_integrators",
            "target_integrator",
            "uniform",
        )
        base_name = _slug(str(merged.get("name", f"job_{job_index:03d}")))
        for device in devices:
            for backend in backends:
                for integrator in integrators:
                    run = dict(merged)
                    for key in (
                        "devices",
                        "df_backends",
                        "target_integrators",
                        "enabled",
                    ):
                        run.pop(key, None)
                    run["device"] = device
                    run["df_backend"] = backend
                    run["target_integrator"] = integrator
                    suffix = (
                        f"_{integrator}" if integrator != "uniform" else ""
                    )
                    run["name"] = f"{base_name}_{device}_{backend}{suffix}"
                    runs.append(run)
    return runs


def _resolve_atom_file(run: dict[str, Any], queue_dir: Path) -> None:
    if run.get("atom_file") is None:
        return
    path = Path(run["atom_file"]).expanduser()
    if not path.is_absolute():
        path = (queue_dir / path).resolve()
    run["atom_file"] = str(path)


def _completed_report(path: Path, expected_fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        report = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    execution_status = report.get("execution_status", report.get("status"))
    report_evidence = report.get("evidence") or {}
    return (
        execution_status == "complete"
        and report_evidence.get("fingerprint_sha256")
        == str(expected_fingerprint)
    )


@contextmanager
def _case_lock(case_dir: Path):
    """Hold an exclusive case lock for the complete child lifetime."""

    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / ".queue.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def _run_case(
    run: dict[str, Any], case_dir: Path, run_id: str
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    archived = _archive_case_artifacts(case_dir)
    run_identity = _new_run_identity(run_id)
    run = dict(run)
    run["run_identity"] = run_identity
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
    attempt_path = case_dir / "attempt.json"
    with (case_dir / "stdout.txt").open("w", encoding="utf-8") as stdout, (
        case_dir / "stderr.txt"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=HERE,
            env=_case_environment(run),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        attempt = {
            "schema": "pbc_dcdft_benchmark_attempt_v1",
            "status": "running",
            "run_identity": run_identity,
            "child_pid": process.pid,
            "child_process_start_ticks": _process_start_ticks(process.pid),
            "archived_previous_artifacts": (
                str(archived) if archived is not None else None
            ),
            "evidence": run.get("evidence"),
        }
        _write_json(attempt_path, attempt)
        try:
            returncode = process.wait()
        except BaseException:
            _terminate_process(process)
            attempt["status"] = "interrupted"
            attempt["finished_at_kst"] = _now_kst()
            attempt["interruption_reason"] = "queue runner was interrupted"
            _write_json(attempt_path, attempt)
            _mark_progress_interrupted(
                case_dir,
                run_identity,
                reason=attempt["interruption_reason"],
            )
            raise
    report_path = case_dir / "report.json"
    report = _read_optional_json(report_path) or {}
    report_identity_matches = report.get("run_identity") == run_identity
    expected_fingerprint = (run.get("evidence") or {}).get(
        "fingerprint_sha256"
    )
    report_fingerprint = (report.get("evidence") or {}).get(
        "fingerprint_sha256"
    )
    report_fingerprint_matches = (
        expected_fingerprint is not None
        and report_fingerprint == expected_fingerprint
    )
    if not (report_identity_matches and report_fingerprint_matches):
        report = {}
    if report:
        status = str(
            report.get("execution_status", report.get("status", "failed"))
        )
    elif returncode < 0:
        status = "interrupted"
    else:
        status = "failed"
    if status == "interrupted":
        _mark_progress_interrupted(
            case_dir,
            run_identity,
            reason=f"benchmark child exited with return code {returncode}",
        )
    attempt["status"] = status
    attempt["finished_at_kst"] = _now_kst()
    attempt["returncode"] = int(returncode)
    attempt["report_identity_matches"] = bool(report_identity_matches)
    attempt["report_fingerprint_matches"] = bool(
        report_fingerprint_matches
    )
    _write_json(attempt_path, attempt)
    return {
        "returncode": returncode,
        "status": status,
        "report": report,
        "run_identity": run_identity,
    }


def _summary_row(
    run_id: str,
    case_dir: Path,
    report: Mapping[str, Any],
    *,
    queue_state: str,
):
    timing = report.get("timing_s") or {}
    accuracy = report.get("accuracy") or {}
    system = report.get("system") or {}
    settings = report.get("numerical_settings") or {}
    memory = report.get("memory") or {}
    vram = memory.get("vram_peak_tracking") or {}
    return {
        "run_id": run_id,
        "status": report.get("status", queue_state),
        "execution_status": report.get(
            "execution_status", report.get("status", queue_state)
        ),
        "accuracy_status": report.get("accuracy_status", "not_requested"),
        "queue_state": queue_state,
        "benchmark_scope": report.get("benchmark_scope"),
        "device": report.get("device"),
        "df_backend": report.get("df_backend"),
        "target_integrator": report.get("target_integrator", "uniform"),
        "xc": report.get("xc"),
        "dispersion": report.get("dispersion"),
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
        "net_gradient_norm_ha_per_bohr": (
            (report.get("gradient_summary") or {}).get(
                "translation_sum_norm_ha_per_bohr"
            )
        ),
        "fingerprint_sha256": (
            (report.get("evidence") or {}).get("fingerprint_sha256")
        ),
        "source_git_commit": (
            ((report.get("evidence") or {}).get("source") or {}).get(
                "git_commit"
            )
        ),
        "host_peak_rss_bytes": memory.get("host_peak_rss_bytes"),
        "peak_observed_cupy_pool_reserved_bytes": memory.get(
            "peak_observed_cupy_pool_reserved_bytes"
        ),
        "minimum_observed_gpu_free_device_bytes": memory.get(
            "minimum_observed_gpu_free_device_bytes"
        ),
        "peak_sampled_gpu_device_used_bytes": vram.get(
            "peak_observed_gpu_device_used_bytes"
        ),
        "peak_sampled_gpu_device_used_delta_from_baseline_bytes": vram.get(
            "peak_observed_gpu_device_used_delta_from_baseline_bytes"
        ),
        "peak_sampled_cupy_pool_reserved_bytes": vram.get(
            "peak_observed_cupy_pool_reserved_bytes"
        ),
        "vram_sample_count": vram.get("sample_count"),
        "report_path": str(case_dir / "report.json"),
    }


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_or_reuse_case(
    run: dict[str, Any],
    case_dir: Path,
    run_id: str,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    expected_fingerprint = run["evidence"]["fingerprint_sha256"]
    with _case_lock(case_dir) as acquired:
        if not acquired:
            print(f"[{run_id}] skip: case lock held by another queue")
            return {
                "report": {},
                "state": "skipped_locked",
                "run_identity": None,
            }

        report_path = case_dir / "report.json"
        lifecycle = _reconcile_previous_attempt(case_dir)
        if lifecycle == "running":
            attempt = _read_optional_json(case_dir / "attempt.json") or {}
            print(f"[{run_id}] skip: active attempt")
            return {
                "report": {},
                "state": "skipped_running",
                "run_identity": attempt.get("run_identity"),
            }
        if (
            _completed_report(report_path, expected_fingerprint)
            and not overwrite
        ):
            report = _read_json(report_path)
            print(f"[{run_id}] skip: matching complete report")
            return {
                "report": report,
                "state": "skipped_complete",
                "run_identity": report.get("run_identity"),
            }

        print(f"[{run_id}] run")
        outcome = _run_case(run, case_dir, run_id)
        print(f"[{run_id}] {outcome['status']}")
        return {
            "report": outcome["report"],
            "state": outcome["status"],
            "run_identity": outcome["run_identity"],
        }


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
    benchmark_scope = _validate_managed_queue(
        queue_path,
        queue,
        output_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    source_record = evidence.source_provenance()
    runs = expand_jobs(queue)
    for index, run in enumerate(runs):
        if benchmark_scope is not None:
            run["benchmark_scope"] = benchmark_scope
        _resolve_atom_file(run, queue_path.parent)
        run["evidence"] = evidence.build_case_evidence(
            run,
            source=source_record,
        )
        run_id = (
            f"{index:03d}_{_structure_slug(run)}_{_slug(run['name'])}"
        )
        if args.only and args.only not in run_id:
            continue
        case_dir = output_root / run_id
        report_path = case_dir / "report.json"
        outcome = _run_or_reuse_case(
            run,
            case_dir,
            run_id,
            overwrite=args.overwrite,
        )
        report = outcome["report"]
        state = outcome["state"]
        run_identity = outcome["run_identity"]
        manifest.append(
            {
                "run_id": run_id,
                "state": state,
                "device": run.get("device"),
                "df_backend": run.get("df_backend"),
                "target_integrator": run.get(
                    "target_integrator", "uniform"
                ),
                "run_identity": run_identity,
                "fingerprint_sha256": run["evidence"][
                    "fingerprint_sha256"
                ],
                "source_git_commit": source_record.get("git_commit"),
                "report_path": str(report_path),
                "progress_path": str(case_dir / "progress.json"),
            }
        )
        summary_rows.append(
            _summary_row(
                run_id,
                case_dir,
                report,
                queue_state=state,
            )
        )
        _write_json(
            output_root / "queue_manifest.json",
            {
                "queue": str(queue_path),
                "benchmark_scope": benchmark_scope,
                "output_root": str(output_root),
                "runs": manifest,
            },
        )
        _write_summary(output_root / "summary.csv", summary_rows)

    failures = sum(
        row.get("execution_status") != "complete"
        or row.get("accuracy_status") == "fail"
        for row in summary_rows
    )
    print(f"output: {output_root}")
    print(f"cases: {len(summary_rows)}, failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
