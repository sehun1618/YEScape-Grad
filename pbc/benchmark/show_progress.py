#!/usr/bin/env python
"""Show live progress snapshots for one benchmark case or a queue directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


def _duration(seconds: Any) -> str:
    if seconds is None:
        return "-"
    value = max(0, int(float(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _percentage(payload: Mapping[str, Any]) -> str:
    fraction = payload.get("stage_fraction")
    if fraction is None:
        return ""
    return f" {100.0 * float(fraction):6.2f}%"


def _counter(payload: Mapping[str, Any]) -> str:
    details = payload.get("details") or {}
    completed = details.get("completed")
    total = details.get("total")
    if completed is None or total is None:
        return ""
    return f" {completed}/{total}"


def _eta(payload: Mapping[str, Any]) -> str:
    fraction = payload.get("stage_fraction")
    if fraction is None or not 0.0 < float(fraction) < 1.0:
        return ""
    elapsed = float(payload.get("stage_elapsed_s", 0.0))
    remaining = elapsed * (1.0 - float(fraction)) / float(fraction)
    return f" eta={_duration(remaining)}"


def format_progress(path: Path, payload: Mapping[str, Any]) -> str:
    """Format one compact status line without requiring external tools."""

    context = payload.get("context") or {}
    name = context.get("run_name") or path.parent.name
    status = payload.get("status", "unknown")
    stage = payload.get("stage", "unknown")
    route = "/".join(
        str(value)
        for value in (context.get("device"), context.get("df_backend"))
        if value
    )
    route_text = f" {route}" if route else ""
    details = payload.get("details") or {}
    matrix = details.get("matrix_gib")
    matrix_text = "" if matrix is None else f" matrix={float(matrix):.2f} GiB"
    return (
        f"[{status}] {name}{route_text}: {stage}"
        f"{_counter(payload)}{_percentage(payload)}"
        f" elapsed={_duration(payload.get('elapsed_s'))}"
        f" stage={_duration(payload.get('stage_elapsed_s'))}"
        f"{_eta(payload)}{matrix_text}"
    )


def find_progress_files(path: Path) -> list[Path]:
    """Resolve a progress file, case directory, or queue result directory."""

    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return [resolved]
    direct = resolved / "progress.json"
    if direct.is_file():
        return [direct]
    if not resolved.exists():
        return []
    return sorted(resolved.rglob("progress.json"))


def read_progress(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_process_lines(path: Path) -> list[str]:
    """Find benchmark processes launched before progress callbacks existed."""

    needle = str(path.expanduser().resolve()).encode()
    matches = []
    for process_dir in Path("/proc").glob("[0-9]*"):
        pid = int(process_dir.name)
        if pid == os.getpid():
            continue
        try:
            command = (process_dir / "cmdline").read_bytes()
        except OSError:
            continue
        if b"dcdft_benchmark.py" not in command or needle not in command:
            continue
        result = subprocess.run(
            [
                "ps", "-p", str(pid),
                "-o", "pid=,etime=,%cpu=,%mem=,stat=,cmd=",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            matches.append(result.stdout.strip())
    return matches


def _show_legacy_status(path: Path) -> bool:
    lines = _legacy_process_lines(path)
    if lines:
        print("[running/legacy] detailed stage unavailable; process is active")
        for line in lines:
            print(f"  {line}")
        return False

    report_path = path.expanduser().resolve() / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        else:
            status = report.get("status", "unknown")
            print(f"[{status}/legacy] report={report_path}")
            return status in {"complete", "failed"}

    print(f"progress.json not found under: {path.expanduser().resolve()}")
    return False


def show(path: Path) -> bool:
    files = find_progress_files(path)
    if not files:
        return _show_legacy_status(path)

    payloads = []
    for progress_path in files:
        try:
            payload = read_progress(progress_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[unreadable] {progress_path}: {exc}")
            continue
        payloads.append(payload)
        print(format_progress(progress_path, payload))
    return bool(payloads) and all(
        payload.get("status") in {"complete", "failed"}
        for payload in payloads
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="progress.json, one case directory, or queue result directory",
    )
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="refresh until every discovered calculation completes or fails",
    )
    args = parser.parse_args()
    if args.watch is not None and args.watch <= 0.0:
        parser.error("--watch interval must be positive")

    if args.watch is None:
        show(args.path)
        return 0

    while True:
        finished = show(args.path)
        if finished:
            return 0
        time.sleep(args.watch)
        print()


if __name__ == "__main__":
    raise SystemExit(main())
