#!/usr/bin/env python
"""Show progress snapshots for one stress case or an output root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _progress_paths(root):
    if root.is_file():
        return [root] if root.name == "progress.json" else []
    direct = root/"progress.json"
    if direct.is_file():
        return [direct]
    return sorted(root.glob("*/progress.json"))


def _work_summary(payload):
    details = payload.get("details") or {}
    reused = details.get("checkpoint_reused_evaluations")
    computed = details.get("computed_evaluations")
    if reused is not None or computed is not None:
        return f"reuse={reused or 0} compute={computed or 0}"
    stage = str(payload.get("stage", ""))
    return "checkpoint" if "checkpoint_reused" in stage else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    root = args.path.expanduser().resolve()
    paths = _progress_paths(root)
    if not paths:
        raise SystemExit(f"no progress.json found under {root}")
    print(
        f"{'case':58} {'status':12} {'stage':34} "
        f"{'elapsed_s':>12} {'checkpoint work':>24}"
    )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"{path.parent.name[:58]:58} "
            f"{str(payload.get('status'))[:12]:12} "
            f"{str(payload.get('stage'))[:34]:34} "
            f"{float(payload.get('elapsed_s', 0.0)):12.1f} "
            f"{_work_summary(payload):>24}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
