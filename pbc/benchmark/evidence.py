"""Stable benchmark fingerprints and software provenance.

The queue runner uses this module before deciding whether an existing report
can be reused.  The case runner writes the same evidence into the final
report, which makes a timing artifact traceable to both its numerical input
and the exact local source tree that produced it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATHS = (
    "pbc/cpu/src",
    "pbc/cpu/pyproject.toml",
    "pbc/gpu/src",
    "pbc/gpu/pyproject.toml",
    "pbc/benchmark/dcdft_benchmark.py",
    "pbc/benchmark/run_benchmark_queue.py",
    "pbc/benchmark/evidence.py",
)
_TRANSIENT_CONFIG_KEYS = {"evidence", "run_identity"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _canonical_value(value.tolist())
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value with deterministic ordering."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _source_files(repo_root: Path):
    for relative in _SOURCE_PATHS:
        path = repo_root / relative
        if path.is_file():
            yield path
            continue
        if not path.is_dir():
            continue
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file():
                continue
            if "__pycache__" in candidate.parts or candidate.suffix == ".pyc":
                continue
            yield candidate


def source_tree_digest(repo_root: Path = REPO_ROOT) -> str:
    """Hash numerical package sources independently of benchmark outputs."""

    digest = hashlib.sha256()
    for path in _source_files(Path(repo_root)):
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _git_output(repo_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def source_provenance(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Describe the Git revision and source-only dirty state."""

    repo_root = Path(repo_root)
    status = _git_output(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *_SOURCE_PATHS,
    )
    tree_digest = source_tree_digest(repo_root)
    return {
        "repository_root": str(repo_root),
        "git_commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git_output(
            repo_root, "rev-parse", "--abbrev-ref", "HEAD"
        ),
        "source_tree_sha256": tree_digest,
        "source_dirty": bool(status),
        "source_dirty_state_sha256": (
            _sha256(status.encode("utf-8")) if status else None
        ),
    }


def build_case_evidence(
    config: Mapping[str, Any],
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the fingerprint used for report reuse and release evidence."""

    effective_config = {
        str(key): value
        for key, value in config.items()
        if key not in _TRANSIENT_CONFIG_KEYS
    }
    atom_file = effective_config.get("atom_file")
    atom_input = None
    if atom_file is not None:
        path = Path(atom_file).expanduser().resolve()
        atom_input = {
            "path": str(path),
            "sha256": _sha256(path.read_bytes()),
        }
    source_record = dict(source or source_provenance())
    fingerprint_payload = {
        "effective_config": effective_config,
        "atom_input": atom_input,
        "source": source_record,
    }
    return {
        "schema": "pbc_dcdft_case_evidence_v1",
        "fingerprint_sha256": _sha256(
            canonical_json_bytes(fingerprint_payload)
        ),
        "effective_config_sha256": _sha256(
            canonical_json_bytes(effective_config)
        ),
        "atom_input": atom_input,
        "source": source_record,
    }


def _package_version(distribution: str, module: str | None = None) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        if module is None:
            return None
    try:
        imported = __import__(module or distribution)
    except ImportError:
        return None
    return getattr(imported, "__version__", None)


def software_provenance(device: str) -> dict[str, Any]:
    """Collect versions needed to reproduce one benchmark process."""

    record: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "packages": {
            "numpy": _package_version("numpy", "numpy"),
            "pyscf": _package_version("pyscf", "pyscf"),
            "pyscf-pbc-dcdft": _package_version(
                "pyscf-pbc-dcdft", "pyscf_pbc_dcdft"
            ),
        },
    }
    if str(device).lower() != "gpu":
        return record

    record["packages"].update(
        {
            "cupy": _package_version("cupy", "cupy"),
            "gpu4pyscf": _package_version("gpu4pyscf", "gpu4pyscf"),
            "gpu4pyscf-pbc-dcdft": _package_version(
                "gpu4pyscf-pbc-dcdft", "gpu4pyscf_pbc_dcdft"
            ),
        }
    )
    try:
        import cupy as cp

        record["cuda"] = {
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        }
    except Exception as error:
        record["cuda"] = {
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return record
