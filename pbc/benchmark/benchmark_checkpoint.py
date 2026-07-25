"""Atomic, fingerprint-bound checkpoints for long benchmark cases.

The benchmark intentionally stores numerical state rather than pickled
PySCF/GPU4PySCF objects.  NPY arrays are portable across processes, while the
JSON manifest records byte checksums so interrupted or stale files are never
silently reused.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np


KST = ZoneInfo("Asia/Seoul")
SCHEMA = "pbc_dcdft_benchmark_checkpoint_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__cuda_array_interface__") and hasattr(value, "get"):
        return _json_value(value.get())
    return str(value)


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "__cuda_array_interface__") and hasattr(value, "get"):
        return np.asarray(value.get())
    if isinstance(value, (list, tuple)):
        return np.stack([_host_array(item) for item in value])
    return np.asarray(value)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CheckpointStore:
    """Manage one case's restart state under ``case/checkpoints``."""

    def __init__(
        self,
        root: Path,
        fingerprint: str,
        *,
        enabled: bool = True,
    ):
        self.root = Path(root)
        self.fingerprint = str(fingerprint)
        self.enabled = bool(enabled)
        self.manifest_path = self.root / "manifest.json"
        self._manifest = self._new_manifest()
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        existing = self._read_manifest()
        if (
            existing is not None
            and existing.get("schema") == SCHEMA
            and existing.get("fingerprint_sha256") == self.fingerprint
        ):
            self._manifest = existing
            return
        if any(self.root.iterdir()):
            shutil.rmtree(self.root)
            self.root.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _new_manifest(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "fingerprint_sha256": self.fingerprint,
            "created_at_kst": datetime.now(KST).isoformat(),
            "updated_at_kst": datetime.now(KST).isoformat(),
            "artifacts": {},
        }

    def _read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.is_file():
            return None
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _write_manifest(self) -> None:
        self._manifest["updated_at_kst"] = datetime.now(KST).isoformat()
        payload = (
            json.dumps(self._manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _atomic_bytes(self.manifest_path, payload)

    @staticmethod
    def _key(key: str) -> str:
        normalized = PurePosixPath(str(key))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(f"checkpoint key must be relative: {key!r}")
        text = normalized.as_posix().strip("/")
        if not text or text == ".":
            raise ValueError("checkpoint key cannot be empty")
        return text

    def _path(self, key: str, suffix: str) -> Path:
        return self.root / f"{self._key(key)}{suffix}"

    def _record(
        self,
        key: str,
        path: Path,
        *,
        kind: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        relative = path.relative_to(self.root).as_posix()
        entry = {
            "kind": kind,
            "path": relative,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "completed_at_kst": datetime.now(KST).isoformat(),
        }
        entry.update(dict(extra or {}))
        self._manifest.setdefault("artifacts", {})[self._key(key)] = entry
        self._write_manifest()

    def _validated_path(self, key: str, kind: str) -> Path | None:
        if not self.enabled:
            return None
        entry = self._manifest.get("artifacts", {}).get(self._key(key))
        if not isinstance(entry, Mapping) or entry.get("kind") != kind:
            return None
        path = self.root / str(entry.get("path", ""))
        try:
            path.resolve().relative_to(self.root.resolve())
        except (OSError, ValueError):
            return None
        if not path.is_file():
            return None
        try:
            if path.stat().st_size != int(entry.get("size_bytes", -1)):
                return None
            if _sha256(path) != entry.get("sha256"):
                return None
        except (OSError, TypeError, ValueError):
            return None
        return path

    def save_array(self, key: str, value: Any) -> Path | None:
        if not self.enabled:
            return None
        path = self._path(key, ".npy")
        array = _host_array(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self._record(
            key,
            path,
            kind="npy",
            extra={"shape": list(array.shape), "dtype": array.dtype.str},
        )
        return path

    def load_array(self, key: str) -> np.ndarray | None:
        path = self._validated_path(key, "npy")
        if path is None:
            return None
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return None

    def save_json(self, key: str, value: Mapping[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        path = self._path(key, ".json")
        payload = (
            json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _atomic_bytes(path, payload)
        self._record(key, path, kind="json")
        return path

    def load_json(self, key: str) -> dict[str, Any] | None:
        path = self._validated_path(key, "json")
        if path is None:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def describe(self) -> dict[str, Any]:
        artifacts = self._manifest.get("artifacts", {})
        return {
            "enabled": self.enabled,
            "directory": str(self.root),
            "fingerprint_sha256": self.fingerprint,
            "artifact_count": len(artifacts),
            "artifact_keys": sorted(artifacts),
        }
