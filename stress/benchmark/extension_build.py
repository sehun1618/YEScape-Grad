"""Build the local stress C extension for the active Python ABI."""

from __future__ import annotations

import ctypes
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import subprocess
import sys

import pyscf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def stress_extension_path():
    package = ROOT/"stress"/"cpu"/"src"/"pyscf_pbc_dcdft_stress"
    for suffix in EXTENSION_SUFFIXES:
        candidate = package/f"_sr3c_strain{suffix}"
        if candidate.is_file():
            return candidate
    return None


def ensure_stress_extension():
    existing = stress_extension_path()
    source_root = ROOT/"stress"/"cpu"
    sources = [source_root/"setup.py"]
    sources.extend(
        (source_root/"src"/"pyscf_pbc_dcdft_stress"/"c_ext").glob("*.c")
    )
    source_mtime = max(path.stat().st_mtime for path in sources)
    if (
        existing is not None
        and existing.stat().st_mtime >= source_mtime
        and _has_current_symbols(existing)
    ):
        return existing
    reason = "missing" if existing is None else "stale"
    print(
        f"BUILD stress C extension for the active Python environment ({reason})",
        flush=True,
    )
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
        cwd=str(source_root), check=True,
    )
    built = stress_extension_path()
    if built is None:
        raise RuntimeError(
            "stress C extension build completed but _sr3c_strain was not found"
        )
    if not _has_current_symbols(built):
        raise RuntimeError(
            "rebuilt stress C extension is missing the fused Fourier symbol"
        )
    return built


def _has_current_symbols(path):
    """Check the stress extension ABI without importing its Python module."""

    pyscf_lib = Path(pyscf.__file__).resolve().parent/"lib"
    dependencies = (
        pyscf_lib/"deps"/"lib"/"libcint.so",
        pyscf_lib/"libcgto.so",
    )
    try:
        for dependency in dependencies:
            ctypes.CDLL(str(dependency), mode=ctypes.RTLD_GLOBAL)
        extension = ctypes.CDLL(str(path))
        getattr(extension, "DCDT_ft_ovlp_ipij_sph")
        getattr(extension, "DCDT_ft_ovlp_ipij_cart")
    except (AttributeError, OSError):
        return False
    return True
