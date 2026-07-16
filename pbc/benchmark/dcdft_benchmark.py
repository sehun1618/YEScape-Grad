#!/usr/bin/env python
"""Benchmark one restricted PBC DC-DFT energy/gradient calculation.

The benchmark deliberately uses only the public APIs of ``pyscf-pbc-dcdft``
and ``gpu4pyscf-pbc-dcdft``.  Analytic gradients and optional full-energy
central finite differences therefore exercise the same user-visible route.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
for source_dir in (
    WORKSPACE / "cpu" / "src",
    WORKSPACE / "gpu" / "src",
):
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


AXIS_NAMES = ("x", "y", "z")
SUPPORTED_DEVICES = {"cpu", "gpu"}
SUPPORTED_DF_BACKENDS = {"fftdf", "gdf", "rsdf"}
KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True)
class ResolvedSystem:
    """Geometry and lattice normalized to the package's atomic units."""

    symbols: tuple[str, ...]
    coordinates_bohr: np.ndarray
    lattice_bohr: np.ndarray

    @property
    def natoms(self) -> int:
        return len(self.symbols)


@dataclass(frozen=True)
class EnergyRun:
    """One complete source-HF plus target-DFA energy evaluation."""

    energy_ha: float
    method: Any
    setup_wall_time_s: float
    wall_time_s: float


def _json_value(value: Any) -> Any:
    """Convert NumPy/CuPy and package metadata to JSON-compatible objects."""

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
    if hasattr(value, "get") and value.__class__.__module__.startswith("cupy"):
        return _json_value(value.get())
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a stable, human-readable JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class ProgressReporter:
    """Write an atomic, throttled snapshot of a running benchmark case."""

    def __init__(
        self,
        path: Path,
        *,
        context: Mapping[str, Any] | None = None,
        minimum_write_interval_s: float = 2.0,
    ):
        self.path = Path(path)
        self.context = dict(context or {})
        self.minimum_write_interval_s = float(minimum_write_interval_s)
        self.started_at = perf_counter()
        self.stage_started_at = self.started_at
        self.last_write_at = float("-inf")
        self.stage: str | None = None
        self.details: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.last_write_error: str | None = None

    def __call__(self, stage: str, **details: Any) -> None:
        self.update(stage, **details)

    def update(self, stage: str, *, force: bool = False, **details: Any) -> None:
        now = perf_counter()
        stage = str(stage)
        stage_changed = stage != self.stage
        if stage_changed:
            self.stage = stage
            self.stage_started_at = now
            self.history.append(
                {
                    "stage": stage,
                    "elapsed_s": now - self.started_at,
                }
            )
        self.details = dict(details)
        if (
            not force
            and not stage_changed
            and now - self.last_write_at < self.minimum_write_interval_s
        ):
            return
        self._write(status="running", now=now)

    def complete(self, **details: Any) -> None:
        now = perf_counter()
        self.stage = "complete"
        self.details = dict(details)
        self.history.append(
            {"stage": "complete", "elapsed_s": now - self.started_at}
        )
        self._write(status="complete", now=now)

    def fail(self, exc: BaseException) -> None:
        now = perf_counter()
        failed_stage = self.stage
        self.stage = "failed"
        self.details = {
            "failed_stage": failed_stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        self.history.append(
            {"stage": "failed", "elapsed_s": now - self.started_at}
        )
        self._write(status="failed", now=now)

    def _write(self, *, status: str, now: float) -> None:
        completed = self.details.get("completed")
        total = self.details.get("total")
        fraction = None
        if completed is not None and total not in (None, 0):
            fraction = float(completed) / float(total)
        payload = {
            "schema": "pbc_dcdft_benchmark_progress_v1",
            "status": status,
            "stage": self.stage,
            "elapsed_s": now - self.started_at,
            "stage_elapsed_s": now - self.stage_started_at,
            "stage_fraction": fraction,
            "updated_at_kst": datetime.now(KST).isoformat(),
            "timezone": "Asia/Seoul",
            "context": self.context,
            "details": self.details,
            "history": self.history,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(_json_value(payload), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
            self.last_write_at = now
            self.last_write_error = None
        except OSError as exc:
            self.last_write_error = str(exc)
            temporary.unlink(missing_ok=True)


def _unit_scale_to_bohr(unit: str) -> float:
    from pyscf.lib import param

    normalized = str(unit).strip().lower().replace("angstroms", "angstrom")
    if normalized in {"bohr", "au", "a.u."}:
        return 1.0
    if normalized in {"angstrom", "ang", "a"}:
        return 1.0 / float(param.BOHR)
    raise ValueError(f"unsupported length unit {unit!r}; use Bohr or Angstrom")


def _read_xyz(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not lines:
        raise ValueError(f"empty XYZ file: {path}")
    try:
        natoms = int(lines[0])
    except ValueError as exc:
        raise ValueError(f"first XYZ line must be an atom count: {path}") from exc
    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"XYZ file declares {natoms} atoms but is incomplete: {path}")

    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for line_number, line in enumerate(atom_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"invalid XYZ atom line {line_number}: {line!r}")
        symbols.append(fields[0])
        coordinates.append([float(value) for value in fields[1:4]])
    return tuple(symbols), np.asarray(coordinates, dtype=float)


def _read_inline_atoms(atom_spec: Iterable[Any]) -> tuple[tuple[str, ...], np.ndarray]:
    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for index, atom in enumerate(atom_spec):
        if isinstance(atom, Mapping):
            symbol = atom.get("element", atom.get("symbol"))
            xyz = atom.get("xyz", atom.get("coordinates"))
        else:
            try:
                symbol, xyz = atom
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"atom[{index}] must be [symbol, [x, y, z]] or a mapping"
                ) from exc
        if symbol is None or xyz is None or len(xyz) != 3:
            raise ValueError(f"invalid inline atom[{index}]: {atom!r}")
        symbols.append(str(symbol))
        coordinates.append([float(value) for value in xyz])
    if not symbols:
        raise ValueError("at least one atom is required")
    return tuple(symbols), np.asarray(coordinates, dtype=float)


def resolve_system(config: Mapping[str, Any], base_dir: Path) -> ResolvedSystem:
    """Resolve XYZ/inline geometry and lattice, then convert them to Bohr."""

    if config.get("atom_file") is not None:
        atom_path = Path(config["atom_file"]).expanduser()
        if not atom_path.is_absolute():
            atom_path = (base_dir / atom_path).resolve()
        symbols, coordinates = _read_xyz(atom_path)
    elif config.get("atom") is not None:
        symbols, coordinates = _read_inline_atoms(config["atom"])
    else:
        raise ValueError("provide either atom_file or atom")

    lattice = np.asarray(config.get("lattice_vectors"), dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError("lattice_vectors must be a 3x3 matrix")
    atom_scale = _unit_scale_to_bohr(config.get("atom_unit", "Angstrom"))
    lattice_scale = _unit_scale_to_bohr(config.get("lattice_unit", "Bohr"))
    return ResolvedSystem(
        symbols=symbols,
        coordinates_bohr=coordinates * atom_scale,
        lattice_bohr=lattice * lattice_scale,
    )


def _validate_config(config: Mapping[str, Any]) -> tuple[str, str]:
    device = str(config.get("device", "cpu")).lower()
    backend = str(config.get("df_backend", "fftdf")).lower()
    if device not in SUPPORTED_DEVICES:
        raise ValueError(f"device must be one of {sorted(SUPPORTED_DEVICES)}")
    if backend not in SUPPORTED_DF_BACKENDS:
        raise ValueError(
            f"df_backend must be one of {sorted(SUPPORTED_DF_BACKENDS)}"
        )
    if int(config.get("charge", 0)) != 0 or int(config.get("spin", 0)) != 0:
        raise NotImplementedError(
            "this benchmark follows the active restricted neutral-singlet scope: "
            "charge=0 and spin=0"
        )
    if config.get("mesh") is not None and config.get("ke_cutoff_ha") is not None:
        raise ValueError("set either mesh or ke_cutoff_ha, not both")
    return device, backend


def build_cell(
    config: Mapping[str, Any],
    system: ResolvedSystem,
    coordinates_bohr: np.ndarray | None = None,
):
    """Build a fixed-lattice PySCF Cell for one geometry."""

    from pyscf.pbc import gto

    coordinates = (
        system.coordinates_bohr
        if coordinates_bohr is None
        else np.asarray(coordinates_bohr, dtype=float)
    )
    cell = gto.Cell()
    cell.atom = list(zip(system.symbols, coordinates.tolist()))
    cell.a = np.asarray(system.lattice_bohr)
    cell.unit = "Bohr"
    cell.basis = config.get("basis", "gth-szv")
    cell.pseudo = config.get("pseudo", "gth-hf")
    cell.charge = int(config.get("charge", 0))
    cell.spin = int(config.get("spin", 0))
    cell.precision = float(config.get("precision", 1.0e-9))
    cell.verbose = int(config.get("verbose", 0))
    cell.max_memory = float(config.get("max_memory_mb", 4000))
    if config.get("mesh") is not None:
        mesh = np.asarray(config["mesh"], dtype=int)
        if mesh.shape != (3,) or np.any(mesh < 1):
            raise ValueError("mesh must contain three positive integers")
        cell.mesh = mesh
    else:
        cell.ke_cutoff = float(config.get("ke_cutoff_ha", 20.0))
    cell.build()
    if cell.nelectron % 2:
        raise ValueError(
            f"restricted closed-shell benchmark requires an even electron count; "
            f"got {cell.nelectron}"
        )
    return cell


def _normalize_kmesh(config: Mapping[str, Any]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in config.get("kmesh", (1, 1, 1)))
    if len(values) != 3 or any(value < 1 for value in values):
        raise ValueError("kmesh must contain three positive integers")
    return values


def _package_modules(device: str):
    if device == "cpu":
        from pyscf_pbc_dcdft import df, dft
    else:
        from gpu4pyscf_pbc_dcdft import df, dft
    return dft, df


def build_method(config: Mapping[str, Any], cell, device: str, backend: str):
    """Construct the requested public DC-DFT method and DF provider."""

    dft, df = _package_modules(device)
    kmesh = _normalize_kmesh(config)
    gamma = kmesh == (1, 1, 1)
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    if gamma:
        method = dft.DCRKS(cell, xc=str(config.get("xc", "PBE")))
    else:
        method = dft.DCKRKS(cell, kpts=kpts, xc=str(config.get("xc", "PBE")))

    provider_kpts = None if gamma else kpts
    if backend == "fftdf":
        method.with_df = df.FFTDF(cell, kpts=provider_kpts, mesh=cell.mesh)
    elif backend == "gdf":
        gdf_kwargs = {
            "kpts": provider_kpts,
            "auxbasis": config.get("auxbasis"),
        }
        if device == "gpu":
            gdf_kwargs["setup_mode"] = config.get(
                "gdf_setup_mode", "native"
            )
        method.with_df = df.GDF(cell, **gdf_kwargs)
    else:
        method.with_df = df.RSDF(
            cell,
            kpts=provider_kpts,
            auxbasis=config.get("auxbasis"),
        )
    method.conv_tol = float(config.get("conv_tol", 1.0e-10))
    conv_tol_grad = config.get("conv_tol_grad", 1.0e-8)
    method.conv_tol_grad = None if conv_tol_grad is None else float(conv_tol_grad)
    method.max_cycle = int(config.get("max_cycle", 80))
    method.exxdiv = str(config.get("exxdiv", "ewald"))
    method.verbose = int(config.get("verbose", 0))
    return method, kpts


def synchronize(device: str) -> None:
    """Make external wall timings include completed CUDA work."""

    if device == "gpu":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def warmup_device(device: str) -> None:
    """Initialize the CUDA context outside measured scientific stages."""

    if device != "gpu":
        return
    import cupy as cp

    probe = cp.arange(4096, dtype=cp.float64)
    cp.sum(probe * probe)
    cp.cuda.Stream.null.synchronize()
    del probe
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()


def _host_peak_rss_bytes() -> int:
    """Return process high-water RSS with platform units normalized."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def memory_snapshot(device: str, stage: str) -> dict[str, Any]:
    """Sample process and optional CUDA memory at one stage boundary."""

    snapshot: dict[str, Any] = {
        "stage": str(stage),
        "host_peak_rss_bytes": _host_peak_rss_bytes(),
    }
    if device == "gpu":
        import cupy as cp

        free_bytes, total_bytes = cp.cuda.Device().mem_info
        pool = cp.get_default_memory_pool()
        snapshot.update(
            {
                "gpu_free_device_bytes": int(free_bytes),
                "gpu_total_device_bytes": int(total_bytes),
                "cupy_pool_used_bytes": int(pool.used_bytes()),
                "cupy_pool_reserved_bytes": int(pool.total_bytes()),
            }
        )
    return snapshot


def summarize_memory(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate sampled high-water values without claiming exact peaks."""

    values = [dict(sample) for sample in samples]
    if not values:
        raise ValueError("at least one memory sample is required")
    report: dict[str, Any] = {
        "measurement": "stage_boundary_observed_high_water",
        "host_peak_rss_bytes": max(
            int(sample["host_peak_rss_bytes"]) for sample in values
        ),
        "samples": values,
    }
    gpu_samples = [
        sample for sample in values if "gpu_free_device_bytes" in sample
    ]
    if gpu_samples:
        report.update(
            {
                "gpu_total_device_bytes": max(
                    int(sample["gpu_total_device_bytes"])
                    for sample in gpu_samples
                ),
                "minimum_observed_gpu_free_device_bytes": min(
                    int(sample["gpu_free_device_bytes"])
                    for sample in gpu_samples
                ),
                "peak_observed_cupy_pool_used_bytes": max(
                    int(sample["cupy_pool_used_bytes"])
                    for sample in gpu_samples
                ),
                "peak_observed_cupy_pool_reserved_bytes": max(
                    int(sample["cupy_pool_reserved_bytes"])
                    for sample in gpu_samples
                ),
            }
        )
    return report


def evaluate_energy(
    config: Mapping[str, Any],
    system: ResolvedSystem,
    coordinates_bohr: np.ndarray,
    device: str,
    backend: str,
) -> EnergyRun:
    """Build one independent geometry and evaluate its full DC-DFT energy."""

    start = perf_counter()
    cell = build_cell(config, system, coordinates_bohr)
    method, _ = build_method(config, cell, device, backend)
    synchronize(device)
    energy_start = perf_counter()
    energy = float(method.kernel())
    synchronize(device)
    wall_time = perf_counter() - energy_start
    return EnergyRun(
        energy_ha=energy,
        method=method,
        setup_wall_time_s=energy_start - start,
        wall_time_s=wall_time,
    )


def _selects_all(spec: Any) -> bool:
    """Accept both ``"all"`` and the common JSON spelling ``["all"]``."""

    if spec is None:
        return True
    if isinstance(spec, str):
        return spec.strip().lower() == "all"
    if isinstance(spec, (list, tuple)) and len(spec) == 1:
        value = spec[0]
        return isinstance(value, str) and value.strip().lower() == "all"
    return False


def _selected_atoms(spec: Any, natoms: int) -> list[int]:
    if _selects_all(spec):
        return list(range(natoms))
    values = [spec] if isinstance(spec, (int, np.integer)) else spec
    atoms = sorted({int(value) for value in values})
    if not atoms:
        raise ValueError("fd_atoms cannot be empty when run_fd is true")
    if any(value < 0 or value >= natoms for value in atoms):
        raise ValueError(f"fd_atoms must be within [0, {natoms - 1}]")
    return atoms


def _selected_axes(spec: Any) -> list[int]:
    if _selects_all(spec):
        return [0, 1, 2]
    if isinstance(spec, (str, int, np.integer)):
        spec = [spec]
    axes: set[int] = set()
    for value in spec:
        if isinstance(value, str):
            normalized = value.lower()
            if normalized not in AXIS_NAMES:
                raise ValueError("fd_axes entries must be x, y, z, 0, 1, or 2")
            axes.add(AXIS_NAMES.index(normalized))
        else:
            axes.add(int(value))
    if any(value not in (0, 1, 2) for value in axes):
        raise ValueError("fd_axes entries must be x, y, z, 0, 1, or 2")
    if not axes:
        raise ValueError("fd_axes cannot be empty when run_fd is true")
    return sorted(axes)


def finite_difference_gradient(
    config: Mapping[str, Any],
    system: ResolvedSystem,
    device: str,
    backend: str,
    progress: ProgressReporter | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float]:
    """Evaluate selected components of the full-energy central FD gradient."""

    step = float(config.get("fd_step_bohr", 1.0e-4))
    if step <= 0.0:
        raise ValueError("fd_step_bohr must be positive")
    atoms = _selected_atoms(config.get("fd_atoms", "all"), system.natoms)
    axes = _selected_axes(config.get("fd_axes", "all"))
    gradient = np.full((system.natoms, 3), np.nan, dtype=float)
    mask = np.zeros((system.natoms, 3), dtype=bool)
    components: list[dict[str, Any]] = []
    total_start = perf_counter()
    total_components = len(atoms) * len(axes)

    for atom_id in atoms:
        for axis in axes:
            if progress is not None:
                progress.update(
                    "finite_difference",
                    completed=len(components),
                    total=total_components,
                    atom=atom_id,
                    axis=AXIS_NAMES[axis],
                )
            plus = np.array(system.coordinates_bohr, copy=True)
            minus = np.array(system.coordinates_bohr, copy=True)
            plus[atom_id, axis] += step
            minus[atom_id, axis] -= step

            component_start = perf_counter()
            plus_run = evaluate_energy(config, system, plus, device, backend)
            minus_run = evaluate_energy(config, system, minus, device, backend)
            derivative = (plus_run.energy_ha - minus_run.energy_ha) / (2.0 * step)
            gradient[atom_id, axis] = derivative
            mask[atom_id, axis] = True
            components.append(
                {
                    "atom": atom_id,
                    "axis": AXIS_NAMES[axis],
                    "energy_plus_ha": plus_run.energy_ha,
                    "energy_minus_ha": minus_run.energy_ha,
                    "fd_gradient_ha_per_bohr": derivative,
                    "plus_setup_wall_s": plus_run.setup_wall_time_s,
                    "plus_energy_wall_s": plus_run.wall_time_s,
                    "minus_setup_wall_s": minus_run.setup_wall_time_s,
                    "minus_energy_wall_s": minus_run.wall_time_s,
                    "component_wall_s": perf_counter() - component_start,
                }
            )
    if progress is not None:
        progress.update(
            "finite_difference",
            force=True,
            completed=total_components,
            total=total_components,
        )
    return gradient, mask, components, perf_counter() - total_start


def _gradient_metadata(gradient_object: Any) -> Mapping[str, Any] | None:
    result = getattr(gradient_object, "result", None)
    if result is None:
        return None
    if hasattr(result, "metadata"):
        return result.metadata()
    return None


def _configure_gradient(config: Mapping[str, Any], gradient_object: Any) -> None:
    """Apply optional public Z-vector controls without backend branching."""

    attributes = {
        "z_solver": str,
        "z_tolerance": float,
        "z_residual_tolerance": float,
        "z_max_cycle": int,
        "z_dense_max_dimension": int,
        "z_nuclear_batch_memory_fraction": float,
        "z_nuclear_batch_max_components": int,
        "gdf_bilinear_mode": str,
    }
    for name, converter in attributes.items():
        if name in config and hasattr(gradient_object, name):
            value = config[name]
            if name == "z_nuclear_batch_max_components" and value is None:
                setattr(gradient_object, name, None)
            else:
                setattr(gradient_object, name, converter(value))


def _device_metadata(device: str) -> dict[str, Any]:
    metadata = {
        "requested_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device == "gpu":
        import cupy as cp

        device_id = int(cp.cuda.Device().id)
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        name = properties.get("name", "unknown")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        metadata.update({"cuda_device_id": device_id, "cuda_device_name": name})
    return metadata


def _accuracy_report(
    analytic: np.ndarray,
    finite_difference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    difference = analytic - finite_difference
    selected = difference[mask]
    rows = []
    for atom_id, axis in np.argwhere(mask):
        rows.append(
            {
                "atom": int(atom_id),
                "axis": AXIS_NAMES[int(axis)],
                "analytic_ha_per_bohr": float(analytic[atom_id, axis]),
                "fd_ha_per_bohr": float(finite_difference[atom_id, axis]),
                "analytic_minus_fd_ha_per_bohr": float(
                    difference[atom_id, axis]
                ),
            }
        )
    return {
        "component_count": int(selected.size),
        "max_abs_error_ha_per_bohr": float(np.max(np.abs(selected))),
        "rms_error_ha_per_bohr": float(np.sqrt(np.mean(selected**2))),
        "mean_abs_error_ha_per_bohr": float(np.mean(np.abs(selected))),
        "components": rows,
    }


def _save_arrays(
    output_dir: Path,
    analytic: np.ndarray,
    fd_gradient: np.ndarray | None,
    fd_mask: np.ndarray | None,
) -> dict[str, str]:
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    analytic_path = arrays_dir / "analytic_gradient_ha_per_bohr.npy"
    np.save(analytic_path, analytic)
    paths = {"analytic_gradient": str(analytic_path)}

    optional_paths = (
        arrays_dir / "fd_gradient_ha_per_bohr.npy",
        arrays_dir / "analytic_minus_fd_gradient_ha_per_bohr.npy",
        arrays_dir / "fd_mask.npy",
    )
    if fd_gradient is None or fd_mask is None:
        for path in optional_paths:
            path.unlink(missing_ok=True)
        return paths

    difference = np.full_like(analytic, np.nan)
    difference[fd_mask] = analytic[fd_mask] - fd_gradient[fd_mask]
    np.save(optional_paths[0], fd_gradient)
    np.save(optional_paths[1], difference)
    np.save(optional_paths[2], fd_mask)
    paths.update(
        {
            "fd_gradient": str(optional_paths[0]),
            "analytic_minus_fd_gradient": str(optional_paths[1]),
            "fd_mask": str(optional_paths[2]),
        }
    )
    return paths


def run_benchmark(
    config: Mapping[str, Any],
    output_dir: Path,
    input_base_dir: Path,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Run one energy/gradient benchmark and write its numerical arrays."""

    device, backend = _validate_config(config)
    warmup_start = perf_counter()
    if progress is not None:
        progress.update("device_warmup", force=True, device=device)
    if bool(config.get("device_warmup", True)):
        warmup_device(device)
    warmup_wall_s = perf_counter() - warmup_start
    memory_samples = [memory_snapshot(device, "after_device_warmup")]
    workflow_start = perf_counter()
    setup_start = perf_counter()
    if progress is not None:
        progress.update("input_and_cell_setup", force=True)
    system = resolve_system(config, input_base_dir)
    cell = build_cell(config, system)
    method, kpts = build_method(config, cell, device, backend)
    if hasattr(method, "progress_callback"):
        method.progress_callback = progress
    setup_wall_s = perf_counter() - setup_start
    memory_samples.append(memory_snapshot(device, "after_input_setup"))

    if progress is not None:
        progress.update(
            "energy_workflow",
            force=True,
            natoms=system.natoms,
            mesh=np.asarray(cell.mesh, dtype=int),
        )
    synchronize(device)
    energy_start = perf_counter()
    energy = float(method.kernel())
    synchronize(device)
    energy_wall_s = perf_counter() - energy_start
    memory_samples.append(memory_snapshot(device, "after_energy"))
    if progress is not None:
        progress.update(
            "energy_complete",
            force=True,
            energy_ha=energy,
            wall_time_s=energy_wall_s,
        )

    gradient_object = method.nuc_grad_method()
    _configure_gradient(config, gradient_object)
    if hasattr(gradient_object, "progress_callback"):
        gradient_object.progress_callback = progress
    if progress is not None:
        progress.update(
            "analytic_gradient",
            force=True,
            backend=backend,
            device=device,
        )
    synchronize(device)
    gradient_start = perf_counter()
    analytic_gradient = np.asarray(gradient_object.kernel(), dtype=float)
    synchronize(device)
    analytic_gradient_wall_s = perf_counter() - gradient_start
    memory_samples.append(memory_snapshot(device, "after_analytic_gradient"))
    if progress is not None:
        progress.update(
            "analytic_gradient_complete",
            force=True,
            wall_time_s=analytic_gradient_wall_s,
        )

    fd_gradient = None
    fd_mask = None
    fd_components = None
    fd_wall_s = None
    accuracy = None
    if bool(config.get("run_fd", False)):
        fd_gradient, fd_mask, fd_components, fd_wall_s = finite_difference_gradient(
            config, system, device, backend, progress=progress
        )
        accuracy = _accuracy_report(analytic_gradient, fd_gradient, fd_mask)
        memory_samples.append(memory_snapshot(device, "after_fd"))

    if progress is not None:
        progress.update("final_reporting", force=True)
    array_paths = _save_arrays(output_dir, analytic_gradient, fd_gradient, fd_mask)
    method_metadata = method.metadata() if hasattr(method, "metadata") else None
    gradient_metadata = _gradient_metadata(gradient_object)
    timing = {
        "device_warmup_wall_s": warmup_wall_s,
        "setup_wall_s": setup_wall_s,
        "energy_wall_s": energy_wall_s,
        "analytic_gradient_wall_s": analytic_gradient_wall_s,
        "analytic_total_wall_s": energy_wall_s + analytic_gradient_wall_s,
        "workflow_without_fd_wall_s": (
            setup_wall_s + energy_wall_s + analytic_gradient_wall_s
        ),
        "fd_wall_s": fd_wall_s,
        "workflow_wall_s": perf_counter() - workflow_start,
        "method_internal_s": getattr(method, "timings_s", None),
        "gradient_internal_s": (
            None if gradient_metadata is None else gradient_metadata.get("timings_s")
        ),
    }
    return {
        "status": "complete",
        "schema_version": 1,
        "run_name": config.get("name"),
        "device": device,
        "df_backend": backend,
        "xc": str(config.get("xc", "PBE")),
        "energy_ha": energy,
        "units": {
            "energy": "Ha",
            "gradient": "Ha/Bohr",
            "coordinates": "Bohr",
            "lattice": "Bohr",
            "fd_step": "Bohr",
        },
        "system": {
            "natoms": system.natoms,
            "symbols": list(system.symbols),
            "coordinates_bohr": system.coordinates_bohr,
            "lattice_vectors_bohr": system.lattice_bohr,
            "charge": 0,
            "spin": 0,
        },
        "numerical_settings": {
            "basis": config.get("basis", "gth-szv"),
            "pseudo": config.get("pseudo", "gth-hf"),
            "auxbasis": config.get("auxbasis"),
            "gdf_setup_mode": (
                config.get("gdf_setup_mode", "native")
                if device == "gpu" and backend == "gdf"
                else None
            ),
            "kmesh": list(_normalize_kmesh(config)),
            "nkpts": len(kpts),
            "kpts": np.asarray(kpts),
            "mesh": np.asarray(cell.mesh, dtype=int),
            "ke_cutoff_ha": getattr(cell, "ke_cutoff", None),
            "precision": cell.precision,
            "conv_tol": method.conv_tol,
            "conv_tol_grad": method.conv_tol_grad,
            "max_cycle": method.max_cycle,
            "num_threads": int(config.get("num_threads", 1)),
            "device_warmup": bool(config.get("device_warmup", True)),
            "z_solver": getattr(gradient_object, "z_solver", None),
            "z_tolerance": getattr(gradient_object, "z_tolerance", None),
            "z_residual_tolerance": getattr(
                gradient_object, "z_residual_tolerance", None
            ),
            "z_max_cycle": getattr(gradient_object, "z_max_cycle", None),
            "z_dense_max_dimension": getattr(
                gradient_object, "z_dense_max_dimension", None
            ),
            "z_nuclear_batch_memory_fraction": getattr(
                gradient_object,
                "z_nuclear_batch_memory_fraction",
                None,
            ),
            "z_nuclear_batch_max_components": getattr(
                gradient_object,
                "z_nuclear_batch_max_components",
                None,
            ),
            "gdf_bilinear_mode": getattr(
                gradient_object,
                "gdf_bilinear_mode",
                None,
            ),
        },
        "fd": {
            "enabled": bool(config.get("run_fd", False)),
            "step_bohr": (
                float(config.get("fd_step_bohr", 1.0e-4))
                if bool(config.get("run_fd", False))
                else None
            ),
            "components": fd_components,
        },
        "accuracy": accuracy,
        "timing_s": timing,
        "arrays": array_paths,
        "method_metadata": method_metadata,
        "gradient_metadata": gradient_metadata,
        "device_metadata": _device_metadata(device),
        "memory": summarize_memory(memory_samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="case JSON")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(input_path.read_text(encoding="utf-8"))
    report_path = output_dir / "report.json"
    progress = ProgressReporter(
        output_dir / "progress.json",
        context={
            "run_name": config.get("name"),
            "device": config.get("device", "cpu"),
            "df_backend": config.get("df_backend", "fftdf"),
            "xc": config.get("xc", "PBE"),
        },
    )
    progress.update("starting", force=True)
    try:
        report = run_benchmark(
            config,
            output_dir,
            input_path.parent,
            progress=progress,
        )
        write_json(report_path, report)
        progress.complete(report_path=str(report_path))
    except Exception as exc:
        failure = {
            "status": "failed",
            "schema_version": 1,
            "run_name": config.get("name"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(report_path, failure)
        progress.fail(exc)
        raise

    timing = report["timing_s"]
    print(f"status: complete")
    print(f"energy: {report['energy_ha']:.12f} Ha")
    print(f"analytic gradient: {timing['analytic_gradient_wall_s']:.6f} s")
    if report["accuracy"] is not None:
        print(
            "max |analytic - FD|: "
            f"{report['accuracy']['max_abs_error_ha_per_bohr']:.6e} Ha/Bohr"
        )
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
