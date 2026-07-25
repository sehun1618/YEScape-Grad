#!/usr/bin/env python
"""Run one restricted PBC DC-DFT analytic-stress benchmark case."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PBC_BENCHMARK = ROOT / "pbc" / "benchmark"
for path in (
    HERE,
    PBC_BENCHMARK,
    ROOT / "stress" / "cpu" / "src",
    ROOT / "stress" / "gpu" / "src",
    ROOT / "pbc" / "cpu" / "src",
    ROOT / "pbc" / "gpu" / "src",
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import dcdft_benchmark as gradient_benchmark  # noqa: E402
from benchmark_checkpoint import CheckpointStore  # noqa: E402
from extension_build import ensure_stress_extension  # noqa: E402
from pyscf_pbc_dcdft_stress import (  # noqa: E402
    EnergyEvaluation,
    FDStress,
    FDStressConfig,
    calculation_state_fingerprint,
    gamma_gdf_analytic_stress,
    gamma_rsdf_analytic_stress,
    kpoint_gdf_analytic_stress,
    kpoint_rsdf_analytic_stress,
    mlip_record_from_results,
    write_mlip_extxyz,
    write_mlip_json,
)


CALCULATION_MODES = {"stress_only", "gradient_and_stress"}
SUPPORTED_BACKENDS = {"gdf", "rsdf"}


@dataclass(frozen=True)
class CachedStressResult:
    """Minimal stress-result interface reconstructed from a checkpoint."""

    stress_voigt_ha_per_bohr3: np.ndarray
    stress_tensor_ha_per_bohr3: np.ndarray
    volume_bohr3: float
    timings_s: dict
    metadata: dict
    payload: dict

    def as_dict(self):
        return dict(self.payload)


def _json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "get"):
        return _json_value(value.get())
    return str(value)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True)+"\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate(config):
    device, backend = gradient_benchmark._validate_config(config)
    mode = str(config.get("calculation_mode", "stress_only")).lower()
    if backend not in SUPPORTED_BACKENDS:
        raise NotImplementedError("analytic stress supports GDF/RSDF only")
    if mode not in CALCULATION_MODES:
        raise ValueError(f"calculation_mode must be one of {sorted(CALCULATION_MODES)}")
    if str(config.get("target_integrator", "uniform")).lower() != "uniform":
        raise NotImplementedError("analytic stress currently requires uniform XC")
    if config.get("dispersion") not in (None, "", "none"):
        raise NotImplementedError("dispersion stress is not implemented")
    tolerance = config.get("stress_fd_tolerance_ha_per_bohr3")
    if tolerance is not None and not bool(config.get("run_stress_fd", False)):
        raise ValueError("stress FD tolerance requires run_stress_fd=true")
    return device, backend, mode


def _stress_driver(device, backend, sampled_k):
    if device == "gpu":
        from gpu4pyscf_pbc_dcdft_stress import gpu_analytic_stress

        return gpu_analytic_stress
    if backend == "gdf":
        return kpoint_gdf_analytic_stress if sampled_k else gamma_gdf_analytic_stress
    return kpoint_rsdf_analytic_stress if sampled_k else gamma_rsdf_analytic_stress


def _stress_options(
    config, device, *, shared_z_vector=None, progress_callback=None,
):
    options = {
        "z_equation": str(config.get("z_equation", "adjoint")),
        "shared_z_vector": shared_z_vector,
    }
    if device == "gpu":
        reciprocal_block_size = config.get(
            "stress_host_reciprocal_block_size"
        )
        df_reciprocal_block_size = config.get(
            "stress_df_host_reciprocal_block_size"
        )
        options.update({
            "z_solver": str(config.get("z_solver", "auto")),
            "host_workers": config.get("stress_host_workers"),
            "pp_response_backend": str(config.get(
                "stress_pp_response_backend", "host_analytic_matrix"
            )),
            "pp_grid_block_size": config.get("stress_pp_grid_block_size"),
            "pp_nonlocal_host_block_size": int(config.get(
                "stress_pp_nonlocal_host_block_size", 8192
            )),
            "detailed_profile": bool(
                config.get("stress_detailed_profile", False)
            ),
            "df_response_backend": str(config.get(
                "stress_df_response_backend",
                "host_analytic_device_factorization",
            )),
            "df_contracted_fd_step": float(config.get(
                "stress_df_contracted_fd_step", 1.0e-5
            )),
            "df_diagonal_cache_bytes": int(
                config.get("stress_df_diagonal_cache_bytes", 256*(1 << 20))
            ),
            "df_device_memory_fraction": float(
                config.get("stress_df_device_memory_fraction", 0.85)
            ),
            "host_reciprocal_block_size": (
                None if reciprocal_block_size is None
                else int(reciprocal_block_size)
            ),
            "df_host_reciprocal_block_size": (
                None if df_reciprocal_block_size is None
                else int(df_reciprocal_block_size)
            ),
            "progress_callback": progress_callback,
        })
    return options


def _host_peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value*1024


def _to_host(value):
    return np.asarray(value.get() if hasattr(value, "get") else value)


class BenchmarkEnergyEvaluator:
    """Fresh same-device energy evaluator for central strain FD."""

    def __init__(
        self, config, device, backend, *, checkpoint=None, progress=None,
        expected_calls=None,
    ):
        self.config = dict(config)
        self.device = device
        self.backend = backend
        self.checkpoint = checkpoint
        self.progress = progress
        self.expected_calls = expected_calls
        self.completed_calls = 0
        self.reused_calls = 0
        self.computed_calls = 0

    @staticmethod
    def _geometry_key(cell):
        digest = hashlib.sha256()
        for value in (
            np.asarray(cell.lattice_vectors(), dtype=np.float64),
            np.asarray(cell.atom_coords(), dtype=np.float64),
            np.asarray(cell.mesh, dtype=np.int64),
        ):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return f"stress_fd/evaluations/{digest.hexdigest()}"

    def _load(self, cell):
        if self.checkpoint is None or not self.checkpoint.enabled:
            return None
        key = self._geometry_key(cell)
        metadata = self.checkpoint.load_json(f"{key}/stage")
        energy = self.checkpoint.load_array(f"{key}/energy_ha")
        density = self.checkpoint.load_array(f"{key}/density")
        if metadata is None or energy is None or energy.size != 1:
            return None
        try:
            value = float(np.asarray(energy).reshape(()))
            if not np.isclose(value, float(metadata["energy_ha"])):
                return None
            return EnergyEvaluation(
                energy_ha=value,
                wall_seconds=float(metadata["wall_seconds"]),
                converged=bool(metadata["converged"]),
                density=density,
                metadata={
                    **dict(metadata.get("metadata") or {}),
                    "checkpoint_reused": True,
                },
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _save(self, cell, evaluation):
        if self.checkpoint is None or not self.checkpoint.enabled:
            return
        key = self._geometry_key(cell)
        self.checkpoint.save_array(
            f"{key}/energy_ha", np.asarray(evaluation.energy_ha)
        )
        if evaluation.density is not None:
            self.checkpoint.save_array(f"{key}/density", evaluation.density)
        self.checkpoint.save_json(
            f"{key}/stage",
            {
                "energy_ha": evaluation.energy_ha,
                "wall_seconds": evaluation.wall_seconds,
                "converged": evaluation.converged,
                "metadata": evaluation.metadata,
                "mesh": np.asarray(cell.mesh, dtype=int),
                "lattice_bohr": np.asarray(cell.lattice_vectors()),
            },
        )

    def seed(self, cell, evaluation):
        """Register the already evaluated central cell for FD reuse."""

        self._save(cell, evaluation)

    def _report_progress(self, *, reused):
        self.completed_calls += 1
        if reused:
            self.reused_calls += 1
        else:
            self.computed_calls += 1
        if self.progress is not None:
            self.progress.update(
                "stress_finite_difference",
                force=True,
                completed=self.completed_calls,
                total=self.expected_calls,
                checkpoint_reused_evaluations=self.reused_calls,
                computed_evaluations=self.computed_calls,
            )

    def checkpoint_summary(self):
        return {
            "evaluation_count": self.completed_calls,
            "reused_evaluations": self.reused_calls,
            "computed_evaluations": self.computed_calls,
        }

    def __call__(self, cell, *, dm0=None, kpts=None):
        del kpts
        cached = self._load(cell)
        if cached is not None:
            self._report_progress(reused=True)
            return cached
        method, _ = gradient_benchmark.build_method(
            self.config, cell, self.device, self.backend
        )
        started = perf_counter()
        energy = float(method.kernel(dm0=dm0))
        gradient_benchmark.synchronize(self.device)
        density = _to_host(method.make_rdm1())
        result = EnergyEvaluation(
            energy_ha=energy,
            wall_seconds=perf_counter()-started,
            converged=bool(method.converged),
            density=density,
            metadata={
                "device": self.device,
                "df_backend": self.backend,
                "checkpoint_reused": False,
            },
        )
        self._save(cell, result)
        self._report_progress(reused=False)
        return result


def _run_stress_fd(
    config, cell, device, backend, progress, *, checkpoint=None,
    central_evaluation=None,
):
    components = tuple(config.get("stress_fd_components", ("xx", "yy", "zz", "yz", "xz", "xy")))
    evaluator = BenchmarkEnergyEvaluator(
        config, device, backend, checkpoint=checkpoint, progress=progress,
        expected_calls=2*len(components)+1,
    )
    if central_evaluation is not None:
        evaluator.seed(cell, central_evaluation)
    fd = FDStress(
        evaluator,
        FDStressConfig(
            step=float(config.get("stress_fd_step", 2.5e-4)),
            components=components,
            fixed_mesh=True,
            warm_start=bool(config.get("stress_fd_warm_start", False)),
        ),
    )
    if progress is not None:
        progress.update(
            "stress_finite_difference", force=True,
            completed=0, total=2*len(components)+1,
        )
    started = perf_counter()
    result = fd.kernel(cell)
    result.metadata["checkpoint"] = evaluator.checkpoint_summary()
    if checkpoint is not None and checkpoint.enabled:
        checkpoint.save_array(
            "stress_fd/stress_voigt_ha_per_bohr3",
            result.stress_voigt_ha_per_bohr3,
        )
        checkpoint.save_array(
            "stress_fd/stress_tensor_ha_per_bohr3",
            result.stress_tensor_ha_per_bohr3,
        )
        checkpoint.save_json("stress_fd/stage", result.as_dict())
    return result, perf_counter()-started


def _save_gradient_checkpoint(
    checkpoint, method, gradient_object, gradient, *, wall_seconds,
):
    if not checkpoint.enabled:
        return
    checkpoint.save_array("gradient/gradient_ha_per_bohr", gradient)
    checkpoint.save_array("gradient/force_ha_per_bohr", -np.asarray(gradient))
    result = gradient_object.result
    z_vector = getattr(result, "z_vector", None)
    blocks = tuple(getattr(z_vector, "blocks", ()) or ())
    for block_id, block in enumerate(blocks):
        checkpoint.save_array(
            f"gradient/z_vector/block_{block_id:04d}", block
        )
    metadata = result.metadata() if hasattr(result, "metadata") else None
    checkpoint.save_json(
        "gradient/stage",
        {
            "energy_ha": float(method.e_tot),
            "wall_seconds": wall_seconds,
            "metadata": metadata,
            "z_vector_block_count": len(blocks),
        },
    )


def _load_gradient_checkpoint(checkpoint, expected_shape):
    stage = checkpoint.load_json("gradient/stage")
    gradient = checkpoint.load_array("gradient/gradient_ha_per_bohr")
    if stage is None or gradient is None or gradient.shape != expected_shape:
        return None
    return np.asarray(gradient, dtype=float), stage


def _save_stress_checkpoint(
    checkpoint, stress, *, energy_ha, state_fingerprint, wall_seconds,
):
    if not checkpoint.enabled:
        return
    checkpoint.save_array(
        "stress/stress_voigt_ha_per_bohr3",
        stress.stress_voigt_ha_per_bohr3,
    )
    checkpoint.save_array(
        "stress/stress_tensor_ha_per_bohr3",
        stress.stress_tensor_ha_per_bohr3,
    )
    z_vector = getattr(stress, "z_vector", None)
    blocks = tuple(getattr(z_vector, "blocks", ()) or ())
    for block_id, block in enumerate(blocks):
        checkpoint.save_array(
            f"stress/z_vector/block_{block_id:04d}", block
        )
    checkpoint.save_json(
        "stress/stage",
        {
            "energy_ha": energy_ha,
            "state_fingerprint": state_fingerprint,
            "wall_seconds": wall_seconds,
            "z_vector_block_count": len(blocks),
            "result": stress.as_dict(),
        },
    )


def _load_stress_checkpoint(checkpoint):
    stage = checkpoint.load_json("stress/stage")
    voigt = checkpoint.load_array("stress/stress_voigt_ha_per_bohr3")
    tensor = checkpoint.load_array("stress/stress_tensor_ha_per_bohr3")
    if (
        stage is None or voigt is None or tensor is None
        or voigt.shape != (6,) or tensor.shape != (3, 3)
    ):
        return None
    payload = dict(stage.get("result") or {})
    return CachedStressResult(
        stress_voigt_ha_per_bohr3=np.asarray(voigt, dtype=float),
        stress_tensor_ha_per_bohr3=np.asarray(tensor, dtype=float),
        volume_bohr3=float(payload["volume_bohr3"]),
        timings_s=dict(payload.get("timings_s") or {}),
        metadata=dict(payload.get("metadata") or {}),
        payload=payload,
    ), stage


def _save_arrays(output_dir, stress, gradient=None, force=None, fd=None):
    arrays = output_dir / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    paths = {}
    values = {
        "stress_voigt_ha_per_bohr3": stress.stress_voigt_ha_per_bohr3,
        "stress_tensor_ha_per_bohr3": stress.stress_tensor_ha_per_bohr3,
    }
    if gradient is not None:
        values["gradient_ha_per_bohr"] = gradient
        values["force_ha_per_bohr"] = force
    if fd is not None:
        values["fd_stress_voigt_ha_per_bohr3"] = fd.stress_voigt_ha_per_bohr3
        values["analytic_minus_fd_stress_voigt_ha_per_bohr3"] = (
            stress.stress_voigt_ha_per_bohr3-fd.stress_voigt_ha_per_bohr3
        )
    for name, value in values.items():
        path = arrays / f"{name}.npy"
        np.save(path, np.asarray(value))
        paths[name] = str(path)
    return paths


def run_case(config, output_dir, input_base_dir, progress=None, vram_tracker=None):
    device, backend, mode = _validate(config)
    checkpoint = CheckpointStore(
        output_dir/"checkpoints",
        str(
            config.get("input_fingerprint")
            or gradient_benchmark._checkpoint_fingerprint(config)
        ),
        enabled=bool(config.get("checkpoint_resume", True)),
    )
    if bool(config.get("device_warmup", True)):
        gradient_benchmark.warmup_device(device)
    if vram_tracker is not None:
        vram_tracker.start()
        if progress is not None:
            progress.memory_provider = vram_tracker.progress_snapshot

    workflow_started = perf_counter()
    if progress is not None:
        progress.update("input_and_cell_setup", force=True)
    system = gradient_benchmark.resolve_system(config, input_base_dir)
    cell = gradient_benchmark.build_cell(config, system)
    sampled_k = tuple(
        int(value) for value in config.get("kmesh", (1, 1, 1))
    ) != (1, 1, 1)
    stress_checkpoint = _load_stress_checkpoint(checkpoint)
    gradient_checkpoint = (
        _load_gradient_checkpoint(checkpoint, (system.natoms, 3))
        if mode == "gradient_and_stress" else None
    )
    need_method = (
        stress_checkpoint is None
        or (mode == "gradient_and_stress" and gradient_checkpoint is None)
    )
    method = None
    if need_method:
        method, _ = gradient_benchmark.build_method(
            config, cell, device, backend
        )
    setup_seconds = perf_counter()-workflow_started

    source_density_warm_start = False
    source_stage = checkpoint.load_json("source/stage")
    central_density = checkpoint.load_array("source/density")
    if method is not None:
        if progress is not None:
            progress.update("source_hf_and_target_energy", force=True)
        source_density_warm_start = central_density is not None
        started = perf_counter()
        energy = float(method.kernel(dm0=central_density))
        gradient_benchmark.synchronize(device)
        energy_seconds = perf_counter()-started
        if not bool(method.converged):
            raise RuntimeError("source HF calculation did not converge")
        central_density = _to_host(method.make_rdm1())
        fingerprint = calculation_state_fingerprint(
            method, density=central_density
        )
        gradient_benchmark._save_source_state_checkpoint(
            checkpoint,
            method,
            energy_ha=energy,
            energy_wall_s=energy_seconds,
        )
    else:
        _, stress_stage = stress_checkpoint
        energy = float(stress_stage["energy_ha"])
        fingerprint = str(stress_stage["state_fingerprint"])
        energy_seconds = float(
            (source_stage or {}).get("energy_wall_s", 0.0)
        )
        if progress is not None:
            progress.update(
                "source_and_stress_checkpoint_reused",
                force=True,
                energy_ha=energy,
            )

    gradient = force = gradient_object = None
    shared_z_vector = None
    gradient_seconds = 0.0
    gradient_metadata = None
    gradient_checkpoint_reused = False
    if mode == "gradient_and_stress":
        if gradient_checkpoint is not None:
            gradient, gradient_stage = gradient_checkpoint
            force = -gradient
            gradient_seconds = float(gradient_stage["wall_seconds"])
            gradient_metadata = gradient_stage.get("metadata")
            gradient_checkpoint_reused = True
            if progress is not None:
                progress.update(
                    "gradient_checkpoint_reused", force=True,
                    wall_time_s=gradient_seconds,
                )
        else:
            if progress is not None:
                progress.update("analytic_gradient", force=True)
            gradient_object = method.nuc_grad_method()
            gradient_benchmark._configure_gradient(config, gradient_object)
            if hasattr(gradient_object, "progress_callback"):
                gradient_object.progress_callback = progress
            started = perf_counter()
            gradient = np.asarray(gradient_object.kernel(), dtype=float)
            gradient_benchmark.synchronize(device)
            gradient_seconds = perf_counter()-started
            force = -gradient
            shared_z_vector = getattr(
                gradient_object.result, "z_vector", None
            )
            if shared_z_vector is None:
                raise RuntimeError(
                    "gradient result did not expose its converged Z-vector"
                )
            result = gradient_object.result
            gradient_metadata = (
                result.metadata() if hasattr(result, "metadata") else None
            )
            _save_gradient_checkpoint(
                checkpoint,
                method,
                gradient_object,
                gradient,
                wall_seconds=gradient_seconds,
            )

    stress_checkpoint_reused = stress_checkpoint is not None
    if stress_checkpoint is not None:
        stress, stress_stage = stress_checkpoint
        stress_seconds = float(stress_stage["wall_seconds"])
        if progress is not None:
            progress.update(
                "analytic_stress_checkpoint_reused", force=True,
                wall_time_s=stress_seconds,
            )
    else:
        if progress is not None:
            progress.update("analytic_stress", force=True)
        driver = _stress_driver(device, backend, sampled_k)
        started = perf_counter()
        stress = driver(
            method, **_stress_options(
                config, device, shared_z_vector=shared_z_vector,
                progress_callback=progress,
            )
        )
        gradient_benchmark.synchronize(device)
        stress_seconds = perf_counter()-started
        _save_stress_checkpoint(
            checkpoint,
            stress,
            energy_ha=energy,
            state_fingerprint=fingerprint,
            wall_seconds=stress_seconds,
        )

    fd_result = None
    fd_seconds = None
    accuracy = None
    accuracy_status = "not_requested"
    if bool(config.get("run_stress_fd", False)):
        central_evaluation = None
        if central_density is not None:
            central_evaluation = EnergyEvaluation(
                energy_ha=energy,
                wall_seconds=energy_seconds,
                converged=True,
                density=central_density,
                metadata={
                    "device": device,
                    "df_backend": backend,
                    "central_source_state": True,
                },
            )
        fd_result, fd_seconds = _run_stress_fd(
            config,
            cell,
            device,
            backend,
            progress,
            checkpoint=checkpoint,
            central_evaluation=central_evaluation,
        )
        selected = np.isfinite(fd_result.stress_voigt_ha_per_bohr3)
        difference = (
            stress.stress_voigt_ha_per_bohr3[selected]
            - fd_result.stress_voigt_ha_per_bohr3[selected]
        )
        maximum = float(np.max(np.abs(difference), initial=0.0))
        tolerance = config.get("stress_fd_tolerance_ha_per_bohr3")
        passed = None if tolerance is None else maximum <= float(tolerance)
        accuracy_status = "not_requested" if passed is None else ("pass" if passed else "fail")
        accuracy = {
            "components": list(config.get("stress_fd_components", ("xx", "yy", "zz", "yz", "xz", "xy"))),
            "max_abs_analytic_minus_fd_ha_per_bohr3": maximum,
            "tolerance_ha_per_bohr3": tolerance,
            "passed": passed,
            "fd": fd_result.as_dict(),
        }

    record = None
    output_paths = {}
    if gradient is not None:
        record = mlip_record_from_results(
            cell, energy, gradient, stress,
            metadata={
                "state_fingerprint": fingerprint,
                "device": device,
                "df_backend": backend,
                "xc": str(config.get("xc", "PBE")),
            },
        )
        output_paths["mlip_json"] = str(
            write_mlip_json(record, output_dir/"mlip_frame.json")
        )
        output_paths["mlip_extxyz"] = str(
            write_mlip_extxyz(record, output_dir/"mlip_frame.xyz")
        )
    output_paths.update(_save_arrays(
        output_dir, stress, gradient=gradient, force=force, fd=fd_result
    ))

    current_energy_seconds = 0.0 if method is None else energy_seconds
    current_gradient_seconds = (
        0.0 if gradient_checkpoint_reused else gradient_seconds
    )
    current_stress_seconds = (
        0.0 if stress_checkpoint_reused else stress_seconds
    )
    timing = {
        "setup_wall_s": setup_seconds,
        "source_hf_and_target_energy_wall_s": energy_seconds,
        "analytic_gradient_wall_s": gradient_seconds if gradient is not None else None,
        "analytic_stress_wall_s": stress_seconds,
        "gradient_and_stress_wall_s": (
            gradient_seconds+stress_seconds if gradient is not None else None
        ),
        "analytic_total_wall_s": energy_seconds+gradient_seconds+stress_seconds,
        "stress_fd_wall_s": fd_seconds,
        "current_attempt_source_hf_and_target_energy_wall_s": (
            current_energy_seconds
        ),
        "current_attempt_analytic_gradient_wall_s": (
            current_gradient_seconds if gradient is not None else None
        ),
        "current_attempt_analytic_stress_wall_s": current_stress_seconds,
        "current_attempt_analytic_total_wall_s": (
            current_energy_seconds
            + current_gradient_seconds
            + current_stress_seconds
        ),
        "workflow_wall_s": perf_counter()-workflow_started,
        "stress_internal_s": dict(stress.timings_s),
    }
    return {
        "status": "complete",
        "execution_status": "complete",
        "accuracy_status": accuracy_status,
        "schema_version": 2,
        "run_name": config.get("name"),
        "benchmark_scope": config.get("benchmark_scope"),
        "input_fingerprint": config.get("input_fingerprint"),
        "calculation_mode": mode,
        "device": device,
        "df_backend": backend,
        "target_integrator": "uniform",
        "xc": str(config.get("xc", "PBE")),
        "system": {
            "natoms": system.natoms,
            "atom_file": config.get("atom_file"),
            "lattice_bohr": np.asarray(cell.lattice_vectors()),
            "volume_bohr3": float(cell.vol),
            "mesh": np.asarray(cell.mesh, dtype=int),
            "kmesh": list(config.get("kmesh", (1, 1, 1))),
            "basis": config.get("basis"),
            "pseudo": config.get("pseudo"),
            "auxbasis": config.get("auxbasis"),
        },
        "energy_ha": energy,
        "state_fingerprint": fingerprint,
        "gradient_ha_per_bohr": gradient,
        "force_ha_per_bohr": force,
        "stress": stress.as_dict(),
        "mlip_record": None if record is None else record.as_dict(),
        "accuracy": accuracy,
        "timing_s": timing,
        "gradient_metadata": gradient_metadata,
        "checkpoint": {
            **checkpoint.describe(),
            "source_density_warm_start": source_density_warm_start,
            "gradient_reused": gradient_checkpoint_reused,
            "stress_reused": stress_checkpoint_reused,
            "state_policy": (
                "NPY numerical state; PySCF objects are reconstructed, not pickled"
            ),
        },
        "output_paths": output_paths,
        "memory": {
            "host_peak_rss_bytes": _host_peak_rss_bytes(),
            "stress_device_memory": stress.metadata.get("device_memory"),
            "stress_df_memory_estimate": stress.metadata.get("df_memory_estimate"),
        },
        "software_provenance": gradient_benchmark.evidence.software_provenance(device),
        "run_identity": config.get("run_identity"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    ensure_stress_extension()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(input_path.read_text(encoding="utf-8"))
    progress = gradient_benchmark.ProgressReporter(
        output_dir/"progress.json",
        context={
            "run_name": config.get("name"),
            "device": config.get("device"),
            "df_backend": config.get("df_backend"),
            "calculation_mode": config.get("calculation_mode"),
            "run_identity": config.get("run_identity"),
        },
    )
    tracker = gradient_benchmark.VRAMPeakTracker(
        str(config.get("device", "cpu")),
        enabled=bool(config.get("track_vram_peak", True)),
        sample_interval_s=float(config.get("vram_sample_interval_s", 0.05)),
        stage_provider=lambda: progress.stage,
    )
    try:
        report = run_case(
            config, output_dir, input_path.parent,
            progress=progress, vram_tracker=tracker,
        )
        report["memory"]["vram_peak_tracking"] = tracker.stop()
        write_json(output_dir/"report.json", report)
        progress.complete(report_path=str(output_dir/"report.json"))
    except Exception as error:
        failure = {
            "status": "failed",
            "execution_status": "failed",
            "accuracy_status": None,
            "schema_version": 2,
            "run_name": config.get("name"),
            "run_identity": config.get("run_identity"),
            "input_fingerprint": config.get("input_fingerprint"),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "memory": {"vram_peak_tracking": tracker.stop()},
        }
        write_json(output_dir/"report.json", failure)
        progress.fail(error)
        raise
    print(f"execution status: {report['execution_status']}")
    print(f"accuracy status: {report['accuracy_status']}")
    print(f"energy: {report['energy_ha']:.12f} Ha")
    print(f"analytic stress: {report['timing_s']['analytic_stress_wall_s']:.6f} s")
    if report["timing_s"]["analytic_gradient_wall_s"] is not None:
        print(f"analytic gradient: {report['timing_s']['analytic_gradient_wall_s']:.6f} s")
    return 0 if report["accuracy_status"] != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
