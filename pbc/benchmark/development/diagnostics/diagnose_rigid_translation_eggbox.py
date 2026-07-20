#!/usr/bin/env python
"""Measure rigid-translation energy variation over one uniform-grid step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


BENCHMARK = Path(__file__).resolve().parents[2]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from dcdft_benchmark import (  # noqa: E402
    _configure_gradient,
    _validate_config,
    build_cell,
    build_method,
    evaluate_energy,
    resolve_system,
    synchronize,
    write_json,
)


PHASE_STEP = 0.125
DEFAULT_PHASES = (
    -0.125,
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
)


def _sum_vector(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    return np.sum(array, axis=0)


def _component_translation_sums(result):
    contributions = result.contributions
    fields = {
        "one_electron": getattr(result, "one_electron", None),
        "target_coulomb": getattr(result, "target_coulomb", None),
        "target_xc": getattr(result, "target_xc", None),
        "target_overlap_pulay": getattr(
            result, "target_overlap_pulay", None
        ),
        "nuclear": getattr(result, "nuclear", None),
        "fused_target_j_z_bilinear": getattr(
            result, "fused_target_j_z_bilinear", None
        ),
        "explicit_channel": contributions.explicit,
        "response_channel": contributions.response,
        "coupled_channel": contributions.coupled,
        "total": contributions.total,
    }
    return {name: _sum_vector(value) for name, value in fields.items()}


def _sampled_periodic_slopes(records, grid_step_norm):
    """Central energy slopes at eight equally spaced grid phases."""

    energies = {
        round(float(record["phase"]), 3): float(record["energy_ha"])
        for record in records
    }
    denominator = 2.0 * PHASE_STEP * float(grid_step_norm)
    slopes = []
    for index in range(8):
        phase = index * PHASE_STEP
        minus = phase - PHASE_STEP
        plus = phase + PHASE_STEP
        slopes.append(
            {
                "phase": phase,
                "slope_ha_per_bohr": (
                    energies[round(plus, 3)]
                    - energies[round(minus, 3)]
                )
                / denominator,
            }
        )
    return slopes


def _analytic_gradient(config, system, coordinates, device, backend):
    cell = build_cell(config, system, coordinates)
    method, _kpts = build_method(config, cell, device, backend)
    synchronize(device)
    start = perf_counter()
    energy = float(method.kernel())
    gradient_object = method.nuc_grad_method()
    _configure_gradient(config, gradient_object)
    gradient = np.asarray(gradient_object.kernel(), dtype=float)
    synchronize(device)
    return {
        "energy_ha": energy,
        "gradient_ha_per_bohr": gradient,
        "translation_sum_vector_ha_per_bohr": np.sum(gradient, axis=0),
        "component_translation_sums_ha_per_bohr": (
            _component_translation_sums(gradient_object.result)
        ),
        "wall_time_s": perf_counter() - start,
    }


def diagnose(
    input_path: Path,
    axis: int,
    *,
    slope_step_bohr: float = 1.0e-4,
    center_phase: float = 0.0,
    ke_cutoff_ha: float | None = None,
    mesh: tuple[int, int, int] | None = None,
    xc: str | None = None,
    df_backend: str | None = None,
    target_integrator: str | None = None,
    target_multigrid_mesh: tuple[int, int, int] | None = None,
) -> dict[str, object]:
    config = json.loads(input_path.read_text(encoding="utf-8"))
    if ke_cutoff_ha is not None and mesh is not None:
        raise ValueError("ke_cutoff_ha and mesh are mutually exclusive")
    if ke_cutoff_ha is not None:
        config.pop("mesh", None)
        config["ke_cutoff_ha"] = float(ke_cutoff_ha)
    if mesh is not None:
        if len(mesh) != 3 or any(int(value) <= 0 for value in mesh):
            raise ValueError("mesh must contain three positive integers")
        config.pop("ke_cutoff_ha", None)
        config["mesh"] = [int(value) for value in mesh]
    if xc is not None:
        config["xc"] = str(xc)
    if df_backend is not None:
        config["df_backend"] = str(df_backend)
    if target_integrator is not None:
        config["target_integrator"] = str(target_integrator)
    if target_multigrid_mesh is not None:
        config["target_multigrid_mesh"] = [
            int(value) for value in target_multigrid_mesh
        ]
    device, backend = _validate_config(config)
    system = resolve_system(config, input_path.parent)
    cell = build_cell(config, system)
    grid_step = np.asarray(system.lattice_bohr[axis]) / int(cell.mesh[axis])
    grid_step_norm = float(np.linalg.norm(grid_step))
    direction = grid_step / grid_step_norm

    records = []
    total_start = perf_counter()
    for phase in DEFAULT_PHASES:
        coordinates = system.coordinates_bohr + float(phase) * grid_step
        run = evaluate_energy(config, system, coordinates, device, backend)
        records.append(
            {
                "phase": float(phase),
                "translation_bohr": float(phase) * grid_step,
                "energy_ha": run.energy_ha,
                "setup_wall_time_s": run.setup_wall_time_s,
                "energy_wall_time_s": run.wall_time_s,
            }
        )

    energies = np.asarray([record["energy_ha"] for record in records])
    phase_to_index = {
        float(phase): index for index, phase in enumerate(DEFAULT_PHASES)
    }
    zero_index = DEFAULT_PHASES.index(0.0)
    one_index = DEFAULT_PHASES.index(1.0)
    zero_energy = float(energies[zero_index])
    for record in records:
        record["energy_minus_phase_zero_ha"] = (
            float(record["energy_ha"]) - zero_energy
        )

    slope_half_width_phase = 0.125
    minus_index = phase_to_index[-slope_half_width_phase]
    plus_index = phase_to_index[slope_half_width_phase]
    translation_interval_bohr = 2.0 * slope_half_width_phase * np.linalg.norm(
        grid_step
    )
    local_slope = (
        energies[plus_index] - energies[minus_index]
    ) / translation_interval_bohr

    slope_step_bohr = float(slope_step_bohr)
    if slope_step_bohr <= 0.0:
        raise ValueError("slope_step_bohr must be positive")
    center_phase = float(center_phase)
    center_coordinates = (
        system.coordinates_bohr + center_phase * grid_step
    )
    displacement = slope_step_bohr * direction
    small_minus = evaluate_energy(
        config,
        system,
        center_coordinates - displacement,
        device,
        backend,
    )
    small_plus = evaluate_energy(
        config,
        system,
        center_coordinates + displacement,
        device,
        backend,
    )
    small_step_slope = (
        small_plus.energy_ha - small_minus.energy_ha
    ) / (2.0 * slope_step_bohr)
    analytic = _analytic_gradient(
        config,
        system,
        center_coordinates,
        device,
        backend,
    )
    analytic_directional_sum = float(
        np.dot(analytic["translation_sum_vector_ha_per_bohr"], direction)
    )
    sampled_slopes = _sampled_periodic_slopes(records, grid_step_norm)
    sampled_slope_values = np.asarray(
        [record["slope_ha_per_bohr"] for record in sampled_slopes]
    )

    return {
        "schema": "pbc_dcdft_rigid_translation_eggbox_v2",
        "input": str(input_path),
        "xc": str(config.get("xc", "PBE")),
        "device": device,
        "df_backend": backend,
        "target_integrator": str(
            config.get("target_integrator", "uniform")
        ),
        "target_multigrid_mesh": config.get("target_multigrid_mesh"),
        "axis": "xyz"[axis],
        "mesh": [int(value) for value in cell.mesh],
        "ke_cutoff_ha": getattr(cell, "ke_cutoff", None),
        "grid_step_bohr": grid_step,
        "grid_step_norm_bohr": grid_step_norm,
        "translation_direction": direction,
        "energy_peak_to_peak_ha": float(np.ptp(energies)),
        "sampled_periodic_slopes": sampled_slopes,
        "worst_sampled_rigid_translation_slope_ha_per_bohr": float(
            np.max(np.abs(sampled_slope_values))
        ),
        "rms_sampled_rigid_translation_slope_ha_per_bohr": float(
            np.sqrt(np.mean(sampled_slope_values**2))
        ),
        "one_grid_step_closure_error_ha": float(
            energies[one_index] - energies[zero_index]
        ),
        "periodic_endpoint_closure_error_ha": float(
            energies[phase_to_index[0.875]] - energies[minus_index]
        ),
        "local_slope_half_width_phase": slope_half_width_phase,
        "local_rigid_translation_slope_ha_per_bohr": float(local_slope),
        "small_step_bohr": slope_step_bohr,
        "analytic_center_phase": center_phase,
        "analytic_center_translation_bohr": center_phase * grid_step,
        "small_step_rigid_translation_slope_ha_per_bohr": float(
            small_step_slope
        ),
        "analytic_directional_translation_sum_ha_per_bohr": (
            analytic_directional_sum
        ),
        "analytic_minus_small_step_fd_ha_per_bohr": float(
            analytic_directional_sum - small_step_slope
        ),
        "analytic": analytic,
        "records": records,
        "wall_time_s": perf_counter() - total_start,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--slope-step-bohr", type=float, default=1.0e-4)
    parser.add_argument("--center-phase", type=float, default=0.0)
    parser.add_argument("--ke-cutoff-ha", type=float)
    parser.add_argument(
        "--mesh",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
    )
    parser.add_argument("--xc")
    parser.add_argument("--df-backend", choices=("fftdf", "gdf", "rsdf"))
    parser.add_argument(
        "--target-integrator",
        choices=("uniform", "multigrid_xc", "multigrid_j_xc"),
    )
    parser.add_argument(
        "--target-multigrid-mesh",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    result = diagnose(
        input_path,
        "xyz".index(args.axis),
        slope_step_bohr=args.slope_step_bohr,
        center_phase=args.center_phase,
        ke_cutoff_ha=args.ke_cutoff_ha,
        mesh=None if args.mesh is None else tuple(args.mesh),
        xc=args.xc,
        df_backend=args.df_backend,
        target_integrator=args.target_integrator,
        target_multigrid_mesh=(
            None
            if args.target_multigrid_mesh is None
            else tuple(args.target_multigrid_mesh)
        ),
    )
    write_json(output_path, result)
    print(f"peak-to-peak: {result['energy_peak_to_peak_ha']:.9e} Ha")
    print(
        "one-grid closure: "
        f"{result['one_grid_step_closure_error_ha']:.9e} Ha"
    )
    print(
        "local rigid-translation slope: "
        f"{result['local_rigid_translation_slope_ha_per_bohr']:.9e} Ha/Bohr"
    )
    print(
        "analytic / small-step FD: "
        f"{result['analytic_directional_translation_sum_ha_per_bohr']:.9e} / "
        f"{result['small_step_rigid_translation_slope_ha_per_bohr']:.9e} "
        "Ha/Bohr"
    )
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
