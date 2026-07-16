#!/usr/bin/env python
"""Measure rigid-translation energy variation over one uniform-grid step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
BENCHMARK = WORKSPACE / "benchmark"
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from dcdft_benchmark import (  # noqa: E402
    _validate_config,
    build_cell,
    evaluate_energy,
    resolve_system,
    write_json,
)


DEFAULT_PHASES = (-0.125, 0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0)


def diagnose(input_path: Path, axis: int) -> dict[str, object]:
    config = json.loads(input_path.read_text(encoding="utf-8"))
    device, backend = _validate_config(config)
    system = resolve_system(config, input_path.parent)
    cell = build_cell(config, system)
    grid_step = np.asarray(system.lattice_bohr[axis]) / int(cell.mesh[axis])

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

    return {
        "schema": "pbc_dcdft_rigid_translation_eggbox_v1",
        "input": str(input_path),
        "xc": str(config.get("xc", "PBE")),
        "device": device,
        "df_backend": backend,
        "axis": "xyz"[axis],
        "mesh": [int(value) for value in cell.mesh],
        "grid_step_bohr": grid_step,
        "grid_step_norm_bohr": float(np.linalg.norm(grid_step)),
        "energy_peak_to_peak_ha": float(np.ptp(energies)),
        "one_grid_step_closure_error_ha": float(
            energies[one_index] - energies[zero_index]
        ),
        "periodic_endpoint_closure_error_ha": float(
            energies[phase_to_index[0.875]] - energies[minus_index]
        ),
        "local_slope_half_width_phase": slope_half_width_phase,
        "local_rigid_translation_slope_ha_per_bohr": float(local_slope),
        "records": records,
        "wall_time_s": perf_counter() - total_start,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    result = diagnose(input_path, "xyz".index(args.axis))
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
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
