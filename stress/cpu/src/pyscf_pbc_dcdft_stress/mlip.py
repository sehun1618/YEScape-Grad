"""Validated MLIP records for PBC DC-DFT energy, force, and stress."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from pyscf.lib import param


HARTREE_TO_EV = 27.211386245988


def stress_metadata_for_mlip(stress_result):
    """Keep the stress owner contract with each exported MLIP frame."""

    source = dict(getattr(stress_result, "metadata", {}) or {})
    mapping = {
        "stage": "stress_stage",
        "production": "stress_production",
        "low_level_numerical_components":
            "stress_low_level_numerical_components",
        "primitive_backend": "stress_primitive_backend",
        "pp_response_backend": "stress_pp_response_backend",
        "df_response_backend": "stress_df_response_backend",
        "source_hf_owner": "stress_source_hf_owner",
        "target_fock_owner": "stress_target_fock_owner",
        "z_vector_owner": "stress_z_vector_owner",
    }
    return {
        target: source[key] for key, target in mapping.items()
        if key in source
    }


@dataclass(frozen=True)
class MLIPRecord:
    """One periodic configuration with explicit unit/sign conventions."""

    symbols: tuple[str, ...]
    positions_bohr: np.ndarray
    lattice_bohr: np.ndarray
    energy_ha: float
    forces_ha_per_bohr: np.ndarray
    stress_tensile_ha_per_bohr3: np.ndarray
    virial_ha: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        natm = len(self.symbols)
        positions = np.asarray(self.positions_bohr, dtype=float)
        lattice = np.asarray(self.lattice_bohr, dtype=float)
        forces = np.asarray(self.forces_ha_per_bohr, dtype=float)
        stress = np.asarray(self.stress_tensile_ha_per_bohr3, dtype=float)
        virial = np.asarray(self.virial_ha, dtype=float)
        if positions.shape != (natm, 3) or forces.shape != (natm, 3):
            raise ValueError("positions and forces must have shape (natm, 3)")
        if lattice.shape != (3, 3):
            raise ValueError("lattice must have shape (3, 3)")
        if stress.shape != (3, 3) or virial.shape != (3, 3):
            raise ValueError("stress and virial must have shape (3, 3)")
        if not all(np.all(np.isfinite(value)) for value in (
            positions, lattice, forces, stress, virial,
        )):
            raise ValueError("MLIP record contains NaN or Inf")
        volume = abs(float(np.linalg.det(lattice)))
        if volume <= 0.0:
            raise ValueError("lattice volume must be positive")
        expected_virial = -volume*stress
        if not np.allclose(virial, expected_virial, atol=1e-10, rtol=1e-10):
            raise ValueError("virial must satisfy W = -volume * tensile stress")
        object.__setattr__(self, "positions_bohr", positions)
        object.__setattr__(self, "lattice_bohr", lattice)
        object.__setattr__(self, "forces_ha_per_bohr", forces)
        object.__setattr__(self, "stress_tensile_ha_per_bohr3", stress)
        object.__setattr__(self, "virial_ha", virial)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def volume_bohr3(self):
        return abs(float(np.linalg.det(self.lattice_bohr)))

    @property
    def positions_angstrom(self):
        return self.positions_bohr*param.BOHR

    @property
    def lattice_angstrom(self):
        return self.lattice_bohr*param.BOHR

    @property
    def energy_ev(self):
        return float(self.energy_ha)*HARTREE_TO_EV

    @property
    def forces_ev_per_angstrom(self):
        return self.forces_ha_per_bohr*HARTREE_TO_EV/param.BOHR

    @property
    def stress_tensile_ev_per_angstrom3(self):
        return (
            self.stress_tensile_ha_per_bohr3
            * HARTREE_TO_EV/(param.BOHR**3)
        )

    def as_dict(self):
        return {
            "symbols": list(self.symbols),
            "positions_bohr": self.positions_bohr.tolist(),
            "lattice_bohr": self.lattice_bohr.tolist(),
            "energy_ha": float(self.energy_ha),
            "forces_ha_per_bohr": self.forces_ha_per_bohr.tolist(),
            "stress_tensile_ha_per_bohr3": (
                self.stress_tensile_ha_per_bohr3.tolist()
            ),
            "virial_ha": self.virial_ha.tolist(),
            "volume_bohr3": self.volume_bohr3,
            "units": {
                "energy": "Hartree",
                "positions": "Bohr",
                "forces": "Hartree/Bohr",
                "stress": "Hartree/Bohr^3",
                "virial": "Hartree",
            },
            "conventions": {
                "force": "force = -nuclear_gradient",
                "stress": "tensile_positive",
                "virial": "W = -volume * stress",
            },
            "metadata": _json_safe(dict(self.metadata)),
        }


def mlip_record_from_results(
    cell, energy_ha, nuclear_gradient_ha_per_bohr, stress_result, *, metadata=None,
):
    """Combine same-state energy, gradient, and stress without sign ambiguity."""

    lattice = np.asarray(cell.lattice_vectors(), dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    reported_volume = float(stress_result.volume_bohr3)
    if not np.isclose(volume, reported_volume, atol=1e-10, rtol=1e-10):
        raise ValueError(
            "stress result volume does not match the supplied calculation cell"
        )
    stress = np.asarray(
        stress_result.stress_tensor_ha_per_bohr3, dtype=float
    )
    gradient = np.asarray(nuclear_gradient_ha_per_bohr, dtype=float)
    return MLIPRecord(
        symbols=tuple(cell.atom_symbol(index) for index in range(cell.natm)),
        positions_bohr=np.asarray(cell.atom_coords(), dtype=float),
        lattice_bohr=lattice,
        energy_ha=float(energy_ha),
        forces_ha_per_bohr=-gradient,
        stress_tensile_ha_per_bohr3=stress,
        virial_ha=-volume*stress,
        metadata={} if metadata is None else metadata,
    )


def write_mlip_json(record, path):
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record.as_dict(), indent=2)+"\n", encoding="utf-8")
    return output


def write_mlip_extxyz(record, path, *, append=False):
    """Write an extxyz frame in eV/Angstrom with explicit DC-DFT fields."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lattice = " ".join(f"{value:.16g}" for value in record.lattice_angstrom.ravel())
    stress = " ".join(
        f"{value:.16g}"
        for value in record.stress_tensile_ev_per_angstrom3.ravel()
    )
    virial = " ".join(f"{value:.16g}" for value in record.virial_ha.ravel())
    provenance = _extxyz_provenance(record.metadata)
    header = (
        f'Lattice="{lattice}" '
        'Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy={record.energy_ev:.16g} '
        f'dcdft_stress_tensile_eV_per_A3="{stress}" '
        f'dcdft_virial_Ha="{virial}" '
        'stress_convention=tensile_positive '
        'virial_convention="W=-V*stress" '
        f'{provenance}pbc="T T T"'
    )
    lines = [str(len(record.symbols)), header]
    for symbol, position, force in zip(
        record.symbols, record.positions_angstrom,
        record.forces_ev_per_angstrom,
    ):
        values = " ".join(
            f"{value:.16g}" for value in np.concatenate((position, force))
        )
        lines.append(f"{symbol} {values}")
    mode = "a" if append else "w"
    with output.open(mode, encoding="utf-8") as handle:
        handle.write("\n".join(lines)+"\n")
    return output


def _extxyz_provenance(metadata):
    names = {
        "device": "dcdft_device",
        "df_backend": "dcdft_df_backend",
        "xc": "dcdft_xc",
        "state_fingerprint": "dcdft_state_fingerprint",
        "stress_production": "dcdft_stress_production",
        "stress_pp_response_backend": "dcdft_stress_pp_backend",
        "stress_df_response_backend": "dcdft_stress_df_backend",
    }
    fields = []
    for source, target in names.items():
        if source not in metadata:
            continue
        value = metadata[source]
        if isinstance(value, bool):
            value = "T" if value else "F"
        text = str(value).replace(" ", "_").replace('"', "")
        fields.append(f"{target}={text}")
    return "" if not fields else " ".join(fields)+" "


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
