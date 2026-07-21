"""D2C-D4 energy and nuclear-gradient correction.

The damping parameters are from Table S1 of Lee et al., JCTC 2024,
DOI 10.1021/acs.jctc.4c00689.  D2C is independent of the HF density and
therefore contributes an explicit geometry derivative, but no Z-vector term.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from importlib.resources import files
import json
from time import perf_counter
from typing import Any

import numpy as np
from pyscf import gto

from ..lib.types import DCEnergyResult


_DISPERSION_ALIASES = {"d2c", "d2cd4", "d2c-d4", "d2c_d4"}


def normalize_dispersion(value: Any) -> str | None:
    """Normalize the public dispersion option without guessing parameters."""

    if value is None or value is False:
        return None
    if value is True:
        return "d2c"
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "off", "false"}:
        return None
    if normalized in _DISPERSION_ALIASES:
        return "d2c"
    raise ValueError("dispersion must be None or 'd2c'")


def _functional_key(xc: str) -> str:
    return "".join(character for character in str(xc).lower() if character.isalnum())


@lru_cache(maxsize=1)
def _parameter_registry() -> dict[str, Any]:
    path = files(__package__).joinpath("parameters", "d2c_d4.json")
    return json.loads(path.read_text(encoding="utf-8"))


def get_d2c_parameters(xc: str) -> dict[str, Any]:
    """Return one published parameter set, rejecting unsupported XC labels."""

    registry = _parameter_registry()
    key = _functional_key(xc)
    try:
        functional = registry["functionals"][key]
    except KeyError as exc:
        supported = ", ".join(
            value["label"] for value in registry["functionals"].values()
        )
        raise ValueError(
            f"no published D2C-D4 parameters for xc={xc!r}; "
            f"supported functionals: {supported}"
        ) from exc
    return {
        "functional": functional["label"],
        "s6": float(registry["global_parameters"]["s6"]),
        "s8": float(functional["s8"]),
        "s9": float(registry["global_parameters"]["s9"]),
        "a1": float(functional["a1"]),
        "a2": float(functional["a2"]),
        "alp": float(registry["global_parameters"]["alp"]),
        "model": str(registry["model"]),
        "damping": str(registry["damping"]),
        "source": dict(registry["source"]),
    }


@dataclass(frozen=True)
class D2CCorrection:
    """One D2C-D4 evaluation in atomic units."""

    energy: float
    gradient: np.ndarray | None
    virial: np.ndarray | None
    parameters: dict[str, Any]
    wall_time_s: float

    def metadata(self) -> dict[str, Any]:
        gradient_max_abs = (
            None
            if self.gradient is None
            else float(np.max(np.abs(self.gradient)))
        )
        return {
            "name": "D2C-D4",
            "energy_ha": self.energy,
            "gradient_available": self.gradient is not None,
            "gradient_max_abs_ha_per_bohr": gradient_max_abs,
            "virial_available": self.virial is not None,
            "parameters": dict(self.parameters),
            "wall_time_s": self.wall_time_s,
            "execution_device": "cpu",
            "density_response": False,
        }


def _import_dftd4():
    try:
        from dftd4.interface import DampingParam, DispersionModel
    except ImportError as exc:
        raise ImportError(
            "D2C-D4 requires the optional dftd4 package; install "
            "pyscf-pbc-dcdft[dispersion] or dftd4>=4.2"
        ) from exc
    return DampingParam, DispersionModel


def _periodic_axes(cell) -> np.ndarray:
    dimension = int(getattr(cell, "dimension", 3))
    if dimension < 0 or dimension > 3:
        raise ValueError(f"unsupported periodic dimension {dimension}")
    return np.arange(3) < dimension


def evaluate_d2c(cell, xc: str, *, gradient: bool) -> D2CCorrection:
    """Evaluate published D2C-D4 parameters for a PySCF periodic cell."""

    start = perf_counter()
    DampingParam, DispersionModel = _import_dftd4()
    parameters = get_d2c_parameters(xc)
    numbers = np.asarray(
        [gto.charge(symbol) for symbol in cell.elements], dtype=np.int32
    )
    model = DispersionModel(
        numbers=numbers,
        positions=np.asarray(cell.atom_coords(), dtype=float),
        charge=float(cell.charge),
        lattice=np.asarray(cell.lattice_vectors(), dtype=float),
        periodic=_periodic_axes(cell),
        model=parameters["model"],
    )
    damping = DampingParam(
        s6=parameters["s6"],
        s8=parameters["s8"],
        s9=parameters["s9"],
        a1=parameters["a1"],
        a2=parameters["a2"],
        alp=parameters["alp"],
    )
    result = model.get_dispersion(param=damping, grad=bool(gradient))
    wall_time_s = perf_counter() - start
    energy = float(np.asarray(result["energy"]).item())
    gradient_array = result.get("gradient")
    virial = result.get("virial")
    if not np.isfinite(energy):
        raise RuntimeError("D2C-D4 returned a non-finite energy")
    if gradient_array is not None:
        gradient_array = np.asarray(gradient_array, dtype=float)
        if gradient_array.shape != (cell.natm, 3):
            raise RuntimeError(
                "D2C-D4 gradient has shape "
                f"{gradient_array.shape}, expected {(cell.natm, 3)}"
            )
    if virial is not None:
        virial = np.asarray(virial, dtype=float)
    return D2CCorrection(
        energy=energy,
        gradient=gradient_array,
        virial=virial,
        parameters=parameters,
        wall_time_s=wall_time_s,
    )


def apply_d2c_energy(
    result: DCEnergyResult, correction: D2CCorrection
) -> DCEnergyResult:
    """Add D2C to a frozen-density energy without changing its HF reference."""

    components = replace(
        result.components,
        dispersion=correction.energy,
        total=result.components.total + correction.energy,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "formula": "E_DFA[n_HF] + E_D2C-D4",
            "dispersion": correction.metadata(),
        }
    )
    return replace(
        result,
        energy=result.energy + correction.energy,
        components=components,
        metadata=metadata,
    )


@dataclass(frozen=True)
class DispersionCorrectedGradientResult:
    """Transparent result wrapper adding D2C to the explicit channel."""

    electronic_result: Any
    dispersion: D2CCorrection

    @property
    def contributions(self):
        base = self.electronic_result.contributions
        gradient = self.dispersion_gradient
        return replace(
            base,
            explicit=base.explicit + gradient,
            total=base.total + gradient,
        )

    @property
    def dispersion_gradient(self) -> np.ndarray:
        if self.dispersion.gradient is None:
            raise RuntimeError("D2C-D4 gradient was not evaluated")
        return self.dispersion.gradient

    @property
    def electronic_gradient(self) -> np.ndarray:
        return np.asarray(self.electronic_result.gradient)

    @property
    def gradient(self) -> np.ndarray:
        return self.electronic_gradient + self.dispersion_gradient

    def metadata(self) -> dict[str, Any]:
        payload = dict(self.electronic_result.metadata())
        payload["dispersion"] = self.dispersion.metadata()
        payload["formula"] = "dE_DFA[n_HF]/dR + dE_D2C-D4/dR"
        component_max_abs = dict(payload.get("component_max_abs", {}))
        component_max_abs["dispersion_d2c"] = float(
            np.max(np.abs(self.dispersion_gradient))
        )
        payload["component_max_abs"] = component_max_abs
        timings = dict(payload.get("timings_s", {}))
        timings["dispersion_d2c_gradient"] = self.dispersion.wall_time_s
        payload["timings_s"] = timings
        host_boundaries = list(payload.get("host_boundaries", []))
        boundary = "periodic D2C-D4 energy/gradient library"
        if boundary not in host_boundaries:
            host_boundaries.append(boundary)
        if host_boundaries:
            payload["host_boundaries"] = host_boundaries
        return payload

    def __getattr__(self, name: str) -> Any:
        return getattr(self.electronic_result, name)


def apply_d2c_gradient(
    result: Any, correction: D2CCorrection
) -> DispersionCorrectedGradientResult:
    """Return a result whose total and explicit gradients include D2C-D4."""

    if correction.gradient is None:
        raise ValueError("gradient correction requires a D2C gradient")
    return DispersionCorrectedGradientResult(result, correction)
