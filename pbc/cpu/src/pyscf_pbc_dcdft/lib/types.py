"""Typed public results for fixed-density energy calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DCEnergyComponents:
    """Terms in ``E_DFA[n_HF]`` in Hartree.

    ``one_electron`` contains kinetic plus electron-ion pseudopotential energy.
    ``coulomb`` and ``xc`` are the tags returned by the target PySCF KS
    effective potential.  Their sum with ``nuclear`` must reproduce ``total``.
    """

    one_electron: float
    coulomb: float
    xc: float
    nuclear: float
    total: float

    @property
    def reconstructed_total(self) -> float:
        return self.one_electron + self.coulomb + self.xc + self.nuclear


@dataclass(frozen=True)
class DCEnergyResult:
    """Immutable fixed-density DC-DFT energy result."""

    energy: float
    reference_energy: float
    target_xc: str
    nkpts: int
    components: DCEnergyComponents
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary with explicit energy units."""

        payload = asdict(self)
        payload["units"] = {"energy": "Ha", "kpoint": "1/Bohr"}
        payload["component_reconstruction_error_ha"] = (
            self.energy - self.components.reconstructed_total
        )
        return payload
