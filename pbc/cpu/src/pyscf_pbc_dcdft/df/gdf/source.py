"""Three-center RSGDF source and its fixed-cell nuclear derivative."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf.pbc.lib.kpts_helper import is_zero

from .source_long_range import (
    long_range_source,
    long_range_source_derivative,
)
from .source_short_range import (
    short_range_source,
    short_range_source_derivative,
)
from .vbar import vbar_source, vbar_source_derivative


@dataclass(frozen=True)
class SourceDerivativeParts:
    """Resolved derivative of one three-center source block."""

    short_range: np.ndarray
    long_range: np.ndarray
    vbar: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """Return ``dB_SR + dB_LR + dB_vbar``."""

        return self.short_range + self.long_range + self.vbar


def three_center_source(
    cell,
    auxcell,
    omega: float,
    *,
    mesh: tuple[int, int, int] | np.ndarray,
    kpti: np.ndarray,
    kptj: np.ndarray,
) -> np.ndarray:
    """Return the full RSGDF pre-solve source ``B(ki,kj)``.

    The result has shape ``(nao**2, naux)``.  ``vbar`` is present only for a
    zero-transfer pair and is evaluated with the pair's absolute k-point.
    """

    ki, kj = _kpoint_pair(kpti, kptj)
    source = short_range_source(
        cell, auxcell, omega, kpti=ki, kptj=kj
    )
    source = source + long_range_source(
        cell,
        auxcell,
        omega,
        mesh=mesh,
        kpti=ki,
        kptj=kj,
    )
    if is_zero(kj - ki):
        source = source + vbar_source(cell, auxcell, omega, kpt=ki)
    return np.asarray(source)


def three_center_source_derivative(
    cell,
    auxcell,
    atom_id: int,
    omega: float,
    *,
    mesh: tuple[int, int, int] | np.ndarray,
    kpti: np.ndarray,
    kptj: np.ndarray,
) -> SourceDerivativeParts:
    """Return all Cartesian derivatives of ``B(ki,kj)`` for one atom."""

    ki, kj = _kpoint_pair(kpti, kptj)
    short_range = short_range_source_derivative(
        cell,
        auxcell,
        atom_id,
        omega,
        kpti=ki,
        kptj=kj,
    )
    long_range = long_range_source_derivative(
        cell,
        auxcell,
        atom_id,
        omega,
        mesh=mesh,
        kpti=ki,
        kptj=kj,
    )
    if is_zero(kj - ki):
        vbar = vbar_source_derivative(
            cell, auxcell, atom_id, omega, kpt=ki
        )
    else:
        vbar = np.zeros_like(short_range)
    return SourceDerivativeParts(
        short_range=np.asarray(short_range),
        long_range=np.asarray(long_range),
        vbar=np.asarray(vbar),
    )


def _kpoint_pair(kpti, kptj):
    first = np.asarray(kpti, dtype=float)
    second = np.asarray(kptj, dtype=float)
    if first.shape != (3,) or second.shape != (3,):
        raise ValueError(
            "kpti and kptj must each have shape (3,), got "
            f"{first.shape} and {second.shape}"
        )
    return first, second
