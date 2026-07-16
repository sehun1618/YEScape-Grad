"""Zero-transfer ``vbar`` correction to the periodic GDF source."""

from __future__ import annotations

import numpy as np
from pyscf.pbc.df import rsdf


def vbar_source(cell, auxcell, omega: float, *, kpt: np.ndarray) -> np.ndarray:
    """Return ``-S(k) pi q_P / (omega**2 Omega)``."""

    point = _kpoint(kpt)
    overlap = np.asarray(
        cell.pbc_intor("int1e_ovlp", hermi=1, kpts=point.reshape(1, 3))
    )[0]
    return -overlap.reshape(cell.nao_nr() ** 2, 1) * _vbar(
        cell, auxcell, omega
    )[None, :]


def vbar_source_derivative(
    cell,
    auxcell,
    atom_id: int,
    omega: float,
    *,
    kpt: np.ndarray,
) -> np.ndarray:
    """Return fixed-cell center derivative of the zero-transfer source."""

    atom = int(atom_id)
    if atom < 0 or atom >= cell.natm:
        raise ValueError(f"atom_id out of range: {atom_id}")
    point = _kpoint(kpt)
    ip = -np.asarray(
        cell.pbc_intor("int1e_ipovlp", kpts=point.reshape(1, 3))
    )[0]
    p0, p1 = np.asarray(cell.aoslice_by_atom(), dtype=int)[atom, 2:4]
    coefficient = _vbar(cell, auxcell, omega)[None, :]
    result = []
    for axis in range(3):
        d_overlap = np.zeros_like(ip[axis])
        d_overlap[p0:p1, :] += ip[axis, p0:p1, :]
        d_overlap[:, p0:p1] += ip[axis, p0:p1, :].T.conj()
        result.append(-d_overlap.reshape(cell.nao_nr() ** 2, 1) * coefficient)
    return np.asarray(result)


def vbar_source_derivatives(
    cell,
    auxcell,
    omega: float,
    *,
    kpt: np.ndarray,
) -> np.ndarray:
    """Return all-atom zero-transfer source derivatives from one overlap."""

    point = _kpoint(kpt)
    ip = -np.asarray(
        cell.pbc_intor("int1e_ipovlp", kpts=point.reshape(1, 3))
    )[0]
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    coefficient = _vbar(cell, auxcell, omega)[None, :]
    result = np.zeros(
        (int(cell.natm), 3, nao, nao, naux), dtype=ip.dtype
    )
    slices = np.asarray(cell.aoslice_by_atom(), dtype=int)
    for atom_id in range(cell.natm):
        p0, p1 = slices[atom_id, 2:4]
        for axis in range(3):
            d_overlap = np.zeros((nao, nao), dtype=ip.dtype)
            d_overlap[p0:p1, :] += ip[axis, p0:p1, :]
            d_overlap[:, p0:p1] += ip[axis, p0:p1, :].T.conj()
            result[atom_id, axis] = (
                -d_overlap[:, :, None] * coefficient[None, :, :]
            )
    return result.reshape(int(cell.natm), 3, nao * nao, naux)


def _vbar(cell, auxcell, omega):
    charges = np.asarray(rsdf.get_aux_chg(auxcell), dtype=float)
    return np.pi * charges / (float(omega) ** 2 * float(cell.vol))


def _kpoint(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"k-point must have shape (3,), got {point.shape}")
    return point
