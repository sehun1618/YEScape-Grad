"""Differentiate periodic GDF Coulomb and exchange matrices from ``dCDERI``.

This module is independent of how the analytic ``dCDERI`` blocks are built.
It defines the tensor algebra consumed by restricted DC-PBE response:

- ``dJ[D]/dR`` for target Coulomb and source HF response,
- ``dK[D]/dR`` for source HF response,
- ``d(J-K/2)[D]/dR`` in the closed-shell convention.

Canonical input blocks use ``ki >= kj``.  Reverse k-point pairs are recovered
by Hermitian symmetry through :mod:`pyscf_pbc_dcdft.df.gdf.storage`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from pyscf.pbc.tools import pbc as pbc_tools

from .storage import Pair, derivative_pair_block, pair_block


@dataclass(frozen=True)
class GDFJKDerivatives:
    """Full fixed-density GDF J/K nuclear derivative matrices.

    Arrays use ``(natm, nkpts, 3, nao, nao)`` and Hartree/Bohr.  General
    k-point results are complex even though every final energy derivative is
    real after k-point summation.
    """

    coulomb: np.ndarray
    exchange: np.ndarray

    @property
    def restricted_hf(self) -> np.ndarray:
        """Return ``d(J-K/2)/dR`` for a closed-shell spatial density."""

        return self.coulomb - 0.5 * self.exchange


def gdf_jk_derivative(
    cderi: Mapping[Pair, np.ndarray],
    dcderi: Mapping[Pair, np.ndarray],
    density: np.ndarray,
) -> GDFJKDerivatives:
    """Return full GDF J and K derivative matrices at fixed density.

    ``density`` follows PySCF's closed-shell convention and therefore already
    includes occupation factor two.  The explicit ``1/nkpts`` in each matrix
    matches periodic ``get_jk``; the later energy contraction contributes its
    own k-point average.
    """

    dm = _density_array(density)
    natm, ncart, _naux, nao, nao2 = _derivative_shape(dcderi)
    if nao != dm.shape[1] or nao2 != dm.shape[2]:
        raise ValueError(
            f"dCDERI AO dimensions {(nao, nao2)} do not match density "
            f"{dm.shape[1:]}"
        )
    coulomb = _coulomb_derivative(cderi, dcderi, dm, natm, ncart)
    exchange = _exchange_derivative(cderi, dcderi, dm, natm, ncart)
    return GDFJKDerivatives(
        coulomb=_real_if_close(coulomb),
        exchange=_real_if_close(exchange),
    )


def contract_coulomb_energy_derivative(
    density: np.ndarray,
    coulomb_derivative: np.ndarray,
) -> np.ndarray:
    """Return ``0.5/Nk sum_k Tr[D_k dJ_k]`` in Hartree/Bohr."""

    dm = _density_array(density)
    derivative = np.asarray(coulomb_derivative)
    _validate_matrix_derivative_shape(derivative, dm)
    value = 0.5 * np.einsum(
        "akxij,kji->ax", derivative, dm, optimize=True
    )
    return np.asarray(value.real / len(dm), dtype=float)


def ewald_exchange_derivative(
    cell,
    kpts: np.ndarray,
    density: np.ndarray,
    overlap_derivative: np.ndarray,
) -> np.ndarray:
    """Differentiate PySCF's GDF ``exxdiv='ewald'`` correction.

    PySCF adds

    ``K_G0(k) = madelung * S(k) D(k) S(k)``

    after the CDERI exchange contraction.  At fixed cell the Madelung scalar
    is constant, but both overlap matrices move with the atom-centered AO
    basis.  ``overlap_derivative`` must therefore contain full Hermitian
    derivatives with shape ``(natm,nkpts,3,nao,nao)``.
    """

    dm = _density_array(density)
    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    derivative = np.asarray(overlap_derivative)
    expected = (derivative.shape[0], len(points), 3, dm.shape[1], dm.shape[2])
    if derivative.shape != expected:
        raise ValueError(
            "overlap derivative must have shape "
            f"(natm,{len(points)},3,{dm.shape[1]},{dm.shape[2]}), got "
            f"{derivative.shape}"
        )
    overlap = np.asarray(
        cell.pbc_intor("int1e_ovlp", hermi=1, kpts=points)
    )
    madelung = float(pbc_tools.madelung(cell, points))
    output = np.empty_like(derivative, dtype=np.complex128)
    for atom_id in range(derivative.shape[0]):
        for kpoint_id in range(len(points)):
            for axis in range(3):
                ds = derivative[atom_id, kpoint_id, axis]
                output[atom_id, kpoint_id, axis] = madelung * (
                    ds @ dm[kpoint_id] @ overlap[kpoint_id]
                    + overlap[kpoint_id] @ dm[kpoint_id] @ ds
                )
    return _real_if_close(output)


def _coulomb_derivative(cderi, dcderi, dm, natm, ncart):
    nkpts, nao, _ = dm.shape
    jaux = np.sum(
        [
            np.einsum(
                "pij,ji->p",
                pair_block(cderi, k_id, k_id),
                dm[k_id],
                optimize=True,
            )
            for k_id in range(nkpts)
        ],
        axis=0,
    )
    out = np.zeros(
        (natm, nkpts, ncart, nao, nao), dtype=np.complex128
    )
    for atom_id in range(natm):
        for axis in range(ncart):
            djaux = np.sum(
                [
                    np.einsum(
                        "pij,ji->p",
                        derivative_pair_block(
                            dcderi, k_id, k_id, atom_id, axis
                        ),
                        dm[k_id],
                        optimize=True,
                    )
                    for k_id in range(nkpts)
                ],
                axis=0,
            )
            for k_id in range(nkpts):
                value = pair_block(cderi, k_id, k_id)
                derivative = derivative_pair_block(
                    dcderi, k_id, k_id, atom_id, axis
                )
                out[atom_id, k_id, axis] = (
                    np.einsum("p,pij->ij", jaux, derivative, optimize=True)
                    + np.einsum("p,pij->ij", djaux, value, optimize=True)
                ) / nkpts
    return out


def _exchange_derivative(cderi, dcderi, dm, natm, ncart):
    nkpts, nao, _ = dm.shape
    out = np.zeros(
        (natm, nkpts, ncart, nao, nao), dtype=np.complex128
    )
    for atom_id in range(natm):
        for axis in range(ncart):
            for ki in range(nkpts):
                matrix = np.zeros((nao, nao), dtype=np.complex128)
                for kj in range(nkpts):
                    c_ij = pair_block(cderi, ki, kj)
                    c_ji = pair_block(cderi, kj, ki)
                    dc_ij = derivative_pair_block(
                        dcderi, ki, kj, atom_id, axis
                    )
                    dc_ji = derivative_pair_block(
                        dcderi, kj, ki, atom_id, axis
                    )
                    # dK = dC D C + C D dC.  Keeping the two orientations
                    # explicit is essential for complex non-Gamma pairs.
                    matrix += np.einsum(
                        "pij,jk,pkl->il", dc_ij, dm[kj], c_ji, optimize=True
                    )
                    matrix += np.einsum(
                        "pij,jk,pkl->il", c_ij, dm[kj], dc_ji, optimize=True
                    )
                out[atom_id, ki, axis] = matrix / nkpts
    return out


def _density_array(density: np.ndarray) -> np.ndarray:
    dm = np.asarray(density)
    if dm.ndim == 2:
        dm = dm[None, :, :]
    if dm.ndim != 3 or dm.shape[1] != dm.shape[2]:
        raise ValueError(
            f"density must have shape (nkpts, nao, nao), got {dm.shape}"
        )
    return dm


def _derivative_shape(
    dcderi: Mapping[Pair, np.ndarray],
) -> tuple[int, int, int, int, int]:
    try:
        first = np.asarray(next(iter(dcderi.values())))
    except StopIteration as exc:
        raise ValueError("dcderi cannot be empty") from exc
    if first.ndim != 5 or first.shape[1] != 3:
        raise ValueError(
            "dCDERI blocks must have shape (natm, 3, naux, nao, nao), "
            f"got {first.shape}"
        )
    expected = first.shape
    if any(np.asarray(block).shape != expected for block in dcderi.values()):
        raise ValueError("all dCDERI blocks must have the same shape")
    return tuple(int(value) for value in expected)


def _validate_matrix_derivative_shape(derivative, density):
    expected_tail = (len(density), 3, density.shape[1], density.shape[2])
    if derivative.ndim != 5 or derivative.shape[1:] != expected_tail:
        raise ValueError(
            "matrix derivative must have shape (natm,nkpts,3,nao,nao); "
            f"got {derivative.shape}"
        )


def _real_if_close(array: np.ndarray, tolerance: float = 1.0e-10) -> np.ndarray:
    value = np.asarray(array)
    if value.size and np.max(np.abs(value.imag)) <= tolerance:
        return value.real
    return value
