"""Device contractions for full periodic GDF J/K nuclear derivatives.

Canonical CDERI and ``dCDERI`` blocks are CuPy arrays.  This module applies
the full product rule to obtain AO matrices ``dJ[D]/dR`` and ``dK[D]/dR``.
The closed-shell density already contains occupation factor two; each matrix
build contains the explicit periodic ``1/Nk`` normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cupy as cp
import numpy as np
from pyscf.pbc.tools import pbc as pbc_tools
from pyscf_pbc_dcdft.df.gdf.storage import Pair


@dataclass(frozen=True)
class GDFJKDerivatives:
    """Device J/K derivatives with ``(atom,k,xyz,AO,AO)`` ordering."""

    coulomb: cp.ndarray
    exchange: cp.ndarray

    @property
    def restricted_hf(self) -> cp.ndarray:
        """Return ``d(J-K/2)/dR`` for a restricted spatial density."""

        return self.coulomb - 0.5 * self.exchange


def gdf_jk_derivative(
    cderi: Mapping[Pair, cp.ndarray],
    dcderi: Mapping[Pair, cp.ndarray],
    density,
) -> GDFJKDerivatives:
    """Contract device ``C``/``dC`` tensors into full fixed-density J/K."""

    dm = _density_array(density)
    natm, ncart, _naux, nao, nao2 = _derivative_shape(dcderi)
    if (nao, nao2) != tuple(dm.shape[1:]):
        raise ValueError(
            f"dCDERI AO shape {(nao, nao2)} does not match {dm.shape[1:]}"
        )
    return GDFJKDerivatives(
        coulomb=_coulomb_derivative(cderi, dcderi, dm, natm, ncart),
        exchange=_exchange_derivative(cderi, dcderi, dm, natm, ncart),
    )


def contract_coulomb_energy_derivative(density, derivative):
    """Return ``0.5/Nk sum_k Tr[D_k dJ_k]`` on the device."""

    dm = _density_array(density)
    matrix_derivative = cp.asarray(derivative)
    _validate_matrix_derivative_shape(matrix_derivative, dm)
    value = 0.5 * cp.einsum(
        "akxij,kji->ax", matrix_derivative, dm, optimize=True
    )
    return value.real / len(dm)


def contract_restricted_hf_energy_derivative(density, derivative):
    """Contract a full ``d(J-K/2)[D]/dR`` matrix derivative.

    The restricted fixed-density two-electron energy is

    ``Q(D) = 1/2 Nk sum_k Tr[D_k (J[D]-K[D]/2)_k]``.

    ``derivative`` must therefore have been built from the same ``density``.
    This full-matrix helper is a reference contraction for validating the
    source-contracted native GDF kernels; production native mode does not
    materialize the five-index ``dCDERI`` tensors consumed here.
    """

    dm = _density_array(density)
    matrix_derivative = _restricted_hf_matrix_derivative(derivative)
    _validate_matrix_derivative_shape(matrix_derivative, dm)
    value = 0.5 * cp.einsum(
        "akxij,kji->ax", matrix_derivative, dm, optimize=True
    )
    return value.real / len(dm)


def contract_restricted_hf_bilinear_energy_derivative(
    reference_density,
    response_density,
    reference_derivative,
    response_derivative,
):
    """Return the exact cross-density RHF GDF derivative from full matrices.

    Let ``L_R[D] = d(J[D]-K[D]/2)/dR``.  Because ``L_R`` is linear in the
    density, the polarization difference obeys

    ``Q(D+T)-Q(D-T) = 1/Nk sum_k Re[Tr(D L_R[T]) + Tr(T L_R[D])]``.

    This form makes the two density roles explicit.  It is the equation-level
    witness for a future signed two-density native CUDA kernel and avoids
    relying on cancellation between two quadratic reference evaluations.
    """

    reference = _density_array(reference_density)
    response = _density_array(response_density)
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    derivative_at_reference = _restricted_hf_matrix_derivative(
        reference_derivative
    )
    derivative_at_response = _restricted_hf_matrix_derivative(
        response_derivative
    )
    _validate_matrix_derivative_shape(derivative_at_reference, reference)
    _validate_matrix_derivative_shape(derivative_at_response, response)
    value = cp.einsum(
        "kji,akxij->ax",
        reference,
        derivative_at_response,
        optimize=True,
    )
    value += cp.einsum(
        "kji,akxij->ax",
        response,
        derivative_at_reference,
        optimize=True,
    )
    return value.real / len(reference)


def ewald_exchange_derivative(
    cell,
    kpts,
    density,
    overlap_derivative,
) -> cp.ndarray:
    """Differentiate the GDF ``exxdiv='ewald'`` overlap correction.

    PySCF geometry and the Madelung scalar are host metadata.  The two matrix
    products for each atom, k-point, and Cartesian direction are CuPy kernels.
    """

    dm = _density_array(density)
    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    derivative = cp.asarray(overlap_derivative)
    expected = (
        derivative.shape[0], len(points), 3, dm.shape[1], dm.shape[2]
    )
    if derivative.shape != expected:
        raise ValueError(
            f"overlap derivative must have shape {expected}, got "
            f"{derivative.shape}"
        )
    overlap = cp.asarray(
        cell.pbc_intor("int1e_ovlp", hermi=1, kpts=points)
    )
    madelung = float(pbc_tools.madelung(cell, points))
    output = cp.empty_like(derivative, dtype=cp.complex128)
    for atom_id in range(derivative.shape[0]):
        for kpoint_id in range(len(points)):
            for axis in range(3):
                ds = derivative[atom_id, kpoint_id, axis]
                output[atom_id, kpoint_id, axis] = madelung * (
                    ds @ dm[kpoint_id] @ overlap[kpoint_id]
                    + overlap[kpoint_id] @ dm[kpoint_id] @ ds
                )
    return output


def pair_block(blocks, ki, kj):
    """Return one device CDERI block using Hermitian pair reversal."""

    key = (int(ki), int(kj))
    if key in blocks:
        return blocks[key]
    reverse = (key[1], key[0])
    if reverse in blocks:
        return blocks[reverse].conj().transpose(0, 2, 1)
    raise KeyError(f"CDERI block for pair {key} is unavailable")


def derivative_pair_block(blocks, ki, kj, atom_id, axis):
    """Return one device ``dCDERI`` block using pair reversal."""

    key = (int(ki), int(kj))
    if key in blocks:
        return blocks[key][int(atom_id), int(axis)]
    reverse = (key[1], key[0])
    if reverse in blocks:
        return (
            blocks[reverse][int(atom_id), int(axis)]
            .conj()
            .transpose(0, 2, 1)
        )
    raise KeyError(f"dCDERI block for pair {key} is unavailable")


def _coulomb_derivative(cderi, dcderi, dm, natm, ncart):
    nkpts, nao, _ = dm.shape
    first = pair_block(cderi, 0, 0)
    jaux = cp.zeros(first.shape[0], dtype=cp.complex128)
    for k_id in range(nkpts):
        jaux += cp.einsum(
            "pij,ji->p", pair_block(cderi, k_id, k_id), dm[k_id]
        )
    output = cp.zeros(
        (natm, nkpts, ncart, nao, nao), dtype=cp.complex128
    )
    for atom_id in range(natm):
        for axis in range(ncart):
            djaux = cp.zeros_like(jaux)
            for k_id in range(nkpts):
                djaux += cp.einsum(
                    "pij,ji->p",
                    derivative_pair_block(
                        dcderi, k_id, k_id, atom_id, axis
                    ),
                    dm[k_id],
                )
            for k_id in range(nkpts):
                value = pair_block(cderi, k_id, k_id)
                derivative = derivative_pair_block(
                    dcderi, k_id, k_id, atom_id, axis
                )
                output[atom_id, k_id, axis] = (
                    cp.einsum("p,pij->ij", jaux, derivative)
                    + cp.einsum("p,pij->ij", djaux, value)
                ) / nkpts
    return output


def _exchange_derivative(cderi, dcderi, dm, natm, ncart):
    nkpts, nao, _ = dm.shape
    output = cp.zeros(
        (natm, nkpts, ncart, nao, nao), dtype=cp.complex128
    )
    for atom_id in range(natm):
        for axis in range(ncart):
            for ki in range(nkpts):
                matrix = cp.zeros((nao, nao), dtype=cp.complex128)
                for kj in range(nkpts):
                    c_ij = pair_block(cderi, ki, kj)
                    c_ji = pair_block(cderi, kj, ki)
                    dc_ij = derivative_pair_block(
                        dcderi, ki, kj, atom_id, axis
                    )
                    dc_ji = derivative_pair_block(
                        dcderi, kj, ki, atom_id, axis
                    )
                    matrix += cp.einsum(
                        "pij,jk,pkl->il", dc_ij, dm[kj], c_ji,
                        optimize=True,
                    )
                    matrix += cp.einsum(
                        "pij,jk,pkl->il", c_ij, dm[kj], dc_ji,
                        optimize=True,
                    )
                output[atom_id, ki, axis] = matrix / nkpts
    return output


def _density_array(value):
    dm = cp.asarray(value)
    if dm.ndim == 2:
        dm = dm[None]
    if dm.ndim != 3 or dm.shape[1] != dm.shape[2]:
        raise ValueError(
            f"density must have shape (nkpts,nao,nao), got {dm.shape}"
        )
    return dm


def _restricted_hf_matrix_derivative(value):
    if isinstance(value, GDFJKDerivatives):
        return value.restricted_hf
    derivative = cp.asarray(value)
    if derivative.ndim != 5:
        raise ValueError(
            "restricted HF matrix derivative must have five dimensions, "
            f"got {derivative.shape}"
        )
    return derivative


def _derivative_shape(blocks):
    try:
        first = next(iter(blocks.values()))
    except StopIteration as exc:
        raise ValueError("dcderi cannot be empty") from exc
    if first.ndim != 5 or first.shape[1] != 3:
        raise ValueError(
            "dCDERI blocks must have (natm,3,naux,nao,nao), got "
            f"{first.shape}"
        )
    expected = first.shape
    if any(block.shape != expected for block in blocks.values()):
        raise ValueError("all dCDERI blocks must have the same shape")
    return tuple(int(value) for value in expected)


def _validate_matrix_derivative_shape(derivative, density):
    expected_tail = (len(density), 3, density.shape[1], density.shape[2])
    if derivative.ndim != 5 or derivative.shape[1:] != expected_tail:
        raise ValueError(
            "matrix derivative must have (natm,nkpts,3,nao,nao), got "
            f"{derivative.shape}"
        )
