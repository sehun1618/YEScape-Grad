"""Adjoints that contract RSDF source derivatives before CDERI assembly.

For one canonical k-point pair, RSDF stores

``C = L^-1 B.T``

where ``B`` has AO-pair-by-auxiliary ordering.  The DC-DFT response needs
only two real scalar derivatives.  This module moves their linear weights
backwards through the triangular solve, producing source adjoints ``W_B``
such that ``Re(sum(W_B * dB))`` equals the established ``dCDERI``
contraction.  A native integral kernel can therefore contract ``dB`` in
place without materializing one source derivative tensor per atom and axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
from cupyx.scipy.linalg import solve_triangular

from ..gdf.jk_derivative import (
    auxiliary_density_contraction,
    pair_block,
)


@dataclass(frozen=True)
class PairSourceAdjoints:
    """Source-order adjoints for the two RSDF response observables."""

    target_coulomb: cp.ndarray
    restricted_hf_bilinear: cp.ndarray


def build_pair_source_adjoints(
    cderi,
    pair,
    cholesky_factor,
    reference_density,
    response_density,
    *,
    auxiliary_reference=None,
    auxiliary_response=None,
) -> PairSourceAdjoints:
    """Return AO-pair-by-auxiliary adjoints for one canonical pair.

    The returned arrays follow the same ``(nao**2, naux)`` ordering as the
    CPU RSDF source builder.  No derivative primitive is generated here.
    """

    reference = _density_array(reference_density)
    response = _density_array(response_density)
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    canonical = (int(pair[0]), int(pair[1]))
    central = pair_block(cderi, *canonical)
    factor = cp.asarray(cholesky_factor)
    if factor.shape != (central.shape[0], central.shape[0]):
        raise ValueError(
            "Cholesky factor and canonical CDERI auxiliary dimensions "
            f"differ: {factor.shape} and {central.shape}"
        )
    if auxiliary_reference is None:
        auxiliary_reference = auxiliary_density_contraction(
            cderi, reference
        )
    if auxiliary_response is None:
        auxiliary_response = auxiliary_density_contraction(cderi, response)

    target_cderi, bilinear_cderi = _cderi_adjoints(
        cderi,
        canonical,
        reference,
        response,
        cp.asarray(auxiliary_reference),
        cp.asarray(auxiliary_response),
    )
    return PairSourceAdjoints(
        target_coulomb=_push_through_triangular_solve(
            factor, target_cderi
        ),
        restricted_hf_bilinear=_push_through_triangular_solve(
            factor, bilinear_cderi
        ),
    )


def contract_source_derivative(source_adjoint, source_derivative):
    """Contract one source adjoint and ``dB`` into a real scalar."""

    adjoint = cp.asarray(source_adjoint)
    derivative = cp.asarray(source_derivative)
    if adjoint.shape != derivative.shape:
        raise ValueError(
            "source adjoint and derivative shapes differ: "
            f"{adjoint.shape} and {derivative.shape}"
        )
    return cp.real(cp.einsum("qp,qp->", adjoint, derivative))


def _cderi_adjoints(
    cderi,
    pair,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
):
    """Build plain-product adjoints for the canonical ``dC`` block."""

    first, second = pair
    central = pair_block(cderi, first, second)
    dtype = cp.result_type(
        central.dtype,
        reference.dtype,
        response.dtype,
        auxiliary_reference.dtype,
        auxiliary_response.dtype,
    )
    target = cp.zeros(central.shape, dtype=dtype)
    bilinear = cp.zeros_like(target)
    nkpts = len(reference)
    normalization = float(nkpts**2)

    if first == second:
        target += cp.einsum(
            "p,ji->pij",
            auxiliary_reference,
            reference[first],
            optimize=True,
        ) / normalization
        bilinear += 2.0 * cp.einsum(
            "p,ji->pij",
            auxiliary_reference,
            response[first],
            optimize=True,
        ) / normalization
        bilinear += 2.0 * cp.einsum(
            "p,ji->pij",
            auxiliary_response,
            reference[first],
            optimize=True,
        ) / normalization

    exchange = cp.zeros_like(target)
    ordered_pairs = (
        (pair,) if first == second else (pair, (second, first))
    )
    for ki, kj in ordered_pairs:
        factor_ij = pair_block(cderi, ki, kj)
        factor_ji = pair_block(cderi, kj, ki)
        derivative_ij = _exchange_adjoint_for_first_factor(
            reference[ki],
            response[kj],
            factor_ji,
        )
        derivative_ij += _exchange_adjoint_for_first_factor(
            response[ki],
            reference[kj],
            factor_ji,
        )
        derivative_ji = _exchange_adjoint_for_second_factor(
            reference[ki],
            factor_ij,
            response[kj],
        )
        derivative_ji += _exchange_adjoint_for_second_factor(
            response[ki],
            factor_ij,
            reference[kj],
        )
        exchange += _return_to_canonical_orientation(
            derivative_ij, pair, ki, kj
        )
        exchange += _return_to_canonical_orientation(
            derivative_ji, pair, kj, ki
        )
    bilinear -= 0.5 * exchange / normalization
    return target, bilinear


def _exchange_adjoint_for_first_factor(left, right, second_factor):
    return cp.einsum(
        "li,jk,pkl->pij",
        left,
        right,
        second_factor,
        optimize=True,
    )


def _exchange_adjoint_for_second_factor(left, first_factor, right):
    return cp.einsum(
        "li,pij,jk->pkl",
        left,
        first_factor,
        right,
        optimize=True,
    )


def _return_to_canonical_orientation(value, canonical, first, second):
    if (int(first), int(second)) == canonical:
        return value
    if (int(second), int(first)) == canonical:
        # Re[A * conj(dC.T)] = Re[conj(A.T) * dC].
        return value.conj().transpose(0, 2, 1)
    raise ValueError(
        f"ordered pair {(first, second)} is unrelated to {canonical}"
    )


def _push_through_triangular_solve(factor, cderi_adjoint):
    """Apply the transpose adjoint of ``C = solve(L, B.T)``."""

    factor_array = cp.asarray(factor)
    adjoint = cp.asarray(cderi_adjoint)
    dtype = cp.result_type(factor_array.dtype, adjoint.dtype)
    factor_array = factor_array.astype(dtype, copy=False)
    adjoint = adjoint.astype(dtype, copy=False)
    flat = adjoint.reshape(adjoint.shape[0], -1)
    solved = solve_triangular(
        factor_array.T,
        flat,
        lower=False,
        check_finite=False,
    )
    return solved.T


def _density_array(value):
    array = cp.asarray(value)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(
            "density must have shape (nkpts,nao,nao), got "
            f"{array.shape}"
        )
    return array
