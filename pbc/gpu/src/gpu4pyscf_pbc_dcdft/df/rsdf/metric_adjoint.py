"""Reverse Cholesky adjoints for RSDF metric derivatives.

For one canonical pair, ``C = L^-1 B.T`` and ``M = L L^H``.  A scalar
observable supplies a plain-product adjoint ``G_C`` such that its derivative is
``Re(sum(G_C * dC))``.  This module pushes ``G_C`` through the Cholesky gauge
and returns ``W_M`` satisfying ``Re(sum(W_M * dM))``.  Native metric kernels can
therefore contract the two-center derivative directly without storing ``dM``
or ``dC`` for every atom and Cartesian axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
from cupyx.scipy.linalg import solve_triangular

from ..gdf.jk_derivative import (
    auxiliary_density_contraction,
    pair_block,
)
from .source_adjoint import build_pair_cderi_adjoints


@dataclass(frozen=True)
class PairMetricAdjoints:
    """Metric-order adjoints for the two RSDF response observables."""

    target_coulomb: cp.ndarray
    restricted_hf_bilinear: cp.ndarray


def build_pair_metric_adjoints(
    cderi,
    pair,
    factorization,
    reference_density,
    response_density,
    *,
    auxiliary_reference=None,
    auxiliary_response=None,
) -> PairMetricAdjoints:
    """Return ``(naux,naux)`` metric adjoints for one canonical pair."""

    reference = _density_array(reference_density)
    response = _density_array(response_density)
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    canonical = (int(pair[0]), int(pair[1]))
    central = pair_block(cderi, *canonical)
    if auxiliary_reference is None:
        auxiliary_reference = auxiliary_density_contraction(
            cderi,
            reference,
        )
    if auxiliary_response is None:
        auxiliary_response = auxiliary_density_contraction(cderi, response)
    target, bilinear = build_pair_cderi_adjoints(
        cderi,
        canonical,
        reference,
        response,
        cp.asarray(auxiliary_reference),
        cp.asarray(auxiliary_response),
    )
    value = cp.asarray(factorization.value)
    if value.shape != central.shape:
        raise ValueError(
            "factorized and provider CDERI blocks have different shapes: "
            f"{value.shape} and {central.shape}"
        )
    return PairMetricAdjoints(
        target_coulomb=push_cderi_adjoint_through_metric(
            factorization,
            target,
        ),
        restricted_hf_bilinear=push_cderi_adjoint_through_metric(
            factorization,
            bilinear,
        ),
)


DEFAULT_METRIC_PRODUCT_INNER_BLOCK = 8192


def contract_metric_derivative(metric_adjoint, metric_derivative):
    """Contract one metric adjoint and ``dM`` into a real scalar."""

    adjoint = cp.asarray(metric_adjoint)
    derivative = cp.asarray(metric_derivative)
    if adjoint.shape != derivative.shape:
        raise ValueError(
            "metric adjoint and derivative shapes differ: "
            f"{adjoint.shape} and {derivative.shape}"
        )
    return cp.real(cp.einsum("pq,pq->", adjoint, derivative))


def push_cderi_adjoint_through_metric(factorization, cderi_adjoint):
    """Push one plain-product CDERI adjoint through the metric factor."""

    factor_array = cp.asarray(factorization.factor)
    value = cp.asarray(factorization.value)
    value_flat = value.reshape(value.shape[0], -1)
    adjoint_flat = cp.asarray(cderi_adjoint).reshape(value.shape[0], -1)

    # dC_metric = -Phi(L^-1 dM L^-H) C.  Phi retains the strict lower
    # triangle and half of the real diagonal in the Cholesky gauge.
    # Build only the small (naux,naux) mixed-dtype result.  Casting either
    # (naux,nao**2) operand to a common complex dtype can add tens of GiB for
    # larger Gamma cells even though the final metric adjoint is small.
    gauge_adjoint = -_plain_mixed_product(adjoint_flat, value_flat.T)
    reduced_adjoint = cp.tril(gauge_adjoint, k=-1)
    reduced_adjoint += cp.diag(0.5 * cp.diag(gauge_adjoint))

    dtype = cp.result_type(factor_array.dtype, reduced_adjoint.dtype)
    factor_array = factor_array.astype(dtype, copy=False)
    reduced_adjoint = reduced_adjoint.astype(dtype, copy=False)
    identity = cp.eye(factor_array.shape[0], dtype=dtype)
    inverse = solve_triangular(
        factor_array,
        identity,
        lower=True,
        check_finite=False,
    )
    return inverse.T @ reduced_adjoint @ inverse.conj()


def _plain_mixed_product(left, right):
    """Return ``left @ right`` without whole-input real/complex promotion."""

    left = cp.asarray(left)
    right = cp.asarray(right)
    if left.dtype.kind != "c" and right.dtype.kind != "c":
        return _real_component_product(left, right)

    real = _real_component_product(left.real, right.real)
    if left.dtype.kind == "c" and right.dtype.kind == "c":
        real -= _real_component_product(left.imag, right.imag)
        imaginary = _real_component_product(left.real, right.imag)
        imaginary += _real_component_product(left.imag, right.real)
    elif left.dtype.kind == "c":
        imaginary = _real_component_product(left.imag, right)
    else:
        imaginary = _real_component_product(left, right.imag)
    return real + 1j * imaginary


def _real_component_product(
    left,
    right,
    *,
    maximum_inner_block=DEFAULT_METRIC_PRODUCT_INNER_BLOCK,
):
    """Multiply real components with bounded mixed-precision promotion."""

    left = cp.asarray(left)
    right = cp.asarray(right)
    if left.dtype == right.dtype:
        return left @ right
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError(
            "metric product operands must be compatible matrices, got "
            f"{left.shape} and {right.shape}"
        )
    dtype = cp.result_type(left.dtype, right.dtype)
    output = cp.zeros((left.shape[0], right.shape[1]), dtype=dtype)
    block = max(1, int(maximum_inner_block))
    for start in range(0, int(left.shape[1]), block):
        stop = min(int(left.shape[1]), start + block)
        left_block = left[:, start:stop].astype(dtype, copy=False)
        right_block = right[start:stop].astype(dtype, copy=False)
        output += left_block @ right_block
    return output


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
