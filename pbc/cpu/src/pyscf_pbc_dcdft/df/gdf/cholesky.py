"""Auxiliary-metric solve and analytic Cholesky response for periodic GDF.

For one momentum transfer, PySCF stores orthonormalized three-index factors

``C = L^-1 B,  M = L L^H``,

where ``B`` is the three-center source tensor and ``M`` is the two-center
auxiliary Coulomb metric.  Nuclear differentiation gives

``dC = L^-1 dB - L^-1 dL C``.

The second term is the GDF metric response.  Omitting it is equivalent to
freezing the auxiliary fitting coefficients and generally produces a force
that disagrees with the derivative of the fitted energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class CDERIDerivativeParts:
    """Central CDERI value and separated source/metric derivatives."""

    value: np.ndarray
    source: np.ndarray
    metric: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.source + self.metric


def solve_cderi(source: np.ndarray, metric: np.ndarray, nao: int) -> np.ndarray:
    """Return ``C = chol(M)^-1 B`` as ``(naux, nao, nao)``."""

    factor = _factor_metric(metric)
    return _solve_factor(factor, source, nao)


def differentiate_cderi(
    source: np.ndarray,
    metric: np.ndarray,
    source_derivative: np.ndarray,
    metric_derivative: np.ndarray,
    nao: int,
) -> CDERIDerivativeParts:
    """Differentiate one explicit GDF source/metric pair.

    Parameters
    ----------
    source, source_derivative
        Arrays with leading flattened AO-pair dimension ``nao**2`` and trailing
        auxiliary dimension.  Units are the corresponding Coulomb integrals
        and their derivative per Bohr.
    metric, metric_derivative
        Hermitian auxiliary metric and its derivative per Bohr.
    nao
        Number of primary AO functions.

    Returns
    -------
    CDERIDerivativeParts
        ``value``, ``source``, and ``metric`` all use
        ``(naux, nao, nao)``.  ``total`` is the analytic ``dCDERI/dR``.
    """

    factor = _factor_metric(metric)
    value = _solve_factor(factor, source, nao)
    source_part = _solve_factor(factor, source_derivative, nao)
    metric_part = _metric_response(factor, value, metric_derivative)
    return CDERIDerivativeParts(
        value=_real_if_close(value),
        source=_real_if_close(source_part),
        metric=_real_if_close(metric_part),
    )


def _factor_metric(metric: np.ndarray) -> np.ndarray:
    array = np.asarray(metric)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"metric must be square, got {array.shape}")
    return scipy.linalg.cholesky(array, lower=True, check_finite=False)


def _solve_factor(
    factor: np.ndarray,
    source: np.ndarray,
    nao: int,
) -> np.ndarray:
    source_matrix = np.asarray(source)
    expected = int(nao) ** 2
    if source_matrix.ndim != 2 or source_matrix.shape[0] != expected:
        raise ValueError(
            f"source must have shape ({expected}, naux), got "
            f"{source_matrix.shape}"
        )
    solved = scipy.linalg.solve_triangular(
        factor,
        source_matrix.T,
        lower=True,
        check_finite=False,
    )
    return solved.reshape(solved.shape[0], int(nao), int(nao))


def _metric_response(
    factor: np.ndarray,
    cderi: np.ndarray,
    metric_derivative: np.ndarray,
) -> np.ndarray:
    d_factor = _differentiate_cholesky(factor, metric_derivative)
    cderi_flat = np.asarray(cderi).reshape(cderi.shape[0], -1)
    rhs = -(d_factor @ cderi_flat)
    response = scipy.linalg.solve_triangular(
        factor,
        rhs,
        lower=True,
        check_finite=False,
    )
    return response.reshape(np.asarray(cderi).shape)


def _differentiate_cholesky(
    factor: np.ndarray,
    metric_derivative: np.ndarray,
) -> np.ndarray:
    """Return ``dL`` in the lower-triangular Cholesky gauge.

    Starting from ``dM = dL L^H + L dL^H``, form
    ``W = L^-1 dM L^-H``.  Its strict lower triangle is copied directly and
    its real diagonal is halved.  This fixes the otherwise redundant unitary
    gauge of the metric square root.
    """

    lower = np.asarray(factor)
    derivative = np.asarray(metric_derivative)
    if derivative.shape != lower.shape:
        raise ValueError(
            f"metric derivative shape {derivative.shape} does not match "
            f"factor {lower.shape}"
        )
    left_solved = scipy.linalg.solve_triangular(
        lower,
        derivative,
        lower=True,
        check_finite=False,
    )
    reduced = scipy.linalg.solve_triangular(
        lower.conj(),
        left_solved.T,
        lower=True,
        check_finite=False,
    ).T
    gauge = np.tril(reduced, k=-1)
    gauge += np.diag(0.5 * np.diag(reduced).real)
    return lower @ gauge


def _real_if_close(array: np.ndarray, tolerance: float = 1.0e-11) -> np.ndarray:
    value = np.asarray(array)
    if value.size and np.max(np.abs(value.imag)) <= tolerance:
        return value.real
    return value
