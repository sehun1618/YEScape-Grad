'''Common Gaussian-DF CDERI source and metric response algebra.'''

import numpy as np
import scipy.linalg


def solve_cderi(source, metric, nao):
    r'''Return ``C = L^{-1} B`` with ``M = L L^dagger``.

    Args:
        source : ndarray of shape (nao*nao, naux)
            Periodic three-center source ``B``.
        metric : ndarray of shape (naux, naux)
            Hermitian positive-definite auxiliary Coulomb metric ``M``.
        nao : int
            Number of primary AO functions.

    Returns:
        ndarray of shape (naux, nao, nao)
            Orthonormalized three-center tensor ``C``.
    '''
    _factor, cderi = cholesky_cderi(source, metric, nao)
    return cderi


def cholesky_cderi(source, metric, nao):
    r'''Factor the auxiliary metric and return ``C = L^{-1}B``.

    Args:
        source : ndarray of shape (nao*nao, naux)
            Periodic three-center source :math:`B`.
        metric : ndarray of shape (naux, naux)
            Hermitian positive-definite auxiliary metric :math:`M`.
        nao : int
            Number of primary AO functions.

    Returns:
        tuple
            Lower Cholesky factor ``L`` and central CDERI ``C`` with shape
            ``(naux,nao,nao)``.

    Notes:
        Atom-stream response reuses the same geometry-central ``L`` and ``C``
        for every nuclear perturbation.  Returning them together prevents a
        hidden :math:`3N` repetition of the metric factorization.
    '''
    factor = factor_metric(metric)
    return factor, solve_cderi_with_factor(factor, source, nao)


def factor_metric(metric):
    r'''Return the lower Cholesky factor of the auxiliary metric.

    This entry point is used when the central CDERI is already owned by the
    PySCF DF provider.  In that production ordering only :math:`M=LL^\dagger`
    has to be reconstructed; rebuilding the zero-order three-center source
    merely to reproduce the provider's :math:`C` would be redundant.
    '''
    return _cholesky(metric)


def solve_cderi_with_factor(factor, source, nao):
    r'''Return :math:`C=L^{-1}B` for a previously factored metric.'''
    return _solve_factor(factor, source, nao)


def grad_cderi(source, metric, source1, metric1, nao):
    r'''Differentiate ``C = L^{-1}B`` in the Cholesky gauge.

    The two returned derivative terms are

    ``C_B^lambda = L^{-1} B^lambda``

    and

    ``C_M^lambda = -L^{-1} L^lambda C``.

    Keeping them separate makes primary/auxiliary source response and metric
    response independently testable.

    Returns:
        tuple
            ``(cderi, source_response, metric_response)`` with canonical
            ``(naux, nao, nao)`` layout.
    '''
    factor, cderi = cholesky_cderi(source, metric, nao)
    source_response, metric_response = grad_cderi_with_factor(
        factor, cderi, source1, metric1, nao)
    return cderi, source_response, metric_response


def grad_cderi_with_factor(factor, cderi, source1, metric1, nao):
    r'''Differentiate one or more CDERI blocks with a fixed central factor.

    Args:
        factor : ndarray of shape (naux, naux)
            Lower Cholesky factor :math:`L` of the central metric.
        cderi : ndarray of shape (naux, nao, nao)
            Central :math:`C=L^{-1}B`.
        source1 : ndarray of shape (..., nao*nao, naux)
            One or more three-center source derivatives :math:`B^\lambda`.
        metric1 : ndarray of shape (..., naux, naux)
            Matching metric derivatives :math:`M^\lambda`.
        nao : int
            Number of primary AO functions.

    Returns:
        tuple
            ``(source_response, metric_response)`` with shape
            ``(...,naux,nao,nao)`` corresponding to
            :math:`L^{-1}B^\lambda` and
            :math:`-L^{-1}L^\lambda C`.

    Notes:
        The leading dimensions may represent a full :math:`3N` reference or one
        atom block.  Production uses the latter so response memory does not
        grow with the total number of atoms.
    '''
    factor = np.asarray(factor)
    cderi = np.asarray(cderi)
    source1 = np.asarray(source1)
    metric1 = np.asarray(metric1)
    if factor.ndim != 2 or factor.shape[0] != factor.shape[1]:
        raise ValueError('Cholesky factor must be square')
    if cderi.shape != (factor.shape[0], nao, nao):
        raise ValueError('CDERI and Cholesky-factor dimensions are inconsistent')

    leading = source1.shape[:-2]
    if source1.shape[-2:] != (nao * nao, factor.shape[0]):
        raise ValueError('batched source derivative dimensions are inconsistent')
    if metric1.shape != leading + factor.shape:
        raise ValueError('batched source and metric leading dimensions differ')

    npert = int(np.prod(leading)) if leading else 1
    rhs = source1.reshape(npert, nao * nao, factor.shape[0])
    rhs = rhs.transpose(2, 0, 1).reshape(factor.shape[0], -1)
    source_response = scipy.linalg.solve_triangular(
        factor, rhs, lower=True, check_finite=False)
    source_response = source_response.reshape(
        factor.shape[0], npert, nao * nao).transpose(1, 0, 2)
    source_response = source_response.reshape(
        leading + (factor.shape[0], nao, nao))

    cderi_flat = cderi.reshape(cderi.shape[0], -1)
    metric1 = metric1.reshape((npert,) + factor.shape)
    factor1 = grad_cholesky(factor, metric1)
    rhs = -np.matmul(factor1, cderi_flat)
    rhs = rhs.transpose(1, 0, 2).reshape(factor.shape[0], -1)
    metric_response = scipy.linalg.solve_triangular(
        factor, rhs, lower=True, check_finite=False)
    metric_response = metric_response.reshape(
        factor.shape[0], npert, nao * nao).transpose(1, 0, 2)
    metric_response = metric_response.reshape(
        leading + (factor.shape[0], nao, nao))
    return source_response, metric_response


def grad_cderi_all(source, metric, source1, metric1, nao):
    r'''Differentiate CDERI for all perturbations with one factorization.

    Args:
        source, metric, nao
            Central ``B``, ``M`` and primary AO dimension accepted by
            :func:`grad_cderi`.
        source1 : ndarray of shape (..., nao*nao, naux)
            Batched three-center source derivatives.
        metric1 : ndarray of shape (..., naux, naux)
            Batched metric derivatives with the same leading dimensions.

    Returns:
        tuple
            ``(cderi, source_response, metric_response)``.  Both response
            arrays have shape ``(..., naux, nao, nao)``.

    Notes:
        ``M`` is geometry-central and must be factorized only once.  All
        :math:`L^{-1}B^\lambda` right-hand sides are solved in one triangular
        call.  The independent ``L^lambda`` matrices are also arranged as
        multiple right-hand sides, preserving their separate Cholesky gauges
        while avoiding one solver dispatch per Cartesian coordinate.
    '''
    factor, cderi = cholesky_cderi(source, metric, nao)
    source_response, metric_response = grad_cderi_with_factor(
        factor, cderi, source1, metric1, nao)
    return cderi, source_response, metric_response


def grad_cholesky(factor, metric1):
    r'''Return ``L^lambda`` from the differentiated Cholesky identity.

    ``metric1`` may contain arbitrary leading perturbation dimensions.  They
    are flattened only for multiple-RHS triangular solves; each perturbation
    retains the lower-triangular Cholesky gauge independently.
    '''
    factor = np.asarray(factor)
    metric1 = np.asarray(metric1)
    if metric1.shape[-2:] != factor.shape:
        raise ValueError('metric derivative and Cholesky factor shapes differ')
    leading = metric1.shape[:-2]
    npert = int(np.prod(leading)) if leading else 1
    naux = factor.shape[0]
    metric1 = metric1.reshape(npert, naux, naux)

    rhs = metric1.transpose(1, 0, 2).reshape(naux, -1)
    left = scipy.linalg.solve_triangular(
        factor, rhs, lower=True, check_finite=False)
    left = left.reshape(naux, npert, naux).transpose(1, 0, 2)
    rhs = left.transpose(2, 0, 1).reshape(naux, -1)
    reduced = scipy.linalg.solve_triangular(
        factor.conj(), rhs, lower=True, check_finite=False)
    reduced = reduced.reshape(naux, npert, naux).transpose(1, 2, 0)
    gauge = np.tril(reduced, k=-1)
    indices = np.arange(naux)
    gauge[:, indices, indices] = .5 * np.diagonal(
        reduced, axis1=-2, axis2=-1).real
    factor1 = np.matmul(factor[None], gauge)
    return factor1.reshape(leading + factor.shape)


def _cholesky(metric):
    metric = np.asarray(metric)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError('auxiliary metric must be square')
    return scipy.linalg.cholesky(
        metric, lower=True, check_finite=False)


def _solve_factor(factor, source, nao):
    source = np.asarray(source)
    if source.shape != (nao * nao, factor.shape[0]):
        raise ValueError(
            'three-center source must have shape '
            f'{(nao * nao, factor.shape[0])}, got {source.shape}'
        )
    value = scipy.linalg.solve_triangular(
        factor, source.T, lower=True, check_finite=False)
    return value.reshape(factor.shape[0], nao, nao)
