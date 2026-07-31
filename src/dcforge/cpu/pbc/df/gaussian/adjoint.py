'''Reverse-adjoint Gaussian-DF response contractions at the Gamma point.'''

import numpy as np
import scipy.linalg
from pyscf import lib

from dcforge.cpu.pbc.df.gaussian.response import response_blocks


def make_cderi_observable_adjoints(cderi, dm0, dz):
    r'''Return CDERI adjoints for target Hartree and source-Fock terms.

    The two observables are the Gaussian-DF parts of

    .. math::

        E_J^\lambda
          = \frac12 \operatorname{Tr}[P_{HF}J^\lambda[P_{HF}]],

    and

    .. math::

        E_{F,Z}^\lambda
          = -2\operatorname{Tr}
            [D_Z(J^\lambda[P_{HF}]-\tfrac12K^\lambda[P_{HF}])].

    If ``A = dE/dC``, a CDERI response contributes
    ``sum(A * C1)``.  Constructing ``A`` once permits the source and metric
    derivatives to be contracted without materializing ``C1``, ``J1`` or
    ``K1`` for every nuclear coordinate.

    Args:
        cderi : ndarray of shape (naux, nao, nao)
            Provider-owned central Gamma CDERI.
        dm0 : ndarray of shape (nao, nao)
            Spin-summed source-RHF density.
        dz : ndarray of shape (nao, nao)
            Hermitian Z-transition density.

    Returns:
        tuple
            Target-Hartree and source-Fock adjoints, each with the same shape
            as ``cderi``.
    '''
    cderi = _real_gamma(cderi, 'cderi')
    dm0 = _real_gamma(dm0, 'dm0')
    dz = _real_gamma(dz, 'dz')
    if dm0.shape != cderi.shape[1:] or dz.shape != dm0.shape:
        raise ValueError('CDERI and density AO dimensions are inconsistent')

    rho0 = lib.einsum('Pij,ji->P', cderi, dm0)
    rhoz = lib.einsum('Pij,ji->P', cderi, dz)
    target_j = lib.einsum('P,ji->Pij', rho0, dm0)

    fock_j = -2 * (
        lib.einsum('P,ji->Pij', rhoz, dm0)
        + lib.einsum('P,ji->Pij', rho0, dz)
    )
    fock_k = np.empty_like(cderi)
    for p, cp in enumerate(cderi):
        fock_k[p] = (
            dm0 @ cp @ dz
            + dz @ cp @ dm0
        ).T
    return target_j, fock_j + fock_k


def push_cderi_adjoints(factor, cderi, adjoints):
    r'''Push one or more observable adjoints through ``C=L^-1 B``.

    Args:
        factor : ndarray of shape (naux, naux)
            Lower Cholesky factor ``L`` of the auxiliary metric.
        cderi : ndarray of shape (naux, nao, nao)
            Provider-owned central CDERI ``C``.
        adjoints : ndarray of shape (nobs, naux, nao, nao)
            Observable derivatives ``dE_o/dC``.

    Returns:
        tuple
            Source adjoints with shape ``(nobs,nao*nao,naux)`` and symmetric
            metric adjoints with shape ``(nobs,naux,naux)``.

    Notes:
        For ``X=L^-T A``, the source term is ``<X,dB>``.  The metric term first
        forms ``-X C^T`` as the adjoint of ``L`` and then applies the reverse
        of the lower-Cholesky gauge used by :func:`grad_cholesky`.
    '''
    factor = _real_gamma(factor, 'factor')
    cderi = _real_gamma(cderi, 'cderi')
    adjoints = _real_gamma(adjoints, 'adjoints')
    if adjoints.ndim == 3:
        adjoints = adjoints[None]
    if adjoints.ndim != 4 or adjoints.shape[1:] != cderi.shape:
        raise ValueError('observable adjoints must match the central CDERI')

    nobs, naux = adjoints.shape[:2]
    if factor.shape != (naux, naux):
        raise ValueError('Cholesky factor and CDERI dimensions differ')
    adjoint_flat = adjoints.reshape(nobs, naux, -1)
    rhs = adjoint_flat.transpose(1, 0, 2).reshape(naux, -1)
    solved = scipy.linalg.solve_triangular(
        factor.T, rhs, lower=False, check_finite=False)
    solved = solved.reshape(naux, nobs, -1).transpose(1, 0, 2)
    source_adjoint = solved.transpose(0, 2, 1)

    cderi_flat = cderi.reshape(naux, -1)
    metric_adjoint = np.empty((nobs, naux, naux), dtype=solved.dtype)
    for obs in range(nobs):
        factor_adjoint = -(solved[obs] @ cderi_flat.T)
        reduced_adjoint = factor.T @ factor_adjoint
        lower = np.tril(reduced_adjoint, k=-1)
        indices = np.arange(naux)
        lower[indices, indices] = .5 * np.diag(reduced_adjoint)
        symmetric = .5 * (lower + lower.T)
        left = scipy.linalg.solve_triangular(
            factor.T, symmetric, lower=False, check_finite=False)
        metric_adjoint[obs] = scipy.linalg.solve_triangular(
            factor.T, left.T, lower=False, check_finite=False).T
    return source_adjoint, metric_adjoint


def contract_response_adjoints(
        source1, metric1, source_adjoint, metric_adjoint):
    r'''Contract source/metric nuclear derivatives into scalar observables.'''
    source1 = np.asarray(source1)
    metric1 = np.asarray(metric1)
    source_adjoint = np.asarray(source_adjoint)
    metric_adjoint = np.asarray(metric_adjoint)
    leading = source1.shape[:-2]
    if metric1.shape[:-2] != leading:
        raise ValueError('source and metric response leading dimensions differ')
    npert = int(np.prod(leading)) if leading else 1
    source_flat = source1.reshape((npert,) + source1.shape[-2:])
    metric_flat = metric1.reshape((npert,) + metric1.shape[-2:])
    source_value = lib.einsum(
        'tmP,omP->ot', source_flat, source_adjoint)
    metric_value = lib.einsum(
        'tPQ,oPQ->ot', metric_flat, metric_adjoint)
    shape = (len(source_adjoint),) + leading
    return source_value.reshape(shape).real, metric_value.reshape(shape).real


def make_contracted_response(
        cell, auxcell, central, factor, source1_gen, metric1_gen,
        dm0, dz, atmlst=None, blksize=None, max_memory=4000):
    r'''Stream primitive derivatives into resolved gradient scalars.

    This is the common GDF/RSDF production kernel.  It preserves separate
    source/metric and target-J/source-Fock terms while discarding each atom
    block of :math:`B^\lambda` and :math:`M^\lambda` immediately after its
    reverse-adjoint contraction.

    Returns:
        tuple
            ``(terms, blksize, peak_explicit_bytes)``.  ``terms`` contains
            ``de_j_source``, ``de_j_metric``, ``de_fock_source`` and
            ``de_fock_metric``, each with shape ``(natm,3)``.
    '''
    central = _real_gamma(central, 'central')
    dm0 = _real_gamma(dm0, 'dm0')
    dz = _real_gamma(dz, 'dz')
    dtype = np.result_type(central, dm0, dz)
    atmlst, blksize = response_blocks(
        cell, atmlst, blksize, cell.nao_nr(), auxcell.nao_nr(),
        dtype, max_memory, storage='contracted')
    adjoints = np.asarray(
        make_cderi_observable_adjoints(central, dm0, dz))
    source_adjoint, metric_adjoint = push_cderi_adjoints(
        factor, central, adjoints)
    output = np.zeros((2, 2, cell.natm, 3), dtype=float)
    peak_bytes = int(
        adjoints.nbytes + source_adjoint.nbytes + metric_adjoint.nbytes)

    for p0 in range(0, len(atmlst), blksize):
        atoms = atmlst[p0:p0 + blksize]
        source1 = source1_gen(atoms)
        metric1 = metric1_gen(atoms)
        source_value, metric_value = contract_response_adjoints(
            source1, metric1, source_adjoint, metric_adjoint)
        output[:, 0, atoms] = source_value
        output[:, 1, atoms] = metric_value
        peak_bytes = max(peak_bytes, int(
            source1.nbytes + metric1.nbytes + adjoints.nbytes
            + source_adjoint.nbytes + metric_adjoint.nbytes))

    terms = {
        'de_j_source': output[0, 0],
        'de_j_metric': output[0, 1],
        'de_fock_source': output[1, 0],
        'de_fock_metric': output[1, 1],
    }
    return terms, blksize, peak_bytes


def _real_gamma(value, name, tol=1e-10):
    '''Return a real Gamma tensor or reject unsupported complex storage.'''
    value = np.asarray(value)
    if np.iscomplexobj(value):
        maximum = float(np.max(np.abs(value.imag))) if value.size else 0.
        if maximum > tol:
            raise NotImplementedError(
                f'{name} has non-Gamma imaginary component {maximum:.3e}')
        value = value.real
    return value
