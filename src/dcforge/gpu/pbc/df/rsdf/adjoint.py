r'''Device-resident reverse auxiliary adjoints for Gamma RSDF response.

The RSDF energy provider stores the central three-index factor as
``C[P,i,j]``.  Its source and metric primitives instead obey

.. math::

   C = L^{-1} B, \qquad M = L L^T.

This module never forms ``dC``, ``dJ``, or ``dK``.  It first turns the two
DC-DFT observables into ``dE/dC`` and then returns the corresponding
``dE/dB`` and ``dE/dM`` tensors.  Native source and metric kernels can
contract those tensors directly with their coordinate derivatives.
'''

import cupy as cp
import numpy as np
from cupyx.scipy.linalg import solve_triangular
from gpu4pyscf.lib.cupy_helper import contract


def make_cderi_observable_adjoints(
        cderi, dm0, dz, auxiliary_block_size=32):
    r'''Return target-J and source-Fock adjoints of the Gamma CDERI.

    Args:
        cderi : cupy.ndarray
            Real central RSDF factor ``C[P,i,j]``.
        dm0 : cupy.ndarray
            Spin-summed source RHF density ``P``.
        dz : cupy.ndarray
            Hermitian Z-transition density ``D_Z``.

    Returns:
        cupy.ndarray
            Two adjoints with shape ``(2,naux,nao,nao)``.  Index 0 belongs
            to ``E_J = Tr[P J[P]] / 2``.  Index 1 belongs to
            ``-2 Tr[D_Z (J[P] - K[P]/2)]``.

    Kwargs:
        auxiliary_block_size : int
            Maximum auxiliary slice used to assemble the two persistent
            adjoints.  This changes only temporary storage; the returned
            arrays and their contraction convention are unchanged.

    Notes:
        These are plain-product adjoints.  If ``C1`` is a CDERI derivative,
        each scalar derivative is ``sum(adjoint * C1)``.  Keeping target-J
        and source-Fock separate preserves the molecular/CPU PBC gradient
        term ledger until the final assembly stage.
    '''
    cderi = _real_gamma(cderi, 'cderi')
    dm0 = _real_gamma(dm0, 'dm0')
    dz = _real_gamma(dz, 'dz')
    if dm0.shape != cderi.shape[1:] or dz.shape != dm0.shape:
        raise ValueError('CDERI and density AO dimensions are inconsistent')
    if auxiliary_block_size < 1:
        raise ValueError('auxiliary_block_size must be positive')

    rho0 = cp.einsum('Pij,ji->P', cderi, dm0, optimize=True)
    rhoz = cp.einsum('Pij,ji->P', cderi, dz, optimize=True)
    naux, nao = cderi.shape[:2]
    maximum = min(naux, int(auxiliary_block_size))
    adjoints = cp.empty((2, naux, nao, nao), dtype=cderi.dtype)
    temporary = cp.empty((maximum, nao, nao), dtype=cderi.dtype)
    for p0 in range(0, naux, maximum):
        p1 = min(naux, p0 + maximum)
        width = p1 - p0
        block = cderi[p0:p1]
        target_j = adjoints[0, p0:p1]
        fock = adjoints[1, p0:p1]
        work = temporary[:width]

        cp.multiply(
            rho0[p0:p1, None, None], dm0[None], out=target_j)
        cp.multiply(
            rhoz[p0:p1, None, None], dm0[None], out=fock)
        cp.multiply(
            rho0[p0:p1, None, None], dz[None], out=work)
        fock += work
        fock *= -2.0

        # Exchange is accumulated into the final Fock-adjoint buffer.  The
        # one reusable ``work`` slice replaces two full ``(naux,nao,nao)``
        # einsum results and the subsequent stack allocation of the previous
        # implementation.
        contract('Pkl,lj->Pkj', block, dz, out=work)
        contract('ik,Pkj->Pji', dm0, work, beta=1.0, out=fock)
        contract('Pkl,lj->Pkj', block, dm0, out=work)
        contract('ik,Pkj->Pji', dz, work, beta=1.0, out=fock)
    return adjoints


def push_cderi_adjoints(factor, cderi, adjoints):
    r'''Push Gamma CDERI adjoints through ``C = L^{-1} B``.

    Args:
        factor : cupy.ndarray
            Lower Cholesky factor ``L`` of the authoritative RSDF metric.
        cderi : cupy.ndarray
            Central factor ``C[P,i,j]``.
        adjoints : cupy.ndarray
            One or more ``dE/dC`` tensors.  The accepted shapes are
            ``(naux,nao,nao)`` and ``(nobservable,naux,nao,nao)``.

    Returns:
        tuple
            ``(source_adjoint, metric_adjoint)`` with shapes
            ``(nobservable,nao*nao,naux)`` and
            ``(nobservable,naux,naux)``.  They satisfy
            ``dE = <source_adjoint,dB> + <metric_adjoint,dM>`` under the
            lower-Cholesky gauge used by the central provider.

    Notes:
        The solve is device-resident.  The explicit metric adjoint is only
        ``O(naux**2)``; the much larger nuclear derivative tensors remain
        absent.  This matches the CPU ``gaussian.adjoint`` convention so
        both implementations can use the same term-wise validation oracle.
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
    solved = solve_triangular(
        factor.T, rhs, lower=False, check_finite=False)
    solved = solved.reshape(naux, nobs, -1).transpose(1, 0, 2)
    source_adjoint = solved.transpose(0, 2, 1)

    cderi_flat = cderi.reshape(naux, -1)
    metric_adjoint = cp.empty((nobs, naux, naux), dtype=solved.dtype)
    indices = cp.arange(naux)
    for observable in range(nobs):
        factor_adjoint = -(solved[observable] @ cderi_flat.T)
        reduced_adjoint = factor.T @ factor_adjoint
        lower = cp.tril(reduced_adjoint, k=-1)
        lower[indices, indices] = .5 * cp.diag(reduced_adjoint)
        symmetric = .5 * (lower + lower.T)
        left = solve_triangular(
            factor.T, symmetric, lower=False, check_finite=False)
        metric_adjoint[observable] = solve_triangular(
            factor.T, left.T, lower=False, check_finite=False).T
    return source_adjoint, metric_adjoint


def make_packed_cderi_observable_adjoint(
        cderi, layout, dm0, dz, observable, auxiliary_block_size=32):
    r'''Return one real-Gamma observable adjoint in unique AO-pair storage.

    Args:
        cderi : cupy.ndarray
            Unique-pair central factor with shape ``(naux,npair)``.
        layout : dict
            Pair metadata returned by
            :func:`dcforge.gpu.pbc.df.cderi.canonical_pair_layout`.
        dm0, dz : cupy.ndarray
            Real symmetric source and Z-transition AO densities.
        observable : {'target_j', 'source_fock'}
            Physical scalar whose ``dE/dC`` is requested.

    Kwargs:
        auxiliary_block_size : int
            Maximum temporary dense auxiliary block.  Only this bounded block
            is expanded; neither central CDERI nor its persistent adjoint is
            ever stored as ``(naux,nao,nao)``.

    Returns:
        cupy.ndarray
            Packed derivative with respect to the unique symmetric CDERI
            variables.  Off-diagonal entries include the exact factor of two
            from the two equal dense matrix elements.
    '''
    cderi = _real_gamma(cderi, 'cderi')
    dm0 = _real_gamma(dm0, 'dm0')
    dz = _real_gamma(dz, 'dz')
    if observable not in ('target_j', 'source_fock'):
        raise ValueError("observable must be 'target_j' or 'source_fock'")
    nao = int(dm0.shape[0])
    if dm0.shape != (nao, nao) or dz.shape != dm0.shape:
        raise ValueError('packed CDERI densities must be square and conformable')
    row = cp.asarray(layout['row'], dtype=cp.int32)
    column = cp.asarray(layout['column'], dtype=cp.int32)
    weights = cp.asarray(layout['weights'], dtype=cderi.dtype)
    if cderi.ndim != 2 or cderi.shape[1] != len(row):
        raise ValueError('packed CDERI and canonical pair layout differ')
    naux = int(cderi.shape[0])
    maximum = max(1, min(naux, int(auxiliary_block_size)))
    output = cp.empty_like(cderi)
    expanded = cp.zeros((maximum, nao, nao), dtype=cderi.dtype)
    for p0 in range(0, naux, maximum):
        p1 = min(naux, p0 + maximum)
        width = p1 - p0
        dense = expanded[:width]
        dense[...] = 0.
        dense[:, row, column] = cderi[p0:p1]
        dense[:, column, row] = cderi[p0:p1]
        both = make_cderi_observable_adjoints(
            dense, dm0, dz, auxiliary_block_size=width)
        selected = both[0 if observable == 'target_j' else 1]
        output[p0:p1] = selected[:, row, column] * weights[None]
    return output


def push_packed_cderi_adjoint(
        factor, cderi, adjoint, layout, pair_block_size=8192,
        overwrite_adjoint=True):
    r'''Push one packed Gamma adjoint through ``C=L^{-1}B``.

    The auxiliary solve is blocked over independent AO-pair columns.  With
    ``overwrite_adjoint=True`` the packed ``dE/dC`` allocation becomes the
    packed ``dE/dB`` allocation, so no second persistent pair tensor exists.
    The metric adjoint is accumulated from the same blocks before the central
    factor can be released.
    '''
    factor = _real_gamma(factor, 'factor')
    cderi = _real_gamma(cderi, 'cderi')
    adjoint = _real_gamma(adjoint, 'adjoint')
    if cderi.ndim != 2 or adjoint.shape != cderi.shape:
        raise ValueError('packed central factor and adjoint must have one shape')
    naux, npair = map(int, cderi.shape)
    if factor.shape != (naux, naux):
        raise ValueError('Cholesky factor and packed CDERI dimensions differ')
    block_size = max(1, min(npair, int(pair_block_size)))
    source = adjoint if overwrite_adjoint else cp.empty_like(adjoint)
    factor_adjoint = cp.zeros((naux, naux), dtype=cderi.dtype)
    for p0 in range(0, npair, block_size):
        p1 = min(npair, p0 + block_size)
        solved = solve_triangular(
            factor.T, adjoint[:, p0:p1], lower=False,
            check_finite=False)
        source[:, p0:p1] = solved
        factor_adjoint -= solved @ cderi[:, p0:p1].T

    reduced_adjoint = factor.T @ factor_adjoint
    indices = cp.arange(naux)
    lower = cp.tril(reduced_adjoint, k=-1)
    lower[indices, indices] = .5 * cp.diag(reduced_adjoint)
    symmetric = .5 * (lower + lower.T)
    left = solve_triangular(
        factor.T, symmetric, lower=False, check_finite=False)
    metric = solve_triangular(
        factor.T, left.T, lower=False, check_finite=False).T
    return source, metric, {
        'pair_block_size': block_size,
        'pair_block_count': int((npair + block_size - 1) // block_size),
        'packed_source_bytes': int(source.nbytes),
        'source_reuses_adjoint_storage': bool(overwrite_adjoint),
        'dense_source_bytes': int(
            naux * _layout_nao(layout) ** 2 * source.dtype.itemsize),
    }


def expand_packed_source_adjoint(source, layout):
    r'''Expand packed-variable ``dE/dB`` into dense per-entry derivatives.

    Off-diagonal packed variables represent two equal dense entries.  Their
    derivative is therefore divided by two before assigning both symmetric
    elements.  The returned layout matches ``contract_source_derivative``.
    '''
    source = _real_gamma(source, 'source')
    if source.ndim != 2:
        raise ValueError('packed source adjoint must have shape (naux,npair)')
    nao = _layout_nao(layout)
    row = cp.asarray(layout['row'], dtype=cp.int32)
    column = cp.asarray(layout['column'], dtype=cp.int32)
    weights = cp.asarray(layout['weights'], dtype=source.dtype)
    if source.shape[1] != len(row):
        raise ValueError('packed source and canonical pair layout differ')
    values = source / weights[None]
    expanded = cp.zeros((source.shape[0], nao, nao), dtype=source.dtype)
    expanded[:, row, column] = values
    expanded[:, column, row] = values
    return expanded.reshape(source.shape[0], nao * nao).T


def _layout_nao(layout):
    if 'nao' in layout:
        return int(layout['nao'])
    address = np.asarray(layout['address'], dtype=np.int64)
    if address.size == 0:
        raise ValueError('canonical pair layout is empty')
    # Every physical AO diagonal belongs to the Gamma shell-pair layout.
    return int(np.sqrt(int(address.max()) + 1))


def _real_gamma(value, name, tol=1.0e-10):
    '''Return a real Gamma tensor or reject unsupported complex storage.'''
    value = cp.asarray(value)
    if value.dtype.kind == 'c':
        maximum = float(cp.max(cp.abs(value.imag)).get()) if value.size else 0.
        if maximum > tol:
            raise NotImplementedError(
                f'{name} has non-Gamma imaginary component {maximum:.3e}')
        value = value.real
    return value
