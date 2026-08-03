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
from cupyx.scipy.linalg import solve_triangular


def make_cderi_observable_adjoints(cderi, dm0, dz):
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

    rho0 = cp.einsum('Pij,ji->P', cderi, dm0, optimize=True)
    rhoz = cp.einsum('Pij,ji->P', cderi, dz, optimize=True)
    target_j = cp.einsum('P,ji->Pij', rho0, dm0, optimize=True)
    fock_j = -2.0 * (
        cp.einsum('P,ji->Pij', rhoz, dm0, optimize=True)
        + cp.einsum('P,ji->Pij', rho0, dz, optimize=True))
    fock_k = cp.einsum('ik,Pkl,lj->Pji', dm0, cderi, dz, optimize=True)
    fock_k += cp.einsum('ik,Pkl,lj->Pji', dz, cderi, dm0, optimize=True)
    return cp.stack((target_j, fock_j + fock_k))


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
