r'''Packed-source-free consumers for the exact real-Gamma RHF adjoint.

This module provides the production MO-direct LR/SR consumers for the
occupied-space representation from
:mod:`dcforge.gpu.pbc.df.gdf.rhf_factorized`.  They avoid materializing the
persistent weighted ``(naux,npair)`` primitive source.  The established
packed implementation remains the explicit numerical oracle and exact
unsupported-representation fallback.

For every auxiliary function ``P``, the exact primitive source is

.. math::

   S_P = C A_P C^T + C K_P U^T + U K_P C^T,

where ``C`` contains occupied RHF orbitals and ``U=C_v Z`` contains the
occupied-column response orbitals.  In one reciprocal block ``F_G``, define

.. math::

   H_G = C^T F_G C, \qquad X_G = C^T F_G U.

The auxiliary-side three-center source is then evaluated directly as

.. math::

   (S:F)_{PG} = A_P:H_G + K_P:X_G + K_P:X_G^T,

and the AO-side derivative source follows by first forming
``A_G=sum_P A_P V_PG`` and ``K_G=sum_P K_P V_PG``.  Only the transaction-local
``(nao,nao,nG)`` source required by ``PBC_ft_aopair_ek_ip1`` is produced.

The short-range helper evaluates the same ``S_P`` only at libpbc's requested
raw AO-pair addresses.  Duplicate and oppositely oriented raw pairs are kept;
no unique-pair multiplicity is applied because ``PBCsr_ejk_int3c2e_ip1``
consumes raw, unweighted values.
'''

import cupy as cp
from gpu4pyscf.lib.cupy_helper import contract
from pyscf import lib

DEFAULT_RAW_PAIR_BLOCK_SIZE = 8192


def contract_rhf_mo_response_long_range_block(
        factors, pair_fourier, auxiliary_fourier, auxiliary_weighted,
        include_metric_source=True):
    r'''Consume one ``RHFMOResponseFactors`` object in an LR transaction.

    This is the private integration boundary for the MO-direct zero-order
    builder in :mod:`rhf_mo_response`.  Attribute access is deliberately
    structural rather than tied to a concrete class import, keeping the pure
    contraction helper independently testable and avoiding an import cycle.
    '''
    return contract_rhf_mo_factorized_long_range_block(
        pair_fourier, auxiliary_fourier, auxiliary_weighted,
        factors.occupied_coefficients, factors.response_orbitals,
        factors.source_factor, factors.response_factor,
        metric_source=(
            factors.metric_source if include_metric_source else None))


def build_rhf_mo_response_short_range_source_block(
        factors, pair_addresses, auxiliary_start=0, auxiliary_stop=None,
        pair_block_size=DEFAULT_RAW_PAIR_BLOCK_SIZE, out=None,
        pair_indices=None):
    r'''Consume one auxiliary slice of ``RHFMOResponseFactors`` for SR.

    ``auxiliary_start:auxiliary_stop`` follows the native SR auxiliary AO
    order.  The returned raw-pair buffer is transaction-local and contains no
    diagonal/off-diagonal multiplicity; it can therefore fill the ``packed``
    argument of ``PBCsr_ejk_int3c2e_ip1`` directly.
    '''
    naux = int(factors.source_factor.shape[0])
    start = int(auxiliary_start)
    stop = naux if auxiliary_stop is None else int(auxiliary_stop)
    if start < 0 or stop <= start or stop > naux:
        raise ValueError(
            'auxiliary interval must satisfy 0 <= start < stop <= naux')
    return build_rhf_mo_factorized_short_range_source_block(
        pair_addresses, factors.occupied_coefficients,
        factors.response_orbitals, factors.source_factor[start:stop],
        factors.response_factor[start:stop],
        pair_block_size=pair_block_size, out=out,
        pair_indices=pair_indices)


def contract_rhf_mo_factorized_long_range_block(
        pair_fourier, auxiliary_fourier, auxiliary_weighted,
        occupied_coefficients, response_orbitals, source_factor,
        response_factor, metric_source=None):
    r'''Build one exact LR derivative transaction without a packed source.

    Args:
        pair_fourier : cupy.ndarray
            Symmetric AO-pair Fourier block ``F_G`` with shape
            ``(nao,nao,nG)`` and complex128 dtype, or its contiguous FP64
            real view with shape ``(nao,nao,2*nG)``.
        auxiliary_fourier : cupy.ndarray
            Auxiliary Fourier values with shape ``(naux,nG)`` and complex128
            dtype, or their contiguous FP64 real view.  They enter only the
            metric source ``-Gamma @ auxiliary_fourier``.
        auxiliary_weighted : cupy.ndarray
            Coulomb-weighted conjugate auxiliary Fourier values
            ``V_PG`` in the same complex/real-view convention.  This is the
            weight used to construct the AO derivative source.
        occupied_coefficients : cupy.ndarray of shape (nao,nocc)
            Real occupied RHF coefficient matrix ``C``.
        response_orbitals : cupy.ndarray of shape (nao,nocc)
            Real response-orbital matrix ``U=C_v Z``.
        source_factor : cupy.ndarray of shape (naux,nocc,nocc)
            Occupied-space matrices ``A_P``.
        response_factor : cupy.ndarray of shape (naux,nocc,nocc)
            Occupied-space matrices ``K_P``.

    Kwargs:
        metric_source : cupy.ndarray or None
            Exact auxiliary metric adjoint ``Gamma`` with shape
            ``(naux,naux)``.  When supplied, the returned auxiliary source
            includes ``-Gamma @ auxiliary_fourier``, matching the native LR
            consumer.  ``None`` returns the three-center ``S:F`` source only.

    Returns:
        tuple
            ``(auxiliary_source, ao_derivative_source)`` as contiguous FP64
            real views with shapes ``(naux,2*nG)`` and
            ``(nao,nao,2*nG)``.  They can be passed directly to the existing
            reciprocal derivative reductions; no persistent AO-pair source
            is allocated.

    Notes:
        Real-view contraction is intentional.  ``C``, ``U``, ``A`` and ``K``
        are real, so the real and imaginary Fourier columns undergo the same
        real linear map independently.  No complex conjugation belongs in
        the AO/MO projection itself; the conjugation is already encoded in
        ``auxiliary_weighted`` by the caller.
    '''
    occupied, response, source_factor, response_factor = _validate_factors(
        occupied_coefficients, response_orbitals, source_factor,
        response_factor)
    nao, nocc = map(int, occupied.shape)
    naux = int(source_factor.shape[0])
    pair_real = _fourier_real_view(
        pair_fourier, 'pair_fourier', leading_shape=(nao, nao))
    auxiliary_real = _fourier_real_view(
        auxiliary_fourier, 'auxiliary_fourier', leading_shape=(naux,))
    weighted_real = _fourier_real_view(
        auxiliary_weighted, 'auxiliary_weighted', leading_shape=(naux,))
    nreal = int(pair_real.shape[-1])
    if (int(auxiliary_real.shape[-1]) != nreal
            or int(weighted_real.shape[-1]) != nreal):
        raise ValueError('all Fourier inputs must describe the same G block')

    # H_G=C^T F_G C and X_G=C^T F_G U.  The first transform is shared.
    transformed = cp.empty((nocc, nao, nreal), dtype=cp.float64)
    contract('pi,pqG->iqG', occupied, pair_real, out=transformed)
    occupied_pair = cp.empty((nocc, nocc, nreal), dtype=cp.float64)
    mixed_pair = cp.empty_like(occupied_pair)
    contract('iqG,qj->ijG', transformed, occupied, out=occupied_pair)
    contract('iqG,qj->ijG', transformed, response, out=mixed_pair)
    del transformed

    auxiliary_source = cp.empty((naux, nreal), dtype=cp.float64)
    contract('rij,ijG->rG', source_factor, occupied_pair,
             out=auxiliary_source)
    contract('rij,ijG->rG', response_factor, mixed_pair,
             alpha=1., beta=1., out=auxiliary_source)
    contract('rij,jiG->rG', response_factor, mixed_pair,
             alpha=1., beta=1., out=auxiliary_source)
    if metric_source is not None:
        metric = _real_matrix(metric_source, 'metric_source')
        if metric.shape != (naux, naux):
            raise ValueError(
                'metric_source must have shape (naux,naux)')
        contract('sr,sG->rG', metric, auxiliary_real,
                 alpha=-1., beta=1., out=auxiliary_source)

    weighted_source = cp.empty((nocc, nocc, nreal), dtype=cp.float64)
    weighted_response = cp.empty_like(weighted_source)
    contract('rij,rG->ijG', source_factor, weighted_real,
             out=weighted_source)
    contract('rij,rG->ijG', response_factor, weighted_real,
             out=weighted_response)

    ao_source = cp.empty((nao, nao, nreal), dtype=cp.float64)
    left = cp.empty((nao, nocc, nreal), dtype=cp.float64)
    contract('pi,ijG->pjG', occupied, weighted_source, out=left)
    contract('pjG,qj->pqG', left, occupied, out=ao_source)
    contract('pi,ijG->pjG', occupied, weighted_response, out=left)
    contract('pjG,qj->pqG', left, response,
             alpha=1., beta=1., out=ao_source)
    contract('pi,ijG->pjG', response, weighted_response, out=left)
    contract('pjG,qj->pqG', left, occupied,
             alpha=1., beta=1., out=ao_source)
    return cp.ascontiguousarray(auxiliary_source), cp.ascontiguousarray(ao_source)


def build_rhf_mo_factorized_short_range_source_block(
        pair_addresses, occupied_coefficients, response_orbitals,
        source_factor_block, response_factor_block,
        pair_block_size=DEFAULT_RAW_PAIR_BLOCK_SIZE, out=None,
        pair_indices=None):
    r'''Build one raw-pair SR source transaction from exact MO factors.

    Args:
        pair_addresses : array_like of int
            Raw flattened sorted-AO addresses ``mu*nao+nu`` in exactly the
            order requested by libpbc.  Duplicates and both orientations are
            retained.
        occupied_coefficients : cupy.ndarray of shape (nao,nocc)
            Real occupied RHF coefficient matrix ``C``.
        response_orbitals : cupy.ndarray of shape (nao,nocc)
            Real response orbitals ``U=C_v Z``.
        source_factor_block : cupy.ndarray of shape (naux_block,nocc,nocc)
            Transaction-local auxiliary slice of ``A_P``.
        response_factor_block : cupy.ndarray of shape (naux_block,nocc,nocc)
            Matching auxiliary slice of ``K_P``.

    Kwargs:
        pair_block_size : int
            Maximum raw AO pairs contracted in one temporary.  A final tail
            is handled without padding or changing pair order.
        out : cupy.ndarray or None
            Optional destination with shape ``(nraw,naux_block)``.  Supplying
            the libpbc batch buffer avoids another transaction allocation.

    Returns:
        cupy.ndarray
            Raw, unweighted source values with shape
            ``(nraw,naux_block)``.  This is the layout consumed by
            ``PBCsr_ejk_int3c2e_ip1``.  Unique-pair weights must not be
            applied at this boundary.
    '''
    occupied, response, source_factor, response_factor = _validate_factors(
        occupied_coefficients, response_orbitals, source_factor_block,
        response_factor_block)
    nao, nocc = map(int, occupied.shape)
    if pair_indices is None:
        addresses = cp.asarray(pair_addresses, dtype=cp.int64).ravel()
        row, column = prepare_rhf_mo_short_range_pair_indices(
            addresses, nao)
        pair_count = int(addresses.size)
    else:
        try:
            row, column = pair_indices
        except (TypeError, ValueError) as error:
            raise ValueError(
                'pair_indices must contain prepared row and column arrays') \
                from error
        row = cp.asarray(row, dtype=cp.int64).ravel()
        column = cp.asarray(column, dtype=cp.int64).ravel()
        pair_count = int(row.size)
        if (column.size != pair_count
                or len(pair_addresses) != pair_count):
            raise ValueError(
                'prepared pair indices must match pair_addresses size')
    maximum = _block_size(
        pair_block_size, pair_count, 'pair_block_size')
    naux_block = int(source_factor.shape[0])
    if out is None:
        output = cp.empty(
            (pair_count, naux_block), dtype=cp.float64)
    else:
        output = cp.asarray(out)
        if output.shape != (pair_count, naux_block):
            raise ValueError(
                'out must have shape (nraw,naux_block)')
        if output.dtype != cp.float64:
            raise ValueError('out must use FP64 dtype')

    # Form each expensive nocc-by-nocc transform once per AO row, not once per
    # raw AO pair.  The previous pair-first ordering repeated the same row
    # transform for every image/shell-pair occurrence and became the Water20
    # bottleneck.  Raw pair ordering, duplicates and orientations enter only
    # the following bounded dot products, so the mathematical source is
    # unchanged.
    left = cp.empty((naux_block, nao, nocc), dtype=cp.float64)
    contract('mi,rij->rmj', occupied, source_factor, out=left)
    _contract_raw_pair_values(
        output, left, occupied, row, column, maximum, accumulate=False)
    contract('mi,rij->rmj', occupied, response_factor, out=left)
    _contract_raw_pair_values(
        output, left, response, row, column, maximum, accumulate=True)
    contract('mi,rij->rmj', response, response_factor, out=left)
    _contract_raw_pair_values(
        output, left, occupied, row, column, maximum, accumulate=True)
    return output


def _contract_raw_pair_values(
        output, left, right, row, column, maximum, accumulate):
    '''Contract an AO-left factor over bounded raw pair slices.'''
    pair_count = int(row.size)
    for q0, q1 in lib.prange(0, pair_count, maximum):
        value = contract(
            'rki,ki->rk', left[:, row[q0:q1]], right[column[q0:q1]])
        if accumulate:
            output[q0:q1] += value.T
        else:
            output[q0:q1] = value.T


def prepare_rhf_mo_short_range_pair_indices(pair_addresses, nao):
    '''Validate raw pair addresses once and return reusable GPU indices.

    The native SR driver calls the factorized source builder for many
    auxiliary slices.  Performing ``min().get()``, ``max().get()`` and
    ``divmod`` inside every slice would add profiler-off synchronization and
    repeated allocations.  This preparation boundary performs those checks
    once per complete derivative traversal.
    '''
    nao = int(nao)
    if nao < 1:
        raise ValueError('nao must be positive')
    addresses = cp.asarray(pair_addresses, dtype=cp.int64).ravel()
    if int(addresses.size) < 1:
        raise ValueError('pair_addresses must contain at least one raw pair')
    minimum = int(addresses.min().get())
    maximum = int(addresses.max().get())
    if minimum < 0 or maximum >= nao * nao:
        raise ValueError('pair address is outside the flattened AO matrix')
    return divmod(addresses, nao)


def _validate_factors(occupied_coefficients, response_orbitals,
                      source_factor, response_factor):
    '''Validate one real-Gamma ``C,U,A,K`` representation.'''
    occupied = _real_matrix(
        occupied_coefficients, 'occupied_coefficients')
    response = _real_matrix(response_orbitals, 'response_orbitals')
    if occupied.shape != response.shape:
        raise ValueError(
            'response_orbitals must have the same shape as '
            'occupied_coefficients')
    if int(occupied.shape[0]) < 1 or int(occupied.shape[1]) < 1:
        raise ValueError('occupied_coefficients dimensions must be positive')
    source = _real_matrix_stack(source_factor, 'source_factor')
    response_source = _real_matrix_stack(
        response_factor, 'response_factor')
    expected = (int(source.shape[0]), int(occupied.shape[1]),
                int(occupied.shape[1]))
    if source.shape != expected or response_source.shape != expected:
        raise ValueError(
            'source_factor and response_factor must have shape '
            '(naux,nocc,nocc)')
    return occupied, response, source, response_source


def _fourier_real_view(value, name, leading_shape):
    '''Return one contiguous complex-Fourier block as its FP64 real view.'''
    array = cp.asarray(value)
    if tuple(array.shape[:-1]) != tuple(leading_shape):
        raise ValueError(
            f'{name} leading dimensions must be {tuple(leading_shape)}')
    if array.dtype == cp.complex128:
        if int(array.shape[-1]) < 1:
            raise ValueError(f'{name} must contain at least one G vector')
        array = cp.ascontiguousarray(array)
        return array.view(cp.float64).reshape(*leading_shape, -1)
    if array.dtype == cp.float64:
        if int(array.shape[-1]) < 2 or int(array.shape[-1]) % 2:
            raise ValueError(
                f'{name} FP64 real view must have an even final dimension')
        return cp.ascontiguousarray(array)
    raise TypeError(f'{name} must use complex128 or its FP64 real view')


def _real_matrix(value, name):
    '''Return a real FP64 matrix or reject a genuinely complex value.'''
    array = cp.asarray(value)
    if array.ndim != 2:
        raise ValueError(f'{name} must be a matrix')
    return _real_array(array, name)


def _real_matrix_stack(value, name):
    '''Return a real FP64 stack of square matrices.'''
    array = cp.asarray(value)
    if (array.ndim != 3 or int(array.shape[0]) < 1
            or int(array.shape[1]) < 1
            or array.shape[1] != array.shape[2]):
        raise ValueError(f'{name} must have shape (naux,nocc,nocc)')
    return _real_array(array, name)


def _real_array(value, name):
    '''Convert a numerical array to FP64 within the real-Gamma domain.'''
    array = cp.asarray(value)
    if cp.iscomplexobj(array):
        imaginary = float(cp.max(cp.abs(array.imag)).get())
        if imaginary > 1.e-10:
            raise NotImplementedError(
                f'{name} must be real in the real-Gamma RHF domain')
        array = array.real
    return cp.asarray(array, dtype=cp.float64)


def _block_size(value, length, name):
    '''Resolve one positive bounded block size, including a final tail.'''
    value = int(value)
    if value < 1:
        raise ValueError(f'{name} must be positive')
    return min(int(length), value)
