r'''Exact MO-factorized real-Gamma GDF adjoint algebra for RHF.

This module defines the shared exact algebra for the native GDF nuclear
response.  The packed density adjoint in :mod:`adjoint` remains the explicit
numerical oracle and unsupported-representation fallback.  Production
``auto`` consumes the same factors through the packed-source-free
``mo_direct`` route; the dense and packed-backed constructors here remain
validation endpoints.

Let ``C`` contain the occupied RHF orbitals and let ``U = C_v Z`` be the
occupied-column response orbital.  The two AO densities are

.. math::

   P = 2 C C^T, \qquad T = U C^T + C U^T.

For the metric-solved three-center tensor ``W = M^{-1} B``, define

.. math::

   O_P = C^T W_P C, \qquad R_P = C^T W_P U,

   a_P = 2\operatorname{Tr}(O_P), \qquad
   b_P = 2\operatorname{Tr}(R_P).

The exact dense primitive source can then be written without a dense AO
density contraction,

.. math::

   S_P = C A_P C^T + C K_P U^T + U K_P C^T,

where

.. math::

   A_P &= 2\delta a_P I
          + s\{4 b_P I - 2(R_P + R_P^T)\}, \\
   K_P &= 2s(a_P I - O_P).

Here ``delta`` is one when target Hartree is included and zero otherwise;
``s`` is ``response_sign``.  The corresponding metric source is formed
directly in occupied space,

.. math::

   \Gamma = \delta a a^T + 2s(ba^T + ab^T)
            - 4s(X + X^T),

with ``X_PQ = sum_ij R_Pij O_Qij``.  No approximation, density truncation,
or change to the auxiliary metric gauge is introduced.
'''

import cupy as cp
from gpu4pyscf.lib.cupy_helper import contract

DEFAULT_AUXILIARY_BLOCK_SIZE = 32
DEFAULT_PAIR_BLOCK_SIZE = 8192


def project_rhf_mo_intermediates(
        solved_three_center, occupied_coefficients, response_orbitals,
        auxiliary_block_size=DEFAULT_AUXILIARY_BLOCK_SIZE):
    r'''Project a dense real-Gamma ``W`` into the occupied response space.

    The projections are evaluated in bounded auxiliary blocks.  Symmetry of
    each real-Gamma ``W_P`` is used to form ``C^T W_P`` from ``(W_P C)^T``;
    consequently only one AO-by-occupied transform is required per block.

    Args:
        solved_three_center : cupy.ndarray of shape (naux, nao, nao)
            Real symmetric metric-solved primitive tensor
            ``W = M^{-1} B``.
        occupied_coefficients : cupy.ndarray of shape (nao, nocc)
            Real occupied RHF coefficient matrix ``C``.  The closed-shell
            factor two is not included in this matrix.
        response_orbitals : cupy.ndarray of shape (nao, nocc)
            Real response-orbital matrix ``U = C_v Z`` using the same
            occupied-column order as ``C``.

    Kwargs:
        auxiliary_block_size : int
            Maximum number of auxiliary rows transformed together.  A final
            non-divisible block is handled without padding the result.

    Returns:
        tuple
            ``(O, R, a, b)`` with ``O`` and ``R`` of shape
            ``(naux,nocc,nocc)`` and ``a`` and ``b`` of shape ``(naux,)``.
            They are exactly ``C.T @ W @ C``, ``C.T @ W @ U``,
            ``2*trace(O)`` and ``2*trace(R)`` in FP64 arithmetic.

    Notes:
        This internal function trusts the native real-Gamma provider contract
        that every ``W_P`` is symmetric.  It does not materialize a second
        full tensor merely to recheck that invariant.
    '''
    solved = _as_real_matrix_stack(
        solved_three_center, 'solved_three_center')
    occupied, response = _validate_orbitals(
        occupied_coefficients, response_orbitals, solved.shape[1])
    maximum = _block_size(
        auxiliary_block_size, int(solved.shape[0]), 'auxiliary_block_size')
    return _project_dense(solved, occupied, response, maximum)


def build_rhf_mo_factorized_primitive_adjoint(
        solved_three_center, occupied_coefficients, response_orbitals,
        include_target_j=False, response_sign=1.,
        auxiliary_block_size=DEFAULT_AUXILIARY_BLOCK_SIZE):
    r'''Build the exact dense primitive source through RHF MO factors.

    This routine materializes a dense source only as a validation endpoint.
    Its result is algebraically identical to
    :func:`dcforge.gpu.pbc.df.gdf.adjoint.build_rhf_primitive_adjoint` for
    ``P=2 C C^T`` and ``T=U C^T+C U^T``.  The existing density-based routine
    remains the numerical oracle and exact fallback; this materializing
    constructor is a validation endpoint rather than the production
    ``mo_direct`` consumer.

    Args:
        solved_three_center : cupy.ndarray of shape (naux, nao, nao)
            Real symmetric metric-solved tensor ``W = M^{-1} B``.
        occupied_coefficients : cupy.ndarray of shape (nao, nocc)
            Real occupied RHF coefficient matrix ``C``.
        response_orbitals : cupy.ndarray of shape (nao, nocc)
            Real response orbitals ``U = C_v Z``.

    Kwargs:
        include_target_j : bool
            Include the target-Hartree source ``a_P P``.  False constructs
            only the signed source-Fock response.
        response_sign : float
            Sign ``s`` of the polarization response.  Only ``+1`` and ``-1``
            are accepted; DC-Forge's native nonvariational convention uses
            ``-1``.
        auxiliary_block_size : int
            Maximum auxiliary rows used for projection and source
            materialization temporaries.

    Returns:
        tuple
            ``(source, metric_source, metadata)``.  ``source`` has shape
            ``(naux,nao,nao)`` and ``metric_source`` has shape
            ``(naux,naux)``.  Both are real FP64 CuPy arrays.  The input
            ``solved_three_center`` is not modified.
    '''
    solved = _as_real_matrix_stack(
        solved_three_center, 'solved_three_center')
    occupied, response = _validate_orbitals(
        occupied_coefficients, response_orbitals, solved.shape[1])
    response_sign = _response_sign(response_sign)
    naux = int(solved.shape[0])
    maximum = _block_size(
        auxiliary_block_size, naux, 'auxiliary_block_size')

    occupied_projection, mixed_projection, auxiliary_reference, \
        auxiliary_response = _project_dense(
            solved, occupied, response, maximum)
    source_factor, response_factor = _build_source_factors(
        occupied_projection, mixed_projection, auxiliary_reference,
        auxiliary_response, include_target_j=include_target_j,
        response_sign=response_sign)
    metric_source = _build_metric_source(
        occupied_projection, mixed_projection, auxiliary_reference,
        auxiliary_response, include_target_j=include_target_j,
        response_sign=response_sign)
    source = _materialize_dense_source(
        occupied, response, source_factor, response_factor, maximum,
        solved.shape[1])

    metadata = _metadata(
        route='real_gamma_rhf_mo_factorized_dense_adjoint', solved=solved,
        occupied=occupied, include_target_j=include_target_j,
        response_sign=response_sign, auxiliary_block_size=maximum)
    metadata.update({
        'source_storage': 'dense_ao_validation',
        'persistent_source_bytes': int(source.nbytes),
        'metric_bytes': int(metric_source.nbytes),
    })
    return source, metric_source, metadata


def build_rhf_mo_factorized_packed_adjoint(
        solved_three_center, layout, occupied_coefficients,
        response_orbitals, include_target_j=False, response_sign=1.,
        auxiliary_block_size=DEFAULT_AUXILIARY_BLOCK_SIZE,
        pair_block_size=DEFAULT_PAIR_BLOCK_SIZE,
        overwrite_solved=False):
    r'''Build the exact factorized source in unique symmetric-pair storage.

    ``solved_three_center`` follows the native packed convention: one
    unweighted value is stored for each unique AO pair.  The returned source
    contains ``S_P[row,column] * weights`` so that

    .. math::

       \Gamma_{PQ}
       = \sum_k source_{Pk}\,W_{Qk}
       = \sum_{ij} S_{Pij}W_{Qij}.

    Source pairs are contracted directly from ``A_P`` and ``K_P`` in bounded
    pair blocks.  A dense ``(naux,nao,nao)`` source is never materialized.
    A bounded dense ``W`` block is used only while projecting the packed
    solved tensor to ``O`` and ``R``.

    Args:
        solved_three_center : cupy.ndarray of shape (naux, npair)
            Unweighted unique-pair storage of real symmetric
            ``W = M^{-1}B``.
        layout : dict
            Canonical pair layout with ``nao``, ``row``, ``column`` and
            ``weights`` entries.  Diagonal weights must be one and
            off-diagonal weights two.
        occupied_coefficients : cupy.ndarray of shape (nao, nocc)
            Real occupied RHF coefficient matrix ``C``.
        response_orbitals : cupy.ndarray of shape (nao, nocc)
            Real response orbitals ``U = C_v Z``.

    Kwargs:
        include_target_j : bool
            Include the target-Hartree primitive source.
        response_sign : float
            Signed polarization coefficient ``s``; only ``+1`` or ``-1``.
        auxiliary_block_size : int
            Maximum auxiliary rows expanded or contracted together.
        pair_block_size : int
            Maximum unique AO pairs gathered in one MO-source contraction.
        overwrite_solved : bool
            Overwrite ``solved_three_center`` with the weighted packed source
            after all ``O`` and ``R`` projections have been completed.  This
            removes a second persistent ``(naux,npair)`` device allocation.
            The default is false because this module remains an experimental
            validation endpoint; the native integration may opt in once it
            no longer needs ``W``.

    Returns:
        tuple
            ``(packed_source, metric_source, metadata)``.  The packed source
            has the same shape as ``solved_three_center`` and uses the
            weighted derivative convention.  ``metric_source`` has shape
            ``(naux,naux)``.  The input solved tensor is unchanged unless
            ``overwrite_solved=True``.
    '''
    solved, row, column, weights, nao = _validate_packed(
        solved_three_center, layout)
    occupied, response = _validate_orbitals(
        occupied_coefficients, response_orbitals, nao)
    response_sign = _response_sign(response_sign)
    naux = int(solved.shape[0])
    maximum = _block_size(
        auxiliary_block_size, naux, 'auxiliary_block_size')
    pair_maximum = _block_size(
        pair_block_size, int(solved.shape[1]), 'pair_block_size')

    occupied_projection, mixed_projection, auxiliary_reference, \
        auxiliary_response = _project_packed(
            solved, row, column, nao, occupied, response, maximum)
    source_factor, response_factor = _build_source_factors(
        occupied_projection, mixed_projection, auxiliary_reference,
        auxiliary_response, include_target_j=include_target_j,
        response_sign=response_sign)
    metric_source = _build_metric_source(
        occupied_projection, mixed_projection, auxiliary_reference,
        auxiliary_response, include_target_j=include_target_j,
        response_sign=response_sign)
    source = _materialize_packed_source(
        occupied, response, source_factor, response_factor,
        row, column, weights, maximum, pair_maximum,
        out=solved if overwrite_solved else None)

    metadata = _metadata(
        route='real_gamma_rhf_mo_factorized_packed_adjoint', solved=solved,
        occupied=occupied, include_target_j=include_target_j,
        response_sign=response_sign, auxiliary_block_size=maximum)
    metadata.update({
        'source_storage': 'weighted_unique_ao_pairs',
        'raw_pair_count': int(layout.get(
            'raw_pair_count', solved.shape[1])),
        'unique_pair_count': int(solved.shape[1]),
        'pair_block_size': pair_maximum,
        'pair_block_count': int(
            (solved.shape[1] + pair_maximum - 1) // pair_maximum),
        'persistent_source_bytes': int(source.nbytes),
        'source_reuses_solved_storage': bool(overwrite_solved),
        'full_source_allocation_avoided_bytes': (
            int(source.nbytes) if overwrite_solved else 0),
        'dense_source_bytes_avoided': int(
            naux * nao * nao * solved.dtype.itemsize - source.nbytes),
        'metric_bytes': int(metric_source.nbytes),
    })
    return source, metric_source, metadata


def _project_dense(solved, occupied, response, maximum):
    '''Return ``O``, ``R``, ``a`` and ``b`` from a dense solved tensor.'''
    naux = int(solved.shape[0])
    nao, nocc = map(int, occupied.shape)
    occupied_projection = cp.empty(
        (naux, nocc, nocc), dtype=solved.dtype)
    mixed_projection = cp.empty_like(occupied_projection)
    solved_occupied = cp.empty(
        (maximum, nao, nocc), dtype=solved.dtype)
    for p0 in range(0, naux, maximum):
        p1 = min(naux, p0 + maximum)
        width = p1 - p0
        transformed = solved_occupied[:width]
        contract('pij,jk->pik', solved[p0:p1], occupied,
                 out=transformed)
        block = contract('pji,jk->pik', transformed, occupied)
        # ``O=C^T W C`` is symmetric in the real-Gamma domain.  Enforcing
        # that exact representation prevents round-off asymmetry from
        # changing the equivalent ``U K C^T`` transpose contribution.
        occupied_projection[p0:p1] = (
            .5 * (block + block.transpose(0, 2, 1)))
        contract('pji,jk->pik', transformed, response,
                 out=mixed_projection[p0:p1])
    auxiliary_reference = 2. * cp.trace(
        occupied_projection, axis1=1, axis2=2)
    auxiliary_response = 2. * cp.trace(
        mixed_projection, axis1=1, axis2=2)
    return (occupied_projection, mixed_projection,
            auxiliary_reference, auxiliary_response)


def _project_packed(
        solved, row, column, nao, occupied, response, maximum):
    '''Expand bounded packed ``W`` blocks and return their MO projections.'''
    naux = int(solved.shape[0])
    nocc = int(occupied.shape[1])
    occupied_projection = cp.empty(
        (naux, nocc, nocc), dtype=solved.dtype)
    mixed_projection = cp.empty_like(occupied_projection)
    dense_solved = cp.zeros((maximum, nao, nao), dtype=solved.dtype)
    solved_occupied = cp.empty(
        (maximum, nao, nocc), dtype=solved.dtype)
    for p0 in range(0, naux, maximum):
        p1 = min(naux, p0 + maximum)
        width = p1 - p0
        dense = dense_solved[:width]
        dense[...] = 0.
        dense[:, row, column] = solved[p0:p1]
        dense[:, column, row] = solved[p0:p1]
        transformed = solved_occupied[:width]
        contract('pij,jk->pik', dense, occupied, out=transformed)
        block = contract('pji,jk->pik', transformed, occupied)
        occupied_projection[p0:p1] = (
            .5 * (block + block.transpose(0, 2, 1)))
        contract('pji,jk->pik', transformed, response,
                 out=mixed_projection[p0:p1])
    auxiliary_reference = 2. * cp.trace(
        occupied_projection, axis1=1, axis2=2)
    auxiliary_response = 2. * cp.trace(
        mixed_projection, axis1=1, axis2=2)
    return (occupied_projection, mixed_projection,
            auxiliary_reference, auxiliary_response)


def _build_source_factors(
        occupied_projection, mixed_projection, auxiliary_reference,
        auxiliary_response, include_target_j, response_sign):
    r'''Return occupied matrices ``A_P`` and ``K_P`` for every auxiliary.'''
    nocc = int(occupied_projection.shape[1])
    diagonal = cp.arange(nocc)
    source_factor = -2. * response_sign * (
        mixed_projection + mixed_projection.transpose(0, 2, 1))
    source_factor[:, diagonal, diagonal] += (
        4. * response_sign * auxiliary_response[:, None])
    if include_target_j:
        source_factor[:, diagonal, diagonal] += (
            2. * auxiliary_reference[:, None])
    response_factor = -2. * response_sign * occupied_projection
    response_factor[:, diagonal, diagonal] += (
        2. * response_sign * auxiliary_reference[:, None])
    return source_factor, response_factor


def _build_metric_source(
        occupied_projection, mixed_projection, auxiliary_reference,
        auxiliary_response, include_target_j, response_sign):
    r'''Build ``Gamma`` directly from ``O``, ``R``, ``a`` and ``b``.'''
    cross = contract(
        'pij,qij->pq', mixed_projection, occupied_projection)
    metric_source = -4. * response_sign * (cross + cross.T)
    metric_source += 2. * response_sign * (
        cp.outer(auxiliary_response, auxiliary_reference)
        + cp.outer(auxiliary_reference, auxiliary_response))
    if include_target_j:
        metric_source += cp.outer(
            auxiliary_reference, auxiliary_reference)
    return metric_source


def _materialize_dense_source(
        occupied, response, source_factor, response_factor, maximum, nao):
    '''Materialize ``C A C^T + C K U^T + U K C^T`` in AO space.'''
    naux = int(source_factor.shape[0])
    nocc = int(occupied.shape[1])
    source = cp.empty((naux, nao, nao), dtype=occupied.dtype)
    left = cp.empty((maximum, nao, nocc), dtype=occupied.dtype)
    temporary = cp.empty((maximum, nao, nao), dtype=occupied.dtype)
    for p0 in range(0, naux, maximum):
        p1 = min(naux, p0 + maximum)
        width = p1 - p0
        left_block = left[:width]
        temporary_block = temporary[:width]
        contract('mi,pij->pmj', occupied, source_factor[p0:p1],
                 out=left_block)
        contract('pmi,ni->pmn', left_block, occupied,
                 out=source[p0:p1])
        contract('mi,pij->pmj', occupied, response_factor[p0:p1],
                 out=left_block)
        contract('pmi,ni->pmn', left_block, response,
                 out=temporary_block)
        source[p0:p1] += temporary_block
        contract('mi,pij->pmj', response, response_factor[p0:p1],
                 out=left_block)
        contract('pmi,ni->pmn', left_block, occupied,
                 out=temporary_block)
        source[p0:p1] += temporary_block
    return source


def _materialize_packed_source(
        occupied, response, source_factor, response_factor,
        row, column, weights, maximum, pair_maximum, out=None):
    '''Contract factorized AO-pair values without a dense AO source.'''
    naux = int(source_factor.shape[0])
    npair = int(row.size)
    nao, nocc = map(int, occupied.shape)
    if out is None:
        source = cp.empty((naux, npair), dtype=occupied.dtype)
    else:
        source = cp.asarray(out)
        if source.shape != (naux, npair) or source.dtype != occupied.dtype:
            raise ValueError(
                'packed source output must match the solved tensor shape '
                'and FP64 dtype')
    left = cp.empty((maximum, nao, nocc), dtype=occupied.dtype)
    for p0 in range(0, naux, maximum):
        p1 = min(naux, p0 + maximum)
        width = p1 - p0
        output = source[p0:p1]
        left_block = left[:width]

        contract('mi,pij->pmj', occupied, source_factor[p0:p1],
                 out=left_block)
        _contract_packed_pairs(
            output, left_block, occupied, row, column, pair_maximum,
            accumulate=False)

        contract('mi,pij->pmj', occupied, response_factor[p0:p1],
                 out=left_block)
        _contract_packed_pairs(
            output, left_block, response, row, column, pair_maximum,
            accumulate=True)

        contract('mi,pij->pmj', response, response_factor[p0:p1],
                 out=left_block)
        _contract_packed_pairs(
            output, left_block, occupied, row, column, pair_maximum,
            accumulate=True)
        output *= weights[None]
    return source


def _contract_packed_pairs(
        output, left, right, row, column, pair_maximum, accumulate):
    '''Contract one MO-left factor over bounded unique AO-pair slices.'''
    npair = int(row.size)
    for k0 in range(0, npair, pair_maximum):
        k1 = min(npair, k0 + pair_maximum)
        value = contract(
            'pki,ki->pk', left[:, row[k0:k1]], right[column[k0:k1]])
        if accumulate:
            output[:, k0:k1] += value
        else:
            output[:, k0:k1] = value


def _metadata(
        route, solved, occupied, include_target_j, response_sign,
        auxiliary_block_size):
    '''Return common JSON-compatible factorization metadata.'''
    naux = int(solved.shape[0])
    nocc = int(occupied.shape[1])
    return {
        'route': route,
        'domain': 'real_gamma_closed_shell_rhf',
        'include_target_j': bool(include_target_j),
        'response_sign': float(response_sign),
        'naux': naux,
        'nao': int(occupied.shape[0]),
        'nocc': nocc,
        'auxiliary_block_size': int(auxiliary_block_size),
        'auxiliary_block_count': int(
            (naux + auxiliary_block_size - 1) // auxiliary_block_size),
        'occupied_projection_bytes': int(naux * nocc * nocc * 8),
        'mixed_projection_bytes': int(naux * nocc * nocc * 8),
        'algebra': 'C_A_Ct_plus_C_K_Ut_plus_U_K_Ct',
        'metric_algebra': 'delta_aa_plus_2s_ba_ab_minus_4s_X_Xt',
        # This packed-backed algebra endpoint remains available only by an
        # explicit ``mo_factorized`` request.  Production ``auto`` uses the
        # separately validated packed-source-free ``mo_direct`` endpoint.
        'production_auto_selected': False,
    }


def _validate_packed(solved_three_center, layout):
    '''Validate the unique symmetric-pair representation and return arrays.'''
    solved = cp.asarray(solved_three_center)
    if solved.ndim != 2:
        raise ValueError(
            'packed solved_three_center must have shape (naux, npair)')
    if solved.shape[0] < 1 or solved.shape[1] < 1:
        raise ValueError('packed solved_three_center dimensions must be positive')
    solved = _as_real_array(solved, 'solved_three_center')
    try:
        nao = int(layout['nao'])
        row = cp.asarray(layout['row'], dtype=cp.int32).ravel()
        column = cp.asarray(layout['column'], dtype=cp.int32).ravel()
        weights = cp.asarray(layout['weights'], dtype=solved.dtype).ravel()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            'layout must define nao, row, column and weights') from error
    if nao < 1:
        raise ValueError('layout nao must be positive')
    npair = int(solved.shape[1])
    if row.size != npair or column.size != npair or weights.size != npair:
        raise ValueError('packed layout arrays and solved pair dimension differ')
    minimum = int(cp.minimum(row, column).min().get())
    maximum = int(cp.maximum(row, column).max().get())
    if minimum < 0 or maximum >= nao:
        raise ValueError('packed layout AO-pair index is out of bounds')
    expected = cp.where(row == column, 1., 2.)
    weight_error = float(cp.max(cp.abs(weights - expected)).get())
    if weight_error > 1.e-12:
        raise ValueError(
            'packed layout weights must be one on diagonal pairs and two '
            'on off-diagonal pairs')
    return solved, row, column, weights, nao


def _validate_orbitals(occupied_coefficients, response_orbitals, nao):
    '''Validate real ``C`` and ``U`` matrices in one AO representation.'''
    occupied = _as_real_array(
        occupied_coefficients, 'occupied_coefficients')
    response = _as_real_array(response_orbitals, 'response_orbitals')
    if occupied.ndim != 2:
        raise ValueError(
            'occupied_coefficients must have shape (nao, nocc)')
    if response.ndim != 2:
        raise ValueError('response_orbitals must have shape (nao, nocc)')
    if occupied.shape[0] != nao:
        raise ValueError(
            'occupied_coefficients AO dimension does not match '
            'solved_three_center')
    if response.shape != occupied.shape:
        raise ValueError(
            'response_orbitals must have the same (nao, nocc) shape as '
            'occupied_coefficients')
    if occupied.shape[1] < 1:
        raise ValueError('occupied_coefficients must contain an occupied MO')
    return occupied, response


def _as_real_matrix_stack(value, name):
    '''Validate one real dense ``(naux,nao,nao)`` tensor.'''
    array = cp.asarray(value)
    if (array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 1
            or array.shape[1] != array.shape[2]):
        raise ValueError(f'{name} must have shape (naux, nao, nao)')
    return _as_real_array(array, name)


def _as_real_array(value, name):
    '''Return a real FP64 CuPy array or reject a complex Gamma value.'''
    array = cp.asarray(value)
    if cp.iscomplexobj(array):
        imaginary = float(cp.max(cp.abs(array.imag)).get())
        if imaginary > 1.e-10:
            raise NotImplementedError(
                f'{name} must be real in the real-Gamma RHF factorization')
        array = array.real
    return cp.asarray(array, dtype=cp.float64)


def _response_sign(value):
    '''Return an accepted signed-polarization coefficient.'''
    value = float(value)
    if value not in (-1., 1.):
        raise ValueError('response_sign must be +1 or -1')
    return value


def _block_size(value, length, name):
    '''Resolve one positive bounded block size without hiding tail blocks.'''
    value = int(value)
    if value < 1:
        raise ValueError(f'{name} must be positive')
    return min(int(length), value)
