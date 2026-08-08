'''Direct real-Gamma GDF primitive adjoints for the DFA@RHF Z response.

The native Gamma derivative is naturally expressed in terms of an
unorthogonalized three-center tensor ``B_P`` and auxiliary metric ``M``.  For
the metric-solved tensor ``W = M^{-1} B``, this module builds the reverse
sources which contract directly with ``dB`` and ``dM``.  It is therefore an
algebraic replacement for the polarization calculation with ``P + D_Z`` and
``P - D_Z`` densities; it never constructs either full-rank signed density.
'''

import os
import time

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import contract, empty_mapped

DEFAULT_ADJOINT_AUXILIARY_BLOCK_SIZE = 128
DEFAULT_ADJOINT_STORAGE = 'host_cache'
DEFAULT_ADJOINT_HOST_CACHE = 'mapped'
DEFAULT_ADJOINT_HOST_MEMORY_FRACTION = .80
DEFAULT_ADJOINT_HOST_MINIMUM_RESERVE_BYTES = 4 * 1024**3
DEFAULT_ADJOINT_HOST_MAX_BYTES = None
DEFAULT_ADJOINT_HOST_CACHE_FALLBACK = True

_ADJOINT_STORAGE_MODES = ('double_pass', 'host_cache')
_ADJOINT_HOST_CACHE_MODES = ('mapped', 'pageable')


def build_rhf_primitive_adjoint(solved_three_center, reference_density,
                                response_density, include_target_j=False,
                                response_sign=1.,
                                auxiliary_block_size=32,
                                overwrite_solved=False):
    r'''Build reverse ``dB`` and ``dM`` sources at real Gamma.

    Let ``W_P = M^{-1} B_P``, ``P`` be the source RHF density, and ``T`` be
    the Hermitian Z-vector transition density.  The returned ``source[P]``
    obeys

    .. math::

       \sum_P \mathrm{Tr}[source_P^T dB_P]
       - \sum_{PQ} metric_{PQ} dM_{PQ}
       = E_J[P]^\lambda
       + s\{Q[P+T]^\lambda - Q[P-T]^\lambda\},

    where ``Q[D] = Tr[D(J[D]-K[D]/2)]/2`` and ``s=response_sign``.
    ``include_target_j=True`` adds the first target-Hartree contribution;
    ``s=-1`` gives the DC-Forge ``de_j_fock_native`` convention.  The formula
    is not a numerical finite difference or an approximation to the
    signed-state result.

    Args:
        solved_three_center : cupy.ndarray of shape (naux, nao, nao)
            Real, AO-symmetric ``W = M^{-1} B`` in the primitive sorted-AO
            representation.
        reference_density : cupy.ndarray of shape (nao, nao)
            Real symmetric source-RHF density ``P`` in the same AO basis.
        response_density : cupy.ndarray of shape (nao, nao)
            Real symmetric adjoint transition density ``T`` in the same AO
            basis.

    Kwargs:
        include_target_j : bool
            Include ``E_J[P]^lambda`` in addition to the Z-vector response.
        response_sign : float
            Sign of the polarization difference.  Only ``+1`` and ``-1``
            are accepted.  ``-1`` corresponds to
            ``E_J[P] - {Q[P+T]-Q[P-T]}``.
        auxiliary_block_size : int
            Number of auxiliary functions used to build temporary AO source
            blocks.  The cap prevents multiple full
            ``(naux, nao, nao)`` temporaries; ``overwrite_solved=True`` also
            reuses the existing persistent tensor for the final source.
        overwrite_solved : bool
            Reuse ``solved_three_center`` as the returned source.  The metric
            adjoint is accumulated in a first blocked pass while all rows of
            ``W`` remain intact; a second pass then overwrites ``W`` with the
            exact source.  This avoids one complete persistent tensor at the
            cost of repeating only the local source-algebra contractions.

    Returns:
        tuple
            ``(source, metric, metadata)``.  ``source`` has shape
            ``(naux, nao, nao)`` and contracts with ``dB``; ``metric`` has
            shape ``(naux, naux)`` and contracts with ``dM``.  ``metadata``
            documents the bounded auxiliary blocking used to assemble them.
    '''
    solved = _as_real_matrix_stack(solved_three_center, 'solved_three_center')
    reference = _as_real_square(reference_density, 'reference_density')
    response = _as_real_square(response_density, 'response_density')
    if reference.shape != response.shape:
        raise ValueError(
            'reference_density and response_density must have one shape, '
            f'got {reference.shape} and {response.shape}')
    if tuple(solved.shape[1:]) != tuple(reference.shape):
        raise ValueError(
            'solved_three_center AO shape does not match density shape: '
            f'{solved.shape[1:]} != {reference.shape}')
    if auxiliary_block_size < 1:
        raise ValueError('auxiliary_block_size must be positive')
    response_sign = float(response_sign)
    if response_sign not in (-1., 1.):
        raise ValueError('response_sign must be +1 or -1')

    naux = int(solved.shape[0])
    auxiliary_reference = contract('pij,ji->p', solved, reference)
    auxiliary_response = contract('pij,ji->p', solved, response)
    maximum = min(naux, int(auxiliary_block_size))
    block_count = 0
    temporary = cp.empty((maximum, *reference.shape), dtype=solved.dtype)
    exchange = cp.empty_like(temporary)
    if overwrite_solved:
        source_block = cp.empty_like(temporary)
        metric = cp.empty((naux, naux), dtype=solved.dtype)
        for p0 in range(0, naux, maximum):
            p1 = min(naux, p0 + maximum)
            width = p1 - p0
            block = source_block[:width]
            _build_source_block(
                block, solved[p0:p1], reference, response,
                auxiliary_reference[p0:p1], auxiliary_response[p0:p1],
                include_target_j=include_target_j,
                response_sign=response_sign,
                temporary=temporary[:width], exchange=exchange[:width])
            contract('pij,qij->pq', block, solved, out=metric[p0:p1])
            block_count += 1
        for p0 in range(0, naux, maximum):
            p1 = min(naux, p0 + maximum)
            width = p1 - p0
            block = source_block[:width]
            _build_source_block(
                block, solved[p0:p1], reference, response,
                auxiliary_reference[p0:p1], auxiliary_response[p0:p1],
                include_target_j=include_target_j,
                response_sign=response_sign,
                temporary=temporary[:width], exchange=exchange[:width])
            solved[p0:p1] = block
        source = solved
    else:
        source = cp.empty_like(solved)
        for p0 in range(0, naux, maximum):
            p1 = min(naux, p0 + maximum)
            width = p1 - p0
            _build_source_block(
                source[p0:p1], solved[p0:p1], reference, response,
                auxiliary_reference[p0:p1], auxiliary_response[p0:p1],
                include_target_j=include_target_j,
                response_sign=response_sign,
                temporary=temporary[:width], exchange=exchange[:width])
            block_count += 1
        metric = contract('pij,qij->pq', source, solved)

    # Differentiating ``W=M^{-1}B`` gives ``-source:W:dM``.  The native
    # metric kernel therefore receives this positive contraction and applies
    # the leading minus sign in the caller.
    metadata = {
        'route': 'direct_real_gamma_primitive_adjoint',
        'include_target_j': bool(include_target_j),
        'response_sign': response_sign,
        'persistent_source_bytes': int(source.nbytes),
        'source_reuses_solved_storage': bool(overwrite_solved),
        'full_source_allocation_avoided_bytes': (
            int(source.nbytes) if overwrite_solved else 0),
        'source_algebra_passes': 2 if overwrite_solved else 1,
        'metric_bytes': int(metric.nbytes),
        'auxiliary_block_size': maximum,
        'auxiliary_block_count': block_count,
    }
    return source, metric, metadata


def build_rhf_primitive_adjoint_channels(
        solved_three_center, reference_density, response_density,
        response_sign=-1., auxiliary_block_size=32):
    r'''Build separate target-J and Z-Fock primitive adjoint channels.

    The production fused GDF response is the sum of two physical
    observables,

    .. math::

       E_J[P]^\lambda
       + s\{Q[P+T]^\lambda - Q[P-T]^\lambda\},

    where ``s=-1`` for the DC-Forge source-Fock convention.  This helper
    returns the two corresponding reverse sources rather than treating the
    fused source/metric decomposition as a physical J/Fock decomposition.
    In particular, each returned channel has its own pair
    ``(source, metric)`` for

    .. math::

       source : dB - metric : dM.

    ``PBCsr_ejk_int3c2e_ip1`` historically accepts only one ``source``
    channel.  The companion two-channel CUDA ABI is therefore required to
    consume both results in one short-range traversal.  Keeping this
    construction separate from that ABI makes the algebra independently
    testable and keeps the ordinary fused route unchanged.

    Args:
        solved_three_center : cupy.ndarray
            Real Gamma ``W=M^{-1}B`` with shape ``(naux, nao, nao)``.
        reference_density : cupy.ndarray
            Real symmetric RHF density ``P`` in the primitive AO order.
        response_density : cupy.ndarray
            Real symmetric Z transition density ``T`` in the same order.

    Kwargs:
        response_sign : float
            Sign of the Z-Fock polarization term.  DC-Forge uses ``-1.``.
        auxiliary_block_size : int
            Maximum auxiliary block used while building each source.

    Returns:
        tuple
            ``(target, fock, metadata)``.  ``target`` and ``fock`` are
            dictionaries with ``source`` and ``metric`` device arrays.
    '''
    fock_source, fock_metric, fock_metadata = (
        build_rhf_primitive_adjoint(
            solved_three_center, reference_density, response_density,
            include_target_j=False, response_sign=response_sign,
            auxiliary_block_size=auxiliary_block_size))
    combined_source, combined_metric, combined_metadata = (
        build_rhf_primitive_adjoint(
            solved_three_center, reference_density, response_density,
            include_target_j=True, response_sign=response_sign,
            auxiliary_block_size=auxiliary_block_size))

    # The J source is recovered from the exact linear identity
    # ``combined = target + fock``.  This keeps `_build_source_block` as the
    # one authoritative definition of the RHF direct-adjoint coefficients.
    target_source = combined_source
    target_source -= fock_source
    target_metric = combined_metric
    target_metric -= fock_metric
    metadata = {
        'route': 'direct_real_gamma_primitive_adjoint_channels',
        'response_sign': float(response_sign),
        'target': {
            'persistent_source_bytes': int(target_source.nbytes),
            'metric_bytes': int(target_metric.nbytes),
            'construction': 'combined_minus_fock',
        },
        'fock': dict(fock_metadata),
        'combined': dict(combined_metadata),
    }
    return ({'source': target_source, 'metric': target_metric},
            {'source': fock_source, 'metric': fock_metric}, metadata)


def build_rhf_packed_primitive_adjoint(
        solved_three_center, layout, reference_density, response_density,
        include_target_j=False, response_sign=1.,
        auxiliary_block_size=32, overwrite_solved=True,
        profile_detail=False, storage='double_pass', host_cache='mapped',
        host_memory_fraction=DEFAULT_ADJOINT_HOST_MEMORY_FRACTION,
        host_minimum_reserve_bytes=(
            DEFAULT_ADJOINT_HOST_MINIMUM_RESERVE_BYTES),
        host_max_bytes=DEFAULT_ADJOINT_HOST_MAX_BYTES,
        host_cache_fallback=DEFAULT_ADJOINT_HOST_CACHE_FALLBACK):
    r'''Build the direct GDF reverse source in unique AO-pair storage.

    This is algebraically identical to :func:`build_rhf_primitive_adjoint`.
    Only a bounded auxiliary slice of ``W=M^{-1}B`` is expanded to a dense AO
    matrix while the persistent solved/source tensor remains
    ``(naux,npair_unique)``.  The packed derivative includes the two symmetric
    dense entries for every off-diagonal pair.

    At real Gamma, ``P``, ``T`` and each expanded ``W_P`` are real symmetric,
    so the two exchange products obey

    .. math::

       T W_P P = (P W_P T)^T.

    Production reuses that transpose and evaluates two rather than four
    dense GEMMs per source block.  ``storage='host_cache'`` additionally
    retains each first-pass packed source in host memory, avoiding the second
    algebra pass before ``solved_three_center`` is overwritten.  A host-memory
    plan is resolved before allocation; an insufficient plan falls back to
    ``double_pass`` only when ``host_cache_fallback=True``.

    ``profile_detail=True`` adds synchronization and CUDA events for
    measurement only.  It does not select a block size or change an
    arithmetic expression.
    '''
    total_start = _detail_profile_start(profile_detail)
    detail_profile = {}
    solved = cp.asarray(solved_three_center, dtype=cp.float64)
    reference = _as_real_square(reference_density, 'reference_density')
    response = _as_real_square(response_density, 'response_density')
    if solved.ndim != 2:
        raise ValueError('packed solved three-center tensor must be rank two')
    if reference.shape != response.shape:
        raise ValueError('reference and response densities must have one shape')
    nao = int(layout['nao'])
    if reference.shape != (nao, nao):
        raise ValueError('packed layout and density AO dimensions differ')
    row = cp.asarray(layout['row'], dtype=cp.int32)
    column = cp.asarray(layout['column'], dtype=cp.int32)
    weights = cp.asarray(layout['weights'], dtype=solved.dtype)
    if solved.shape[1] != len(row):
        raise ValueError('packed solved tensor and pair layout differ')
    response_sign = float(response_sign)
    if response_sign not in (-1., 1.):
        raise ValueError('response_sign must be +1 or -1')
    if not overwrite_solved:
        raise ValueError(
            'packed primitive adjoint requires overwrite_solved=True')
    if storage not in _ADJOINT_STORAGE_MODES:
        raise ValueError(
            f'storage must be one of {_ADJOINT_STORAGE_MODES}, got {storage!r}')
    if host_cache not in _ADJOINT_HOST_CACHE_MODES:
        raise ValueError(
            'host_cache must be one of '
            f'{_ADJOINT_HOST_CACHE_MODES}, got {host_cache!r}')
    naux = int(solved.shape[0])
    maximum = max(1, min(naux, int(auxiliary_block_size)))
    reference_pair = reference[row, column] * weights
    response_pair = response[row, column] * weights
    auxiliary_reference = solved @ reference_pair
    auxiliary_response = solved @ response_pair
    source = solved if overwrite_solved else cp.empty_like(solved)
    metric = cp.empty((naux, naux), dtype=solved.dtype)
    dense_solved = cp.zeros((maximum, nao, nao), dtype=solved.dtype)
    dense_source = cp.empty_like(dense_solved)
    temporary = cp.empty_like(dense_solved)
    exchange = cp.empty_like(dense_solved)
    packed_buffer = cp.empty((maximum, solved.shape[1]), dtype=solved.dtype)
    _detail_profile_stop(
        detail_profile, 'setup_and_projection_seconds', total_start)

    def make_block(p0, p1):
        width = p1 - p0
        w_block = dense_solved[:width]
        w_block[...] = 0.
        w_block[:, row, column] = solved[p0:p1]
        w_block[:, column, row] = solved[p0:p1]
        s_block = dense_source[:width]
        _build_source_block(
            s_block, w_block, reference, response,
            auxiliary_reference[p0:p1], auxiliary_response[p0:p1],
            include_target_j=include_target_j,
            response_sign=response_sign,
            temporary=temporary[:width], exchange=exchange[:width])
        packed = packed_buffer[:width]
        cp.multiply(s_block[:, row, column], weights[None], out=packed)
        return packed

    # Metric needs the original W.  Accumulate it before any in-place source
    # overwrite.  The default provider policy caches the exact first-pass
    # source in host memory; the low-memory oracle repeats only the bounded
    # local algebra in a second pass.
    host_plan = plan_adjoint_host_cache(
        solved.nbytes, memory_fraction=host_memory_fraction,
        minimum_reserve_bytes=host_minimum_reserve_bytes,
        maximum_bytes=host_max_bytes, requested_storage=storage,
        fallback=host_cache_fallback)
    resolved_storage = host_plan['resolved_storage']
    host_source = None
    if resolved_storage == 'host_cache':
        try:
            if host_cache == 'mapped':
                host_source = empty_mapped(solved.shape, dtype=solved.dtype)
            else:
                host_source = np.empty(solved.shape, dtype=solved.dtype)
        except (MemoryError, OSError, cp.cuda.memory.OutOfMemoryError,
                cp.cuda.runtime.CUDARuntimeError) as error:
            if not host_cache_fallback:
                raise
            resolved_storage = 'double_pass'
            host_plan['resolved_storage'] = resolved_storage
            host_plan['fallback_reason'] = (
                f'{type(error).__name__}: {error}')

    metric_pass_start = _detail_profile_start(profile_detail)
    metric_pass_events = []
    host_write_seconds = 0.
    for p0 in range(0, naux, maximum):
        p1 = min(naux, p0 + maximum)
        if profile_detail:
            source_start = cp.cuda.Event()
            source_end = cp.cuda.Event()
            contraction_end = cp.cuda.Event()
            source_start.record()
        packed = make_block(p0, p1)
        if profile_detail:
            source_end.record()
        contract('pk,qk->pq', packed, solved, out=metric[p0:p1])
        if profile_detail:
            contraction_end.record()
        if host_source is not None:
            transfer_start = _detail_profile_start(profile_detail)
            packed.get(out=host_source[p0:p1], blocking=True)
            if transfer_start is not None:
                host_write_seconds += time.perf_counter() - transfer_start
        if profile_detail:
            metric_pass_events.append(
                (source_start, source_end, contraction_end))
    _detail_profile_stop(
        detail_profile, 'metric_source_pass_seconds', metric_pass_start)
    if profile_detail:
        source_gpu_seconds = sum(
            cp.cuda.get_elapsed_time(start, middle)
            for start, middle, _end in metric_pass_events) * 1.e-3
        contraction_gpu_seconds = sum(
            cp.cuda.get_elapsed_time(middle, end)
            for _start, middle, end in metric_pass_events) * 1.e-3
        detail_profile['metric_pass_source_algebra_gpu_seconds'] = (
            source_gpu_seconds)
        detail_profile['metric_pass_contraction_gpu_seconds'] = (
            contraction_gpu_seconds)
        detail_profile['metric_pass_host_and_event_overhead_seconds'] = max(
            0., detail_profile['metric_source_pass_seconds']
            - source_gpu_seconds - contraction_gpu_seconds)

    overwrite_start = _detail_profile_start(profile_detail)
    host_read_seconds = 0.
    if host_source is None:
        for p0 in range(0, naux, maximum):
            p1 = min(naux, p0 + maximum)
            source[p0:p1] = make_block(p0, p1)
        overwrite_name = 'source_overwrite_pass_seconds'
    else:
        for p0 in range(0, naux, maximum):
            p1 = min(naux, p0 + maximum)
            transfer_start = _detail_profile_start(profile_detail)
            source[p0:p1].set(host_source[p0:p1])
            if transfer_start is not None:
                cp.cuda.Stream.null.synchronize()
                host_read_seconds += time.perf_counter() - transfer_start
        # ``empty_mapped`` may permit asynchronous H2D copies.  Keep its
        # owner alive until every block has reached device storage, including
        # ordinary non-profiled gradients where no timing synchronization is
        # otherwise requested.
        cp.cuda.Stream.null.synchronize()
        overwrite_name = 'source_host_read_seconds'
    _detail_profile_stop(detail_profile, overwrite_name, overwrite_start)
    if profile_detail:
        cp.cuda.Stream.null.synchronize()
        detail_profile['host_cache_write_seconds'] = host_write_seconds
        detail_profile['host_cache_read_seconds'] = host_read_seconds
        detail_profile['total_seconds'] = time.perf_counter() - total_start
        accounted = sum(
            detail_profile[name] for name in (
                'setup_and_projection_seconds', 'metric_source_pass_seconds',
                overwrite_name))
        detail_profile['unattributed_seconds'] = max(
            0., detail_profile['total_seconds'] - accounted)
    dense_bytes = naux * nao * nao * solved.dtype.itemsize
    metadata = {
        'route': 'direct_real_gamma_packed_primitive_adjoint',
        'include_target_j': bool(include_target_j),
        'response_sign': response_sign,
        'raw_pair_count': int(layout['raw_pair_count']),
        'unique_pair_count': int(layout['unique_pair_count']),
        'persistent_source_bytes': int(source.nbytes),
        'dense_source_bytes_avoided': int(dense_bytes - source.nbytes),
        'source_reuses_solved_storage': bool(overwrite_solved),
        'source_algebra': 'symmetric_transpose',
        'source_algebra_passes': 1 if host_source is not None else 2,
        'storage_requested': storage,
        'storage': resolved_storage,
        'host_cache': host_cache if host_source is not None else None,
        'host_cache_bytes': (
            0 if host_source is None else int(host_source.nbytes)),
        'host_transfer_bytes': (
            0 if host_source is None else 2 * int(host_source.nbytes)),
        'host_cache_plan': host_plan,
        'metric_bytes': int(metric.nbytes),
        'auxiliary_block_size': maximum,
        'auxiliary_block_count': int((naux + maximum - 1) // maximum),
        'profile_detail': bool(profile_detail),
        'profile': detail_profile if profile_detail else None,
    }
    return source, metric, metadata


def _detail_profile_start(enabled):
    '''Synchronize and start one opt-in packed-adjoint wall timer.'''
    if not enabled:
        return None
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


def _detail_profile_stop(profile, name, start):
    '''Finish one opt-in packed-adjoint wall timer without changing arrays.'''
    if start is None:
        return
    cp.cuda.Stream.null.synchronize()
    profile[name] = time.perf_counter() - start


def plan_adjoint_host_cache(
        source_bytes, memory_fraction=DEFAULT_ADJOINT_HOST_MEMORY_FRACTION,
        minimum_reserve_bytes=DEFAULT_ADJOINT_HOST_MINIMUM_RESERVE_BYTES,
        maximum_bytes=DEFAULT_ADJOINT_HOST_MAX_BYTES,
        requested_storage=DEFAULT_ADJOINT_STORAGE,
        fallback=DEFAULT_ADJOINT_HOST_CACHE_FALLBACK,
        available_bytes=None):
    '''Resolve whether one packed source fits the host-cache policy.

    The query uses currently available host memory, so an already resident
    mapped CDERI is included naturally in the system's consumed memory.  The
    source must fit both the fractional budget and the explicit reserve.  No
    scientific tensor, auxiliary basis or GDF cutoff changes when the plan
    falls back to the exact two-pass execution.

    Args:
        source_bytes : int
            Bytes required by one packed primitive source.

    Kwargs:
        memory_fraction : float
            Maximum fraction of currently available host memory assigned to
            the cache.
        minimum_reserve_bytes : int
            Host memory retained after allocation for Python, CDERI and later
            derivative stages.
        maximum_bytes : int or None
            Optional absolute cap for the cache.
        requested_storage : {'host_cache', 'double_pass'}
            Requested execution route.
        fallback : bool
            Use ``double_pass`` when the cache does not fit.  False raises a
            pre-allocation ``MemoryError`` instead.
        available_bytes : int or None
            Test/diagnostic override for the live host-memory query.

    Returns:
        dict
            JSON-compatible requested/resolved plan and byte budgets.
    '''
    source_bytes = int(source_bytes)
    if source_bytes < 0:
        raise ValueError('source_bytes must be non-negative')
    if requested_storage not in _ADJOINT_STORAGE_MODES:
        raise ValueError(
            'requested_storage must be one of '
            f'{_ADJOINT_STORAGE_MODES}, got {requested_storage!r}')
    memory_fraction = float(memory_fraction)
    if not 0. < memory_fraction <= 1.:
        raise ValueError('memory_fraction must be in (0, 1]')
    minimum_reserve_bytes = int(minimum_reserve_bytes)
    if minimum_reserve_bytes < 0:
        raise ValueError('minimum_reserve_bytes must be non-negative')
    if maximum_bytes is not None:
        maximum_bytes = int(maximum_bytes)
        if maximum_bytes < 0:
            raise ValueError('maximum_bytes must be non-negative or None')
    if available_bytes is None:
        available_bytes = _available_host_memory()
    available_bytes = int(available_bytes)
    fractional_budget = int(available_bytes * memory_fraction)
    reserve_budget = max(0, available_bytes - minimum_reserve_bytes)
    budget = min(fractional_budget, reserve_budget)
    if maximum_bytes is not None:
        budget = min(budget, maximum_bytes)
    fits = source_bytes <= budget
    resolved = requested_storage
    fallback_reason = None
    if requested_storage == 'host_cache' and not fits:
        fallback_reason = (
            f'packed source requires {source_bytes} bytes but the host-cache '
            f'budget is {budget} bytes')
        if not fallback:
            raise MemoryError(fallback_reason)
        resolved = 'double_pass'
    return {
        'requested_storage': requested_storage,
        'resolved_storage': resolved,
        'source_bytes': source_bytes,
        'available_host_bytes': available_bytes,
        'memory_fraction': memory_fraction,
        'minimum_reserve_bytes': minimum_reserve_bytes,
        'maximum_bytes': maximum_bytes,
        'budget_bytes': budget,
        'fits': bool(fits),
        'fallback_enabled': bool(fallback),
        'fallback_reason': fallback_reason,
    }


def _available_host_memory():
    '''Return currently available host bytes without adding a dependency.'''
    try:
        with open('/proc/meminfo', encoding='ascii') as stream:
            for line in stream:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    pages = os.sysconf('SC_AVPHYS_PAGES')
    page_size = os.sysconf('SC_PAGE_SIZE')
    return int(pages) * int(page_size)


def _build_source_block(source, solved, reference, response,
                        auxiliary_reference, auxiliary_response,
                        include_target_j, response_sign,
                        temporary=None, exchange=None):
    '''Fill one bounded auxiliary block of the exact cross-density source.'''
    if temporary is None:
        temporary = cp.empty_like(source)
    if exchange is None:
        exchange = cp.empty_like(source)

    cp.multiply(auxiliary_response[:, None, None], reference[None], out=source)
    cp.multiply(auxiliary_reference[:, None, None], response[None], out=temporary)
    source += temporary
    source *= 2.

    contract('ij,pjk->pik', reference, solved, out=temporary)
    contract('pik,kl->pil', temporary, response, out=exchange)
    source -= exchange
    # At real Gamma all three matrices are symmetric.  Hence
    # ``T W P = (P W T)^T`` and the second two-GEMM chain is exactly the
    # transpose already present in ``exchange``.  `_as_real_square` and the
    # packed construction enforce this domain before reaching this helper.
    source -= exchange.transpose(0, 2, 1)
    source *= response_sign

    if include_target_j:
        cp.multiply(auxiliary_reference[:, None, None], reference[None],
                    out=temporary)
        source += temporary


def _as_real_matrix_stack(value, name):
    '''Validate a real ``(naux, nao, nao)`` primitive tensor.'''
    array = cp.asarray(value)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(f'{name} must have shape (naux, nao, nao)')
    if cp.iscomplexobj(array):
        imaginary = float(cp.max(cp.abs(array.imag)).get())
        if imaginary > 1e-10:
            raise NotImplementedError(f'{name} must be real at Gamma')
        array = array.real
    return cp.asarray(array, dtype=cp.float64)


def _as_real_square(value, name):
    '''Validate a real square Gamma AO density without symmetrizing it.'''
    array = cp.asarray(value)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f'{name} must be a square Gamma AO matrix')
    if cp.iscomplexobj(array):
        imaginary = float(cp.max(cp.abs(array.imag)).get())
        if imaginary > 1e-10:
            raise NotImplementedError(f'{name} must be real at Gamma')
        array = array.real
    asymmetry = float(cp.max(cp.abs(array - array.T)).get())
    if asymmetry > 1e-9:
        raise ValueError(f'{name} must be symmetric at Gamma')
    return cp.asarray(array, dtype=cp.float64)


# Compatibility aliases for the pre-R2SCAN internal spelling.  The reverse
# adjoint depends on the source RHF density and Z transition density, not on
# whether the target semilocal functional is PBE or R2SCAN.
build_rhf_pbe_primitive_adjoint = build_rhf_primitive_adjoint
build_rhf_pbe_primitive_adjoint_channels = build_rhf_primitive_adjoint_channels
