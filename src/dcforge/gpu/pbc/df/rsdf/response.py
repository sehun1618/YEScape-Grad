'''Term-resolved native Gamma RSDF response for DFA@RHF gradients.

This is the GPU counterpart of the CPU contracted Gaussian response.  It
uses the provider's compressed central factor only to construct device
reverse adjoints, then discards that dense response representation before
contracting nuclear source and metric derivatives.  The resulting four
terms retain their mathematical ownership throughout the gradient ledger.
'''

import time

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import get_avail_mem

from dcforge.gpu.pbc.df.gaussian.ewald import cross_derivative
from dcforge.gpu.pbc.df.rsdf.adjoint import (
    make_cderi_observable_adjoints,
    make_packed_cderi_observable_adjoint,
    make_packed_cderi_observable_adjoints,
    push_cderi_adjoints,
    push_packed_cderi_adjoint,
    push_packed_cderi_adjoints,
)
from dcforge.gpu.pbc.df.rsdf.metric import (
    build_metric_contraction_context,
    contract_metric_derivative,
)
from dcforge.gpu.pbc.df.rsdf.source import (
    build_source_optimizer,
    contract_packed_source_derivative,
    contract_source_derivative,
)


def make_gradient_response(with_df, dm0, dz, exxdiv='ewald',
                           with_target_j=True):
    r'''Return the four Gaussian RSDF scalar response terms.

    Args:
        with_df : dcforge.gpu.pbc.df.rsdf.RSDF
            Built Gamma RSDF provider.
        dm0 : cupy.ndarray
            Spin-summed source-RHF AO density ``P``.
        dz : cupy.ndarray
            Hermitian Z-transition AO density ``D_Z``.

    Kwargs:
        exxdiv : {'ewald', None}
            Source-HF exchange-divergence convention.  It is kept separate
            from RSDF source/metric response because the Ewald correction is
            an overlap derivative, not a three- or two-center DF term.
        with_target_j : bool
            Include the RSDF target-Hartree observable.  Set this false when
            multigrid owns target J; the source-HF Fock response remains
            mandatory because it defines the Z-vector Lagrangian.

    Returns:
        tuple
            ``(terms, metadata)``.  ``terms`` is a mapping containing
            ``de_j_source``, ``de_j_metric``, ``de_fock_source``,
            ``de_fock_metric`` and ``de_fock_ewald``, each of shape
            ``(natm,3)`` in Ha/Bohr.

    Notes:
        The first observable is the target Hartree derivative.  The second
        is the source-Fock Z bilinear with the DC-DFT sign already included.
        Source and metric remain independently parameterized by
        ``(omega, mesh_compact)`` and ``(omega_j2c, mesh_j2c)``.
    '''
    _check_provider(with_df, dm0, dz)
    mode = str(with_df.response_mode).lower()
    if mode == 'dense':
        return _make_dense_gradient_response(
            with_df, dm0, dz, exxdiv, with_target_j=with_target_j)
    if mode == 'auto':
        mode, memory_plan = _resolve_auto_response_mode(
            with_df, with_target_j)
        try:
            terms, metadata = (
                _make_fused_packed_gradient_response(
                    with_df, dm0, dz, exxdiv,
                    with_target_j=with_target_j)
                if mode == 'fused_packed' else
                _make_sequential_packed_gradient_response(
                    with_df, dm0, dz, exxdiv,
                    with_target_j=with_target_j))
        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as error:
            if mode != 'fused_packed':
                raise
            cp.get_default_memory_pool().free_all_blocks()
            terms, metadata = _make_sequential_packed_gradient_response(
                with_df, dm0, dz, exxdiv,
                with_target_j=with_target_j)
            memory_plan['resolved_mode'] = 'sequential_packed'
            memory_plan['fallback_reason'] = (
                f'fused allocation failed safely: '
                f'{type(error).__name__}: {error}')
        memory_plan['resolved_route_workspace_bytes'] = int(
            metadata['logical_persistent_workspace_bytes'])
        metadata['auto_memory_plan'] = memory_plan
        return terms, metadata
    if mode == 'fused_packed':
        return _make_fused_packed_gradient_response(
            with_df, dm0, dz, exxdiv, with_target_j=with_target_j)
    if mode != 'sequential_packed':
        raise ValueError(
            "RSDF response_mode must be 'auto', 'fused_packed', "
            "'sequential_packed', or 'dense'")
    return _make_sequential_packed_gradient_response(
        with_df, dm0, dz, exxdiv, with_target_j=with_target_j)


def _make_sequential_packed_gradient_response(
        with_df, dm0, dz, exxdiv, with_target_j):
    '''Evaluate packed target/Fock observables one lifetime at a time.'''
    start = time.perf_counter()
    cderi, layout = with_df.packed_cderi(
        auxiliary_block_size=with_df.response_auxiliary_block_size)
    factor = with_df.central_metric_factor()
    central_bytes = int(cderi.nbytes)
    factor_bytes = int(factor.nbytes)
    dense_cderi_bytes = int(
        cderi.shape[0] * int(layout['nao']) ** 2 * cderi.dtype.itemsize)
    dense_cderi_bytes_avoided = dense_cderi_bytes - central_bytes
    observables = []
    if with_target_j:
        observables.append('target_j')
    observables.append('source_fock')
    values = {}
    observable_metadata = {}
    maximum_observable_workspace_bytes = 0
    memory_releases = []
    component_profile = _new_component_profile(with_df)
    component_started = _component_start(component_profile)
    source_optimizer = build_source_optimizer(with_df)
    _component_stop(
        component_profile, 'source_optimizer_seconds', component_started)
    metric_context = None
    for observable in observables:
        observable_start = time.perf_counter()
        component_started = _component_start(component_profile)
        adjoint = make_packed_cderi_observable_adjoint(
            cderi, layout, dm0, dz, observable,
            auxiliary_block_size=with_df.response_auxiliary_block_size)
        _component_stop(
            component_profile, 'observable_adjoint_seconds',
            component_started)
        adjoint_bytes = int(adjoint.nbytes)
        component_started = _component_start(component_profile)
        source, metric, push_metadata = push_packed_cderi_adjoint(
            factor, cderi, adjoint, layout,
            pair_block_size=with_df.response_pair_block_size,
            overwrite_adjoint=True)
        _component_stop(
            component_profile, 'reverse_solve_seconds', component_started)
        dense_source_bytes_avoided = int(push_metadata[
            'dense_source_bytes'])
        del adjoint
        memory_releases.append(_release_unused_pool_blocks(
            with_df, f'{observable}_reverse_to_source'))
        component_started = _component_start(component_profile)
        source_value, source_metadata = contract_packed_source_derivative(
            with_df, source[None], layout, optimizer=source_optimizer)
        _component_stop(
            component_profile, 'source_contraction_seconds',
            component_started)
        del source
        memory_releases.append(_release_unused_pool_blocks(
            with_df, f'{observable}_source_to_metric'))
        metric_bytes = int(metric.nbytes)
        if metric_context is None:
            component_started = _component_start(component_profile)
            metric_context = build_metric_contraction_context(with_df)
            _component_stop(
                component_profile, 'metric_context_seconds',
                component_started)
        component_started = _component_start(component_profile)
        metric_value = contract_metric_derivative(
            with_df, metric[None], context=metric_context)
        _component_stop(
            component_profile, 'metric_contraction_seconds',
            component_started)
        del metric
        memory_releases.append(_release_unused_pool_blocks(
            with_df, f'{observable}_metric_complete'))
        values[observable] = (source_value[0], metric_value[0])
        observable_metadata[observable] = {
            'adjoint_bytes': adjoint_bytes,
            'dense_source_bytes': 0,
            'dense_source_bytes_avoided': dense_source_bytes_avoided,
            'metric_bytes': metric_bytes,
            'push': push_metadata,
            'source': source_metadata,
            'seconds': time.perf_counter() - observable_start,
        }
        maximum_observable_workspace_bytes = max(
            maximum_observable_workspace_bytes,
            adjoint_bytes + metric_bytes)
    del cderi, factor
    memory_releases.append(_release_unused_pool_blocks(
        with_df, 'packed_observables_to_ewald'))
    component_started = _component_start(component_profile)
    ewald = cross_derivative(
        with_df.cell, cp.asarray(dm0), cp.asarray(dz), exxdiv)
    _component_stop(
        component_profile, 'ewald_seconds', component_started)
    zeros = cp.zeros_like(ewald)
    target_source, target_metric = values.get('target_j', (zeros, zeros))
    fock_source, fock_metric = values['source_fock']
    terms = {
        'de_j_source': target_source,
        'de_j_metric': target_metric,
        'de_fock_source': fock_source,
        'de_fock_metric': fock_metric,
        'de_fock_ewald': ewald,
    }
    metadata = {
        'response_mode': 'sequential_packed_reverse_adjoint',
        'with_target_j': bool(with_target_j),
        'observable_order': observables,
        'central_cderi_storage': 'ephemeral_device_unique_gamma_pairs',
        'derivative_tensor_storage': 'none',
        'raw_pair_count': int(layout['raw_pair_count']),
        'unique_pair_count': int(layout['unique_pair_count']),
        'packed_cderi_bytes': central_bytes,
        'central_factor_bytes': factor_bytes,
        'dense_cderi_bytes_avoided': dense_cderi_bytes_avoided,
        'observable_auxiliary_block_size': int(
            with_df.response_auxiliary_block_size),
        'pair_solve_block_size': int(with_df.response_pair_block_size),
        'observables': observable_metadata,
        # Exact logical bytes of arrays owned concurrently by this route.
        # CuPy allocator fragmentation and third-party kernel workspaces are
        # recorded separately by the process-level peak telemetry.
        'logical_persistent_workspace_bytes': int(
            central_bytes + factor_bytes
            + maximum_observable_workspace_bytes),
        'profile': _finish_component_profile(
            component_profile, observable_metadata),
        # Preserve the established location of reciprocal traversal metadata.
        'source': observable_metadata['source_fock']['source'],
        'shared_context': {
            'source_optimizer_reused': True,
            'metric_context_reused': len(observables) > 1,
            'metric_context_device_bytes': int(
                metric_context.device_resident_bytes),
            'metric_context_host_seconds': float(
                metric_context.host_seconds),
            'metric_context_host_transfer_bytes': int(
                metric_context.host_transfer_bytes),
        },
        'memory_releases': memory_releases,
        'metric': {
            'owner': (
                'PySCF exact screened ip1/ip2 primitive plus GPU4PySCF '
                'reciprocal contraction'),
            'source_omega': float(with_df.omega),
            'metric_omega': float(with_df.omega_j2c),
        },
        'ewald_exchange': str(exxdiv).lower() == 'ewald',
        'total_seconds': time.perf_counter() - start,
    }
    return terms, metadata


def _make_fused_packed_gradient_response(
        with_df, dm0, dz, exxdiv, with_target_j):
    '''Evaluate selected packed observables through one stacked traversal.'''
    start = time.perf_counter()
    observables = (
        ('target_j', 'source_fock') if with_target_j
        else ('source_fock',))
    cderi, layout = with_df.packed_cderi(
        auxiliary_block_size=with_df.response_auxiliary_block_size)
    factor = with_df.central_metric_factor()
    central_bytes = int(cderi.nbytes)
    factor_bytes = int(factor.nbytes)
    dense_cderi_bytes = int(
        cderi.shape[0] * int(layout['nao']) ** 2 * cderi.dtype.itemsize)

    component_profile = _new_component_profile(with_df)
    adjoint_start = _component_start(component_profile)
    adjoints = make_packed_cderi_observable_adjoints(
        cderi, layout, dm0, dz, observables,
        auxiliary_block_size=with_df.response_auxiliary_block_size)
    _component_stop(
        component_profile, 'observable_adjoint_seconds', adjoint_start)
    adjoint_bytes = int(adjoints.nbytes)

    reverse_start = _component_start(component_profile)
    sources, metrics, push_metadata = push_packed_cderi_adjoints(
        factor, cderi, adjoints, layout,
        pair_block_size=with_df.response_pair_block_size,
        overwrite_adjoint=True)
    _component_stop(
        component_profile, 'reverse_solve_seconds', reverse_start)
    packed_source_bytes = int(sources.nbytes)
    metric_bytes = int(metrics.nbytes)
    del cderi, factor, adjoints
    memory_releases = [_release_unused_pool_blocks(
        with_df, 'fused_reverse_to_source')]

    component_started = _component_start(component_profile)
    source_value, source_metadata = contract_packed_source_derivative(
        with_df, sources, layout)
    _component_stop(
        component_profile, 'source_contraction_seconds', component_started)
    del sources
    memory_releases.append(_release_unused_pool_blocks(
        with_df, 'fused_source_to_metric'))

    component_started = _component_start(component_profile)
    metric_context = build_metric_contraction_context(with_df)
    _component_stop(
        component_profile, 'metric_context_seconds', component_started)
    component_started = _component_start(component_profile)
    metric_value = contract_metric_derivative(
        with_df, metrics, context=metric_context)
    _component_stop(
        component_profile, 'metric_contraction_seconds', component_started)
    del metrics
    memory_releases.append(_release_unused_pool_blocks(
        with_df, 'fused_metric_to_ewald'))
    component_started = _component_start(component_profile)
    ewald = cross_derivative(
        with_df.cell, cp.asarray(dm0), cp.asarray(dz), exxdiv)
    _component_stop(
        component_profile, 'ewald_seconds', component_started)
    zeros = cp.zeros_like(ewald)
    index = {name: position for position, name in enumerate(observables)}
    target_source = (
        source_value[index['target_j']] if with_target_j else zeros)
    target_metric = (
        metric_value[index['target_j']] if with_target_j else zeros)
    fock_source = source_value[index['source_fock']]
    fock_metric = metric_value[index['source_fock']]
    terms = {
        'de_j_source': target_source,
        'de_j_metric': target_metric,
        'de_fock_source': fock_source,
        'de_fock_metric': fock_metric,
        'de_fock_ewald': ewald,
    }
    metadata = {
        'response_mode': 'fused_packed_reverse_adjoint',
        'with_target_j': bool(with_target_j),
        'observable_order': list(observables),
        'central_cderi_storage': 'ephemeral_device_unique_gamma_pairs',
        'derivative_tensor_storage': 'none',
        'raw_pair_count': int(layout['raw_pair_count']),
        'unique_pair_count': int(layout['unique_pair_count']),
        'packed_cderi_bytes': central_bytes,
        'central_factor_bytes': factor_bytes,
        'dense_cderi_bytes_avoided': dense_cderi_bytes - central_bytes,
        'observable_adjoint_bytes': adjoint_bytes,
        'packed_source_bytes': packed_source_bytes,
        'dense_source_bytes': 0,
        'dense_source_bytes_avoided': int(push_metadata[
            'dense_source_bytes']),
        'metric_adjoint_bytes': metric_bytes,
        # ``sources`` overwrites ``adjoints``.  Count that allocation once,
        # together with the central factor and stacked metric adjoint.
        'logical_persistent_workspace_bytes': int(
            central_bytes + factor_bytes + packed_source_bytes + metric_bytes),
        'observable_auxiliary_block_size': int(
            with_df.response_auxiliary_block_size),
        'pair_solve_block_size': int(with_df.response_pair_block_size),
        'push': push_metadata,
        'source': source_metadata,
        'shared_context': {
            'source_optimizer_count': 1,
            'metric_context_count': 1,
            'metric_context_device_bytes': int(
                metric_context.device_resident_bytes),
            'metric_context_host_seconds': float(
                metric_context.host_seconds),
            'metric_context_host_transfer_bytes': int(
                metric_context.host_transfer_bytes),
        },
        'profile': _finish_component_profile(
            component_profile, {'fused': {'source': source_metadata}}),
        'memory_releases': memory_releases,
        'metric': {
            'owner': (
                'PySCF exact screened ip1/ip2 primitive plus GPU4PySCF '
                'reciprocal contraction'),
            'source_omega': float(with_df.omega),
            'metric_omega': float(with_df.omega_j2c),
        },
        'ewald_exchange': str(exxdiv).lower() == 'ewald',
        'total_seconds': time.perf_counter() - start,
    }
    return terms, metadata


def _resolve_auto_response_mode(with_df, with_target_j):
    '''Choose fused or sequential packed response from a bounded preflight.'''
    report = with_df.cderi_memory_report(
        response_observables=2 if with_target_j else 1)
    nobs = 2 if with_target_j else 1
    naux = int(with_df.get_naoaux())
    packed = int(report['packed_response_one_observable_bytes'])
    itemsize = np.dtype(np.float64).itemsize
    # Fused direct-packed response retains one packed adjoint/source per
    # observable, one metric adjoint, and bounded sorted-pair projections.
    # The latter is conservatively estimated by one additional packed tensor;
    # no persistent ``naux*nao**2`` source belongs to this route.
    predicted = nobs * (2 * packed + naux * naux * itemsize)
    available = int(get_avail_mem(exclude_memory_pool=True))
    fraction = float(getattr(
        with_df, 'response_memory_fraction', .5))
    minimum_reserve = int(getattr(
        with_df, 'response_minimum_reserve_bytes', 4 * 1024**3))
    fused_maximum = int(getattr(
        with_df, 'response_fused_max_bytes', 4 * 1024**3))
    if fused_maximum < 0:
        raise ValueError('response_fused_max_bytes must be nonnegative')
    reserve = max(minimum_reserve, int((1. - fraction) * available))
    budget = max(0, available - reserve)
    memory_fit = predicted <= budget
    promotion_fit = predicted <= fused_maximum
    mode = (
        'fused_packed' if memory_fit and promotion_fit
        else 'sequential_packed')
    if not memory_fit:
        fallback_reason = (
            f'predicted fused response {predicted} bytes exceeds bounded '
            f'budget {budget} bytes')
    elif not promotion_fit:
        fallback_reason = (
            f'predicted fused response {predicted} bytes exceeds '
            f'production promotion ceiling {fused_maximum} bytes')
    else:
        fallback_reason = None
    return mode, {
        'requested_mode': 'auto',
        'resolved_mode': mode,
        'available_device_bytes': available,
        'memory_fraction': fraction,
        'minimum_reserve_bytes': minimum_reserve,
        'resolved_reserve_bytes': reserve,
        'budget_bytes': budget,
        'predicted_fused_bytes': predicted,
        'fused_promotion_max_bytes': fused_maximum,
        'memory_fit': memory_fit,
        'promotion_fit': promotion_fit,
        'fallback_reason': fallback_reason,
    }


def _make_dense_gradient_response(
        with_df, dm0, dz, exxdiv, with_target_j=True):
    '''Retain the original dense two-observable route as a numerical oracle.'''
    start = time.perf_counter()
    cderi = with_df.canonical_cderi()
    factor = with_df.central_metric_factor()
    adjoints = make_cderi_observable_adjoints(cderi, dm0, dz)
    if not with_target_j:
        adjoints = adjoints[1:]
    observable_auxiliary_block_size = min(32, int(cderi.shape[0]))
    observable_adjoint_bytes = int(adjoints.nbytes)
    source_adjoint, metric_adjoint = push_cderi_adjoints(
        factor, cderi, adjoints)
    source_adjoint_bytes = int(source_adjoint.nbytes)
    metric_adjoint_bytes = int(metric_adjoint.nbytes)
    del cderi, factor, adjoints
    memory_releases = [_release_unused_pool_blocks(
        with_df, 'reverse_cholesky_to_source')]
    reverse_seconds = time.perf_counter() - start
    source_value, source_metadata = contract_source_derivative(
        with_df, source_adjoint)
    del source_adjoint
    memory_releases.append(_release_unused_pool_blocks(
        with_df, 'source_to_metric'))
    metric_value = contract_metric_derivative(with_df, metric_adjoint)
    del metric_adjoint
    memory_releases.append(_release_unused_pool_blocks(
        with_df, 'metric_to_ewald'))
    ewald = cross_derivative(
        with_df.cell, cp.asarray(dm0), cp.asarray(dz), exxdiv)
    if with_target_j:
        target_source, target_metric = source_value[0], metric_value[0]
        fock_source, fock_metric = source_value[1], metric_value[1]
    else:
        target_source = target_metric = cp.zeros_like(ewald)
        fock_source, fock_metric = source_value[0], metric_value[0]
    terms = {
        'de_j_source': target_source,
        'de_j_metric': target_metric,
        'de_fock_source': fock_source,
        'de_fock_metric': fock_metric,
        'de_fock_ewald': ewald,
    }
    metadata = {
        'response_mode': 'dense_reverse_adjoint_native_contraction',
        'with_target_j': bool(with_target_j),
        'central_cderi_storage': 'ephemeral_device_dense_gamma',
        'derivative_tensor_storage': 'none',
        'observable_auxiliary_block_size': observable_auxiliary_block_size,
        'observable_adjoint_bytes': observable_adjoint_bytes,
        'source_adjoint_bytes': source_adjoint_bytes,
        'metric_adjoint_bytes': metric_adjoint_bytes,
        'reverse_cholesky_seconds': reverse_seconds,
        'source': source_metadata,
        'memory_releases': memory_releases,
        'metric': {
            'owner': (
                'PySCF exact screened ip1/ip2 primitive plus GPU4PySCF '
                'reciprocal contraction'),
            'source_omega': float(with_df.omega),
            'metric_omega': float(with_df.omega_j2c),
        },
        'ewald_exchange': str(exxdiv).lower() == 'ewald',
    }
    return terms, metadata


def _release_unused_pool_blocks(with_df, boundary):
    '''Release dead CuPy allocations at one RSDF response stage boundary.

    Args:
        with_df : dcforge.gpu.pbc.df.rsdf.RSDF
            Provider that owns the response memory policy.
        boundary : str
            Human-readable transition between two mathematical stages.

    Returns:
        dict
            Pool usage before and after the optional release, in bytes.

    Notes:
        ``MemoryPool.free_all_blocks`` does not invalidate live arrays.  The
        preceding ``del`` statements establish which large intermediates no
        longer belong to a later term; synchronization then makes their
        temporary allocations eligible for return to CUDA.  This changes
        allocation lifetime only, never the RSDF contraction or its order.
    '''
    cp.cuda.Stream.null.synchronize()
    pool = cp.get_default_memory_pool()
    before_used = int(pool.used_bytes())
    before_total = int(pool.total_bytes())
    enabled = bool(with_df.response_release_cached_blocks)
    if enabled:
        pool.free_all_blocks()
    return {
        'boundary': boundary,
        'enabled': enabled,
        'used_bytes': before_used,
        'cached_bytes_before': before_total - before_used,
        'cached_bytes_after': int(pool.total_bytes()) - int(pool.used_bytes()),
    }


def _new_component_profile(with_df):
    '''Return one response timing ledger without synchronizing production.'''
    return {
        'enabled': bool(getattr(
            with_df, 'response_profile_components', False)),
        'cuda_synchronized': bool(getattr(
            with_df, 'response_profile_components', False)),
        'observable_adjoint_seconds': 0.,
        'reverse_solve_seconds': 0.,
        'dense_source_expansion_seconds': 0.,
        'source_optimizer_seconds': 0.,
        'source_contraction_seconds': 0.,
        'source_short_range_seconds': 0.,
        'source_reciprocal_seconds': 0.,
        'metric_context_seconds': 0.,
        'metric_contraction_seconds': 0.,
        'ewald_seconds': 0.,
    }


def _component_start(profile):
    if profile['enabled']:
        cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


def _component_stop(profile, name, started):
    if profile['enabled']:
        cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - started
    profile[name] += seconds
    return seconds


def _finish_component_profile(profile, observables):
    '''Attach CUDA-event source subterms while retaining exclusive owners.'''
    sources = [value.get('source') for value in observables.values()]
    sources = [value for value in sources if isinstance(value, dict)]
    profile['source_short_range_seconds'] = float(sum(
        value.get('short_range_seconds', 0.) for value in sources))
    profile['source_reciprocal_seconds'] = float(sum(
        value.get('reciprocal_seconds', 0.) for value in sources))
    return profile


def _check_provider(with_df, dm0, dz):
    '''Validate the real-Gamma RSDF response representation before kernels.'''
    if getattr(with_df, 'backend_name', None) != 'rsdf':
        raise TypeError('RSDF response requires a dcforge GPU RSDF provider')
    if with_df.cell.dimension != 3:
        raise NotImplementedError('GPU RSDF response requires a 3D cell')
    kpts = np.asarray(with_df.kpts, dtype=float).reshape(-1, 3)
    if len(kpts) != 1 or np.max(np.abs(kpts)) > 1.0e-12:
        raise NotImplementedError('GPU RSDF response requires Gamma point')
    for name, value in (('dm0', dm0), ('dz', dz)):
        value = cp.asarray(value)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError(f'{name} must be a square Gamma AO density')
        imaginary = float(cp.max(cp.abs(value.imag)).get())
        asymmetry = float(cp.max(cp.abs(value - value.conj().T)).get())
        if imaginary > 1.0e-10:
            raise NotImplementedError(f'GPU RSDF response requires real {name}')
        if asymmetry > 1.0e-9:
            raise ValueError(f'{name} must be Hermitian')
