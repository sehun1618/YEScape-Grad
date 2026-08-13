'''Opt-in wall-time and memory-boundary profiling for GPU PBC stress.'''

import resource
import time
from pathlib import Path

import cupy as cp


def new_profile(enabled):
    '''Return a profiler ledger or ``None`` without synchronizing by default.'''
    if not enabled:
        return None
    return {
        'schema': 'dcforge.pbc.stress.stage_profile.v1',
        'timings_seconds': {},
        'memory_boundaries': {},
        'boundary_maxima': None,
        'cuda_stream_synchronized': True,
        'timings_are_non_overlapping': True,
        'cupy_process_memory_scope': (
            'default_memory_pool_only; native CUDA allocations are not '
            'included'),
        'cuda_global_memory_scope': (
            'whole selected device; allocations from other processes are '
            'included'),
    }


def stage_start(profile):
    '''Synchronize and snapshot the beginning of one opt-in stage.'''
    if profile is None:
        return None
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter(), memory_snapshot()


def stage_stop(profile, name, state):
    '''Synchronize and retain one non-overlapping stage boundary record.'''
    if state is None:
        return
    cp.cuda.Stream.null.synchronize()
    started, before = state
    elapsed = time.perf_counter() - started
    after = memory_snapshot()
    name = str(name)
    if name in profile['timings_seconds']:
        raise RuntimeError(f'duplicate stress profiler stage {name!r}')
    profile['timings_seconds'][name] = float(elapsed)
    profile['memory_boundaries'][name] = {
        'before': before,
        'after': after,
        'boundary_delta': {
            key: float(after[key] - before[key])
            for key in before
        },
    }


def finish_profile(profile, total_started):
    '''Finalize total wall time and boundary maxima without double counting.'''
    if profile is None:
        return None
    cp.cuda.Stream.null.synchronize()
    profile['stress_total_seconds'] = float(
        time.perf_counter() - total_started)
    snapshots = [
        boundary[position]
        for boundary in profile['memory_boundaries'].values()
        for position in ('before', 'after')
    ]
    if snapshots:
        profile['boundary_maxima'] = {
            key: max(snapshot[key] for snapshot in snapshots)
            for key in snapshots[0]
        }
        profile['boundary_peak_stage_by_metric'] = {
            key: max(
                (
                    (snapshot[key], name, position)
                    for name, boundary in profile[
                        'memory_boundaries'].items()
                    for position, snapshot in boundary.items()
                    if position in ('before', 'after')
                ),
                key=lambda item: item[0],
            )[1:]
            for key in snapshots[0]
        }
    else:
        profile['boundary_maxima'] = memory_snapshot()
        profile['boundary_peak_stage_by_metric'] = None
    profile['covered_stage_seconds'] = float(sum(
        profile['timings_seconds'].values()))
    profile['unattributed_seconds'] = max(
        0., profile['stress_total_seconds']
        - profile['covered_stage_seconds'])
    return profile


def memory_snapshot():
    '''Return host, CuPy-pool and whole-device memory in MiB.'''
    pool = cp.get_default_memory_pool()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    current_rss_mib = _current_host_rss_mib()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        'host_current_rss_mib': current_rss_mib,
        'host_peak_rss_mib': max(
            current_rss_mib, float(usage.ru_maxrss / 1024.)),
        'cupy_pool_used_mib': float(pool.used_bytes() / 1024. ** 2),
        'cupy_pool_reserved_mib': float(pool.total_bytes() / 1024. ** 2),
        'cuda_global_used_mib': float(
            (total_bytes - free_bytes) / 1024. ** 2),
        'cuda_global_free_mib': float(free_bytes / 1024. ** 2),
    }


def _current_host_rss_mib():
    '''Return current process RSS without confusing it with ``ru_maxrss``.'''
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return float(line.split()[1]) / 1024.
    except OSError:
        pass
    return float('nan')


__all__ = [
    'finish_profile',
    'memory_snapshot',
    'new_profile',
    'stage_start',
    'stage_stop',
]
