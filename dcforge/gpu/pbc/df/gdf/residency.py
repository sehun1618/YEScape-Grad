'''Opt-in device residency and transfer telemetry for real-Gamma CDERI.'''

import threading
import time

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import get_avail_mem
from gpu4pyscf.pbc.df import rsdf_builder
from pyscf import lib

DEFAULT_DEVICE_RESIDENCY_MEMORY_FRACTION = .8
DEFAULT_DEVICE_RESIDENCY_MINIMUM_RESERVE_BYTES = 4 * 1024**3


def init_cderi_execution_options(provider):
    '''Initialize inert CDERI execution controls on one provider.'''
    provider.cderi_device_residency = 'off'
    # Fraction of auxiliary rows requested on device.  This is independent
    # of ``memory_fraction``, which limits how much of currently free VRAM the
    # allocation may consume.  Keeping the two controls separate makes a
    # 25/50/100-percent residency sweep mean what its label says.
    provider.cderi_device_residency_fraction = 1.
    provider.cderi_device_residency_memory_fraction = (
        DEFAULT_DEVICE_RESIDENCY_MEMORY_FRACTION)
    provider.cderi_device_residency_minimum_reserve_bytes = (
        DEFAULT_DEVICE_RESIDENCY_MINIMUM_RESERVE_BYTES)
    provider.cderi_device_residency_max_bytes = None
    provider.cderi_device_residency_plan = None
    provider.cderi_transfer_profile = False
    provider.cderi_transfer_report = None
    provider._cderi_transfer_lock = threading.RLock()


def reset_cderi_execution_state(provider):
    '''Clear geometry-dependent residency and accumulated transfer telemetry.'''
    provider.cderi_device_residency_plan = None
    provider.cderi_transfer_report = None


def plan_device_resident_cderi(provider, available_bytes=None):
    r'''Return a fail-closed plan for retaining compressed CDERI on one GPU.

    The plan is exact after a build and a lower-triangle estimate beforehand.
    Device residency changes only ownership and H2D traffic.  It does not
    unpack the factor persistently and does not change an integral or metric.
    '''
    requested = _residency_mode(provider.cderi_device_residency)
    resident_fraction = float(provider.cderi_device_residency_fraction)
    fraction = float(provider.cderi_device_residency_memory_fraction)
    reserve = int(provider.cderi_device_residency_minimum_reserve_bytes)
    maximum = provider.cderi_device_residency_max_bytes
    if not 0. < fraction <= 1.:
        raise ValueError(
            'cderi_device_residency_memory_fraction must be in (0,1]')
    if not 0. < resident_fraction <= 1.:
        raise ValueError(
            'cderi_device_residency_fraction must be in (0,1]')
    if reserve < 0:
        raise ValueError(
            'cderi_device_residency_minimum_reserve_bytes must be non-negative')
    if maximum is not None and int(maximum) <= 0:
        raise ValueError(
            'cderi_device_residency_max_bytes must be positive or None')
    if available_bytes is None:
        available_bytes = int(get_avail_mem(exclude_memory_pool=True))
    else:
        available_bytes = int(available_bytes)
    if available_bytes < 0:
        raise ValueError('available device bytes must be non-negative')

    reasons = []
    kpts = np.asarray(provider.kpts).reshape(-1, 3)
    if len(kpts) != 1 or np.max(np.abs(kpts)) >= 1.e-12:
        reasons.append('device residency requires one real Gamma point')
    if int(provider.cell.dimension) != 3:
        reasons.append('device residency requires a 3D Cell')
    if int(rsdf_builder.num_devices) != 1:
        reasons.append('device residency is single-GPU only')
    if getattr(provider, 'gdf_precision_profile', 'fp64') != 'fp64':
        reasons.append('device residency currently requires FP64 GDF')
    negative = getattr(provider, '_cderip', None)
    if negative:
        reasons.append('device residency requires a positive auxiliary metric')
    factors = getattr(provider, '_cderi', None)
    if factors is not None and set(factors) != {0}:
        reasons.append('device residency requires one Gamma factor block')

    exact = factors is not None
    requested_rows = resident_rows = 0
    existing_rows = 0
    if exact:
        raw = factors[0]
        naux = int(raw.shape[0])
        row_bytes = int(np.prod(raw.shape[1:], dtype=np.int64)
                        * np.dtype(raw.dtype).itemsize)
        requested_rows = min(
            naux, max(1, int(np.ceil(naux * resident_fraction))))
        required = requested_rows * row_bytes
        source_classes = {
            type(value).__name__ for value in factors.values()}
        if isinstance(raw, cp.ndarray):
            existing_rows = naux
        elif isinstance(raw, DeviceResidentCDERIPrefix):
            existing_rows = int(raw.prefix_rows)
        already_resident = existing_rows >= requested_rows
    else:
        metadata = getattr(
            provider, '_cderi_preflight_memory_metadata', None) or {}
        full_estimate = int(metadata.get('lower_triangle_bytes_estimate', 0))
        required = int(np.ceil(full_estimate * resident_fraction))
        naux = row_bytes = 0
        source_classes = set()
        already_resident = False
        if required <= 0:
            reasons.append(
                'build or preflight is required for a residency byte estimate')

    limits = [int(available_bytes * fraction), max(0, available_bytes - reserve)]
    if maximum is not None:
        limits.append(int(maximum))
    budget = min(limits)
    if exact and row_bytes:
        resident_rows = (
            existing_rows if already_resident else
            max(existing_rows, min(requested_rows, budget // row_bytes)))
    fits = required <= budget
    supported = not reasons
    capacity_ok = (
        (fits or already_resident) if requested == 'required' else
        (resident_rows > 0 or (not exact and fits) or already_resident))
    feasible = requested != 'off' and supported and capacity_ok
    return {
        'requested': requested,
        'supported': supported,
        'feasible': feasible,
        'already_resident': already_resident,
        'estimate_is_exact': exact,
        'required_device_bytes': int(required),
        'requested_factor_fraction': resident_fraction,
        'requested_auxiliary_rows': int(requested_rows),
        'planned_auxiliary_rows': int(resident_rows),
        'existing_resident_auxiliary_rows': int(existing_rows),
        'planned_factor_fraction': (
            None if not exact or naux == 0 else resident_rows / naux),
        'row_bytes': int(row_bytes),
        'available_device_bytes': int(available_bytes),
        'memory_fraction': fraction,
        'minimum_reserve_bytes': reserve,
        'maximum_resident_bytes': (
            None if maximum is None else int(maximum)),
        'resolved_budget_bytes': int(budget),
        'fits_budget': bool(fits),
        'source_storage_classes': sorted(source_classes),
        'unsupported_reasons': reasons,
        'resolved': (
            ('device' if existing_rows == naux or (
                exact and resident_rows == naux) else 'hybrid_device_prefix')
            if feasible else
            'off' if requested == 'off' else 'host_streamed'),
        'fallback_reason': _fallback_reason(
            requested, supported, fits, already_resident, reasons,
            required, budget),
        'capacity_adjustment': (
            f'requested {requested_rows} rows; budget permits '
            f'{resident_rows} rows'
            if (exact and 0 < resident_rows < requested_rows) else None),
    }


def prepare_device_resident_cderi(provider):
    r'''Apply an explicitly requested bounded full-factor device residency.

    ``'auto'`` falls back to the original host/HDF5 streamed factor after an
    unsupported representation, capacity rejection, or allocation failure.
    ``'required'`` raises instead.  The provider mapping is replaced only after
    every block has been copied and the current CUDA stream has completed, so a
    failed attempt cannot leave a partial scientific state.
    '''
    requested = _residency_mode(provider.cderi_device_residency)
    if getattr(provider, '_cderi', None) is None:
        raise RuntimeError('CDERI must be built before device residency')
    plan = plan_device_resident_cderi(provider)
    provider.cderi_device_residency_plan = plan
    if requested == 'off' or plan['already_resident']:
        return plan
    if (not plan['supported']
            or (requested == 'required' and not plan['fits_budget'])
            or int(plan.get('planned_auxiliary_rows', 0)) < 1):
        if requested == 'required':
            raise MemoryError(
                'required device-resident CDERI is unavailable: '
                f'{plan["fallback_reason"]}')
        return plan

    raw = provider._cderi[0]
    if (isinstance(raw, DeviceResidentCDERIPrefix)
            and int(plan['planned_auxiliary_rows']) <= raw.prefix_rows):
        # ``auto`` may request a larger prefix than the current VRAM budget
        # permits.  Keeping the already-live exact prefix is preferable to a
        # same-size reallocation, and most importantly makes repeated calls
        # idempotent.
        plan.update({
            'resolved': 'hybrid_device_prefix',
            'feasible': True,
            'reused_existing_residency': True,
            'resident_storage_class': type(raw.resident).__name__,
            'resident_dtype': np.dtype(raw.resident.dtype).name,
            'resident_shape': [int(value) for value in raw.resident.shape],
            'resident_device_bytes': int(raw.resident.nbytes),
            'resident_auxiliary_rows': int(raw.prefix_rows),
            'resident_factor_fraction': (
                raw.prefix_rows / int(raw.shape[0])),
            'resident_device_pointer': int(raw.resident.data.ptr),
            'copy_seconds': 0.,
            'host_to_device_bytes': 0,
        })
        provider.cderi_device_residency_plan = plan
        return plan
    # A previous partial transaction keeps the complete original factor in
    # ``host``.  Reconfiguration must copy from that owner, never from a
    # nested hybrid slice (whose prefix is a CuPy array and deliberately
    # rejects implicit NumPy conversion).
    source = raw.host if isinstance(raw, DeviceResidentCDERIPrefix) else raw
    resident_rows = int(plan['planned_auxiliary_rows'])
    rows = max(1, int(getattr(
        provider, 'cderi_storage_block_size', 256)))
    started = time.perf_counter()
    try:
        resident = cp.empty(
            (resident_rows, *source.shape[1:]), dtype=source.dtype)
        for p0, p1 in lib.prange(0, resident_rows, rows):
            block = source[p0:p1]
            if isinstance(block, cp.ndarray):
                resident[p0:p1] = block
            else:
                resident[p0:p1].set(np.asarray(block))
        cp.cuda.get_current_stream().synchronize()
    except (MemoryError, cp.cuda.memory.OutOfMemoryError) as error:
        plan.update({
            'resolved': 'host_streamed',
            'feasible': False,
            'allocation_error': f'{type(error).__name__}: {error}',
            'copy_seconds': time.perf_counter() - started,
            'host_to_device_bytes': 0,
        })
        if requested == 'required':
            raise MemoryError(
                'required device-resident CDERI allocation failed') from error
        return plan
    except cp.cuda.runtime.CUDARuntimeError as error:
        # CUDA status 2 is cudaErrorMemoryAllocation.  Any other runtime
        # failure, especially cudaErrorIllegalAddress, poisons the CUDA
        # context and must terminate the isolated worker instead of silently
        # continuing with the host-streamed path.
        if int(getattr(error, 'status', -1)) != 2:
            raise
        plan.update({
            'resolved': 'host_streamed',
            'feasible': False,
            'allocation_error': f'{type(error).__name__}: {error}',
            'copy_seconds': time.perf_counter() - started,
            'host_to_device_bytes': 0,
        })
        if requested == 'required':
            raise MemoryError(
                'required device-resident CDERI allocation failed') from error
        return plan

    full = resident_rows == int(source.shape[0])
    if full:
        provider._cderi = {0: resident}
        # A provider-owned temporary HDF5 backing is no longer part of the
        # live scientific state after a complete device transaction succeeds.
        from dcforge.gpu.pbc.df.cderi import cleanup_cderi_storage

        cleanup_cderi_storage(provider)
    else:
        provider._cderi = {0: DeviceResidentCDERIPrefix(source, resident)}
    plan.update({
        'resolved': 'device' if full else 'hybrid_device_prefix',
        'feasible': True,
        'resident_storage_class': type(resident).__name__,
        'resident_dtype': np.dtype(resident.dtype).name,
        'resident_shape': [int(value) for value in resident.shape],
        'resident_device_bytes': int(resident.nbytes),
        'resident_auxiliary_rows': resident_rows,
        'resident_factor_fraction': resident_rows / int(source.shape[0]),
        'resident_device_pointer': int(resident.data.ptr),
        'copy_block_size': rows,
        'copy_seconds': time.perf_counter() - started,
        'host_to_device_bytes': int(resident.nbytes),
        'fallback_reason': None,
    })
    provider.cderi_device_residency_plan = plan
    return plan


class DeviceResidentCDERIPrefix:
    '''Read-only hybrid array for a device prefix and host/disk suffix.

    GPU4PySCF's Gamma iterator asks only for auxiliary-axis slices.  Slices
    wholly inside the prefix are returned as CuPy views and therefore incur
    no H2D copy.  Suffix slices preserve the original host/HDF5 object.  A
    boundary-crossing slice is assembled on device from those two pieces.
    '''

    def __init__(self, host, resident):
        if resident.ndim != len(host.shape):
            raise ValueError('resident prefix rank differs from host factor')
        if tuple(resident.shape[1:]) != tuple(host.shape[1:]):
            raise ValueError('resident prefix AO-pair shape differs from host')
        if np.dtype(resident.dtype) != np.dtype(host.dtype):
            raise TypeError('resident prefix dtype differs from host factor')
        if not 0 < int(resident.shape[0]) < int(host.shape[0]):
            raise ValueError('hybrid prefix must be nonempty and incomplete')
        self.host = host
        self.resident = resident
        self.prefix_rows = int(resident.shape[0])
        self.shape = tuple(int(value) for value in host.shape)
        self.dtype = np.dtype(host.dtype)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.nbytes = self.size * self.dtype.itemsize
        self.resident_nbytes = int(resident.nbytes)
        self.last_slice_h2d_bytes = 0
        self.last_slice_resident_bytes = 0

    def __getitem__(self, key):
        row_key, tail = _split_row_key(key, self.ndim)
        start, stop, step = row_key.indices(self.shape[0])
        if step != 1:
            raise IndexError('hybrid CDERI supports contiguous row slices')
        if stop <= self.prefix_rows:
            value = self.resident[start:stop]
            self.last_slice_h2d_bytes = 0
            self.last_slice_resident_bytes = int(value.nbytes)
        elif start >= self.prefix_rows:
            value = self.host[start:stop]
            self.last_slice_h2d_bytes = int(
                np.prod(value.shape, dtype=np.int64) * self.dtype.itemsize)
            self.last_slice_resident_bytes = 0
        else:
            host = self.host[self.prefix_rows:stop]
            value = cp.concatenate((
                self.resident[start:self.prefix_rows],
                cp.asarray(host)), axis=0)
            self.last_slice_h2d_bytes = int(
                np.prod(host.shape, dtype=np.int64) * self.dtype.itemsize)
            self.last_slice_resident_bytes = int(
                self.resident[start:self.prefix_rows].nbytes)
        return value[(slice(None), *tail)] if tail else value


def _split_row_key(key, ndim):
    if isinstance(key, tuple):
        if not key:
            raise IndexError('empty CDERI index')
        row, tail = key[0], key[1:]
    else:
        row, tail = key, ()
    if not isinstance(row, slice):
        raise IndexError('hybrid CDERI requires an auxiliary-axis slice')
    if len(tail) > ndim - 1:
        raise IndexError('too many CDERI indices')
    return row, tail


def profiled_gamma_loop(provider, blocks, *, unpack):
    '''Yield native Gamma blocks while measuring their load/unpack boundary.'''
    if not provider.cderi_transfer_profile:
        yield from blocks
        return
    report = _transfer_report(provider)
    with provider._cderi_transfer_lock:
        report['loop_calls'] += 1
    iterator = iter(blocks)
    while True:
        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        started = time.perf_counter()
        try:
            block = next(iterator)
        except StopIteration:
            return
        except BaseException:
            with provider._cderi_transfer_lock:
                report['loop_failures'] += 1
            raise
        stream.synchronize()
        elapsed = time.perf_counter() - started
        _dense, packed, sign = block
        logical = int(packed.nbytes)
        mapping = provider._cderi if sign > 0 else provider._cderip
        source = None if mapping is None else mapping.get(0)
        if isinstance(source, cp.ndarray):
            h2d = 0
            resident_bytes = logical
        elif isinstance(source, DeviceResidentCDERIPrefix):
            # The upstream iterator converts every yielded slice with
            # ``asarray`` before this profiler sees it.  The hybrid object
            # therefore records the ownership split at ``__getitem__`` time.
            h2d = int(source.last_slice_h2d_bytes)
            resident_bytes = int(source.last_slice_resident_bytes)
        else:
            h2d = logical
            resident_bytes = 0
        with provider._cderi_transfer_lock:
            report['block_count'] += 1
            report['logical_cderi_bytes'] += logical
            report['host_to_device_cderi_bytes'] += h2d
            report['device_resident_logical_bytes'] += resident_bytes
            report['load_and_unpack_seconds'] += elapsed
            report['unpack_requested_block_count'] += int(bool(unpack))
            if h2d == 0:
                report['device_resident_block_count'] += 1
            else:
                report['host_streamed_block_count'] += 1
        yield block


def _transfer_report(provider):
    report = provider.cderi_transfer_report
    if report is None:
        report = {
            'enabled': True,
            'diagnostics_only': True,
            'cuda_synchronized_per_block': True,
            'timing_scope': 'CDERI load plus optional dense unpack',
            'loop_calls': 0,
            'loop_failures': 0,
            'block_count': 0,
            'unpack_requested_block_count': 0,
            'host_streamed_block_count': 0,
            'device_resident_block_count': 0,
            'logical_cderi_bytes': 0,
            'host_to_device_cderi_bytes': 0,
            'device_resident_logical_bytes': 0,
            'load_and_unpack_seconds': 0.,
        }
        provider.cderi_transfer_report = report
    return report


def _array_nbytes(value):
    nbytes = getattr(value, 'nbytes', None)
    if nbytes is not None:
        return int(nbytes)
    return int(value.size) * np.dtype(value.dtype).itemsize


def _residency_mode(value):
    value = str(value).strip().lower()
    if value not in ('off', 'auto', 'required'):
        raise ValueError(
            "cderi_device_residency must be 'off', 'auto', or 'required'")
    return value


def _fallback_reason(requested, supported, fits, resident, reasons,
                     required, budget):
    if requested == 'off':
        return 'device residency was not requested'
    if resident:
        return None
    if not supported:
        return '; '.join(reasons)
    if not fits:
        return (
            f'factor requires {required} bytes but bounded device budget is '
            f'{budget} bytes')
    return None


__all__ = [
    'DEFAULT_DEVICE_RESIDENCY_MEMORY_FRACTION',
    'DEFAULT_DEVICE_RESIDENCY_MINIMUM_RESERVE_BYTES',
    'DeviceResidentCDERIPrefix',
    'init_cderi_execution_options',
    'plan_device_resident_cderi',
    'prepare_device_resident_cderi',
    'profiled_gamma_loop',
    'reset_cderi_execution_state',
]
