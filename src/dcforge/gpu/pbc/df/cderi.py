'''Memory ownership and out-of-core storage for GPU periodic CDERI factors.

GPU4PySCF stores the final Gamma factors as AO-pair-compressed NumPy arrays
backed by CUDA mapped host memory.  That representation is already much
smaller than a dense ``(naux,nao,nao)`` tensor, but its lifetime and its
host-memory cost are otherwise implicit.  This module makes both explicit
without changing a DF integral, metric, or contraction.
'''

import os
import tempfile
from pathlib import Path

import cupy as cp
import h5py
import numpy as np
from gpu4pyscf.lib.cupy_helper import empty_mapped, get_avail_mem
from pyscf.pbc.df import df as cpu_df

CDERI_KEYS = {
    'cderi_storage', 'cderi_storage_path', 'cderi_storage_block_size',
    'cderi_preflight_max_bytes', 'cderi_memory_metadata',
    'cderi_storage_resolved', 'cderi_auto_memory_fraction',
    'cderi_auto_minimum_reserve_bytes', 'cderi_auto_max_bytes',
    'cderi_auto_sidecar_factor', 'cderi_auto_plan',
    '_cderi_preflight_memory_metadata',
    'release_cderi_after_response', '_cderi_storage_files',
}

DEFAULT_CDERI_AUTO_MEMORY_FRACTION = .8
DEFAULT_CDERI_AUTO_MINIMUM_RESERVE_BYTES = 16 * 1024**3


class H5CDERI:
    '''Read-only HDF5 array implementing GPU4PySCF's sliced CDERI contract.

    The upstream GDF iterator only requires ``shape``, ``dtype``, ``size``
    and auxiliary-axis slicing.  Opening the dataset for each slice keeps no
    process-global HDF5 handle alive, which is important for scanner resets
    and forked queue workers.
    '''

    def __init__(self, path, dataset, shape, dtype):
        self.path = str(Path(path).resolve())
        self.dataset = str(dataset)
        self.shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.nbytes = self.size * self.dtype.itemsize

    def __getitem__(self, key):
        with h5py.File(self.path, 'r') as handle:
            return np.asarray(handle[self.dataset][key])


def init_cderi_options(provider):
    '''Initialize PySCF-style provider options for CDERI memory ownership.'''
    provider.cderi_storage = 'mapped_host'
    provider.cderi_storage_resolved = 'mapped_host'
    provider.cderi_storage_path = None
    provider.cderi_storage_block_size = 256
    provider.cderi_preflight_max_bytes = None
    provider.cderi_memory_metadata = None
    provider.cderi_auto_memory_fraction = DEFAULT_CDERI_AUTO_MEMORY_FRACTION
    provider.cderi_auto_minimum_reserve_bytes = (
        DEFAULT_CDERI_AUTO_MINIMUM_RESERVE_BYTES)
    provider.cderi_auto_max_bytes = None
    # Additional persistent host storage required by the selected downstream
    # algorithm, expressed relative to the exact FP64 central factor.  The
    # production mixed-precision source and response paths use one FP32 copy,
    # hence 0.5.  The ordinary FP64 path leaves this at zero.
    provider.cderi_auto_sidecar_factor = 0.
    provider.cderi_auto_plan = None
    provider._cderi_preflight_memory_metadata = None
    provider.release_cderi_after_response = False
    provider._cderi_storage_files = []


def preflight_cderi_memory(provider):
    r'''Estimate the central CDERI before native construction.

    The exact compressed pair count is owned by GPU4PySCF's sorted shell-pair
    optimizer and is known only after that optimizer has run.  The preflight
    therefore reports both the physical lower-triangle estimate and the
    conservative physical-square bound.  ``cderi_preflight_max_bytes`` is
    compared with the lower-triangle estimate; the exact post-build report is
    retained separately and never inferred from this estimate.
    '''
    auxcell = provider.auxcell
    if auxcell is None:
        auxcell = cpu_df.make_auxcell(
            provider.cell, provider.auxbasis, provider.exp_to_discard)
    nao = int(provider.cell.nao_nr())
    naux = int(auxcell.nao_nr())
    kpts = np.asarray(provider.kpts).reshape(-1, 3)
    real_gamma = len(kpts) == 1 and np.max(np.abs(kpts)) < 1e-12
    dtype = np.dtype(np.float64 if real_gamma else np.complex128)
    factor_blocks = 1 if real_gamma else len(kpts)
    lower_pairs = nao * (nao + 1) // 2
    lower_bytes = factor_blocks * naux * lower_pairs * dtype.itemsize
    square_bytes = factor_blocks * naux * nao * nao * dtype.itemsize
    available_host = _available_host_bytes()
    report = {
        'phase': 'preflight',
        'storage_policy': str(provider.cderi_storage),
        'physical_nao': nao,
        'physical_naux': naux,
        'sampling': 'real_gamma' if real_gamma else 'sampled_k',
        'factor_block_estimate': int(factor_blocks),
        'dtype': dtype.name,
        'lower_triangle_pair_estimate': lower_pairs,
        'lower_triangle_bytes_estimate': int(lower_bytes),
        'physical_square_bytes_bound': int(square_bytes),
        'available_host_bytes': available_host,
        'available_device_bytes': _available_device_bytes(),
        'configured_max_bytes': provider.cderi_preflight_max_bytes,
    }
    limit = provider.cderi_preflight_max_bytes
    if limit is not None and lower_bytes > int(limit):
        raise MemoryError(
            'estimated compressed CDERI exceeds cderi_preflight_max_bytes: '
            f'estimate={lower_bytes} bytes, limit={int(limit)} bytes; choose '
            'a smaller explicit auxbasis, cderi_storage=\'hdf5\', or raise '
            'the explicit limit')
    provider._cderi_preflight_memory_metadata = dict(report)
    provider.cderi_auto_plan = resolve_cderi_storage(
        provider, central_bytes=lower_bytes,
        available_host_bytes=available_host, phase='preflight')
    report['storage_policy_resolved'] = provider.cderi_storage_resolved
    report['auto_storage_plan'] = provider.cderi_auto_plan
    provider.cderi_memory_metadata = report
    return report


def finalize_cderi_storage(provider):
    '''Apply the selected storage policy and record the exact built footprint.'''
    exact_bytes = _central_bytes(provider)
    preflight = getattr(provider, '_cderi_preflight_memory_metadata', None) or {}
    provider.cderi_auto_plan = resolve_cderi_storage(
        provider, central_bytes=exact_bytes,
        available_host_bytes=preflight.get('available_host_bytes'),
        phase='post_build')
    policy = provider.cderi_storage_resolved
    if policy == 'hdf5':
        _externalize_to_hdf5(provider)
    provider.cderi_memory_metadata = cderi_memory_report(provider)
    return provider.cderi_memory_metadata


def cderi_memory_report(provider, response_observables=2):
    '''Return exact central storage and explicit dense-response estimates.'''
    entries = []
    total = 0
    for sign, mapping in (('positive', provider._cderi),
                          ('negative', getattr(provider, '_cderip', None))):
        if mapping is None:
            continue
        for key, value in mapping.items():
            nbytes = _nbytes(value)
            total += nbytes
            entry = {
                'sign': sign,
                'key': int(key),
                'shape': [int(item) for item in value.shape],
                'dtype': np.dtype(value.dtype).name,
                'bytes': int(nbytes),
                'storage_class': type(value).__name__,
            }
            storage = getattr(value, 'storage', None)
            if storage is not None:
                entry['compatibility_dtype'] = np.dtype(value.dtype).name
                entry['physical_storage_dtype'] = np.dtype(
                    storage.dtype).name
                entry['physical_storage_class'] = type(storage).__name__
            if isinstance(value, H5CDERI):
                entry['path'] = value.path
                entry['dataset'] = value.dataset
            entries.append(entry)
    nao = int(provider.cell.nao_nr())
    naux = max((entry['shape'][0] for entry in entries), default=0)
    pair_count = max((entry['shape'][1] for entry in entries), default=0)
    dtype_size = max(
        (np.dtype(entry['dtype']).itemsize for entry in entries), default=8)
    nobs = int(response_observables)
    factor_blocks = len(entries)
    dense_one = naux * nao * nao * dtype_size
    return {
        'phase': 'built' if entries else 'released_or_unbuilt',
        'storage_policy': str(provider.cderi_storage),
        'storage_policy_resolved': str(getattr(
            provider, 'cderi_storage_resolved', provider.cderi_storage)),
        'auto_storage_plan': getattr(provider, 'cderi_auto_plan', None),
        'entries': entries,
        'central_bytes': int(total),
        'physical_nao': nao,
        'physical_naux': (
            None if provider.auxcell is None
            else int(provider.auxcell.nao_nr())),
        'factor_naux': int(naux),
        'factor_block_count': int(factor_blocks),
        'actual_compressed_pair_count': int(pair_count),
        'dense_response_one_observable_bytes': int(dense_one),
        'dense_response_observable_count': nobs,
        'dense_response_observables_bytes': int(nobs * dense_one),
        'dense_response_all_factor_blocks_bytes': int(
            nobs * factor_blocks * dense_one),
        'packed_response_one_observable_bytes': int(
            naux * pair_count * dtype_size),
        'available_host_bytes': _available_host_bytes(),
        'available_device_bytes': _available_device_bytes(),
    }


def cderi_block(provider, key, p0, p1, *, positive=True):
    '''Return one host CDERI auxiliary slice for mapped or HDF5 storage.'''
    mapping = provider._cderi if positive else provider._cderip
    if mapping is None or key not in mapping:
        raise KeyError(f'CDERI block {key} is unavailable')
    return np.asarray(mapping[key][p0:p1])


def copy_gamma_cderi_fp32(provider, block_size=None):
    '''Return one mapped-host FP32 copy of the built real-Gamma factor.

    Args:
        provider : dcforge.gpu.pbc.df.gdf.GDF
            Built positive-metric real-Gamma provider.

    Kwargs:
        block_size : int or None
            Number of auxiliary rows converted at once.  ``None`` uses the
            provider's CDERI storage block size.

    Returns:
        tuple
            ``(storage, report)`` where ``storage`` is a persistent mapped-host
            FP32 array and ``report`` records its exact ownership and copy time.

    Notes:
        Conversion happens once.  Subsequent source-SCF or CPHF iterations read
        this FP32 array directly; they never round an HDF5 FP64 slice again.
    '''
    import time

    if provider._cderi is None:
        provider.build(j_only=False)
    if getattr(provider, '_cderip', None):
        raise NotImplementedError(
            'mixed precision requires a positive real-Gamma metric')
    if set(provider._cderi) != {0}:
        raise NotImplementedError(
            'mixed precision requires one real-Gamma CDERI factor')
    # ``mixed`` and ``fp32_build`` providers already own the sole persistent
    # FP32 factor.  Reuse that exact allocation for source SCF and CPHF rather
    # than creating a second half-sized sidecar.  The compatibility view
    # advertises FP64 slices only to unmodified GPU4PySCF consumers.
    from dcforge.gpu.pbc.df.gdf.precision import get_fp32_storage

    resident = get_fp32_storage(provider)
    if resident is not None:
        return resident, {
            'storage_class': type(resident).__name__,
            'storage_dtype': np.dtype(resident.dtype).name,
            'shape': [int(value) for value in resident.shape],
            'bytes': int(resident.nbytes),
            'copy_block_size': None,
            'copy_seconds': 0.,
            'source_storage_class': type(provider._cderi[0]).__name__,
            'source_storage_dtype': np.dtype(
                provider._cderi[0].dtype).name,
            'reused_provider_storage': True,
        }
    source = provider._cderi[0]
    block_size = (
        provider.cderi_storage_block_size
        if block_size is None else int(block_size))
    if block_size < 1:
        raise ValueError('FP32 CDERI copy block_size must be positive')
    started = time.perf_counter()
    output = empty_mapped(source.shape, dtype=np.float32)
    for p0 in range(0, int(source.shape[0]), block_size):
        p1 = min(int(source.shape[0]), p0 + block_size)
        output[p0:p1] = cderi_block(provider, 0, p0, p1)
    return output, {
        'storage_class': type(output).__name__,
        'storage_dtype': np.dtype(output.dtype).name,
        'shape': [int(value) for value in output.shape],
        'bytes': int(output.nbytes),
        'copy_block_size': int(block_size),
        'copy_seconds': time.perf_counter() - started,
        'source_storage_class': type(source).__name__,
        'source_storage_dtype': np.dtype(source.dtype).name,
        'reused_provider_storage': False,
    }


def resolve_cderi_storage(
        provider, central_bytes, available_host_bytes=None, phase='preflight'):
    '''Resolve ``auto`` to mapped-host or HDF5 from a live host-memory budget.

    ``central_bytes`` is the estimated pre-build or exact post-build FP64
    footprint.  ``cderi_auto_sidecar_factor`` reserves additional persistent
    storage such as the 0.5-size FP32 mixed-precision factor.  The calculation
    is deterministic for the recorded memory snapshot and never changes an
    integral, threshold, or density-fitting representation.
    '''
    requested = str(provider.cderi_storage).strip().lower()
    if requested not in ('mapped_host', 'hdf5', 'auto'):
        raise ValueError(
            "cderi_storage must be 'mapped_host', 'hdf5', or 'auto'")
    central_bytes = int(central_bytes)
    sidecar_factor = float(provider.cderi_auto_sidecar_factor)
    if sidecar_factor < 0.:
        raise ValueError('cderi_auto_sidecar_factor must be nonnegative')
    fraction = float(provider.cderi_auto_memory_fraction)
    if not 0. < fraction <= 1.:
        raise ValueError('cderi_auto_memory_fraction must be in (0,1]')
    reserve = int(provider.cderi_auto_minimum_reserve_bytes)
    if reserve < 0:
        raise ValueError(
            'cderi_auto_minimum_reserve_bytes must be nonnegative')
    maximum = provider.cderi_auto_max_bytes
    if maximum is not None and int(maximum) <= 0:
        raise ValueError('cderi_auto_max_bytes must be positive or None')
    available = (
        _available_host_bytes()
        if available_host_bytes is None else int(available_host_bytes))
    sidecar_bytes = int(np.ceil(central_bytes * sidecar_factor))
    required = central_bytes + sidecar_bytes
    budgets = []
    if available is not None:
        budgets.extend((int(available * fraction), max(0, available - reserve)))
    if maximum is not None:
        budgets.append(int(maximum))
    budget = min(budgets) if budgets else None
    if requested == 'auto':
        resolved = (
            'mapped_host'
            if budget is not None and required <= budget else 'hdf5')
        reason = (
            'central factor and declared sidecar fit the host-memory budget'
            if resolved == 'mapped_host' else
            'host-memory capacity is unknown or the declared resident set '
            'exceeds the budget')
    else:
        resolved = requested
        reason = 'explicit user storage policy'
    provider.cderi_storage_resolved = resolved
    return {
        'phase': str(phase),
        'requested': requested,
        'resolved': resolved,
        'reason': reason,
        'central_bytes': central_bytes,
        'sidecar_factor': sidecar_factor,
        'sidecar_bytes': sidecar_bytes,
        'required_resident_bytes': required,
        'available_host_bytes_before_build': available,
        'memory_fraction': fraction,
        'minimum_reserve_bytes': reserve,
        'maximum_resident_bytes': (
            None if maximum is None else int(maximum)),
        'resolved_budget_bytes': budget,
    }


def release_cderi(provider):
    '''Release central factors and any provider-owned out-of-core files.'''
    before = cderi_memory_report(provider)
    provider._cderi = None
    provider._cderip = None
    provider._cderi_idx = None
    cleanup_cderi_storage(provider)
    provider.cderi_memory_metadata = {
        **before,
        'phase': 'released',
        'released_central_bytes': before['central_bytes'],
        'entries': [],
        'central_bytes': 0,
    }
    return provider.cderi_memory_metadata


def cleanup_cderi_storage(provider):
    '''Remove temporary HDF5 files owned by one provider instance.'''
    paths = list(getattr(provider, '_cderi_storage_files', ()))
    provider._cderi_storage_files = []
    for path in paths:
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass


def canonical_pair_layout(pair_address, nao):
    r'''Return unique symmetric AO pairs and the raw-to-unique map.

    GPU4PySCF compression follows shell blocks, so a pair list may contain
    both ``(i,j)`` and ``(j,i)`` for different contracted Cartesian blocks.
    The returned unique layout removes that storage redundancy while keeping
    an inverse map capable of reconstructing the exact upstream layout.
    '''
    address = np.asarray(cp.asnumpy(pair_address), dtype=np.int64).ravel()
    row, column = divmod(address, int(nao))
    upper = np.maximum(row, column)
    lower = np.minimum(row, column)
    canonical = upper * int(nao) + lower
    unique, first, inverse = np.unique(
        canonical, return_index=True, return_inverse=True)
    unique_row, unique_column = divmod(unique, int(nao))
    weights = np.where(unique_row == unique_column, 1., 2.)
    return {
        'nao': int(nao),
        'address': unique.astype(np.int64),
        'row': unique_row.astype(np.int32),
        'column': unique_column.astype(np.int32),
        'weights': weights.astype(np.float64),
        'first': first.astype(np.int64),
        'inverse': inverse.astype(np.int64),
        'raw_pair_count': len(address),
        'unique_pair_count': len(unique),
    }


def load_unique_gamma_cderi(provider, auxiliary_block_size=256):
    '''Load the real-Gamma central factor in the unique AO-pair layout.'''
    if provider._cderi is None:
        provider.build(j_only=False)
    if getattr(provider, '_cderip', None) is not None:
        raise NotImplementedError('packed Gamma response requires positive metric')
    raw = provider._cderi[0]
    layout = canonical_pair_layout(provider._cderi_idx[0], provider.cell.nao_nr())
    naux = int(raw.shape[0])
    first = layout['first']
    output = cp.empty((naux, len(first)), dtype=raw.dtype)
    block_size = max(1, min(naux, int(auxiliary_block_size)))
    for p0 in range(0, naux, block_size):
        p1 = min(naux, p0 + block_size)
        block = cp.asarray(cderi_block(provider, 0, p0, p1))
        output[p0:p1] = block[:, first]
    return output, layout


def _externalize_to_hdf5(provider):
    if provider._cderi is None:
        return
    cleanup_cderi_storage(provider)
    path = _new_storage_path(provider)
    block_size = max(1, int(provider.cderi_storage_block_size))
    replacements = {}
    with h5py.File(path, 'w') as handle:
        for label, mapping in (('positive', provider._cderi),
                               ('negative', provider._cderip)):
            if mapping is None:
                replacements[label] = None
                continue
            stored = {}
            for key, value in mapping.items():
                dataset_name = f'{label}/{int(key)}'
                rows = min(block_size, int(value.shape[0]))
                columns = min(8192, int(value.shape[1]))
                dataset = handle.create_dataset(
                    dataset_name, shape=value.shape, dtype=value.dtype,
                    chunks=(max(1, rows), max(1, columns)))
                for p0 in range(0, int(value.shape[0]), block_size):
                    p1 = min(int(value.shape[0]), p0 + block_size)
                    dataset[p0:p1] = np.asarray(value[p0:p1])
                stored[key] = H5CDERI(
                    path, dataset_name, value.shape, value.dtype)
            replacements[label] = stored
    provider._cderi = replacements['positive']
    provider._cderip = replacements['negative']
    provider._cderi_storage_files = [str(path)]


def _new_storage_path(provider):
    requested = provider.cderi_storage_path
    if requested is None:
        descriptor, path = tempfile.mkstemp(
            prefix='dcforge-cderi-', suffix='.h5')
        os.close(descriptor)
        return Path(path)
    requested = Path(requested).expanduser().resolve()
    if requested.suffix.lower() in ('.h5', '.hdf5'):
        requested.parent.mkdir(parents=True, exist_ok=True)
        descriptor, path = tempfile.mkstemp(
            prefix=requested.stem + '-', suffix=requested.suffix,
            dir=requested.parent)
    else:
        requested.mkdir(parents=True, exist_ok=True)
        descriptor, path = tempfile.mkstemp(
            prefix='dcforge-cderi-', suffix='.h5', dir=requested)
    os.close(descriptor)
    return Path(path)


def _nbytes(value):
    nbytes = getattr(value, 'nbytes', None)
    if nbytes is not None:
        return int(nbytes)
    return int(np.prod(value.shape, dtype=np.int64) * np.dtype(value.dtype).itemsize)


def _central_bytes(provider):
    '''Return the exact bytes represented by all positive/negative factors.'''
    total = 0
    for mapping in (provider._cderi, getattr(provider, '_cderip', None)):
        if mapping is not None:
            total += sum(_nbytes(value) for value in mapping.values())
    return int(total)


def _available_host_bytes():
    try:
        pages = os.sysconf('SC_AVPHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        return int(pages * page_size)
    except (OSError, ValueError):
        return None


def _available_device_bytes():
    try:
        return int(get_avail_mem(exclude_memory_pool=True))
    except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError):
        return None
