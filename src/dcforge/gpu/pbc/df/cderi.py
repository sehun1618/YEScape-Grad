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
from gpu4pyscf.lib.cupy_helper import get_avail_mem
from pyscf.pbc.df import df as cpu_df

CDERI_KEYS = {
    'cderi_storage', 'cderi_storage_path', 'cderi_storage_block_size',
    'cderi_preflight_max_bytes', 'cderi_memory_metadata',
    'release_cderi_after_response', '_cderi_storage_files',
}


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
    provider.cderi_storage_path = None
    provider.cderi_storage_block_size = 256
    provider.cderi_preflight_max_bytes = None
    provider.cderi_memory_metadata = None
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
        'available_host_bytes': _available_host_bytes(),
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
    provider.cderi_memory_metadata = report
    return report


def finalize_cderi_storage(provider):
    '''Apply the selected storage policy and record the exact built footprint.'''
    policy = str(provider.cderi_storage).lower()
    if policy not in ('mapped_host', 'hdf5'):
        raise ValueError("cderi_storage must be 'mapped_host' or 'hdf5'")
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
