'''Mixed-precision CDERI construction for GPU real-Gamma GDF.

This module owns only the *representation* boundary used by
``RHFDFT.gdf_precision_profile``: how the central GDF factor is formed and
which bits remain in persistent storage.  Source SCF, Z-vector and derivative
arithmetic are owned by their respective modules.

``mixed`` retains GPU4PySCF's native FP64 primitive integrals, auxiliary
metric, factorization and metric transformation, then writes each completed
CDERI block directly to mapped-host FP32 storage.  A full mapped-host FP64
CDERI is never materialized.

``fp32_build`` is more aggressive.  Primitive integral kernels and the
auxiliary metric/factorization remain FP64, but the metric factor is cast to
FP32 before the large short-range transformation, and the long-range Fourier
contraction uses complex64/FP32.  The result is again stored as mapped-host
FP32.

Neither route is an FP64 reference.  Promoting a quantized factor to FP64 changes
the arithmetic dtype, not the information content, and cannot restore bits
discarded during storage or construction.
'''

import ctypes
import threading
from contextlib import contextmanager, nullcontext

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import empty_mapped
from pyscf import lib

PROFILES = ('fp64', 'mixed', 'fp32_build')


class GDFMixedPrecisionWarning(UserWarning):
    '''Warn that a GDF result uses a persistent FP32 central factor.'''


class GDFBuildPrecisionWarning(RuntimeWarning):
    '''Warn that the CDERI transformation itself was evaluated in FP32.'''


class FP32CDERIView:
    '''Expose persistent FP32 storage through the native sliced contract.

    GPU4PySCF consumers historically inspect a float64 ``dtype`` and request
    float64 slices.  This compatibility view satisfies that interface while
    keeping exactly one persistent FP32 owner.  Each requested slice is
    promoted on demand; the promotion is not a reconstruction of the original
    FP64 factor.
    '''

    def __init__(self, storage):
        if np.dtype(storage.dtype) != np.dtype(np.float32):
            raise TypeError('FP32CDERIView requires float32 storage')
        self.storage = storage
        self.shape = storage.shape
        self.ndim = storage.ndim
        self.size = storage.size
        # Native GPU4PySCF consumers request FP64 slices.  The persistent
        # information remains FP32 even though this compatibility view
        # advertises the dtype expected by those consumers.
        self.dtype = np.dtype(np.float64)
        self.nbytes = storage.nbytes

    def __getitem__(self, key):
        return np.asarray(self.storage[key], dtype=np.float64)


def normalize_profile(profile):
    '''Return one canonical public precision profile name.'''
    profile = str(profile).strip().lower()
    if profile not in PROFILES:
        raise ValueError(
            f'gdf_precision_profile must be one of {PROFILES}, got '
            f'{profile!r}')
    return profile


@contextmanager
def _direct_fp32_storage():
    '''Make a native FP64 transformation store its final CDERI as FP32.'''
    from gpu4pyscf.pbc.df import rsdf_builder

    original_empty = rsdf_builder.empty_mapped
    original_store = rsdf_builder.libvhf_rys.store_col_segment
    registry = {}
    lock = threading.Lock()

    def allocate(shape, dtype=np.float64, *args, **kwargs):
        if np.dtype(dtype) != np.dtype(np.float64):
            return original_empty(shape, *args, dtype=dtype, **kwargs)
        output = empty_mapped(shape, dtype=np.float32)
        with lock:
            registry[int(output.ctypes.data)] = output
        return output

    def store(destination, source, naux, npair, p0, p1):
        destination_address = int(
            destination.data
            if hasattr(destination, 'data') else destination.value)
        naux = int(naux.value)
        npair = int(npair.value)
        p0 = int(p0.value)
        p1 = int(p1.value)
        with lock:
            output = registry.get(destination_address)
        if output is None:
            return original_store(
                destination, source, ctypes.c_int(naux),
                ctypes.c_int(npair), ctypes.c_int(p0), ctypes.c_int(p1))
        width = p1 - p0
        memory = cp.cuda.UnownedMemory(
            int(source.value), naux * width * np.dtype(np.float64).itemsize,
            output)
        pointer = cp.cuda.MemoryPointer(memory, 0)
        block = cp.ndarray(
            (naux, width), dtype=cp.float64, memptr=pointer, order='C')
        output[:, p0:p1] = cp.asnumpy(block.astype(cp.float32))

    rsdf_builder.empty_mapped = allocate
    rsdf_builder.libvhf_rys.store_col_segment = store
    try:
        yield {
            'route': 'native_fp64_transform_direct_fp32_storage',
            'primitive_integrals': 'FP64/complex128',
            'auxiliary_metric_and_factorization': 'FP64',
            'short_and_long_range_metric_transformation': 'FP64/complex128',
            'persistent_cderi': 'mapped-host float32',
            'final_fp64_cderi_materialized': False,
        }
    finally:
        rsdf_builder.libvhf_rys.store_col_segment = original_store
        rsdf_builder.empty_mapped = original_empty


def _compressed_cderi_gamma_fp32(
        rsdf_builder, ledger, cell, auxcell, omega=None,
        linear_dep_threshold=None):
    '''Build a positive real-Gamma CDERI with FP32 transformations.'''
    if linear_dep_threshold is None:
        linear_dep_threshold = rsdf_builder.LINEAR_DEP_THR
    if int(getattr(cell, 'dimension', 3)) != 3:
        raise NotImplementedError('FP32 CDERI build requires a 3D Cell')
    kmesh = np.ones(3, dtype=int)
    log = rsdf_builder.logger.new_logger(cell)
    timer = log.init_timer()
    physical_omega = 0. if omega is None else abs(float(omega))
    partition_omega = max(
        physical_omega, float(rsdf_builder._guess_omega(cell)))

    intopt = rsdf_builder.SRInt3c2eOpt(
        cell, auxcell, omega=-partition_omega,
        bvk_kmesh=kmesh).build()
    sorted_cell = intopt.cell
    sorted_auxcell = intopt.auxcell
    bvk_ncells = len(intopt.bvkmesh_Ls)
    metric_factors, negative_metric_size = (
        rsdf_builder._precontract_j2c_aux_coeff(
            sorted_auxcell, None, physical_omega, partition_omega,
            linear_dep_threshold))
    if negative_metric_size:
        raise NotImplementedError(
            'FP32 CDERI build requires a positive Gamma metric')
    if len(metric_factors) != 1:
        raise RuntimeError('real-Gamma build must produce one metric factor')
    factor64 = cp.asarray(metric_factors[0], dtype=cp.float64)
    naux_cart, naux = map(int, factor64.shape)
    factor32 = factor64.astype(cp.float32)

    cderi_idx = intopt.pair_and_diag_indices()
    pair_count = len(cderi_idx[0])
    mesh = (
        intopt.mesh if physical_omega < partition_omega
        else np.ones(3, dtype=int))
    coulomb = rsdf_builder._weighted_coulG_kpts(
        sorted_auxcell, mesh, physical_omega, partition_omega)
    vectors = sorted_cell.get_Gv(mesh)
    grid_count = len(vectors)

    free_bytes = int(rsdf_builder.get_avail_mem(exclude_memory_pool=True))
    free_bytes -= int(factor64.nbytes + factor32.nbytes)
    free_bytes -= grid_count * naux * np.dtype(np.complex64).itemsize
    denominator = naux_cart * bvk_ncells * 16 * 4
    batch_size = min(pair_count, max(1, free_bytes // denominator))
    cderi = rsdf_builder.empty_mapped(
        (naux, pair_count), dtype=np.float32)

    if int(rsdf_builder.num_devices) != 1:
        raise NotImplementedError('FP32 CDERI build is single-GPU only')
    nsp_per_block = rsdf_builder.ft_ao.ft_ao_scheme()[0]
    shell_pairs = sorted_cell.aggregate_shl_pairs(
        intopt.bas_ij_cache, nsp_per_block)
    evaluator, _aux_sorting, pair_offsets = intopt.int3c2e_evaluator(
        ao_pair_batch_size=batch_size,
        bas_ij_aggregated=shell_pairs)[:3]
    batch_count = int(len(pair_offsets) - 1)
    ft_opt = rsdf_builder.ft_ao.FTOpt.from_intopt(intopt)
    ft_evaluator, ft_offsets = ft_opt.ft_evaluator(
        batch_size, bas_ij_aggregated=shell_pairs)
    if not np.array_equal(pair_offsets, ft_offsets):
        raise RuntimeError('SR and LR AO-pair batches do not match')

    auxiliary_fourier64 = rsdf_builder.ft_ao.ft_ao(
        sorted_auxcell, vectors).T.conj()
    auxiliary_fourier32 = auxiliary_fourier64.astype(cp.complex64)
    del auxiliary_fourier64
    weighted_auxiliary32 = factor32.T.dot(auxiliary_fourier32)
    weighted_auxiliary32 *= cp.asarray(coulomb, dtype=cp.float32)
    del auxiliary_fourier32

    available = free_bytes - naux_cart * batch_size * 16 * 2
    grid_block = int(
        available // (16 * (batch_size + naux * 2))) // 32 * 32
    if grid_block < 1:
        raise RuntimeError('insufficient GPU memory for FP32 CDERI build')
    grid_block = min(grid_block, grid_count)
    fourier_buffer = cp.empty(
        batch_size * grid_block, dtype=cp.complex128)
    primitive_buffer = cp.empty(
        batch_size * naux_cart * bvk_ncells, dtype=cp.complex128)

    ledger.update({
        'route': 'real_gamma_fp32_cderi_transformation',
        'primitive_integrals': 'FP64/complex128 native GPU4PySCF kernels',
        'auxiliary_metric_and_factorization': 'FP64',
        'metric_factor_dtype': 'float32 after FP64 factorization',
        'short_range_metric_transformation': 'float32',
        'long_range_metric_and_fourier_contraction': 'complex64/float32',
        'persistent_cderi': 'mapped-host float32',
        'final_fp64_cderi_materialized': False,
        'single_gpu': True,
        'cderi_bytes': int(cderi.nbytes),
        'naux': naux,
        'naux_cart': naux_cart,
        'compressed_pair_count': pair_count,
        'grid_count': grid_count,
        'ao_pair_batch_size': int(batch_size),
        'reciprocal_grid_block_size': int(grid_block),
        'batch_count': batch_count,
        'completed_batches': 0,
    })
    del factor64, metric_factors

    for batch_id in range(batch_count):
        primitive = evaluator(
            shl_pair_batch_id=batch_id, out=primitive_buffer)
        if primitive.size == 0:
            continue
        short32 = cp.asarray(primitive.sum(axis=1).T.real, dtype=cp.float32)
        transformed32 = factor32.T.dot(short32)
        for g0, g1 in lib.prange(0, grid_count, grid_block):
            pair_fourier64 = ft_evaluator(
                vectors[g0:g1], batch_id, out=fourier_buffer)
            reciprocal32 = weighted_auxiliary32[:, g0:g1].dot(
                pair_fourier64.astype(cp.complex64).T)
            transformed32 += reciprocal32.real
        p0 = int(pair_offsets[batch_id])
        p1 = int(pair_offsets[batch_id + 1])
        if p1 - p0 != int(transformed32.shape[1]):
            raise RuntimeError('FP32 CDERI AO-pair interval mismatch')
        cderi[:, p0:p1] = cp.asnumpy(cp.ascontiguousarray(transformed32))
        ledger['completed_batches'] = batch_id + 1
        timer = log.timer_debug1('FP32 transform/store', *timer)

    cp.cuda.get_current_stream().synchronize()
    return {0: cderi}, None, cderi_idx


@contextmanager
def _fp32_transformation():
    '''Install the real-Gamma FP32 transformation for one provider build.'''
    from gpu4pyscf.pbc.df import rsdf_builder

    original = rsdf_builder.compressed_cderi_gamma_point
    ledger = {'route': 'real_gamma_fp32_cderi_transformation'}

    def build(cell, auxcell, omega=None,
              linear_dep_threshold=rsdf_builder.LINEAR_DEP_THR):
        return _compressed_cderi_gamma_fp32(
            rsdf_builder, ledger, cell, auxcell, omega,
            linear_dep_threshold)

    rsdf_builder.compressed_cderi_gamma_point = build
    try:
        yield ledger
    finally:
        rsdf_builder.compressed_cderi_gamma_point = original


def build_context(profile):
    '''Return the scoped native-build adapter for one precision profile.

    The adapters temporarily replace narrow GPU4PySCF builder hooks and always
    restore them on context exit.  They are process-global upstream entry
    points, so the surrounding provider build serializes the operation.  The
    returned ledger describes what was evaluated and what was persisted; it
    is part of the scientific provenance of a non-FP64 Hamiltonian.
    '''
    profile = normalize_profile(profile)
    if profile == 'fp64':
        return nullcontext({
            'route': 'native_fp64',
            'persistent_cderi': 'mapped-host float64',
        })
    if profile == 'mixed':
        return _direct_fp32_storage()
    return _fp32_transformation()


def install_fp32_view(provider):
    '''Install and return persistent FP32 storage after one Gamma build.'''
    if provider._cderip is not None:
        raise NotImplementedError(
            'FP32 CDERI profiles require a positive Gamma metric')
    if set(provider._cderi) != {0}:
        raise NotImplementedError(
            'FP32 CDERI profiles require one Gamma factor')
    raw = provider._cderi[0]
    if np.dtype(raw.dtype) != np.dtype(np.float32):
        raise TypeError('provider did not build a persistent FP32 CDERI')
    provider._cderi = {0: FP32CDERIView(raw)}
    return raw


def get_fp32_storage(provider):
    '''Return provider-owned persistent FP32 storage, if present.'''
    if provider._cderi is None or set(provider._cderi) != {0}:
        return None
    value = provider._cderi[0]
    if isinstance(value, FP32CDERIView):
        return value.storage
    if np.dtype(value.dtype) == np.dtype(np.float32):
        return value
    return None


__all__ = [
    'PROFILES',
    'FP32CDERIView',
    'GDFBuildPrecisionWarning',
    'GDFMixedPrecisionWarning',
    'build_context',
    'get_fp32_storage',
    'install_fp32_view',
    'normalize_profile',
]
