'''GPU4PySCF multigrid-v2 adapter for large inverse-FFT real views.'''
from contextlib import contextmanager
from threading import RLock

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.dft import multigrid_v2

MAX_32BIT_ADDRESS_SPAN = np.iinfo(np.int32).max

_MULTIGRID_IFFT_LOCK = RLock()
_MULTIGRID_IFFT_STACK = []


def _new_ifft_real_stats():
    '''Return empty runtime metadata for one multigrid integrator.'''
    return {
        'policy': 'chunked_large_ifft_real_materialization',
        'max_address_span_bytes': int(MAX_32BIT_ADDRESS_SPAN),
        'ifft_calls': 0,
        'materialized_calls': 0,
        'copy_blocks': 0,
        'largest_complex_address_span_bytes': 0,
        'largest_real_output_bytes': 0,
        'largest_copy_block_bytes': 0,
    }


def _address_span(array):
    '''Return the byte span addressed by a nonempty CuPy array view.'''
    if array.size == 0:
        return 0
    return int(sum(
        (size - 1) * abs(stride)
        for size, stride in zip(array.shape, array.strides, strict=True)
    ) + array.dtype.itemsize)


def _materialize_large_ifft_real(array, ni):
    r'''Return a safe real representation of one large inverse FFT.

    GPU4PySCF 1.8.x repeatedly evaluates

    .. math::

       \rho_c(\mathbf r) = \Re\left[\mathcal F^{-1}
       \rho_c(\mathbf G)\right]

    by taking ``ifft_output.real``.  For a complex128 C-contiguous array this
    real view has a 16-byte last-axis stride.  On the local CuPy 14.0.1/CUDA
    13 stack, a later elementwise operation can select 32-bit indexing from
    the element count even though the view's byte-address span exceeds
    ``INT32_MAX``.  The resulting offset overflow causes
    ``CUDA_ERROR_ILLEGAL_ADDRESS``.

    Args:
        array : cupy.ndarray
            Complex output returned by GPU4PySCF's inverse FFT.
        ni : MultiGridNumInt
            Active DC-Forge integrator carrying runtime diagnostics.

    Returns:
        cupy.ndarray
            The original complex array when its real view is address-safe;
            otherwise a C-contiguous real array containing the same real
            values.

    Notes:
        Every audited GPU4PySCF 1.8.x multigrid-v2 caller immediately takes
        ``.real``
        from ``ifft_in_place``.  Returning the already materialized real array
        in the large-span branch therefore preserves the upstream equation
        and dtype seen after ``.real``.  The copy itself is divided into
        source spans below ``INT32_MAX``; it does not change the FFT mesh,
        XC functional, density, or channel ordering.
    '''
    real_view = array.real
    address_span = _address_span(real_view)
    stats = ni.ifft_real_stats
    stats['ifft_calls'] += 1
    stats['largest_complex_address_span_bytes'] = max(
        stats['largest_complex_address_span_bytes'], address_span)
    if address_span <= MAX_32BIT_ADDRESS_SPAN:
        return array

    if not array.flags.c_contiguous:
        raise ValueError(
            'large GPU multigrid inverse-FFT output must be C-contiguous')
    complex_flat = array.reshape(-1)
    real_flat = cp.empty(complex_flat.size, dtype=real_view.dtype)
    block_size = max(
        1, MAX_32BIT_ADDRESS_SPAN // array.dtype.itemsize)
    for p0 in range(0, complex_flat.size, block_size):
        p1 = min(p0 + block_size, complex_flat.size)
        real_flat[p0:p1] = complex_flat[p0:p1].real
        stats['copy_blocks'] += 1
        stats['largest_copy_block_bytes'] = max(
            stats['largest_copy_block_bytes'],
            (p1 - p0) * array.dtype.itemsize)
    stats['materialized_calls'] += 1
    stats['largest_real_output_bytes'] = max(
        stats['largest_real_output_bytes'], real_flat.nbytes)
    return real_flat.reshape(array.shape)


@contextmanager
def safe_multigrid_ifft_real(ni):
    r'''Apply large-real-view protection during one upstream MG operation.

    GPU4PySCF 1.8.x resolves ``ifft_in_place`` as a module global and exposes
    no per-``MultiGridNumInt`` hook.  A process-wide re-entrant lock therefore
    protects this narrow synchronous replacement, and ``finally`` restores
    the exact upstream function on success or failure.  Nested calls such as
    ``nr_rks_fxc`` to ``cache_xc_kernel1`` reuse the active adapter.

    Args:
        ni : MultiGridNumInt
            Active DC-Forge target-XC numerical integrator.
    '''
    with _MULTIGRID_IFFT_LOCK:
        if _MULTIGRID_IFFT_STACK:
            if _MULTIGRID_IFFT_STACK[-1] is not ni:
                raise RuntimeError(
                    'nested GPU multigrid calls must use one integrator')
            _MULTIGRID_IFFT_STACK.append(ni)
            try:
                yield
            finally:
                _MULTIGRID_IFFT_STACK.pop()
            return

        original_ifft = multigrid_v2.ifft_in_place

        def ifft_in_place(array):
            output = original_ifft(array)
            return _materialize_large_ifft_real(output, ni)

        _MULTIGRID_IFFT_STACK.append(ni)
        multigrid_v2.ifft_in_place = ifft_in_place
        try:
            yield
        finally:
            multigrid_v2.ifft_in_place = original_ifft
            _MULTIGRID_IFFT_STACK.pop()


class MultiGridNumInt(multigrid_v2.MultiGridNumInt):
    '''GPU4PySCF multigrid-v2 with safe large inverse-FFT real views.

    The scientific API and numerical hierarchy are inherited unchanged.
    Only the representation of a real inverse-FFT result is made contiguous
    when its complex-strided address span exceeds the 32-bit range.
    '''

    _keys = getattr(multigrid_v2.MultiGridNumInt, '_keys', set()) | {
        'ifft_real_stats',
    }

    def __init__(self, cell):
        multigrid_v2.MultiGridNumInt.__init__(self, cell)
        self.ifft_real_stats = _new_ifft_real_stats()

    def reset(self, cell=None):
        '''Reset geometry-dependent MG tasks and IFFT diagnostics.'''
        multigrid_v2.MultiGridNumInt.reset(self, cell)
        self.ifft_real_stats = _new_ifft_real_stats()
        return self

    def get_j(self, dm, hermi=1, kpts=None, kpts_band=None):
        '''Return multigrid J with safe inverse-FFT real views.'''
        with safe_multigrid_ifft_real(self):
            return multigrid_v2.get_j_kpts(
                self, dm, hermi, kpts, kpts_band)

    def get_rho(self, dm, kpts=None):
        '''Return the real-space density with safe inverse-FFT real views.'''
        with safe_multigrid_ifft_real(self):
            return multigrid_v2.get_rho(self, dm, kpts)

    def nr_rks(self, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
               kpts=None, kpts_band=None, with_j=False, verbose=None):
        '''Evaluate RKS XC energy/potential with safe real views.'''
        with safe_multigrid_ifft_real(self):
            return multigrid_v2.nr_rks(
                self, cell, grids, xc_code, dm_kpts, relativity, hermi,
                kpts, kpts_band, with_j, verbose)

    def nr_uks(self, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
               kpts=None, kpts_band=None, with_j=False, verbose=None):
        '''Evaluate UKS XC energy/potential with safe real views.'''
        with safe_multigrid_ifft_real(self):
            return multigrid_v2.nr_uks(
                self, cell, grids, xc_code, dm_kpts, relativity, hermi,
                kpts, kpts_band, with_j, verbose)

    def nr_rks_fxc(self, cell, grids, xc_code, dm0, dms, hermi=0,
                   fxc=None, kpts=None, with_j=False):
        '''Apply the RKS XC kernel with safe response-density real views.'''
        with safe_multigrid_ifft_real(self):
            return multigrid_v2.MultiGridNumInt.nr_rks_fxc(
                self, cell, grids, xc_code, dm0, dms, hermi, fxc,
                kpts, with_j)

    def nr_uks_fxc(self, cell, grids, xc_code, dm0, dms, hermi=0,
                   fxc=None, kpts=None, with_j=False):
        '''Apply the UKS XC kernel with safe response-density real views.'''
        with safe_multigrid_ifft_real(self):
            return multigrid_v2.MultiGridNumInt.nr_uks_fxc(
                self, cell, grids, xc_code, dm0, dms, hermi, fxc,
                kpts, with_j)

    def cache_xc_kernel1(self, cell, grids, xc_code, dm, spin=0, kpts=None,
                         is_rhf=None):
        '''Build the XC kernel with safe ground-state-density real views.'''
        with safe_multigrid_ifft_real(self):
            return multigrid_v2.MultiGridNumInt.cache_xc_kernel1(
                self, cell, grids, xc_code, dm, spin, kpts, is_rhf)


def get_veff_ip1(ni, xc_code, dm_kpts, hermi=1, kpts=None, with_j=True,
                 with_pseudo_vloc_orbital_derivative=True, verbose=None):
    '''Return the upstream explicit MG derivative with safe real views.'''
    if not isinstance(ni, MultiGridNumInt):
        raise TypeError(
            'safe GPU multigrid derivative requires DC-Forge '
            'MultiGridNumInt')
    with safe_multigrid_ifft_real(ni):
        return multigrid_v2.get_veff_ip1(
            ni, xc_code, dm_kpts, hermi=hermi, kpts=kpts, with_j=with_j,
            with_pseudo_vloc_orbital_derivative=(
                with_pseudo_vloc_orbital_derivative),
            verbose=verbose)
