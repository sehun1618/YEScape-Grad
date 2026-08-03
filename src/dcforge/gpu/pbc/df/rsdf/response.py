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

from dcforge.gpu.pbc.df.gaussian.ewald import cross_derivative
from dcforge.gpu.pbc.df.rsdf.adjoint import (
    make_cderi_observable_adjoints,
    push_cderi_adjoints,
)
from dcforge.gpu.pbc.df.rsdf.metric import contract_metric_derivative
from dcforge.gpu.pbc.df.rsdf.source import contract_source_derivative


def make_gradient_response(with_df, dm0, dz, exxdiv='ewald'):
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
    start = time.perf_counter()
    cderi = with_df.canonical_cderi()
    factor = with_df.central_metric_factor()
    adjoints = make_cderi_observable_adjoints(cderi, dm0, dz)
    source_adjoint, metric_adjoint = push_cderi_adjoints(
        factor, cderi, adjoints)
    # Neither native primitive needs C, L, or dE/dC after reverse-Cholesky
    # propagation.  Dropping their references shortens the high-water mark
    # before the two independent nuclear contractions start.
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
    terms = {
        'de_j_source': source_value[0],
        'de_j_metric': metric_value[0],
        'de_fock_source': source_value[1],
        'de_fock_metric': metric_value[1],
        'de_fock_ewald': ewald,
    }
    metadata = {
        'response_mode': 'dense_reverse_adjoint_native_contraction',
        'central_cderi_storage': 'ephemeral_device_dense_gamma',
        'derivative_tensor_storage': 'none',
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

