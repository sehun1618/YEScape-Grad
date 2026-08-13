'''Analytic real-Gamma RSDF homogeneous-strain response.'''

import cupy as cp
import numpy as np
from cupyx.scipy.linalg import solve_triangular

from dcforge.gpu.pbc.df.gaussian.stress import (
    make_gaussian_strain_response,
)


def make_rsdf_strain_response(
        with_df, dm0, dz, reciprocal_block_size=None,
        auxiliary_block_size=None, profile_components=False,
        overlap_short_long_range=True):
    r'''Return target-J and source-Fock analytic RSDF strain channels.

    The active provider defines

    .. math::

        B=B_{SR}(\omega_s)+B_{LR}(\omega_s)+B_{vbar},

    .. math::

        M=M_{SR}(\omega_m)+M_{LR}(\omega_m)+M_{G=0},

    with independently resolved ``(omega,mesh_compact)`` and
    ``(omega_j2c,mesh_j2c)``.  Its central factor obeys
    ``C=L^{-1}B`` and ``M=LL^T``; hence the reverse-strain primitive starts
    from ``W=M^{-1}B=L^{-T}C`` without rebuilding either source or metric.

    Args:
        with_df : dcforge.gpu.pbc.df.rsdf.RSDF
            Built real-Gamma RSDF provider.
        dm0, dz : cupy.ndarray
            Source-RHF and Hermitian Z-transition AO densities.

    Kwargs:
        reciprocal_block_size, auxiliary_block_size
            Execution-only memory bounds.
        profile_components : bool
            Record synchronized component timings.
        overlap_short_long_range : bool
            Overlap independent host SR and device LR contractions.

    Returns:
        dict
            Provider-neutral two-observable source/metric strain ledger.
    '''
    _check_real_gamma_rsdf(with_df, dm0, dz)
    if getattr(with_df, '_cderi', None) is None or with_df.auxcell is None:
        with_df.build(j_only=False)
    settings = with_df.response_settings()
    cderi = with_df.canonical_cderi()
    factor = with_df.central_metric_factor()
    naux = int(cderi.shape[0])
    flat = cderi.reshape(naux, -1)
    solved = solve_triangular(
        factor.T, flat, lower=False, check_finite=False).reshape(cderi.shape)

    metric = cp.asarray(with_df._central_metric)
    reconstruction = factor @ factor.T
    residual = float(cp.max(cp.abs(reconstruction - metric)).get())
    scale = max(1., float(cp.max(cp.abs(metric)).get()))
    tolerance = max(
        1.e-10,
        1000. * np.finfo(np.float64).eps * naux * scale)
    if residual > tolerance:
        raise RuntimeError(
            'RSDF central Cholesky factor does not reconstruct its metric: '
            f'{residual:.3e} > {tolerance:.3e}')
    metadata = {
        'mode': 'provider_authoritative_rsdf_cholesky',
        'metric_owner': 'active_RSDF_central_metric',
        'source_owner': 'active_RSDF_CDERI',
        'decomposition': 'CD',
        'source_omega': float(settings['source_omega']),
        'source_mesh': list(settings['source_mesh']),
        'metric_omega': float(settings['metric_omega']),
        'metric_mesh': list(settings['metric_mesh']),
        'physical_auxiliary_dimension': naux,
        'retained_metric_rank': int(factor.shape[0]),
        'factor_reconstruction_max_abs': residual,
        'factor_reconstruction_tolerance': tolerance,
        'transparent_source_rebuilt': False,
        'gauge_alignment_performed': False,
        'host_to_device_bytes': 0,
    }
    return make_gaussian_strain_response(
        with_df, dm0, dz, solved, metadata,
        source_omega=settings['source_omega'],
        source_mesh=settings['source_mesh'],
        metric_omega=settings['metric_omega'],
        metric_mesh=settings['metric_mesh'],
        source_precision=float(with_df.precision_R),
        metric_precision=float(with_df.precision_j2c),
        reciprocal_block_size=reciprocal_block_size,
        auxiliary_block_size=auxiliary_block_size,
        profile_components=profile_components,
        overlap_short_long_range=overlap_short_long_range)


def _check_real_gamma_rsdf(with_df, dm0, dz):
    '''Validate the production RSDF stress domain before large allocation.'''
    if getattr(with_df, 'backend_name', None) != 'rsdf':
        raise TypeError('RSDF strain response requires an RSDF provider')
    kpts = np.asarray(with_df.kpts, dtype=float).reshape(-1, 3)
    if len(kpts) != 1 or np.max(np.abs(kpts)) > 1.e-12:
        raise NotImplementedError('GPU RSDF stress requires real Gamma point')
    if int(with_df.cell.dimension) != 3:
        raise NotImplementedError('GPU RSDF stress requires a 3D Cell')
    nao = int(with_df.cell.nao_nr())
    for name, value in (('dm0', dm0), ('dz', dz)):
        array = cp.asarray(value)
        if array.shape != (nao, nao):
            raise ValueError(f'{name} must have shape ({nao},{nao})')
        if cp.iscomplexobj(array):
            imaginary = float(cp.max(cp.abs(array.imag)).get())
            if imaginary > 1.e-10:
                raise NotImplementedError(
                    f'GPU RSDF stress requires real {name}')


__all__ = ['make_rsdf_strain_response']
