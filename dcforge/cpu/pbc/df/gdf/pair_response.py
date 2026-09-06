'''Sampled-k GDF source, metric and CDERI nuclear response.'''

import numpy as np
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder

from dcforge.cpu.pbc.df.gaussian.cderi import (
    cholesky_cderi,
    grad_cderi_with_factor,
)
from dcforge.cpu.pbc.df.gaussian.source import (
    grad_three_center_source_kpts,
    three_center_source_kpts,
)
from dcforge.cpu.pbc.df.gaussian.storage import (
    canonical_kpt_pairs,
    load_cderi_pairs,
)
from dcforge.cpu.pbc.df.gdf.metric import (
    auxiliary_metric,
    grad_auxiliary_metric,
    metric_mesh,
)


def response_settings_kpts(with_df):
    '''Return the PySCF GDF omega, source mesh and metric mesh.'''
    _check_provider(with_df)
    if with_df.auxcell is None or getattr(with_df, '_cderi', None) is None:
        with_df.build(j_only=False)
    if with_df.resolved_omega is not None:
        return (
            float(with_df.resolved_omega),
            np.asarray(with_df.resolved_source_mesh, dtype=int).copy(),
            np.asarray(with_df.resolved_metric_mesh, dtype=int).copy(),
        )
    kpts = np.asarray(with_df.kpts, dtype=float).reshape(-1, 3)
    builder = _RSGDFBuilder(with_df.cell, with_df.auxcell, kpts)
    if with_df.mesh is not None:
        builder.mesh = np.asarray(with_df.mesh, dtype=int)
    builder.linear_dep_threshold = float(with_df.linear_dep_threshold)
    builder.build()
    return (
        float(builder.omega), np.asarray(builder.mesh, dtype=int),
        metric_mesh(with_df.cell, with_df.auxcell, builder.omega),
    )


def make_pair_cderi_response(
        with_df, pair, atmlst=None, value_tol=1e-7, settings=None,
        central=None):
    r'''Return ``C(ki,kj)`` and its separated analytic derivatives.

    For momentum transfer ``q=k_j-k_i``, the provider factor obeys

    ``C = L(q)^-1 B(ki,kj)``, ``M(q)=L(q)L(q)^dagger``.

    Its derivative is split into the three-center source and auxiliary-
    metric terms.  Reconstructed central values are only a convention check;
    derivatives are applied to the provider-authoritative stored CDERI.
    '''
    _check_provider(with_df)
    if getattr(with_df, '_cderi', None) is None or with_df.auxcell is None:
        with_df.build(j_only=False)
    kpts = np.asarray(with_df.kpts, dtype=float).reshape(-1, 3)
    ki, kj = (int(pair[0]), int(pair[1]))
    if not (0 <= kj <= ki < len(kpts)):
        raise ValueError(
            'canonical k-point pair must satisfy 0 <= kj <= ki < nkpts')
    if atmlst is None:
        atmlst = range(with_df.cell.natm)
    atmlst = [int(ia) for ia in atmlst]
    if settings is None:
        settings = response_settings_kpts(with_df)
    omega, source_mesh, metric_mesh1 = settings
    if central is None:
        central = load_cderi_pairs(with_df, ((ki, kj),))[(ki, kj)]

    cell = with_df.cell
    auxcell = with_df.auxcell
    nao = cell.nao_nr()
    kpti = kpts[ki]
    kptj = kpts[kj]
    q = kptj - kpti
    source = three_center_source_kpts(
        cell, auxcell, omega, source_mesh, kpti, kptj,
        max_memory=with_df.max_memory)
    metric = auxiliary_metric(
        cell, auxcell, omega, metric_mesh1, q)
    factor, reconstructed = cholesky_cderi(source, metric, nao)
    parity = float(np.max(np.abs(reconstructed - central)))
    if parity > float(value_tol):
        raise RuntimeError(
            'sampled-k analytic GDF source/metric convention does not '
            f'reproduce pair {(ki, kj)} storage: '
            f'{parity:.3e} > {float(value_tol):.3e}')

    source1 = grad_three_center_source_kpts(
        cell, auxcell, omega, source_mesh, kpti, kptj,
        atmlst=atmlst, max_memory=with_df.max_memory)
    metric1 = grad_auxiliary_metric(
        cell, auxcell, omega, metric_mesh1, q, atmlst)
    cderi1_source, cderi1_metric = grad_cderi_with_factor(
        factor, central, source1, metric1, nao)
    info = {
        'pair': (ki, kj), 'q': q.copy(), 'omega': float(omega),
        'source_mesh': np.asarray(source_mesh, dtype=int).copy(),
        'metric_mesh': np.asarray(metric_mesh1, dtype=int).copy(),
        'value_parity_max_abs': parity,
    }
    return (
        np.asarray(central),
        np.asarray(cderi1_source + cderi1_metric),
        np.asarray(cderi1_source), np.asarray(cderi1_metric), info,
    )


def make_cderi_response_kpts(
        with_df, atmlst=None, value_tol=1e-7, include_total=True):
    '''Return central and derivative maps for all canonical k-point pairs.'''
    _check_provider(with_df)
    if getattr(with_df, '_cderi', None) is None or with_df.auxcell is None:
        with_df.build(j_only=False)
    kpts = np.asarray(with_df.kpts, dtype=float).reshape(-1, 3)
    pairs = canonical_kpt_pairs(len(kpts))
    central = load_cderi_pairs(with_df, pairs)
    settings = response_settings_kpts(with_df)
    cderi1 = {} if include_total else None
    source1 = {}
    metric1 = {}
    records = []
    parity = 0.0
    for pair in pairs:
        values = make_pair_cderi_response(
            with_df, pair, atmlst=atmlst, value_tol=value_tol,
            settings=settings, central=central[pair])
        _central, total, source, metric, info = values
        if include_total:
            cderi1[pair] = total
        source1[pair] = source
        metric1[pair] = metric
        records.append(info)
        parity = max(parity, info['value_parity_max_abs'])
    with_df.cderi_value_parity_max_abs = parity
    with_df.cderi_central_validated = True
    with_df.cderi_response_storage = 'canonical_pair_full'
    with_df.cderi_response_blksize = len(pairs)
    with_df.cderi_response_batch_bytes = int(sum(
        value.nbytes for mapping in (
            (source1, metric1) if cderi1 is None
            else (cderi1, source1, metric1))
        for value in mapping.values()))
    with_df.cderi_pair_response_records = tuple(records)
    return central, cderi1, source1, metric1


def _check_provider(with_df):
    if with_df.cell.dimension != 3:
        raise NotImplementedError(
            'sampled-k GDF response requires a fixed 3D Cell')
    if getattr(with_df, 'j2c_eig_always', False):
        raise NotImplementedError(
            'sampled-k GDF response requires Cholesky metric factorization')
    kpts = np.asarray(with_df.kpts).reshape(-1, 3)
    if len(kpts) < 1:
        raise ValueError('sampled-k GDF provider has no k-points')
