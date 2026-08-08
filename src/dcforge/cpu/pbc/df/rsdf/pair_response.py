'''Sampled-k RSDF source, metric and CDERI nuclear response.'''

import numpy as np

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
from dcforge.cpu.pbc.df.rsdf.metric import (
    auxiliary_metric_kpts,
    grad_auxiliary_metric_kpts,
)


def response_settings_kpts(with_df):
    '''Return independently resolved sampled-k RSDF source/metric settings.'''
    _check_provider(with_df)
    if getattr(with_df, '_cderi', None) is None or with_df.auxcell is None:
        with_df.build(j_only=False)
    required = ('omega', 'mesh_compact', 'omega_j2c', 'mesh_j2c')
    missing = [key for key in required if getattr(with_df, key, None) is None]
    if missing:
        raise RuntimeError(
            'sampled-k RSDF settings are unresolved: ' + ', '.join(missing))
    return (
        float(with_df.omega), np.asarray(with_df.mesh_compact, dtype=int),
        float(with_df.omega_j2c), np.asarray(with_df.mesh_j2c, dtype=int),
    )


def make_pair_cderi_response(
        with_df, pair, atmlst=None, value_tol=1e-7, settings=None,
        central=None):
    r'''Return one sampled-k RSDF CDERI pair and separated derivatives.

    For ``q=k_j-k_i`` the source uses RSDF's resolved
    ``omega/mesh_compact/precision_R`` while the metric independently uses
    ``omega_j2c/mesh_j2c``.  The common Cholesky derivative is

    ``C^lambda=L^-1 B^lambda-L^-1 L^lambda C``.
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
    source_omega, source_mesh, metric_omega, metric_mesh = settings
    if central is None:
        central = load_cderi_pairs(with_df, ((ki, kj),))[(ki, kj)]

    cell = with_df.cell
    auxcell = with_df.auxcell
    nao = cell.nao_nr()
    kpti = kpts[ki]
    kptj = kpts[kj]
    q = kptj - kpti
    source = three_center_source_kpts(
        cell, auxcell, source_omega, source_mesh, kpti, kptj,
        precision=with_df.precision_R, max_memory=with_df.max_memory)
    metric = auxiliary_metric_kpts(
        cell, auxcell, metric_omega, metric_mesh, q)
    factor, reconstructed = cholesky_cderi(source, metric, nao)
    parity = float(np.max(np.abs(reconstructed - central)))
    if parity > float(value_tol):
        raise RuntimeError(
            'sampled-k analytic RSDF source/metric convention does not '
            f'reproduce pair {(ki, kj)} storage: '
            f'{parity:.3e} > {float(value_tol):.3e}')

    source1 = grad_three_center_source_kpts(
        cell, auxcell, source_omega, source_mesh, kpti, kptj,
        precision=with_df.precision_R, atmlst=atmlst,
        max_memory=with_df.max_memory)
    metric1 = grad_auxiliary_metric_kpts(
        cell, auxcell, metric_omega, metric_mesh, q, atmlst)
    # The provider-stored factor is the authoritative zero-order object used
    # by source SCF and target energy.  ``reconstructed`` is only a parity
    # witness for the independently generated B/M convention.  Inserting the
    # witness into ``-L^-1 L^lambda C`` differentiates a nearby reconstructed
    # functional instead of the provider's actual finite representation; a
    # sub-nanohartree central gap can then leave a microhartree/Bohr response
    # error.  Gamma contracted response already follows this ownership rule.
    cderi1_source, cderi1_metric = grad_cderi_with_factor(
        factor, central, source1, metric1, nao)
    info = {
        'pair': (ki, kj), 'q': q.copy(),
        'source_omega': float(source_omega),
        'source_mesh': np.asarray(source_mesh, dtype=int).copy(),
        'metric_omega': float(metric_omega),
        'metric_mesh': np.asarray(metric_mesh, dtype=int).copy(),
        'value_parity_max_abs': parity,
    }
    return (
        np.asarray(central),
        np.asarray(cderi1_source + cderi1_metric),
        np.asarray(cderi1_source), np.asarray(cderi1_metric), info,
    )


def make_cderi_response_kpts(
        with_df, atmlst=None, value_tol=1e-7, include_total=True):
    '''Return central and derivative maps for all canonical RSDF pairs.

    ``include_total=False`` avoids retaining the redundant
    ``source1 + metric1`` map when a caller contracts the two resolved terms
    separately.  Source and metric values, equations and pair ordering are
    unchanged.
    '''
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
    if getattr(with_df, 'backend_name', None) != 'rsdf':
        raise TypeError('sampled-k RSDF response requires dcforge RSDF')
    if with_df.cell.dimension != 3:
        raise NotImplementedError(
            'sampled-k RSDF response requires a fixed 3D Cell')
    if getattr(with_df, 'j2c_eig_always', False):
        raise NotImplementedError(
            'sampled-k RSDF response requires Cholesky metric factorization')
    kpts = np.asarray(with_df.kpts).reshape(-1, 3)
    if len(kpts) < 1:
        raise ValueError('sampled-k RSDF provider has no k-points')
