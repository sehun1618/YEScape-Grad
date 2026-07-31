'''Gamma-point public-RSDF CDERI value and analytic response.'''

import numpy as np

from dcforge.cpu.pbc.df.gaussian.adjoint import make_contracted_response
from dcforge.cpu.pbc.df.gaussian.cderi import (
    cholesky_cderi,
    factor_metric,
    grad_cderi_with_factor,
)
from dcforge.cpu.pbc.df.gaussian.jk import grad_jk
from dcforge.cpu.pbc.df.gaussian.response import response_blocks
from dcforge.cpu.pbc.df.gaussian.source import (
    gen_grad_three_center_source,
    grad_three_center_source,
    three_center_source,
)
from dcforge.cpu.pbc.df.gaussian.storage import load_cderi
from dcforge.cpu.pbc.df.rsdf.metric import (
    auxiliary_metric,
    gen_grad_auxiliary_metric,
    grad_auxiliary_metric,
)


def make_cderi_response(with_df, value_tol=1e-7):
    r'''Return central RSDF CDERI and analytic :math:`C^\lambda`.

    Source and metric range-separation parameters are intentionally distinct:
    ``omega/mesh_compact`` define :math:`B`, whereas
    ``omega_j2c/mesh_j2c`` define :math:`M`.  They must never be silently
    replaced by one common setting.
    '''
    data = _response_data(with_df, value_tol)
    cell, auxcell, nao = data[:3]
    factor, reconstructed, central = data[5:8]
    source_omega, source_mesh, metric_omega, metric_mesh = data[9:]
    del data
    source1 = grad_three_center_source(
        cell, auxcell, source_omega, source_mesh, with_df.precision_R,
        max_memory=with_df.max_memory)
    metric1 = grad_auxiliary_metric(
        cell, auxcell, metric_omega, metric_mesh)
    cderi1_source, cderi1_metric = grad_cderi_with_factor(
        factor, reconstructed, source1, metric1, nao)
    cderi1 = cderi1_source + cderi1_metric

    with_df.cderi_response_storage = 'full'
    with_df.cderi_response_blksize = cell.natm
    with_df.cderi_response_batch_bytes = int(
        source1.nbytes + metric1.nbytes
        + cderi1_source.nbytes + cderi1_metric.nbytes + cderi1.nbytes)
    with_df.cderi1_source = _real_if_close(cderi1_source)
    with_df.cderi1_metric = _real_if_close(cderi1_metric)
    return (
        _real_if_close(central), _real_if_close(cderi1),
        with_df.cderi1_source, with_df.cderi1_metric,
    )


def make_jk_response(with_df, dm, atmlst=None, blksize=None,
                     value_tol=1e-7):
    r'''Contract atom-stream RSDF CDERI response into J/K derivatives.

    ``omega/mesh_compact`` generate each atom block of
    :math:`B^\lambda`; ``omega_j2c/mesh_j2c`` independently generate
    :math:`M^\lambda`.  The block is transformed into
    :math:`C_B^\lambda+C_M^\lambda`, contracted, and discarded before the
    next block.

    Args:
        with_df : dcforge.cpu.pbc.df.rsdf.RSDF
            Built Gamma-point RSDF provider.
        dm : ndarray of shape (nao, nao)
            Spin-summed RHF density.

    Kwargs:
        atmlst : sequence of int or None
            Requested atom indices.
        blksize : int
            Number of atoms retained in one CDERI response block.  ``None``
            selects a dimension- and ``max_memory``-aware block of at most
            eight atoms.
        value_tol : float
            Central stored/reconstructed CDERI parity threshold.

    Returns:
        tuple
            ``(cderi, jk1, jk1_source, jk1_metric)`` using the same contract
            as GDF :func:`make_jk_response`.
    '''
    data = _response_data(with_df, value_tol)
    cell, auxcell, nao, _source, _metric = data[:5]
    factor, reconstructed, central = data[5:8]
    source_omega, source_mesh, metric_omega, metric_mesh = data[9:]
    del data, _source, _metric
    shape = (cell.natm, 3, nao, nao)
    dtype = np.result_type(central, dm)
    atmlst, blksize = response_blocks(
        cell, atmlst, blksize, nao, auxcell.nao_nr(), dtype,
        with_df.max_memory)
    vj1_source = np.zeros(shape, dtype=dtype)
    vk1_source = np.zeros(shape, dtype=dtype)
    vj1_metric = np.zeros(shape, dtype=dtype)
    vk1_metric = np.zeros(shape, dtype=dtype)
    batch_bytes = 0
    source1_gen = gen_grad_three_center_source(
        cell, auxcell, source_omega, source_mesh, with_df.precision_R,
        max_memory=with_df.max_memory)
    with_df.fourier_response_aux_storage = (
        source1_gen.fourier_aux_storage)
    with_df.fourier_response_ao_pair_bytes = int(
        source1_gen.fourier_ao_pair_bytes)
    metric1_gen = gen_grad_auxiliary_metric(
        cell, auxcell, metric_omega, metric_mesh)

    for p0 in range(0, len(atmlst), blksize):
        atoms = atmlst[p0:p0 + blksize]
        source1 = source1_gen(atoms)
        metric1 = metric1_gen(atoms)
        cderi1_source, cderi1_metric = grad_cderi_with_factor(
            factor, reconstructed, source1, metric1, nao)
        vj_source, vk_source = grad_jk(central, cderi1_source, dm)
        vj_metric, vk_metric = grad_jk(central, cderi1_metric, dm)
        vj1_source[atoms] = vj_source
        vk1_source[atoms] = vk_source
        vj1_metric[atoms] = vj_metric
        vk1_metric[atoms] = vk_metric
        batch_bytes = max(batch_bytes, sum(array.nbytes for array in (
            source1, metric1, cderi1_source, cderi1_metric,
            vj_source, vk_source, vj_metric, vk_metric)))

    with_df.cderi_response_storage = 'atom_stream'
    with_df.cderi_response_blksize = blksize
    with_df.cderi_response_batch_bytes = int(batch_bytes)
    with_df.cderi1_source = None
    with_df.cderi1_metric = None
    jk1_source = (_real_if_close(vj1_source), _real_if_close(vk1_source))
    jk1_metric = (_real_if_close(vj1_metric), _real_if_close(vk1_metric))
    jk1 = (
        _real_if_close(vj1_source + vj1_metric),
        _real_if_close(vk1_source + vk1_metric),
    )
    return _real_if_close(central), jk1, jk1_source, jk1_metric


def make_gradient_response(with_df, dm0, dz, atmlst=None, blksize=None,
                           validate_central=False, value_tol=1e-7):
    r'''Contract the RSDF response directly into resolved gradient terms.

    Source and metric retain their independent RSDF omega/mesh ownership.  The
    provider central CDERI and reconstructed metric factor define the reverse
    adjoints; no full CDERI, J, or K derivative is retained.
    '''
    data = _response_data(
        with_df, value_tol, validate_central=validate_central)
    cell, auxcell = data[:2]
    factor, _reconstructed, central = data[5:8]
    source_omega, source_mesh, metric_omega, metric_mesh = data[9:]
    del data, _reconstructed
    source1_gen = gen_grad_three_center_source(
        cell, auxcell, source_omega, source_mesh, with_df.precision_R,
        max_memory=with_df.max_memory)
    metric1_gen = gen_grad_auxiliary_metric(
        cell, auxcell, metric_omega, metric_mesh)
    terms, blksize, peak_bytes = make_contracted_response(
        cell, auxcell, central, factor, source1_gen, metric1_gen,
        dm0, dz, atmlst=atmlst, blksize=blksize,
        max_memory=with_df.max_memory)
    with_df.cderi_response_storage = 'contracted'
    with_df.cderi_response_blksize = blksize
    with_df.cderi_response_batch_bytes = int(peak_bytes)
    with_df.cderi1_source = None
    with_df.cderi1_metric = None
    with_df.fourier_response_aux_storage = (
        source1_gen.fourier_aux_storage)
    with_df.fourier_response_ao_pair_bytes = int(
        source1_gen.fourier_ao_pair_bytes)
    return terms


def response_settings(with_df):
    '''Return source omega/mesh and independent metric omega/mesh.'''
    required = ('omega', 'mesh_compact', 'omega_j2c', 'mesh_j2c')
    missing = [key for key in required if getattr(with_df, key, None) is None]
    if missing:
        raise RuntimeError(
            'build RSDF before requesting response settings: '
            + ', '.join(missing))
    return (
        float(with_df.omega), np.asarray(with_df.mesh_compact, dtype=int),
        float(with_df.omega_j2c), np.asarray(with_df.mesh_j2c, dtype=int),
    )


def _response_data(with_df, value_tol, validate_central=True):
    '''Build one central RSDF factor and verify provider parity.'''
    _check_provider(with_df)
    if getattr(with_df, '_cderi', None) is None or with_df.auxcell is None:
        with_df.build(j_only=False)
    source_omega, source_mesh, metric_omega, metric_mesh = response_settings(
        with_df)
    cell = with_df.cell
    auxcell = with_df.auxcell
    nao = cell.nao_nr()
    metric = auxiliary_metric(
        cell, auxcell, metric_omega, metric_mesh)
    central = load_cderi(with_df)
    if validate_central:
        source = three_center_source(
            cell, auxcell, source_omega, source_mesh, with_df.precision_R,
            max_memory=with_df.max_memory)
        factor, reconstructed = cholesky_cderi(source, metric, nao)
        parity = float(np.max(np.abs(reconstructed - central)))
        if parity > float(value_tol):
            raise RuntimeError(
                'analytic RSDF source/metric convention does not reproduce '
                f'PySCF CDERI storage: {parity:.3e} > {float(value_tol):.3e}')
    else:
        source = None
        factor = factor_metric(metric)
        reconstructed = central
        parity = None
    with_df.cderi_value_parity_max_abs = parity
    with_df.cderi_central_validated = bool(validate_central)
    return (
        cell, auxcell, nao, source, metric, factor, reconstructed, central,
        parity, source_omega, source_mesh, metric_omega, metric_mesh,
    )


def _check_provider(with_df):
    if getattr(with_df, 'backend_name', None) != 'rsdf':
        raise TypeError('RSDF response requires dcforge RSDF provider')
    kpts = np.asarray(with_df.kpts).reshape(-1, 3)
    if len(kpts) != 1 or np.max(np.abs(kpts)) > 1e-10:
        raise NotImplementedError('only Gamma-point RSDF response is implemented')
    if with_df.cell.dimension != 3:
        raise NotImplementedError(
            'only fixed three-dimensional cells are supported')
    if getattr(with_df, 'j2c_eig_always', False):
        raise NotImplementedError(
            'RSDF response requires Cholesky metric factorization')


def _real_if_close(value, tol=1e-10):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tol:
        return value.real
    return value
