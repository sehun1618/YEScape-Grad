'''Gamma-point GDF CDERI value and analytic nuclear response.'''

import numpy as np
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder

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
from dcforge.cpu.pbc.df.gdf.metric import (
    auxiliary_metric,
    gen_grad_auxiliary_metric,
    grad_auxiliary_metric,
    metric_mesh,
)


def make_cderi_response(with_df, value_tol=1e-7):
    r'''Return central CDERI and :math:`C^\lambda` for a built GDF provider.

    For :math:`C=L^{-1}B` and :math:`M=LL^\dagger`, this function assembles

    .. math::

        C^\lambda = L^{-1}B^\lambda
          - L^{-1}L^\lambda C.

    Args:
        with_df : pyscf.pbc.df.GDF
            Built Gamma-point provider.  Its ``auxcell`` must be PySCF's
            charge-normalized ``make_modrho_basis`` cell; substituting a plain
            ``df.make_auxcell`` changes the auxiliary gauge and sign.
        value_tol : float
            Maximum allowed difference between independently reconstructed
            and provider-stored central CDERI.

    Returns:
        tuple
            ``(cderi, cderi1, cderi1_source, cderi1_metric)`` with shapes
            ``(naux,nao,nao)`` and ``(natm,3,naux,nao,nao)``.

    Notes:
        The central factor actually used by SCF is retained from provider
        storage.  Analytic derivatives are accepted only after the explicit
        source/metric reconstruction reproduces that factor in the same
        Cholesky gauge.  This detects a wrong auxcell, omega, mesh, sign, or
        metric convention before it can contaminate a gradient.
    '''
    data = _response_data(with_df, value_tol)
    cell, auxcell, nao = data[:3]
    omega, source_mesh, metric_mesh1 = data[9:]
    source1 = grad_three_center_source(
        cell, auxcell, omega, source_mesh,
        max_memory=with_df.max_memory)
    metric1 = grad_auxiliary_metric(
        cell, auxcell, omega, metric_mesh1)

    factor, reconstructed, central = data[5:8]
    del data
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
        _real_if_close(central),
        _real_if_close(cderi1),
        with_df.cderi1_source,
        with_df.cderi1_metric,
    )


def make_jk_response(with_df, dm, atmlst=None, blksize=None,
                     value_tol=1e-7):
    r'''Contract atom-stream GDF CDERI response into :math:`J^\lambda/K^\lambda`.

    Args:
        with_df : dcforge.cpu.pbc.df.gdf.GDF
            Built Gamma-point provider whose stored CDERI defines the central
            factor used by the source RHF calculation.
        dm : ndarray of shape (nao, nao)
            Spin-summed RHF density including occupation factor two.

    Kwargs:
        atmlst : sequence of int or None
            Nuclear centers required by the caller.  ``None`` means all atoms.
        blksize : int
            Number of atoms whose :math:`B^\lambda`, :math:`M^\lambda`, and
            :math:`C^\lambda` tensors may coexist.  ``None`` selects a
            dimension- and ``max_memory``-aware block of at most eight atoms.
        value_tol : float
            Stored/reconstructed central-CDERI parity threshold.

    Returns:
        tuple
            ``(cderi, jk1, jk1_source, jk1_metric)``.  Each JK tuple contains
            Coulomb and exchange derivatives with shape
            ``(natm,3,nao,nao)``.  Entries outside ``atmlst`` are zero.

    Notes:
        This is the production ordering.  It keeps the central Cholesky factor
        but discards every atom block of :math:`C_B^\lambda` and
        :math:`C_M^\lambda` immediately after the J/K contraction.  The full
        :func:`make_cderi_response` remains the small-system numerical oracle.
    '''
    data = _response_data(with_df, value_tol)
    cell, auxcell, nao, _source, _metric = data[:5]
    factor, reconstructed, central = data[5:8]
    omega, source_mesh, metric_mesh1 = data[9:]
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
        cell, auxcell, omega, source_mesh,
        max_memory=with_df.max_memory)
    with_df.fourier_response_aux_storage = (
        source1_gen.fourier_aux_storage)
    with_df.fourier_response_ao_pair_bytes = int(
        source1_gen.fourier_ao_pair_bytes)
    metric1_gen = gen_grad_auxiliary_metric(
        cell, auxcell, omega, metric_mesh1)

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
    r'''Contract the GDF response directly into resolved gradient terms.

    The provider-owned central CDERI is authoritative.  Production rebuilds
    only the auxiliary metric factor and streams :math:`B^\lambda` and
    :math:`M^\lambda` into reverse adjoints.  ``validate_central=True`` also
    rebuilds zero-order :math:`B` to verify the provider gauge; the full and
    atom-stream reference paths always perform that diagnostic reconstruction.
    '''
    data = _response_data(
        with_df, value_tol, validate_central=validate_central)
    cell, auxcell = data[:2]
    factor, _reconstructed, central = data[5:8]
    omega, source_mesh, metric_mesh1 = data[9:]
    del data, _reconstructed
    source1_gen = gen_grad_three_center_source(
        cell, auxcell, omega, source_mesh,
        max_memory=with_df.max_memory)
    metric1_gen = gen_grad_auxiliary_metric(
        cell, auxcell, omega, metric_mesh1)
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
    '''Return ``omega``, source mesh, and metric mesh used by PySCF GDF.'''
    _check_provider(with_df)
    if with_df.auxcell is None:
        with_df.build(j_only=False)
    builder = _RSGDFBuilder(
        with_df.cell, with_df.auxcell, np.zeros((1, 3)))
    if with_df.mesh is not None:
        builder.mesh = np.asarray(with_df.mesh, dtype=int)
    builder.linear_dep_threshold = float(with_df.linear_dep_threshold)
    builder.build()
    return (
        float(builder.omega),
        np.asarray(builder.mesh, dtype=int),
        metric_mesh(with_df.cell, with_df.auxcell, builder.omega),
    )


def _response_data(with_df, value_tol, validate_central=True):
    '''Build one central GDF factor and verify its provider representation.'''
    _check_provider(with_df)
    if getattr(with_df, '_cderi', None) is None or with_df.auxcell is None:
        with_df.build(j_only=False)
    omega, source_mesh, metric_mesh1 = response_settings(with_df)
    cell = with_df.cell
    auxcell = with_df.auxcell
    nao = cell.nao_nr()
    metric = auxiliary_metric(cell, auxcell, omega, metric_mesh1)
    central = load_cderi(with_df)
    if validate_central:
        source = three_center_source(
            cell, auxcell, omega, source_mesh,
            max_memory=with_df.max_memory)
        factor, reconstructed = cholesky_cderi(source, metric, nao)
        parity = float(np.max(np.abs(reconstructed - central)))
        if parity > float(value_tol):
            raise RuntimeError(
                'analytic GDF source/metric convention does not reproduce '
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
        parity,
        omega, source_mesh, metric_mesh1,
    )


def _check_provider(with_df):
    kpts = np.asarray(with_df.kpts).reshape(-1, 3)
    if len(kpts) != 1 or np.max(np.abs(kpts)) > 1e-10:
        raise NotImplementedError('only Gamma-point GDF response is implemented')
    if with_df.cell.dimension != 3:
        raise NotImplementedError(
            'only fixed three-dimensional cells are supported')
    if getattr(with_df, 'j2c_eig_always', False):
        raise NotImplementedError(
            'GDF response requires Cholesky metric factorization')


def _real_if_close(value, tol=1e-10):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tol:
        return value.real
    return value
