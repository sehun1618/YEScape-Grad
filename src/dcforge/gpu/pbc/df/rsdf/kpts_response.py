'''Explicit sampled-k RSDF derivative oracle for the GPU workflow.'''

import cupy as cp
import numpy as np
from pyscf import lib

from dcforge.cpu.pbc.df.gaussian.jk import (
    grad_ewald_exchange_kpts,
    grad_jk_kpts,
)
from dcforge.cpu.pbc.df.rsdf import RSDF as CPURSDF
from dcforge.cpu.pbc.df.rsdf import make_cderi_response_kpts
from dcforge.cpu.pbc.grad.krhf_nv import make_s1_ao
from dcforge.gpu.pbc.df.gaussian.kpts_response import (
    make_native_gradient_response,
)
from dcforge.gpu.pbc.df.rsdf.rsdf import RSDF


def make_kpts_full_coulomb_response_probe(
        mf_hf, dm0, dz, atmlst=None):
    r'''Return a GPU-native full-Coulomb partition-independence probe.

    The provider-resolved source ``omega`` and BvK mesh construct one
    GPU4PySCF range-separated integral optimizer.  Each native observable is
    requested for the complete Coulomb operator (``omega=0``), so changing
    the internal SR/LR partition does not change the physical Hamiltonian.
    RSDF's independently resolved ``omega_j2c`` remains authoritative for
    its central SCF factors.  The native result is compared with the
    CPU-oracle and finite-difference routes only as a diagnostic; agreement
    is not assumed.

    GPU4PySCF exposes only the combined source, metric, and Ewald derivative
    from this endpoint.  It uses one common range separator internally and
    therefore does *not* differentiate the finite RSDF representation whose
    central source and metric have independent settings.  This function is a
    diagnostic continuum/full-Coulomb probe, not the production RSDF
    derivative.  The resolved functions below retain the provider-exact
    response.

    Args:
        mf_hf : gpu4pyscf.pbc.scf.khf.KRHF
            Converged sampled-k source KRHF with a DC-Forge RSDF provider.
        dm0, dz : cupy.ndarray of shape (nkpts,nao,nao)
            Hermitian source and Z-transition densities.

    Kwargs:
        atmlst : sequence of int or None
            Requested nuclear centers.

    Returns:
        tuple
            Native device scalar terms and response provenance metadata.
    '''
    provider = mf_hf.with_df
    if not isinstance(provider, RSDF):
        raise TypeError('sampled-k RSDF response requires GPU RSDF')
    if provider.auxcell is None or provider._cderi is None:
        provider.build(j_only=False)
    terms, metadata = make_native_gradient_response(
        mf_hf, dm0, dz, provider.omega, provider.kmesh,
        atmlst=atmlst, backend_name='rsdf')
    settings = provider.response_settings()
    metadata.update({
        'source_omega': settings['source_omega'],
        'source_mesh': settings['source_mesh'],
        'metric_omega': settings['metric_omega'],
        'metric_mesh': settings['metric_mesh'],
        'central_representation': (
            'independent RSDF source/metric parameters'),
        'native_observable': 'complete omega=0 Coulomb operator',
        'production_eligible': False,
        'disqualification': (
            'one common native range split does not differentiate the '
            'provider central RSDF source/metric finite representation'),
    })
    return terms, metadata


def make_kpts_contracted_gradient_response(
        with_df, dm0, dz, s1, atmlst=None, exxdiv='ewald'):
    r'''Return scalar sampled-k RSDF response terms with bounded device state.

    This is the K8.7 production ordering.  The CPU correctness provider still
    builds canonical q-pair CDERI derivatives, but each source or metric J/K
    matrix response is immediately contracted with ``P`` and ``D_Z`` and
    released.  Only resolved ``(natm_selected,3)`` scalars cross to the GPU.
    The full-matrix :func:`make_kpts_gradient_response` remains the numerical
    oracle.

    Args:
        with_df : dcforge.gpu.pbc.df.rsdf.RSDF
            Built sampled-k GPU provider.
        dm0, dz : cupy.ndarray of shape (nkpts,nao,nao)
            Source and Hermitian Z-transition densities.
        s1 : cupy.ndarray of shape (nkpts,3,nao,nao)
            AO-row overlap derivative for Ewald exchange.

    Kwargs:
        atmlst : sequence of int or None
            Requested nuclear centers.  The CPU primitive is currently built
            for all atoms, but only these scalar rows are transferred.
        exxdiv : {'ewald', None}
            Source exact-exchange divergence convention.

    Returns:
        tuple
            Device scalar term dictionary and transfer/storage metadata.
    '''
    if not isinstance(with_df, RSDF):
        raise TypeError('sampled-k RSDF response requires GPU RSDF')
    if exxdiv not in ('ewald', None):
        raise NotImplementedError(
            'sampled-k RSDF response supports exxdiv=None or ewald')
    dm0_host = _density_to_host(with_df, dm0, 'dm0')
    dz_host = _density_to_host(with_df, dz, 'dz')
    s1_host = cp.asnumpy(cp.asarray(s1))
    atoms = np.arange(with_df.cell.natm, dtype=int)
    if atmlst is not None:
        atoms = np.asarray([int(ia) for ia in atmlst], dtype=int)

    witness = _cpu_witness(with_df)
    cderi, _total1, source1, metric1 = make_cderi_response_kpts(
        witness, include_total=False)
    nkpts = len(with_df.kpts)
    host_terms = {}
    peak_matrix_bytes = 0
    for suffix, response1 in (
            ('source', source1), ('metric', metric1)):
        vj1, vk1 = grad_jk_kpts(cderi, response1, dm0_host)
        vj1_z, vk1_z = grad_jk_kpts(cderi, response1, dz_host)
        peak_matrix_bytes = max(peak_matrix_bytes, sum(
            value.nbytes for value in (vj1, vk1, vj1_z, vk1_z)))
        vhf1 = vj1 - .5 * vk1
        vhf1_z = vj1_z - .5 * vk1_z
        host_terms[f'de_j_{suffix}'] = (
            .5 * lib.einsum('axkij,kji->ax', vj1, dm0_host).real
            / nkpts)
        host_terms[f'de_fock_{suffix}'] = (
            -2 * lib.einsum('axkij,kji->ax', vhf1, dz_host).real
            / nkpts)
        host_terms[f'de_fock_{suffix}_reverse'] = (
            -2 * lib.einsum(
                'axkij,kji->ax', vhf1_z, dm0_host).real / nkpts)
        del vj1, vk1, vj1_z, vk1_z, vhf1, vhf1_z

    zero = np.zeros((with_df.cell.natm, 3))
    de_ewald = zero.copy()
    de_ewald_reverse = zero.copy()
    ewald_matrix_bytes = 0
    if exxdiv == 'ewald':
        s1ao = make_s1_ao(with_df.cell, s1_host)
        vk1 = grad_ewald_exchange_kpts(
            with_df.cell, with_df.kpts, dm0_host, s1ao)
        vk1_z = grad_ewald_exchange_kpts(
            with_df.cell, with_df.kpts, dz_host, s1ao)
        ewald_matrix_bytes = int(vk1.nbytes + vk1_z.nbytes)
        de_ewald = (
            lib.einsum('axkij,kji->ax', vk1, dz_host).real / nkpts)
        de_ewald_reverse = (
            lib.einsum(
                'axkij,kji->ax', vk1_z, dm0_host).real / nkpts)
    host_terms['de_fock_ewald'] = de_ewald
    host_terms['de_fock_ewald_reverse'] = de_ewald_reverse

    selected = {
        name: cp.asarray(value[atoms])
        for name, value in host_terms.items()
    }
    settings = witness.response_settings()
    derivative_shape = (
        with_df.cell.natm, 3, len(with_df.kpts),
        with_df.cell.nao_nr(), with_df.cell.nao_nr())
    full_matrix_bytes = int(
        np.prod(derivative_shape) * np.dtype(np.complex128).itemsize)
    metadata = {
        'response_backend': 'dcforge_cpu_rsdf_contracted_oracle',
        'response_storage': 'contracted_scalar',
        'scientific_host_fallback': True,
        'host_input_bytes': int(
            dm0_host.nbytes + dz_host.nbytes + s1_host.nbytes),
        'host_result_bytes': int(sum(
            value[atoms].nbytes for value in host_terms.values())),
        'host_workspace_bytes': int(
            getattr(witness, 'cderi_response_batch_bytes', 0)),
        'host_peak_jk_matrix_bytes_estimate': int(max(
            peak_matrix_bytes, ewald_matrix_bytes)),
        'device_full_matrix_bytes_avoided': 10 * full_matrix_bytes,
        'central_value_parity_max_abs': float(
            witness.cderi_value_parity_max_abs),
        'source_omega': settings['source_omega'],
        'source_mesh': settings['source_mesh'],
        'metric_omega': settings['metric_omega'],
        'metric_mesh': settings['metric_mesh'],
        'exxdiv': exxdiv,
    }
    return selected, metadata


def make_kpts_gradient_response(with_df, dm0, dz, s1, exxdiv='ewald'):
    r'''Return source/metric-resolved sampled-k RSDF matrix derivatives.

    GPU4PySCF 1.8.0 has no sampled-k RSDF nuclear-response endpoint that
    preserves independent source ``(omega,mesh_compact)`` and metric
    ``(omega_j2c,mesh_j2c)`` settings.  K8.5 therefore uses the validated
    DC-Forge CPU sampled-k RSDF response as an explicit correctness oracle.
    Source KRHF, central RSDF J/K, target energy, CPHF and Z-vector remain on
    the GPU.  Density inputs and derivative matrices crossing this boundary
    are counted in the returned metadata.

    Args:
        with_df : dcforge.gpu.pbc.df.rsdf.RSDF
            Built sampled-k GPU provider.
        dm0, dz : cupy.ndarray of shape (nkpts,nao,nao)
            Source and Z-transition densities.
        s1 : cupy.ndarray of shape (nkpts,3,nao,nao)
            AO-row overlap derivative for the Ewald exchange correction.

    Kwargs:
        exxdiv : {'ewald', None}
            Source exact-exchange divergence convention.

    Returns:
        tuple
            Device derivative matrices and provenance metadata.
    '''
    if not isinstance(with_df, RSDF):
        raise TypeError('sampled-k RSDF response requires GPU RSDF')
    if exxdiv not in ('ewald', None):
        raise NotImplementedError(
            'sampled-k RSDF response supports exxdiv=None or ewald')
    dm0_host = _density_to_host(with_df, dm0, 'dm0')
    dz_host = _density_to_host(with_df, dz, 'dz')
    s1_host = cp.asnumpy(cp.asarray(s1))

    witness = _cpu_witness(with_df)
    cderi, _total1, source1, metric1 = make_cderi_response_kpts(
        witness)
    vj_source, vk_source = grad_jk_kpts(cderi, source1, dm0_host)
    vj_metric, vk_metric = grad_jk_kpts(cderi, metric1, dm0_host)
    vj_source_z, vk_source_z = grad_jk_kpts(cderi, source1, dz_host)
    vj_metric_z, vk_metric_z = grad_jk_kpts(cderi, metric1, dz_host)

    vk_ewald = np.zeros_like(vk_source)
    vk_ewald_z = np.zeros_like(vk_source_z)
    if exxdiv == 'ewald':
        s1ao = make_s1_ao(with_df.cell, s1_host)
        vk_ewald = grad_ewald_exchange_kpts(
            with_df.cell, with_df.kpts, dm0_host, s1ao)
        vk_ewald_z = grad_ewald_exchange_kpts(
            with_df.cell, with_df.kpts, dz_host, s1ao)

    host_arrays = {
        'vj1_source': np.asarray(vj_source),
        'vk1_source': np.asarray(vk_source),
        'vj1_metric': np.asarray(vj_metric),
        'vk1_metric': np.asarray(vk_metric),
        'vk1_ewald': np.asarray(vk_ewald),
        'vj1_source_z': np.asarray(vj_source_z),
        'vk1_source_z': np.asarray(vk_source_z),
        'vj1_metric_z': np.asarray(vj_metric_z),
        'vk1_metric_z': np.asarray(vk_metric_z),
        'vk1_ewald_z': np.asarray(vk_ewald_z),
    }
    device_arrays = {
        name: cp.asarray(value) for name, value in host_arrays.items()
    }
    settings = witness.response_settings()
    metadata = {
        'response_backend': 'dcforge_cpu_rsdf_matrix_oracle',
        'scientific_host_fallback': True,
        'host_input_bytes': int(
            dm0_host.nbytes + dz_host.nbytes + s1_host.nbytes),
        'host_result_bytes': int(sum(
            value.nbytes for value in host_arrays.values())),
        'host_workspace_bytes': int(
            getattr(witness, 'cderi_response_batch_bytes', 0)),
        'central_value_parity_max_abs': float(
            witness.cderi_value_parity_max_abs),
        'source_omega': settings['source_omega'],
        'source_mesh': settings['source_mesh'],
        'metric_omega': settings['metric_omega'],
        'metric_mesh': settings['metric_mesh'],
        'exxdiv': exxdiv,
    }
    return device_arrays, metadata


def _cpu_witness(with_df):
    '''Build a CPU RSDF provider with the exact GPU central settings.'''
    witness = CPURSDF(
        with_df.cell, kpts=with_df.kpts, auxbasis=with_df.auxbasis)
    for name in (
            'use_bvk', 'precision_R', 'precision_G', 'npw_max',
            '_omega_min', 'omega', 'ke_cutoff', 'mesh_compact',
            'omega_j2c', 'mesh_j2c', 'precision_j2c',
            'j2c_eig_always', 'linear_dep_threshold', 'exp_to_discard'):
        value = getattr(with_df, name)
        if isinstance(value, np.ndarray):
            value = value.copy()
        setattr(witness, name, value)
    witness.max_memory = with_df.max_memory
    witness.build(j_only=False)
    return witness


def _density_to_host(with_df, density, name):
    '''Validate one sampled-k Hermitian density and copy it to the host.'''
    density = cp.asarray(density, order='C')
    shape = (
        len(with_df.kpts), with_df.cell.nao_nr(),
        with_df.cell.nao_nr())
    if density.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    error = float(cp.max(cp.abs(
        density - density.conj().transpose(0, 2, 1))).get())
    if error > 1e-9:
        raise ValueError(f'{name} must be Hermitian')
    return cp.asnumpy(density)
