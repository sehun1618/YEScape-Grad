'''Provider-neutral real-Gamma Gaussian-DF strain response assembly.

GDF and RSDF differ in how the central three-center source and metric are
represented.  Once a provider supplies the exact solved primitive
``W=M^{-1}B`` and its independently resolved source/metric parameters, the
reverse observable algebra and the analytic strain consumers are identical.
This module owns that shared boundary; provider modules remain responsible
for constructing ``W`` without changing their Hamiltonian.
'''

import time

import cupy as cp
import numpy as np

from dcforge.gpu.pbc.df.gdf.adjoint import (
    build_rhf_primitive_adjoint_channel_stack,
)


def stress_auxiliary_block_size(with_df):
    '''Return the provider's bounded reverse-adjoint auxiliary block.'''
    value = getattr(with_df, 'adjoint_auxiliary_block_size', None)
    if value is None:
        value = getattr(with_df, 'response_auxiliary_block_size', 32)
    value = int(value)
    if value < 1:
        raise ValueError('stress auxiliary block size must be positive')
    return value


def stress_reciprocal_controls(with_df):
    '''Return execution-only reciprocal controls shared by GDF and RSDF.'''
    fraction = getattr(with_df, 'reciprocal_memory_fraction', None)
    reserve = getattr(with_df, 'reciprocal_minimum_reserve_bytes', None)
    maximum = getattr(with_df, 'reciprocal_max_block_size', None)
    alignment = getattr(with_df, 'reciprocal_block_alignment', None)
    if fraction is None:
        fraction = with_df.source_reciprocal_memory_fraction
    if reserve is None:
        reserve = with_df.source_reciprocal_minimum_reserve_bytes
    if maximum is None:
        maximum = with_df.source_reciprocal_max_block_size
    if alignment is None:
        alignment = with_df.source_reciprocal_block_alignment
    return {
        'memory_fraction': float(fraction),
        'minimum_reserve_bytes': int(reserve),
        'maximum_block_size': (
            None if maximum is None else int(maximum)),
        'block_alignment': int(alignment),
    }


def make_gaussian_strain_response(
        with_df, dm0, dz, solved_three_center, central_metadata,
        source_omega, source_mesh, metric_omega, metric_mesh,
        source_precision=None, metric_precision=None,
        reciprocal_block_size=None, auxiliary_block_size=None,
        profile_components=False, overlap_short_long_range=True,
        cderi_gauge_aligned_parity=None):
    r'''Contract the shared Gaussian-DF source and metric strain ledger.

    The provider supplies :math:`W=M^{-1}B`.  This function constructs the
    target-Hartree and source-Fock reverse channels once and evaluates

    .. math::

        A_B:B^{(xy)} + A_M:M^{(xy)}

    with separate source ``(omega,mesh)`` and metric ``(omega,mesh)``
    parameters.  For GDF the two omega values are equal; RSDF must preserve
    its independently resolved values.

    Args:
        with_df : GDF or RSDF provider
            Built real-Gamma Gaussian-DF provider.
        dm0, dz : cupy.ndarray
            Source RHF and Hermitian Z-transition AO densities.
        solved_three_center : cupy.ndarray
            Exact ``M^{-1}B`` with shape ``(naux,nao,nao)``.
        central_metadata : dict
            Provider provenance for the construction of ``M^{-1}B``.
        source_omega, source_mesh
            Parameters defining ``B_SR+B_LR+B_vbar``.
        metric_omega, metric_mesh
            Parameters defining ``M_SR+M_LR+M_G0``.

    Kwargs:
        source_precision, metric_precision : float or None
            Independent screened-integral precision values.  ``None`` keeps
            the GDF primitive defaults.
        reciprocal_block_size, auxiliary_block_size
            Execution-only memory bounds; neither changes the representation.
        profile_components : bool
            Synchronize named component boundaries for diagnostics only.
        overlap_short_long_range : bool
            Overlap independent host SR and device LR contractions.
        cderi_gauge_aligned_parity : float or None
            Optional transparent-oracle diagnostic owned by GDF.

    Returns:
        dict
            Two-channel analytic source/metric strain ledger.
    '''
    # Imported lazily to avoid a module cycle while the established GDF
    # primitive implementation is exposed through this provider-neutral
    # assembler.  These functions accept any provider owning a Cell,
    # auxiliary Cell and the common reciprocal execution policy.
    from dcforge.gpu.pbc.df.gdf.stress import (
        contract_auxiliary_metric_strain,
        contract_three_center_source_strain,
        copy_source_adjoint_to_short_range_workspace,
    )

    cell = with_df.cell
    auxcell = with_df.auxcell
    nao = int(cell.nao_nr())
    source_omega = float(source_omega)
    metric_omega = float(metric_omega)
    source_mesh = np.asarray(source_mesh, dtype=int)
    metric_mesh = np.asarray(metric_mesh, dtype=int)
    if auxiliary_block_size is None:
        auxiliary_block_size = stress_auxiliary_block_size(with_df)
    auxiliary_block_size = int(auxiliary_block_size)
    timings = {} if profile_components else None

    started = _component_start(profile_components)
    source_adjoint, metric_adjoint, adjoint_metadata = (
        build_rhf_primitive_adjoint_channel_stack(
            solved_three_center, cp.asarray(dm0), cp.asarray(dz),
            response_sign=-1.,
            auxiliary_block_size=auxiliary_block_size))
    _component_stop(timings, 'device_adjoint', started, synchronize=True)

    # The primitive metric is symmetric.  The channel builder returns both
    # ordered legs of A_B:W; -sym(A_B W^T)/2 is the physical dE/dM source.
    metric_adjoint *= -.5
    del solved_three_center
    source_view = source_adjoint.transpose(0, 2, 3, 1).reshape(
        2, nao * nao, int(auxcell.nao_nr()))
    started = _component_start(profile_components)
    source_workspace, source_transfer = (
        copy_source_adjoint_to_short_range_workspace(
            cell, auxcell, source_view))
    metric_host = cp.asnumpy(metric_adjoint)
    _component_stop(timings, 'adjoint_device_to_host', started)

    started = _component_start(profile_components)
    source_response = contract_three_center_source_strain(
        with_df, source_omega, source_mesh, source_view,
        source_short_range_workspace=source_workspace,
        precision=source_precision,
        reciprocal_block_size=reciprocal_block_size,
        overlap_short_long_range=overlap_short_long_range)
    _component_stop(timings, 'source_strain', started)
    del source_workspace, source_view, source_adjoint

    started = _component_start(profile_components)
    metric_response = contract_auxiliary_metric_strain(
        with_df, metric_omega, metric_mesh, metric_adjoint,
        metric_adjoint_host=metric_host, precision=metric_precision,
        reciprocal_block_size=reciprocal_block_size,
        overlap_short_long_range=overlap_short_long_range)
    _component_stop(timings, 'metric_strain', started)
    raw = (np.asarray(source_response['raw_contraction'])
           + np.asarray(metric_response['raw_contraction']))
    if raw.shape != (2, 3, 3) or not np.all(np.isfinite(raw)):
        raise RuntimeError(
            'GPU Gaussian-DF strain response returned an invalid tensor')

    backend = str(getattr(with_df, 'backend_name', 'gaussian_df'))
    return {
        'name': 'C',
        'primitive_analytic': True,
        'production_analytic': True,
        'gpu_native_complete': False,
        'low_level_numerical_components': [],
        'storage': (
            'provider_cderi_device_reverse_adjoint_'
            'hybrid_analytic_direct_voigt'),
        'provider_backend': backend,
        'observable_names': ('target_j', 'source_fock'),
        'raw_contraction': raw,
        'source': source_response,
        'metric': metric_response,
        # Retain the former GDF key for compatibility while exposing the
        # mathematically necessary independent RSDF values explicitly.
        'omega': source_omega if source_omega == metric_omega else None,
        'source_omega': source_omega,
        'metric_omega': metric_omega,
        'source_mesh': source_mesh,
        'metric_mesh': metric_mesh,
        'reciprocal_block_size_requested': (
            None if reciprocal_block_size is None
            else int(reciprocal_block_size)),
        'auxiliary_block_size': auxiliary_block_size,
        'central_context': central_metadata,
        'cderi_gauge_aligned_parity_max_abs': (
            None if cderi_gauge_aligned_parity is None
            else float(cderi_gauge_aligned_parity)),
        'provider_factor_reconstruction_max_abs': central_metadata.get(
            'factor_reconstruction_max_abs'),
        'provider_factor_reconstruction_tolerance': central_metadata.get(
            'factor_reconstruction_tolerance'),
        'device_adjoint': True,
        'device_metric_solve': True,
        'device_native_contracted_primitives': (
            'central_M_provider_reconstruction',
            'central_M_inverse_B_from_provider_CDERI',
            'B_LR_AO_pair_strain', 'B_LR_adjoint_contractions',
            'B_vbar_overlap_strain', 'M_LR_adjoint_contractions', 'M_G0'),
        'host_primitive_boundary': (
            'B_SR', 'B_LR_auxiliary_Fourier_strain',
            'B_LR_weighted_Coulomb_strain', 'M_SR',
            'M_LR_auxiliary_Fourier_strain',
            'M_LR_weighted_Coulomb_strain'),
        'silent_cpu_fallback': False,
        'resolved_representation': {
            'omega': (
                source_omega if source_omega == metric_omega else None),
            'source_omega': source_omega,
            'source_mesh': source_mesh.tolist(),
            'source_precision': (
                None if source_precision is None else float(source_precision)),
            'metric_omega': metric_omega,
            'metric_mesh': metric_mesh.tolist(),
            'metric_precision': (
                None if metric_precision is None else float(metric_precision)),
        },
        'host_to_device_bytes': int(
            central_metadata.get('host_to_device_bytes', 0)),
        'device_to_host_bytes': int(
            source_transfer['transferred_bytes'] + metric_host.nbytes),
        'source_device_to_host': source_transfer,
        'overlap_short_long_range': bool(overlap_short_long_range),
        'adjoint': adjoint_metadata,
        'component_wall_seconds': timings,
    }


def _component_start(enabled):
    return time.perf_counter() if enabled else None


def _component_stop(timings, name, started, synchronize=False):
    if started is None:
        return
    if synchronize:
        cp.cuda.Stream.null.synchronize()
    timings[str(name)] = float(time.perf_counter() - started)


__all__ = [
    'make_gaussian_strain_response',
    'stress_auxiliary_block_size',
    'stress_reciprocal_controls',
]
