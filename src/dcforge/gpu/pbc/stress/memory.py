'''Memory preflight for the experimental GPU GDF stress route.'''

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import get_avail_mem

from dcforge.gpu.pbc.df.gaussian.stress import (
    stress_auxiliary_block_size,
    stress_reciprocal_controls,
)


def stress_memory_preflight(
        mf_grad, strict=False, memory_fraction=.8,
        minimum_reserve_bytes=4 * 1024 ** 3):
    r'''Estimate the additional workspace of one analytic stress evaluation.

    The production route reconstructs :math:`M^{-1}B` from the active
    provider CDERI and its GPU4PySCF metric inverse factor, then writes the
    target-J and source-Fock adjoints directly into their final two-channel
    arrays.  Let

    .. math::

        A=n_{aux}n_{AO}^2,\qquad M=n_{aux}^2.

    The largest persistent reverse-adjoint live set is ``3 A + 2 M`` real
    numbers: one solved source, two final source channels and two final metric
    channels.  Bounded auxiliary workspaces are added explicitly below.
    Provider-factor validation is also AO-pair blocked, so it no longer
    creates two additional complete ``A`` tensors merely to report a
    residual.  This is an implementation live-set estimate, not a promise
    about CUDA-library workspaces or CuPy allocator retention.

    The reciprocal :math:`B_{LR}` contraction is not included in that dense
    count.  It already uses the provider's free-VRAM planner and retries an
    allocation failure with a smaller G-vector transaction.  The XC image
    moment is likewise bounded independently by the multigrid pair-task
    implementation.  Reporting these policies separately prevents a block
    size from being mistaken for a change in the physical GDF or MG mesh.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.rhf_dft.Gradients
            Gradient/stress driver owning a converged GPU GDF workflow.

    Kwargs:
        strict : bool
            Raise :class:`MemoryError` when the conservative dense-device
            estimate exceeds the live provider budget.  The default is
            advisory because CuPy can reuse already reserved pool blocks and
            the estimate intentionally counts overlapping lifetimes.
        memory_fraction : float
            Fraction of live free VRAM available to the dense stress phases.
            This is independent of the smaller reciprocal-block fraction.
        minimum_reserve_bytes : int
            Live VRAM retained for CUDA libraries and later phases.

    Returns:
        dict
            Dimensions, byte estimates, live CUDA/CuPy state, fit decision,
            and the exact bounded/retry policy used by later primitives.
    '''
    mf = mf_grad.base
    with_df = mf.with_df
    cell = mf.cell
    auxcell = getattr(with_df, 'auxcell', None)
    if auxcell is None:
        raise RuntimeError(
            'stress memory preflight requires a built GDF auxiliary Cell')

    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    mo_occ = np.asarray(cp.asnumpy(cp.asarray(mf.mo_occ)))
    nocc = int(np.count_nonzero(mo_occ > 0))
    nvir = int(np.count_nonzero(mo_occ == 0))
    ngrids = int(np.prod(np.asarray(mf._numint.mesh, dtype=int)))
    return _stress_memory_report(
        with_df, nao, naux, nocc, nvir, ngrids,
        strict=strict, stage='converged_central_state',
        memory_fraction=memory_fraction,
        minimum_reserve_bytes=minimum_reserve_bytes)


def stress_memory_preflight_from_cell(
        cell, with_df, auxbasis, strict=False, memory_fraction=.8,
        minimum_reserve_bytes=4 * 1024 ** 3):
    '''Estimate stress workspace before source HF or CDERI construction.

    This queue-facing preflight builds only PySCF's auxiliary-basis metadata
    to determine ``naux``.  It does not build three-center integrals, run SCF,
    evaluate target XC, or allocate the estimated dense GPU workspace.
    '''
    from pyscf.pbc import df

    auxcell = df.make_auxcell(cell, auxbasis)
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    nocc = int(cell.nelectron // 2)
    nvir = int(nao - nocc)
    ngrids = int(np.prod(np.asarray(cell.mesh, dtype=int)))
    return _stress_memory_report(
        with_df, nao, naux, nocc, nvir, ngrids,
        strict=strict, stage='before_source_hf_and_cderi',
        memory_fraction=memory_fraction,
        minimum_reserve_bytes=minimum_reserve_bytes)


def _stress_memory_report(
        with_df, nao, naux, nocc, nvir, ngrids,
        strict=False, stage=None, memory_fraction=.8,
        minimum_reserve_bytes=4 * 1024 ** 3):
    '''Assemble one common live-device stress memory report.'''
    real_bytes = np.dtype(np.float64).itemsize

    source_elements = naux * nao * nao
    metric_elements = naux * naux
    nobs = 2
    auxiliary_block_size = min(
        naux, stress_auxiliary_block_size(with_df))
    adjoint_temporary_elements = (
        2 * auxiliary_block_size * nao * nao)
    provider_reconstruction_elements = (
        2 * source_elements + 4 * metric_elements)
    reverse_adjoint_elements = (
        3 * source_elements + 2 * metric_elements
        + adjoint_temporary_elements)
    source_staging_block_bytes = 128 * 1024 ** 2
    source_strain_elements = (
        2 * source_elements
        + source_staging_block_bytes // real_bytes)
    metric_strain_elements = 2 * metric_elements
    gdf_device_elements = max(
        provider_reconstruction_elements,
        reverse_adjoint_elements,
        source_strain_elements,
        metric_strain_elements)
    # r2SCAN carries rho, three Cartesian derivatives and tau.  The factor
    # four allows one complex reciprocal and one real mesh representation
    # plus LibXC weights without claiming an exact implementation peak.
    xc_device_elements = 4 * 5 * ngrids
    orbital_device_elements = (
        3 * nao * nao + nao * (nocc + nvir) + nocc * nvir)
    gdf_device_bytes = int(real_bytes * gdf_device_elements)
    xc_device_bytes = int(real_bytes * xc_device_elements)
    orbital_device_bytes = int(real_bytes * orbital_device_elements)
    estimated_device_bytes = int(max(
        gdf_device_bytes + orbital_device_bytes,
        xc_device_bytes + orbital_device_bytes))

    # The SR callback consumes unique symmetric AO pairs in its final C ABI
    # workspace.  Ordered dense AO pairs are never resident on the host.
    npair = nao * (nao + 1) // 2
    packed_source_elements = nobs * npair * naux
    source_result_elements = nobs * 9 * npair
    metric_host_elements = nobs * metric_elements
    host_elements = (
        packed_source_elements + source_result_elements
        + metric_host_elements)
    estimated_host_bytes = int(real_bytes * host_elements)

    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    available_bytes = int(get_avail_mem(exclude_memory_pool=True))
    fraction = float(memory_fraction)
    reserve_bytes = int(minimum_reserve_bytes)
    if not 0. < fraction <= 1.:
        raise ValueError('stress memory_fraction must be in (0,1]')
    if reserve_bytes < 0:
        raise ValueError('stress minimum_reserve_bytes must be nonnegative')
    budget_bytes = int(max(
        0, min(available_bytes * fraction,
               available_bytes - reserve_bytes)))
    fits = estimated_device_bytes <= budget_bytes

    pool = cp.get_default_memory_pool()
    reciprocal_controls = stress_reciprocal_controls(with_df)
    report = {
        'schema': 'dcforge.pbc.stress.memory_preflight.v2',
        'stage': stage,
        'estimate_kind': (
            'production_live_set_excluding_allocator_retention_and_'
            'bounded_library_workspaces'),
        'strict': bool(strict),
        'dimensions': {
            'nao': nao,
            'naux': naux,
            'nocc': nocc,
            'nvir': nvir,
            'multigrid_points': ngrids,
        },
        'estimated_bytes': {
            'gdf_reverse_adjoint_device': gdf_device_bytes,
            'multigrid_xc_device': xc_device_bytes,
            'orbital_and_density_device': orbital_device_bytes,
            'peak_additional_device': estimated_device_bytes,
            'explicit_host_analytic_adjoints': estimated_host_bytes,
            'ordered_dense_host_source_avoided': int(
                nobs * source_elements * real_bytes),
            'packed_host_source': int(
                packed_source_elements * real_bytes),
            'host_short_range_result_workspace': int(
                source_result_elements * real_bytes),
            'host_metric_adjoints': int(
                metric_host_elements * real_bytes),
        },
        'gdf_live_set_elements': {
            'source_tensor_A': int(source_elements),
            'metric_tensor_M': int(metric_elements),
            'provider_reconstruction_phase': int(
                provider_reconstruction_elements),
            'reverse_adjoint_phase': int(reverse_adjoint_elements),
            'source_strain_phase_excluding_reciprocal_transaction': int(
                source_strain_elements),
            'metric_strain_phase_excluding_reciprocal_transaction': int(
                metric_strain_elements),
            'adjoint_auxiliary_block_size': int(auxiliary_block_size),
        },
        'live_device_bytes': {
            'runtime_free': int(free_bytes),
            'runtime_total': int(total_bytes),
            'provider_available_excluding_pool': available_bytes,
            'cupy_pool_used': int(pool.used_bytes()),
            'cupy_pool_reserved': int(pool.total_bytes()),
        },
        'provider_budget': {
            'memory_fraction': fraction,
            'minimum_reserve_bytes': reserve_bytes,
            'estimated_workspace_budget_bytes': budget_bytes,
            'reciprocal_block_memory_fraction': float(
                reciprocal_controls['memory_fraction']),
            'reciprocal_block_minimum_reserve_bytes': int(
                reciprocal_controls['minimum_reserve_bytes']),
        },
        # Retain the original public key for queue/report compatibility.  In
        # v2 it refers to the production live-set estimate described above.
        'fits_conservative_device_estimate': bool(fits),
        'fits_production_live_set_estimate': bool(fits),
        'bounded_policies': {
            'gdf_long_range_source': (
                'provider reciprocal free-VRAM planner with transactional '
                'block-halving allocation-OOM retry'),
            'multigrid_xc_image_moment': (
                'density-size-aware image blocks bounded near 128 MiB'),
            'full_forward_cderi_derivative': False,
            'production_derivative': 'reverse_adjoint_direct_voigt',
            'source_adjoint_host_staging': (
                'bounded_device_unique_pair_pack_into_final_C_workspace'),
            'provider_factor_residual': (
                'bounded_AO_pair_columns_without_full_tensor_temporaries'),
        },
        'unbounded_development_boundaries': (
            'host_analytic_source_and_metric_integral_traversals',
            'device_two_channel_dense_reverse_adjoint'),
    }
    if strict and not fits:
        gib = 1024. ** 3
        raise MemoryError(
            'stress production live-set estimate '
            f'{estimated_device_bytes / gib:.2f} GiB exceeds the live '
            f'{budget_bytes / gib:.2f} GiB provider budget; use a smaller '
            'system/basis or keep strict_memory_preflight=false to collect '
            'an advisory run')
    return report


__all__ = [
    'stress_memory_preflight',
    'stress_memory_preflight_from_cell',
]
