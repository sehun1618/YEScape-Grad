'''Small contract checks for migrated real-Gamma GDF accelerations.'''

import time
import warnings
from types import SimpleNamespace

import cupy as cp
import numpy as np
import pytest

from dcforge.gpu.pbc.df.gdf.build_memory import plan_native_build_memory
from dcforge.gpu.pbc.df.gdf.metric import metric_inverse_from_source
from dcforge.gpu.pbc.df.gdf.precision import (
    GDFBuildPrecisionWarning,
    GDFMixedPrecisionWarning,
)
from dcforge.gpu.pbc.energy import RHFDFT
from tests.common import make_water_cell

pytestmark = [pytest.mark.gpu, pytest.mark.pbc]

GIB = 1024**3


@pytest.mark.parametrize(
    'available, expected, limiter',
    (
        (3 * GIB, int(3 * GIB * .25), 'memory_fraction'),
        (8 * GIB, 2 * GIB, 'memory_fraction'),
        (24 * GIB, 4 * GIB, 'max_working_bytes'),
        (96 * GIB, 4 * GIB, 'max_working_bytes'),
    ),
)
def test_native_build_budget_is_device_aware(available, expected, limiter):
    '''The native builder budget scales down and retains its large-GPU cap.'''
    plan = plan_native_build_memory(available, .25, 2 * GIB, 4 * GIB)

    assert plan['effective_bytes'] == expected
    assert plan['limiting_constraint'] == limiter


def test_gdf_acceleration_defaults_have_explicit_reference_routes():
    '''Record which migrated accelerations are active without building CDERI.'''
    mf = RHFDFT(make_water_cell()).density_fit(auxbasis='gth-szv')
    provider = mf.with_df
    mf_grad = mf.nuc_grad_method()

    assert provider.native_build_memory_fraction == pytest.approx(.25)
    assert provider.native_build_minimum_reserve_bytes == 2 * GIB
    assert provider.native_build_max_working_bytes == 4 * GIB
    assert provider.reciprocal_gamma_conjugate_symmetry is True
    assert provider.adjoint_auxiliary_block_size == 128
    assert provider.adjoint_storage == 'host_cache'
    assert provider.adjoint_host_cache == 'mapped'
    assert provider.gdf_precision_profile == 'fp64'
    assert provider.cderi_storage == 'mapped_host'
    assert provider.cderi_device_residency == 'off'
    assert provider.omega is None
    assert provider.source_metric_factor_metadata is None
    assert provider.source_metric_factor_use_count == 0

    assert mf.source_precision_policy == 'fp64'
    assert mf_grad.precision_policy == 'mixed'
    assert mf_grad.df_cphf_response_mode == 'direct_vo'
    assert mf_grad.df_cphf_response_cache == 'auto'
    assert mf_grad.gdf_response_mode == 'auto'
    assert mf_grad.gdf_adjoint_backend == 'auto'
    assert mf_grad.gdf_derivative_precision == 'fp64'
    assert mf_grad.terminal_response_reuse is True


def test_terminal_response_reuse_preserves_r2scan_force():
    '''The final audited Z action must replace, not approximate, Pulay fvind.'''
    mf = RHFDFT(
        make_water_cell(), xc='R2SCAN').density_fit(auxbasis='gth-szv')
    mf = mf.multigrid_numint(with_j=False)
    with pytest.warns(GDFBuildPrecisionWarning):
        mf.set_gdf_precision_profile('fp32_build')
    mf.conv_tol = 1.e-8
    mf.conv_tol_grad = 1.e-5
    mf.max_cycle = 60
    assert np.isfinite(mf.kernel())

    gradients = {}
    timings = {}
    drivers = {}
    for reuse in (False, True):
        mf_grad = mf.nuc_grad_method()
        mf_grad.terminal_response_reuse = reuse
        mf_grad.cphf_conv_tol = 1.e-8
        mf_grad.cphf_residual_tol = 1.e-6
        started = time.perf_counter()
        gradients[reuse] = cp.asnumpy(mf_grad.kernel())
        timings[reuse] = time.perf_counter() - started
        drivers[reuse] = mf_grad

    difference = gradients[True] - gradients[False]
    print({
        'reuse_false_seconds': timings[False],
        'reuse_true_seconds': timings[True],
        'gradient_rms_ha_per_bohr': float(np.sqrt(np.mean(difference**2))),
        'gradient_max_ha_per_bohr': float(np.max(np.abs(difference))),
    })
    np.testing.assert_allclose(
        gradients[True], gradients[False], atol=1.e-9, rtol=0.)
    assert drivers[False].terminal_response_reuse_used is False
    assert drivers[True].terminal_response_reuse_used is True
    assert drivers[True].terminal_response_reuse_metadata['matched'] is True
    assert drivers[True].z_residual_max_abs <= 1.e-6


def test_source_metric_factor_keeps_only_retained_subspace():
    '''The derivative pseudoinverse must not revive a discarded metric mode.'''
    factor = cp.asarray([[.5], [0.]], dtype=cp.float64)
    owner = SimpleNamespace(source_metric_factor_use_count=0)
    auxcell = SimpleNamespace(
        cell=SimpleNamespace(cart=False),
        ao_loc=np.asarray([0, 2]),
        ctr_coeff=np.eye(2),
        apply_CT_mat_C=lambda value: value,
    )
    intopt = SimpleNamespace(
        auxcell=auxcell,
        _dcforge_source_metric_factor=factor,
        _dcforge_source_metric_factor_metadata={'negative_rank': 0},
        _dcforge_source_metric_factor_owner=owner,
    )

    inverse = metric_inverse_from_source(
        intopt, cp.diag(cp.asarray([4., 1.e-16])))

    np.testing.assert_allclose(
        cp.asnumpy(inverse), np.diag([.25, 0.]), atol=0., rtol=0.)
    assert owner.source_metric_factor_use_count == 1


def test_precision_profiles_remain_explicit_opt_in():
    '''Moderate/aggressive Hamiltonians must warn and remain reversible.'''
    mf = RHFDFT(make_water_cell()).density_fit(auxbasis='gth-szv')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        returned = mf.set_gdf_precision_profile('mixed', block_size=17)
    assert returned is mf
    assert mf.with_df.gdf_precision_profile == 'mixed'
    assert mf.source_precision_policy == 'persistent_mixed'
    assert mf.with_df.gdf_derivative_precision == 'mixed'
    mf_grad = mf.nuc_grad_method()
    assert mf_grad.precision_policy == 'mixed'
    assert mf_grad.gdf_derivative_precision == 'mixed'
    assert any(issubclass(item.category, GDFMixedPrecisionWarning)
               for item in caught)

    with pytest.warns(GDFBuildPrecisionWarning):
        mf.set_gdf_precision_profile('fp32_build')
    mf_grad = mf.nuc_grad_method()
    assert mf.source_precision_policy == 'persistent_mixed'
    assert mf_grad.precision_policy == 'mixed'
    assert mf_grad.gdf_derivative_precision == 'mixed'

    mf.set_gdf_precision_profile('fp64')
    assert mf.with_df.gdf_precision_profile == 'fp64'
    assert mf.source_precision_policy == 'fp64'
    assert mf.with_df.gdf_derivative_precision == 'fp64'
    mf_grad = mf.nuc_grad_method()
    # Z3 remains the response-only real-Gamma default on an FP64 Hamiltonian.
    assert mf_grad.precision_policy == 'mixed'
    assert mf_grad.gdf_derivative_precision == 'fp64'
