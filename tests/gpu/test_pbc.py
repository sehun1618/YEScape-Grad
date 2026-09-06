'''GPU real-Gamma GDF and MultiGrid migration checks.'''

import cupy as cp
import numpy as np
import pytest
from gpu4pyscf.pbc.dft import gen_grid

from dcforge.gpu.pbc.df.fft import FFTDF
from dcforge.gpu.pbc.df.gdf import GDF
from dcforge.gpu.pbc.energy import RHFDFT
from dcforge.gpu.pbc.energy.rhf_dft import (
    _signature_value,
    _target_fock_is_current,
    _target_fock_signature,
)
from tests.common import as_numpy, make_water_cell

pytestmark = [pytest.mark.gpu, pytest.mark.pbc, pytest.mark.slow]


def test_target_fock_signature_accepts_etb_basis_dictionary():
    '''ETB dictionaries are mappings, not CuPy arrays with a zero-arg get.'''
    first = {'O': [[0, [2., 1.]]], 'H': [[0, [1., 1.]]]}
    second = {'H': [[0, [1., 1.]]], 'O': [[0, [2., 1.]]]}

    assert _signature_value(first) == _signature_value(second)


def test_fftdf_gradient_has_complete_ledger():
    cell = make_water_cell()
    mf = RHFDFT(cell, xc='PBE').fft_density_fit()
    mf.conv_tol = 1e-8
    mf.conv_tol_grad = 1e-6
    mf.max_cycle = 60
    assert np.isfinite(mf.kernel())
    mf_grad = mf.nuc_grad_method()
    mf_grad.grid_response = False
    mf_grad.cphf_conv_tol = 1e-8
    mf_grad.cphf_residual_tol = 1e-7
    gradient = as_numpy(mf_grad.kernel())
    assert np.allclose(gradient, mf_grad.de_terms, atol=1e-9, rtol=0)
    assert mf_grad.target_fock_reused


def test_gdf_r2scan_multigrid_xc_gradient():
    cell = make_water_cell()
    mf = RHFDFT(cell, xc='R2SCAN').density_fit(auxbasis='gth-szv')
    mf = mf.multigrid_numint(with_j=False)
    mf.conv_tol = 1e-8
    mf.conv_tol_grad = 1e-6
    mf.max_cycle = 60
    energy = mf.kernel()
    factor = mf.with_df._source_metric_factor
    factor_metadata = mf.with_df.source_metric_factor_metadata
    assert isinstance(factor, cp.ndarray)
    assert factor_metadata['owner'] == 'GPU4PySCF source CDERI build'
    assert factor_metadata['storage'] == 'device'
    assert factor_metadata['bytes'] == factor.nbytes
    assert factor_metadata['retained_rank'] > 0
    assert mf.with_df.cderi_memory_report()[
        'source_metric_factor_bytes'] == factor.nbytes
    mf_grad_fp64 = mf.nuc_grad_method()
    mf_grad_fp64.precision_policy = 'fp64'
    mf_grad_fp64.cphf_conv_tol = 1e-8
    mf_grad_fp64.cphf_residual_tol = 1e-7
    gradient_fp64 = as_numpy(mf_grad_fp64.kernel())

    # Real-Gamma GDF defaults to the guarded Z3 response action.  It still
    # accepts the solution only after reconstructing the FP64 physical
    # residual; the explicit FP64 calculation above remains the reference.
    mf_grad = mf.nuc_grad_method()
    mf_grad.cphf_conv_tol = 1e-8
    mf_grad.cphf_residual_tol = 1e-7
    gradient = as_numpy(mf_grad.kernel())
    assert np.isfinite(energy)
    assert np.all(np.isfinite(gradient))
    assert np.allclose(gradient, gradient_fp64, atol=1e-8, rtol=0)
    assert np.allclose(gradient, mf_grad.de_terms, atol=1e-9, rtol=0)
    assert mf_grad.precision_policy == 'mixed'
    assert mf_grad.z_residual_max_abs <= mf_grad.cphf_residual_tol
    assert mf_grad.xc_deriv_backend == 'multigrid_v2_xc_only'
    assert mf_grad.target_fock_reused
    assert mf.with_df._source_metric_factor is factor
    assert mf.with_df.source_metric_factor_use_count >= 2

    mf.with_df.reset()
    assert mf.with_df._source_metric_factor is None
    assert mf.with_df.source_metric_factor_metadata is None
    assert mf.with_df.source_metric_factor_use_count == 0


def test_gamma_target_fock_cache_tracks_density_grid_and_provider_state():
    cell = make_water_cell()
    mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
    dm = cp.eye(cell.nao_nr())
    mf.dm_hf = dm.copy()
    mf.target_fock = cp.zeros_like(dm)
    mf.target_fock_signature = _target_fock_signature(mf, dm)

    assert _target_fock_is_current(mf, dm)
    assert mf.target_fock_validation_host_bytes >= 2 * dm.nbytes

    changed_dm = dm.copy()
    changed_dm[0, 0] += 1.e-6
    assert mf.target_fock_signature != _target_fock_signature(mf, changed_dm)
    assert not _target_fock_is_current(mf, changed_dm)

    grid_mesh = np.asarray(mf.grids.mesh).copy()
    mf.grids.mesh = grid_mesh + 2
    assert not _target_fock_is_current(mf, dm)
    mf.grids.mesh = grid_mesh

    mf.with_df.linear_dep_threshold *= 10
    assert not _target_fock_is_current(mf, dm)


def test_gamma_rejects_foreign_cell_and_split_df_providers():
    cell = make_water_cell()
    foreign_cell = make_water_cell()
    dm = cp.eye(cell.nao_nr())

    mf = RHFDFT(cell, xc='PBE')
    foreign_provider = FFTDF(foreign_cell, mesh=cell.mesh)
    mf.with_df = mf.mf_hf.with_df = foreign_provider
    with pytest.raises(ValueError, match='same Cell'):
        mf.energy_tot(dm=dm)

    mf = RHFDFT(cell, xc='PBE')
    mf.mf_hf.cell = foreign_cell
    with pytest.raises(ValueError, match='same Cell'):
        mf.energy_tot(dm=dm)

    for target_provider in (FFTDF(cell), GDF(cell, auxbasis='gth-szv')):
        mf = RHFDFT(cell, xc='PBE')
        mf.with_df = target_provider
        with pytest.raises(ValueError, match='share one'):
            mf.energy_tot(dm=dm)

    mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
    mf.with_df = FFTDF(cell)
    with pytest.raises(ValueError, match='share one'):
        mf.nuc_grad_method().kernel()


def test_gamma_rejects_atom_centered_or_foreign_uniform_xc_grid():
    cell = make_water_cell()
    mf = RHFDFT(cell, xc='PBE')
    mf.grids = gen_grid.BeckeGrids(cell)
    mf.mf_hf.kernel = lambda *args, **kwargs: pytest.fail(
        'source RHF must not run for an unsupported target grid')
    with pytest.raises(NotImplementedError, match='UniformGrids'):
        mf.kernel()

    foreign_cell = make_water_cell()
    mf = RHFDFT(cell, xc='PBE')
    mf.grids = gen_grid.UniformGrids(foreign_cell)
    with pytest.raises(ValueError, match='same Cell'):
        mf.energy_tot(dm=cp.eye(cell.nao_nr()))


@pytest.mark.parametrize('provider', ['fft', 'gdf'])
def test_gamma_public_provider_switch_keeps_one_owner(provider):
    cell = make_water_cell()
    mf = RHFDFT(cell, xc='PBE')
    if provider == 'fft':
        mf = mf.fft_density_fit()
    else:
        mf = mf.density_fit(auxbasis='gth-szv')

    assert mf.with_df is mf.mf_hf.with_df
    assert mf.with_df.cell is cell
    assert mf.mf_hf.cell is cell
