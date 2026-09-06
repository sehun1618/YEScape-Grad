'''CPU real-Gamma FFTDF/GDF migration checks.'''

import os

import numpy as np
import pytest
from pyscf.pbc.dft import gen_grid

from dcforge.cpu.pbc.df.fft import FFTDF
from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.energy import RHFDFT
from dcforge.cpu.pbc.energy.rhf_dft import (
    _signature_value,
    _target_fock_is_current,
    _target_fock_signature,
)
from tests.common import make_water_cell

pytestmark = [pytest.mark.cpu, pytest.mark.pbc, pytest.mark.slow]


def test_target_fock_signature_accepts_etb_basis_dictionary():
    first = {'O': [[0, [2., 1.]]], 'H': [[0, [1., 1.]]]}
    second = {'H': [[0, [1., 1.]]], 'O': [[0, [2., 1.]]]}

    assert _signature_value(first) == _signature_value(second)


def _run(cell, *, with_gradient=True):
    mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-7
    mf.max_cycle = 60
    energy = mf.kernel()
    if not with_gradient:
        return energy, None, None
    mf_grad = mf.nuc_grad_method()
    mf_grad.cphf_conv_tol = 1e-9
    mf_grad.cphf_residual_tol = 1e-8
    gradient = mf_grad.kernel()
    return energy, gradient, mf_grad


def test_gdf_gradient_has_complete_ledger():
    _energy, gradient, mf_grad = _run(make_water_cell())
    assert gradient.shape == (3, 3)
    assert np.all(np.isfinite(gradient))
    assert np.allclose(gradient, mf_grad.de_terms, atol=1e-10, rtol=0)
    assert mf_grad.z_residual_max_abs <= mf_grad.cphf_residual_tol
    assert mf_grad.target_fock_reused


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
    gradient = mf_grad.kernel()
    assert np.allclose(gradient, mf_grad.de_terms, atol=1e-10, rtol=0)
    assert np.max(np.abs(mf_grad.de_grid)) == 0
    assert mf_grad.target_fock_reused


def test_gamma_target_fock_cache_tracks_density_grid_and_provider_state():
    cell = make_water_cell()
    mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
    dm = np.eye(cell.nao_nr())
    mf.dm_hf = dm.copy()
    mf.target_fock = np.zeros_like(dm)
    mf.target_fock_signature = _target_fock_signature(mf, dm)

    assert _target_fock_is_current(mf, dm)

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
    dm = np.eye(cell.nao_nr())

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
        mf.energy_tot(dm=np.eye(cell.nao_nr()))


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


@pytest.mark.skipif(
    os.environ.get('DCFORGE_RUN_PBC_FD') != '1',
    reason='set DCFORGE_RUN_PBC_FD=1 to run displaced source-RHF calculations',
)
def test_gdf_selected_finite_difference():
    cell = make_water_cell()
    _energy, gradient, _mf_grad = _run(cell)
    step = 1e-4
    energies = []
    for sign in (1, -1):
        displaced = cell.copy()
        coords = cell.atom_coords().copy()
        coords[1, 2] += sign * step
        displaced.set_geom_(coords, unit='Bohr')
        energy, _gradient, _mf_grad = _run(
            displaced, with_gradient=False)
        energies.append(energy)
    finite_difference = (energies[0] - energies[1]) / (2 * step)
    assert abs(gradient[1, 2] - finite_difference) < 1e-6
