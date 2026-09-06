'''CPU sampled-k GDF PBE@KRHF energy and analytic-gradient checks.'''

import numpy as np
import pytest
from pyscf.pbc import dft, gto

from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.energy import KRHFDFT, RHFDFT


def make_cell(coords=None):
    cell = gto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 1.4'
    cell.a = np.eye(3) * 6.0
    cell.unit = 'Bohr'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-8
    cell.mesh = [15, 15, 15]
    cell.verbose = 0
    cell.build()
    if coords is not None:
        cell.set_geom_(coords, unit='Bohr')
    return cell


def make_water_cell(coords=None):
    cell = gto.Cell()
    cell.atom = '''
    O   0.10  -0.20   0.30
    H   1.55   0.15   0.05
    H  -0.35   1.25  -0.25
    '''
    cell.a = np.eye(3) * 7.0
    cell.unit = 'Bohr'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-10
    cell.mesh = [21, 21, 21]
    cell.verbose = 0
    cell.build()
    if coords is not None:
        cell.set_geom_(coords, unit='Bohr')
    return cell


def run_target(cell, kpts=None):
    if kpts is None:
        kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    mf = KRHFDFT(
        cell, kpts=kpts, xc='PBE', kpts_wrap_around=True)
    mf.density_fit(auxbasis='gth-szv')
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.kernel()
    return mf


def finite_difference_gradient(make_cell):
    cell = make_cell()
    coords = cell.atom_coords()
    step = 1e-4
    numerical = np.zeros_like(coords)
    for atom in range(cell.natm):
        for axis in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numerical[atom, axis] = (
                run_target(make_cell(plus)).e_dft
                - run_target(make_cell(minus)).e_dft) / (2 * step)
    return numerical


@pytest.fixture(scope='module')
def result():
    mf = run_target(make_cell())
    mf_grad = mf.nuc_grad_method()
    analytic = mf_grad.kernel()

    numerical = finite_difference_gradient(make_cell)
    return mf, mf_grad, analytic, numerical


@pytest.fixture(scope='module')
def water_result():
    mf = run_target(make_water_cell())
    analytic = mf.nuc_grad_method().kernel()
    numerical = finite_difference_gradient(make_water_cell)
    return mf, analytic, numerical


def test_sampled_k_gdf_energy_uses_shared_provider(result):
    mf, _, _, _ = result
    assert isinstance(mf.with_df, GDF)
    assert mf.with_df is mf.mf_hf.with_df
    assert mf.with_df.capability()['sampling'] == 'full_bz'

    mf_dft = dft.KRKS(
        mf.cell, kpts=mf.kpts, xc='PBE', exxdiv='ewald')
    mf_dft.with_df = mf.with_df
    dm = mf.mf_hf.make_rdm1()
    h1e = mf_dft.get_hcore(mf.cell, mf.kpts)
    vhf = mf_dft.get_veff(mf.cell, dm, kpts=mf.kpts)
    expected = mf_dft.energy_tot(dm=dm, h1e=h1e, vhf=vhf)
    assert mf.e_dft == pytest.approx(expected, abs=1e-11)


def test_sampled_k_gdf_gradient_matches_all_3n_fd(result):
    _, mf_grad, analytic, numerical = result
    np.testing.assert_allclose(analytic, numerical, atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(
        analytic, mf_grad.de_terms, atol=3e-14, rtol=0.0)
    assert mf_grad.zvec_method_used == 'matrix_free'
    assert mf_grad.z_residual_max_abs < 1e-10


def test_sampled_k_gdf_response_keeps_source_metric_and_ewald_terms(result):
    mf, mf_grad, _, _ = result
    np.testing.assert_allclose(
        mf_grad.vj1,
        mf_grad.vj1_source + mf_grad.vj1_metric,
        atol=3e-14, rtol=0.0)
    np.testing.assert_allclose(
        mf_grad.vk1 - mf_grad.vk1_ewald,
        mf_grad.vk1_source + mf_grad.vk1_metric,
        atol=3e-14, rtol=0.0)
    assert mf.with_df.cderi_central_validated
    assert mf.with_df.cderi_value_parity_max_abs < 3e-9


def test_sampled_k_gdf_water_gradient_matches_all_3n_fd(water_result):
    mf, analytic, numerical = water_result
    np.testing.assert_allclose(analytic, numerical, atol=2e-6, rtol=0.0)
    assert mf.with_df.cderi_central_validated


def test_sampled_k_gdf_energy_is_invariant_to_kpoint_permutation(result):
    mf, _, _, _ = result
    permuted = run_target(make_cell(), np.asarray(mf.kpts)[::-1])
    assert permuted.e_dft == pytest.approx(mf.e_dft, abs=1e-10)


def test_one_gamma_gdf_collapses_to_rhfdft_energy_and_gradient():
    cell = make_cell()
    gamma = np.zeros((1, 3))
    mf_gamma = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
    mf_kgamma = KRHFDFT(
        cell, kpts=gamma, xc='PBE').density_fit(auxbasis='gth-szv')
    mf_gamma.conv_tol = mf_kgamma.conv_tol = 1e-12
    mf_gamma.kernel()
    mf_kgamma.kernel()

    assert mf_kgamma.e_dft == pytest.approx(mf_gamma.e_dft, abs=1e-10)
    np.testing.assert_allclose(
        mf_kgamma.nuc_grad_method().kernel(),
        mf_gamma.nuc_grad_method().kernel(), atol=1e-9, rtol=0.0)
