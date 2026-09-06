'''Sampled-k LDA/GGA and k-resolved tau-MGGA validation.'''

import numpy as np
import pytest
from pyscf.pbc import dft, gto

from dcforge.cpu.pbc.energy import KRHFDFT, RHFDFT

pytestmark = [pytest.mark.cpu, pytest.mark.pbc, pytest.mark.slow]


def make_h2_cell(z=1.4):
    cell = gto.Cell()
    cell.atom = f'H 0 0 0; H 0 0 {z}'
    cell.a = np.eye(3) * 6.0
    cell.unit = 'Bohr'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-8
    cell.mesh = [15, 15, 15]
    cell.verbose = 0
    cell.build()
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


def run_target(cell, xc, provider='fft'):
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    mf = KRHFDFT(
        cell, kpts=kpts, xc=xc, kpts_wrap_around=True)
    if provider == 'gdf':
        mf.density_fit(auxbasis='gth-szv')
    elif provider != 'fft':
        raise ValueError("provider must be 'fft' or 'gdf'")
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.kernel()
    return mf


@pytest.mark.parametrize(
    'xc', ['LDA,VWN', 'BLYP', 'TPSS', 'SCAN', 'M06-L', 'R2SCAN'])
def test_fixed_density_energy_matches_independent_pyscf(xc):
    cell = make_h2_cell()
    mf = run_target(cell, xc)
    reference = dft.KRKS(
        cell, kpts=mf.kpts, xc=xc, exxdiv='ewald')
    reference.with_df = mf.with_df
    dm = mf.mf_hf.make_rdm1()
    h1e = reference.get_hcore(cell, mf.kpts)
    vhf = reference.get_veff(cell, dm, kpts=mf.kpts)

    assert mf.e_dft == pytest.approx(
        reference.energy_tot(dm=dm, h1e=h1e, vhf=vhf), abs=1e-10)


@pytest.mark.parametrize(
    ('xc', 'provider'), [
        ('LDA,VWN', 'fft'),
        ('LDA,VWN', 'gdf'),
        ('BLYP', 'fft'),
        ('BLYP', 'gdf'),
        ('TPSS', 'fft'),
        ('TPSS', 'gdf'),
        ('R2SCAN', 'fft'),
        ('R2SCAN', 'gdf'),
    ])
def test_h2_selected_relaxed_source_fd(xc, provider):
    step = 1e-4
    mf = run_target(make_h2_cell(), xc, provider)
    mf_grad = mf.nuc_grad_method()
    analytic = mf_grad.kernel()[1, 2]
    e_plus = run_target(make_h2_cell(1.4 + step), xc, provider).e_dft
    e_minus = run_target(make_h2_cell(1.4 - step), xc, provider).e_dft
    numerical = (e_plus - e_minus) / (2 * step)

    assert analytic == pytest.approx(numerical, abs=2e-6)
    assert mf_grad.z_residual_max_abs < 1e-10
    assert mf_grad.term_recomposition_max_abs < 3e-14


@pytest.mark.parametrize(
    ('xc', 'provider'), [
        ('TPSS', 'fft'),
        ('TPSS', 'gdf'),
        ('SCAN', 'fft'),
        ('SCAN', 'gdf'),
        ('M06-L', 'fft'),
        ('M06-L', 'gdf'),
        ('R2SCAN', 'fft'),
        ('R2SCAN', 'gdf'),
    ])
def test_water_tau_mgga_selected_fd_exercises_complex_path(xc, provider):
    step = 1e-4
    cell = make_water_cell()
    mf = run_target(cell, xc, provider)
    mf_grad = mf.nuc_grad_method()
    analytic = mf_grad.kernel()[0, 2]
    coords = cell.atom_coords()
    plus = coords.copy()
    minus = coords.copy()
    plus[0, 2] += step
    minus[0, 2] -= step
    e_plus = run_target(make_water_cell(plus), xc, provider).e_dft
    e_minus = run_target(make_water_cell(minus), xc, provider).e_dft
    numerical = (e_plus - e_minus) / (2 * step)

    assert analytic == pytest.approx(numerical, abs=2e-6)
    assert mf_grad.xc_deriv_backend == 'uniform_grid'


def test_one_gamma_r2scan_collapses_to_rhfdft():
    cell = make_h2_cell()
    gamma = np.zeros((1, 3))
    mf_gamma = RHFDFT(cell, xc='R2SCAN')
    mf_kgamma = KRHFDFT(cell, kpts=gamma, xc='R2SCAN')
    mf_gamma.conv_tol = mf_kgamma.conv_tol = 1e-12
    mf_gamma.kernel()
    mf_kgamma.kernel()

    assert mf_kgamma.e_dft == pytest.approx(mf_gamma.e_dft, abs=1e-10)
    np.testing.assert_allclose(
        mf_kgamma.nuc_grad_method().kernel(),
        mf_gamma.nuc_grad_method().kernel(), atol=1e-9, rtol=0.0)


@pytest.mark.parametrize('provider', ['fft', 'gdf'])
def test_cpu_sampled_k_multigrid_fails_before_source_scf(provider):
    cell = make_h2_cell()
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    mf = KRHFDFT(cell, kpts=kpts, xc='PBE')
    if provider == 'gdf':
        mf.density_fit(auxbasis='gth-szv')

    with pytest.raises(NotImplementedError, match='Gamma-only'):
        mf.multigrid_numint()
    assert not mf.mf_hf.converged


def test_sampled_k_xc_capability_and_unsupported_boundaries():
    cell = make_h2_cell()
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    mf = KRHFDFT(cell, kpts=kpts, xc='PBE')
    assert mf.with_df.capability()['target'] == ['LDA', 'GGA', 'MGGA']
    assert mf.density_fit(
        auxbasis='gth-szv').with_df.capability()['target'] == [
            'LDA', 'GGA', 'MGGA']

    with pytest.raises(NotImplementedError, match='pure semilocal'):
        KRHFDFT(cell, kpts=kpts, xc='PBE0').kernel()
    mf_nlc = KRHFDFT(cell, kpts=kpts, xc='PBE')
    mf_nlc.nlc = 'VV10'
    with pytest.raises(NotImplementedError, match='NLC/VV10'):
        mf_nlc.kernel()
