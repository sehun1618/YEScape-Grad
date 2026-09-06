'''CPU full-BZ FFTDF PBE@KRHF energy checks.'''

import numpy as np
import pytest
from pyscf.pbc import df, dft, gto, scf
from pyscf.pbc.dft import krks
from pyscf.pbc.lib.kpts import KPoints

from dcforge.cpu.pbc.df.fft import FFTDF
from dcforge.cpu.pbc.energy import KRHFDFT, RHFDFT
from dcforge.cpu.pbc.kpts import (
    equal_kpts_weights,
    kpoint_fingerprint,
    normalize_kpts,
    validate_insulating_occupations,
    weighted_inner_product,
)

pytestmark = [pytest.mark.cpu, pytest.mark.pbc, pytest.mark.slow]


def _make_cell():
    cell = gto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 1.4'
    cell.a = np.eye(3) * 6.0
    cell.unit = 'Bohr'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-7
    cell.mesh = [15, 15, 15]
    cell.verbose = 0
    cell.build()
    return cell


def _make_kpts(cell):
    return cell.make_kpts([2, 1, 1], wrap_around=True)


@pytest.fixture(scope='module')
def sampled_result():
    cell = _make_cell()
    mf = KRHFDFT(
        cell, kpts=_make_kpts(cell), xc='PBE',
        kpts_wrap_around=True)
    mf.kernel()
    return mf


def test_full_bz_contract_and_weighted_complex_inner_product():
    kpts = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    weights = equal_kpts_weights(kpts)
    x = np.array([[[1.0 + 1.0j]], [[2.0 - 1.0j]]])
    y = np.array([[[2.0 + 0.0j]], [[1.0 + 2.0j]]])

    assert normalize_kpts(kpts).shape == (2, 3)
    np.testing.assert_allclose(weights, [0.5, 0.5])
    assert weighted_inner_product(x, y, weights) == pytest.approx(1.0)
    assert kpoint_fingerprint(kpts, weights, wrap_around=True) != (
        kpoint_fingerprint(kpts[::-1], weights, wrap_around=True))

    with pytest.raises(NotImplementedError, match='KPoints/IBZ'):
        normalize_kpts(KPoints(_make_cell(), kpts))
    with pytest.raises(NotImplementedError, match='fractional occupations'):
        validate_insulating_occupations(
            [[-1.0, 1.0], [-0.9, 1.1]],
            [[1.9, 0.1], [2.0, 0.0]])
    for near_integer in ([2.0, 1e-12], [1.999999, 0.0]):
        with pytest.raises(NotImplementedError, match='exactly 2 or 0'):
            validate_insulating_occupations(
                [[-1.0, 1.0], [-0.9, 1.1]],
                [near_integer, [2.0, 0.0]])


def test_krhfdft_uses_one_fftdf_without_target_scf(sampled_result):
    mf = sampled_result

    assert isinstance(mf, krks.KRKS)
    assert isinstance(mf.with_df, FFTDF)
    assert mf.with_df is mf.mf_hf.with_df
    np.testing.assert_array_equal(mf.kpts, mf.mf_hf.kpts)
    np.testing.assert_array_equal(mf.kpts, mf.with_df.kpts)
    np.testing.assert_allclose(mf.kpts_weights, 0.5)
    assert mf.dm_hf.shape == (2, mf.cell.nao_nr(), mf.cell.nao_nr())
    assert mf.target_fock.shape == mf.dm_hf.shape
    assert mf.source_gap > 0.0
    assert mf.with_df.capability()['sampling'] == 'full_bz'
    assert mf.with_df.capability()['analytic_gradient']


def test_krhfdft_matches_independent_pyscf_supplied_density(sampled_result):
    mf = sampled_result
    cell = mf.cell
    kpts = np.asarray(mf.kpts).copy()
    provider = df.FFTDF(cell, kpts=kpts)
    provider.mesh = np.asarray(cell.mesh).copy()
    mf_hf = scf.KRHF(cell, kpts=kpts, exxdiv='ewald')
    mf_hf.with_df = provider
    mf_hf.conv_tol = mf.conv_tol
    mf_hf.max_cycle = mf.max_cycle
    e_hf = mf_hf.kernel()
    assert mf_hf.converged

    mf_dft = dft.KRKS(cell, kpts=kpts, xc='PBE', exxdiv='ewald')
    mf_dft.with_df = provider
    dm_hf = mf_hf.make_rdm1()
    h1e = mf_dft.get_hcore(cell, kpts)
    vhf = mf_dft.get_veff(cell, dm_hf, kpts=kpts)
    e_dft = mf_dft.energy_tot(dm=dm_hf, h1e=h1e, vhf=vhf)

    assert mf.e_hf == pytest.approx(e_hf, abs=1e-10)
    assert mf.e_dft == pytest.approx(e_dft, abs=1e-10)


def test_explicit_gamma_collapses_to_rhfdft():
    cell = _make_cell()
    e_gamma = RHFDFT(cell, xc='PBE').kernel()
    e_kgamma = KRHFDFT(
        cell, kpts=np.zeros((1, 3)), xc='PBE').kernel()

    assert e_kgamma == pytest.approx(e_gamma, abs=1e-10)


def test_sampled_k_boundaries_fail_before_expensive_work():
    cell = _make_cell()
    kpts = _make_kpts(cell)
    with pytest.raises(NotImplementedError, match='nonuniform'):
        KRHFDFT(cell, kpts=kpts, kpts_weights=[0.4, 0.6])

    assert KRHFDFT(cell, kpts=kpts).nuc_grad_method().base.cell is cell
