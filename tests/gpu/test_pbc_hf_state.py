'''GPU PBC gradient-ready source-state checkpoint checks.'''

import numpy as np
import pytest
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import KRHFDFT, RHFDFT
from tests.common import as_numpy, make_water_cell

pytestmark = [pytest.mark.gpu, pytest.mark.pbc, pytest.mark.slow]


def _make_sampled_cell():
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


def _forbid_source_scf(mf):
    def fail(*args, **kwargs):
        raise AssertionError('restored workflow reran source SCF')
    mf.mf_hf.kernel = fail


def test_gamma_gdf_state_restores_energy_and_gradient(tmp_path):
    cell = make_water_cell()
    mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-7
    energy = mf.kernel()
    gradient = as_numpy(mf.nuc_grad_method().kernel())
    checkpoint = tmp_path / 'gpu-gamma-gdf.chk'
    mf.save_hf_state(checkpoint)

    restored = RHFDFT.from_hf_state(checkpoint, xc='PBE')
    _forbid_source_scf(restored)
    restored_energy = restored.kernel(run_hf=False)
    restored_gradient = as_numpy(restored.nuc_grad_method().kernel())

    assert restored.source_validation.origin == 'restored'
    assert restored_energy == pytest.approx(energy, abs=1e-9)
    np.testing.assert_allclose(
        restored_gradient, gradient, atol=1e-8, rtol=0)


def test_sampled_k_fftdf_state_restores_energy_and_gradient(tmp_path):
    cell = _make_sampled_cell()
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    mf = KRHFDFT(
        cell, kpts=kpts, xc='PBE', kpts_wrap_around=True)
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-7
    energy = mf.kernel()
    gradient = as_numpy(mf.nuc_grad_method().kernel())
    checkpoint = tmp_path / 'gpu-sampled-k-fft.chk'
    mf.save_hf_state(checkpoint)

    restored = KRHFDFT.from_hf_state(checkpoint, xc='PBE')
    _forbid_source_scf(restored)
    restored_energy = restored.kernel(run_hf=False)
    restored_gradient = as_numpy(restored.nuc_grad_method().kernel())

    assert restored.source_validation.origin == 'restored'
    assert restored_energy == pytest.approx(energy, abs=1e-9)
    np.testing.assert_allclose(
        restored_gradient, gradient, atol=1e-8, rtol=0)
