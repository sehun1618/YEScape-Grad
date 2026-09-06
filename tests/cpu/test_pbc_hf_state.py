'''CPU PBC gradient-ready source-state checkpoint checks.'''

import numpy as np
import pytest
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import KRHFDFT, RHFDFT
from dcforge.tools.pbc_hf_state import PBCStateError
from tests.common import make_water_cell

pytestmark = [pytest.mark.cpu, pytest.mark.pbc, pytest.mark.slow]


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
    gradient = mf.nuc_grad_method().kernel()
    checkpoint = tmp_path / 'gamma-gdf.chk'
    mf.save_hf_state(checkpoint)

    restored = RHFDFT.from_hf_state(checkpoint, xc='PBE')
    _forbid_source_scf(restored)
    restored_energy = restored.kernel(run_hf=False)
    restored_gradient = restored.nuc_grad_method().kernel()

    assert restored.source_validation.origin == 'restored'
    assert restored_energy == pytest.approx(energy, abs=1e-10)
    np.testing.assert_allclose(restored_gradient, gradient, atol=1e-9, rtol=0)

    wrong_provider = RHFDFT(cell, xc='PBE').fft_density_fit()
    with pytest.raises(PBCStateError, match='configurations differ'):
        wrong_provider.restore_hf_state(checkpoint)


def test_sampled_k_fftdf_state_restores_energy_and_gradient(tmp_path):
    cell = _make_sampled_cell()
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    mf = KRHFDFT(
        cell, kpts=kpts, xc='PBE', kpts_wrap_around=True)
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-7
    energy = mf.kernel()
    gradient = mf.nuc_grad_method().kernel()
    checkpoint = tmp_path / 'sampled-k-fft.chk'
    mf.save_hf_state(checkpoint)

    restored = KRHFDFT.from_hf_state(checkpoint, xc='PBE')
    _forbid_source_scf(restored)
    restored_energy = restored.kernel(run_hf=False)
    restored_gradient = restored.nuc_grad_method().kernel()

    assert restored.source_validation.origin == 'restored'
    assert restored_energy == pytest.approx(energy, abs=1e-10)
    np.testing.assert_allclose(restored_gradient, gradient, atol=1e-9, rtol=0)


def test_state_mutation_revokes_reuse_authority(tmp_path):
    cell = make_water_cell()
    mf = RHFDFT(cell, xc='PBE').fft_density_fit()
    mf.conv_tol = 1e-9
    mf.kernel_hf()
    checkpoint = tmp_path / 'gamma-fft.chk'
    mf.save_hf_state(checkpoint)

    restored = RHFDFT.from_hf_state(checkpoint, xc='PBE')
    restored.mf_hf.mo_energy[0] += 1e-4
    with pytest.raises(PBCStateError, match='changed after validation'):
        restored.kernel(run_hf=False)
    assert restored.source_validation is None
