'''GPU sampled-k FFTDF/GDF fixed-density energy checks.'''

import numpy as np
import pytest
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import KRHFDFT as CPUKRHFDFT
from dcforge.gpu.pbc.energy import KRHFDFT as GPUKRHFDFT

pytestmark = [pytest.mark.gpu, pytest.mark.pbc, pytest.mark.slow]


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


def _run_pair(provider):
    cell = _make_cell()
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    cpu = CPUKRHFDFT(
        cell, kpts=kpts, xc='PBE', kpts_wrap_around=True)
    gpu = GPUKRHFDFT(
        cell, kpts=kpts, xc='PBE', kpts_wrap_around=True)
    if provider == 'gdf':
        cpu = cpu.density_fit(auxbasis='gth-szv')
        gpu = gpu.density_fit(auxbasis='gth-szv')
    for mf in (cpu, gpu):
        mf.conv_tol = 1e-9
        mf.conv_tol_grad = 1e-7
        mf.max_cycle = 60
    return cpu, gpu, cpu.kernel(), gpu.kernel()


@pytest.mark.parametrize('provider', ['fft', 'gdf'])
def test_sampled_k_energy_matches_cpu(provider):
    cpu, gpu, e_cpu, e_gpu = _run_pair(provider)

    assert gpu.with_df is gpu.mf_hf.with_df
    assert gpu.with_df.capability()['sampling'] == 'full_bz'
    assert gpu.dm_hf.shape == (2, gpu.cell.nao_nr(), gpu.cell.nao_nr())
    assert gpu.target_fock.shape == gpu.dm_hf.shape
    assert gpu.source_gap > 0.0
    assert e_gpu == pytest.approx(e_cpu, abs=1e-8)
    assert gpu.e_hf == pytest.approx(cpu.e_hf, abs=1e-8)
    if provider == 'gdf':
        assert gpu.with_df.sampled_k_hcore_host_calls > 0
        assert gpu.with_df.sampled_k_hcore_owner == (
            'PySCF GDF.get_pp sampled-k')


def test_sampled_k_gdf_gradient_constructs_and_multigrid_requires_cutoff():
    cell = _make_cell()
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    mf = GPUKRHFDFT(
        cell, kpts=kpts, xc='PBE').density_fit(auxbasis='gth-szv')

    assert mf.nuc_grad_method().base is mf
    with pytest.raises(ValueError, match='ke_cutoff'):
        mf.multigrid_numint()
