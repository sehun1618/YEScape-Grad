'''GPU sampled-k XC-only MultiGrid energy and gradient checks.'''

import cupy as cp
import numpy as np
import pytest
from gpu4pyscf.pbc.dft import multigrid_v2
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import KRHFDFT

pytestmark = [pytest.mark.gpu, pytest.mark.pbc, pytest.mark.slow]


def _make_cell(coords=None, ke_cutoff=30):
    cell = gto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 1.4'
    cell.a = np.eye(3) * 6.0
    cell.unit = 'Bohr'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-7
    if ke_cutoff is None:
        cell.mesh = [15, 15, 15]
    else:
        cell.ke_cutoff = ke_cutoff
    cell.verbose = 0
    cell.build()
    if coords is not None:
        cell.set_geom_(coords, unit='Bohr')
    return cell


def _run(cell, provider, xc, multigrid, kpts=None):
    if kpts is None:
        kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    mf = KRHFDFT(
        cell, kpts=kpts, xc=xc, kpts_wrap_around=True)
    if provider == 'gdf':
        mf = mf.density_fit(auxbasis='gth-szv')
    if multigrid:
        mf = mf.multigrid_numint(with_j=False)
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-7
    mf.max_cycle = 60
    mf.kernel()
    return mf


@pytest.mark.parametrize('provider', ['fft', 'gdf'])
@pytest.mark.parametrize('xc', ['PBE', 'R2SCAN'])
def test_xc_only_multigrid_preserves_j_and_matches_relaxed_fd(provider, xc):
    cell = _make_cell()
    uniform = _run(cell, provider, xc, multigrid=False)
    mg = uniform.multigrid_numint(with_j=False)
    mg.kernel(run_hf=False)
    mf_grad = mg.nuc_grad_method()
    gradient = mf_grad.kernel()

    assert mg.with_df is uniform.with_df
    assert mg.with_df is mg.mf_hf.with_df
    assert isinstance(mg._numint, multigrid_v2.MultiGridNumInt)
    assert mg.multigrid_backend == 'gpu4pyscf_multigrid_v2_xc_only'
    assert mg.e_coul == pytest.approx(uniform.e_coul, abs=1e-11)
    assert ((mg.e_dft - uniform.e_dft)
            == pytest.approx(mg.e_xc - uniform.e_xc, abs=1e-11))
    fock = cp.asarray(mg.target_fock)
    cp.testing.assert_allclose(
        fock, fock.conj().transpose(0, 2, 1), atol=1e-10, rtol=0)
    assert mf_grad.xc_deriv_backend == 'multigrid_v2_xc_only'
    assert mf_grad.z_residual_max_abs < 1e-8
    assert mf_grad.term_recomposition_max_abs < 1e-12

    coords = cell.atom_coords()
    step = 1e-4
    energies = []
    for sign in (1.0, -1.0):
        displaced = coords.copy()
        displaced[0, 2] += sign * step
        energies.append(_run(
            _make_cell(displaced), provider, xc, multigrid=True).e_dft)
    numerical = (energies[0] - energies[1]) / (2 * step)
    assert gradient[0, 2] == pytest.approx(numerical, abs=2e-6)


def test_sampled_k_multigrid_rejects_unvalidated_controls():
    cell = _make_cell()
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    mf = KRHFDFT(cell, kpts=kpts, xc='PBE')

    with pytest.raises(NotImplementedError, match=r'J\+XC'):
        mf.multigrid_numint(with_j=True)
    with pytest.raises(ValueError, match='separate'):
        mf.multigrid_numint(mesh=[17, 17, 17])

    no_cutoff = _make_cell(ke_cutoff=None)
    kpts = no_cutoff.make_kpts([2, 1, 1], wrap_around=True)
    mf = KRHFDFT(no_cutoff, kpts=kpts, xc='PBE')
    with pytest.raises(ValueError, match='ke_cutoff'):
        mf.multigrid_numint()


def test_r2scan_fft_multigrid_is_invariant_to_kpoint_order():
    cell = _make_cell()
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    reference = _run(cell, 'fft', 'R2SCAN', True, kpts=kpts)
    reordered = _run(
        _make_cell(), 'fft', 'R2SCAN', True, kpts=kpts[::-1].copy())
    gradient = reference.nuc_grad_method().kernel()
    reordered_gradient = reordered.nuc_grad_method().kernel()

    assert reordered.e_dft == pytest.approx(reference.e_dft, abs=1e-9)
    np.testing.assert_allclose(
        reordered_gradient, gradient, atol=1e-8, rtol=0)


def test_reordered_gdf_kpoints_fail_before_cderi_build():
    cell = _make_cell()
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)[::-1].copy()
    mf = KRHFDFT(
        cell, kpts=kpts, xc='PBE',
        kpts_wrap_around=True).density_fit(auxbasis='gth-szv')

    with pytest.raises(NotImplementedError, match='canonical'):
        mf.kernel()
    assert mf.with_df._cderi is None
