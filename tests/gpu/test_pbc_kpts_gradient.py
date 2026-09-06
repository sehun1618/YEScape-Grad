'''GPU sampled-k FFTDF analytic-gradient checks.'''
import cupy as cp
import numpy as np
import pytest
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import KRHFDFT as CPUKRHFDFT
from dcforge.gpu.pbc.energy import KRHFDFT as GPUKRHFDFT
from dcforge.gpu.pbc.grad.krhf_zvec import (
    gen_hf_response,
    make_fvo,
    make_mo_partitions,
    solve_z_vector,
)

pytestmark = [pytest.mark.gpu, pytest.mark.pbc, pytest.mark.slow]


def _make_cell(coords=None):
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
    if coords is not None:
        cell.set_geom_(coords, unit='Bohr')
    return cell


def _run(workflow, cell, xc='PBE', provider='fft'):
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    mf = workflow(
        cell, kpts=kpts, xc=xc, kpts_wrap_around=True)
    if provider == 'gdf':
        mf = mf.density_fit(auxbasis='gth-szv')
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-7
    mf.max_cycle = 60
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def result():
    cell = _make_cell()
    cpu = _run(CPUKRHFDFT, cell)
    gpu = _run(GPUKRHFDFT, cell)
    cpu_gradient = cpu.nuc_grad_method().kernel()
    gpu_grad = gpu.nuc_grad_method()
    gpu_grad.cphf_conv_tol = 1e-9
    gpu_grad.cphf_residual_tol = 1e-8
    gpu_gradient = gpu_grad.kernel()
    return cpu, gpu, gpu_grad, cpu_gradient, gpu_gradient


@pytest.fixture(scope='module')
def gdf_result():
    cell = _make_cell()
    cpu = _run(CPUKRHFDFT, cell, provider='gdf')
    gpu = _run(GPUKRHFDFT, cell, provider='gdf')
    cpu_gradient = cpu.nuc_grad_method().kernel()
    gpu_grad = gpu.nuc_grad_method()
    gpu_gradient = gpu_grad.kernel()
    return cpu, gpu, gpu_grad, cpu_gradient, gpu_gradient


def test_gradient_matches_cpu_and_recomposes(result):
    _cpu, _gpu, mf_grad, cpu_gradient, gpu_gradient = result

    np.testing.assert_allclose(gpu_gradient, cpu_gradient, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        gpu_gradient, mf_grad.de_terms, atol=1e-12, rtol=0)
    assert mf_grad.zvec_method_used == 'matrix_free'
    assert mf_grad.z_residual_max_abs < 1e-8
    assert mf_grad.fock_derivative_reverse_error < 1e-10


def test_matrix_free_z_vector_matches_dense_reference(result):
    _cpu, mf, mf_grad, _cpu_gradient, _gpu_gradient = result
    Co, Cv, _gaps = make_mo_partitions(
        mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    fvo = make_fvo(mf, Co, Cv)
    dense, info = solve_z_vector(
        mf.mo_energy, mf.mo_occ, Co, Cv, fvo,
        gen_hf_response(mf.mf_hf), residual_tol=1e-10,
        dense_reference_size=512, kpts_weights=mf.kpts_weights)

    assert info['residual_max_abs'] < 1e-10
    for actual, expected in zip(mf_grad.zvec, dense, strict=True):
        cp.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_gdf_gradient_matches_cpu_and_records_native_boundary(gdf_result):
    _cpu, _gpu, mf_grad, cpu_gradient, gpu_gradient = gdf_result

    np.testing.assert_allclose(gpu_gradient, cpu_gradient, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        gpu_gradient, mf_grad.de_terms, atol=1e-12, rtol=0)
    assert mf_grad.zvec_method_used == 'matrix_free'
    assert mf_grad.z_residual_max_abs < 1e-8
    assert mf_grad.provider_response_backend == 'gpu4pyscf_native_combined'
    assert mf_grad.gdf_response_metadata['quadratic_native_calls'] == 3
    assert mf_grad.gdf_response_metadata['native_bvk_kmesh'] == [2, 1, 1]
    assert mf_grad.gdf_native_host_result_bytes > 0
    assert mf_grad.gaussian_hcore_host_calls == 1
    assert mf_grad.gaussian_hcore_contraction_calls == 2
    assert mf_grad.vj1 is None
    assert mf_grad.vk1 is None


def test_r2scan_gradient_matches_cpu():
    cell = _make_cell()
    cpu = _run(CPUKRHFDFT, cell, xc='R2SCAN')
    gpu = _run(GPUKRHFDFT, cell, xc='R2SCAN')
    cpu_gradient = cpu.nuc_grad_method().kernel()
    gpu_grad = gpu.nuc_grad_method()
    gpu_gradient = gpu_grad.kernel()

    assert gpu.e_dft == pytest.approx(cpu.e_dft, abs=1e-8)
    np.testing.assert_allclose(gpu_gradient, cpu_gradient, atol=1e-8, rtol=0)
    assert gpu_grad.z_residual_max_abs < 1e-8


def test_gdf_r2scan_gradient_matches_cpu():
    cell = _make_cell()
    cpu = _run(CPUKRHFDFT, cell, xc='R2SCAN', provider='gdf')
    gpu = _run(GPUKRHFDFT, cell, xc='R2SCAN', provider='gdf')
    cpu_gradient = cpu.nuc_grad_method().kernel()
    gpu_grad = gpu.nuc_grad_method()
    gpu_gradient = gpu_grad.kernel()

    assert gpu.e_dft == pytest.approx(cpu.e_dft, abs=1e-8)
    np.testing.assert_allclose(gpu_gradient, cpu_gradient, atol=1e-8, rtol=0)
    assert gpu_grad.z_residual_max_abs < 1e-8


@pytest.mark.fd
def test_selected_relaxed_source_finite_difference(result):
    _cpu, _gpu, _mf_grad, _cpu_gradient, gpu_gradient = result
    cell = _make_cell()
    coords = cell.atom_coords()
    step = 1e-4
    energies = []
    for sign in (1.0, -1.0):
        displaced = coords.copy()
        displaced[0, 2] += sign * step
        energies.append(_run(GPUKRHFDFT, _make_cell(displaced)).e_dft)
    numerical = (energies[0] - energies[1]) / (2 * step)

    assert gpu_gradient[0, 2] == pytest.approx(numerical, abs=2e-6)


@pytest.mark.fd
def test_gdf_selected_relaxed_source_finite_difference(gdf_result):
    _cpu, _gpu, _mf_grad, _cpu_gradient, gpu_gradient = gdf_result
    cell = _make_cell()
    coords = cell.atom_coords()
    step = 1e-4
    energies = []
    for sign in (1.0, -1.0):
        displaced = coords.copy()
        displaced[0, 2] += sign * step
        energies.append(_run(
            GPUKRHFDFT, _make_cell(displaced), provider='gdf').e_dft)
    numerical = (energies[0] - energies[1]) / (2 * step)

    assert gpu_gradient[0, 2] == pytest.approx(numerical, abs=2e-6)
