'''CPU sampled-k FFTDF PBE@KRHF analytic-gradient checks.'''

import numpy as np
import pytest
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import KRHFDFT, RHFDFT
from dcforge.cpu.pbc.grad.krhf_nv import make_s1_oo_vo

FD_STEP = 1e-4
FD_TOL = 2e-6


pytestmark = [pytest.mark.cpu, pytest.mark.pbc, pytest.mark.slow]


def _make_cell(coords=None):
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


def _run_target(cell, kmesh=(2, 1, 1)):
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    mf = KRHFDFT(
        cell, kpts=kpts, xc='PBE', kpts_wrap_around=True)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def _displaced_energy(cell, atom, axis, displacement):
    coords = cell.atom_coords().copy()
    coords[atom, axis] += displacement
    return _run_target(_make_cell(coords)).e_dft


@pytest.fixture(scope='module')
def h2_result():
    cell = _make_cell()
    mf = _run_target(cell)
    # The shared fixture is the independent dense reference.  Public production
    # calculations default to matrix-free and are compared against it below.
    mf_grad = mf.nuc_grad_method().set(
        zvec_method='dense', cphf_residual_tol=1e-10)
    gradient = mf_grad.kernel()
    return cell, mf, mf_grad, gradient


def test_overlap_projection_retains_complex_bloch_conjugation():
    cell = _make_cell()
    nkpts = 2
    nao = cell.nao_nr()
    rng = np.random.default_rng(91)
    Co = rng.normal(size=(nkpts, nao, 1)) + 1j * rng.normal(
        size=(nkpts, nao, 1))
    Cv = rng.normal(size=(nkpts, nao, nao - 1)) + 1j * rng.normal(
        size=(nkpts, nao, nao - 1))
    s1 = rng.normal(size=(nkpts, 3, nao, nao)) + 1j * rng.normal(
        size=(nkpts, 3, nao, nao))

    s1oo, s1vo = make_s1_oo_vo(
        cell, Co, Cv, s1, range(cell.natm))
    reference_oo = np.empty_like(s1oo)
    reference_vo = np.empty_like(s1vo)
    for atom_id, aoslice in enumerate(cell.aoslice_by_atom()):
        p0, p1 = aoslice[2:]
        for axis in range(3):
            for kpoint in range(nkpts):
                full = np.zeros((nao, nao), dtype=np.complex128)
                full[p0:p1] += s1[kpoint, axis, p0:p1]
                full[:, p0:p1] += s1[kpoint, axis, p0:p1].T.conj()
                reference_oo[atom_id, axis, kpoint] = (
                    Co[kpoint].conj().T @ full @ Co[kpoint])
                reference_vo[atom_id, axis, kpoint] = (
                    Cv[kpoint].conj().T @ full @ Co[kpoint])

    np.testing.assert_allclose(s1oo, reference_oo, atol=2e-14, rtol=0)
    np.testing.assert_allclose(s1vo, reference_vo, atol=2e-14, rtol=0)


def test_term_ledger_and_response_diagnostics(h2_result):
    cell, mf, mf_grad, gradient = h2_result

    np.testing.assert_allclose(
        gradient, mf_grad.de_terms, atol=2e-14, rtol=0)
    assert mf_grad.term_recomposition_max_abs < 2e-14
    assert mf_grad.term_imaginary_residual_max_abs < 1e-10
    assert mf_grad.z_residual_max_abs < 1e-10
    assert mf_grad.z_solver == 'dense_real_linear'
    assert mf_grad.target_fock_reused
    np.testing.assert_allclose(
        mf_grad.translation_residual, gradient.sum(axis=0), atol=0, rtol=0)
    assert mf_grad.translation_residual_max_abs == pytest.approx(
        np.max(np.abs(gradient.sum(axis=0))))
    assert gradient.shape == (cell.natm, 3)
    assert mf.target_fock.shape[0] == 2


def test_all_coordinates_match_relaxed_source_finite_difference(h2_result):
    cell, _, _, analytic = h2_result
    numerical = np.zeros_like(analytic)
    for atom in range(cell.natm):
        for axis in range(3):
            e_plus = _displaced_energy(cell, atom, axis, FD_STEP)
            e_minus = _displaced_energy(cell, atom, axis, -FD_STEP)
            numerical[atom, axis] = (e_plus - e_minus) / (2 * FD_STEP)

    np.testing.assert_allclose(analytic, numerical, atol=FD_TOL, rtol=0)


def test_explicit_gamma_gradient_collapses_to_rhfdft():
    cell = _make_cell()
    mf_gamma = RHFDFT(cell, xc='PBE')
    mf_kgamma = KRHFDFT(cell, kpts=np.zeros((1, 3)), xc='PBE')
    mf_gamma.conv_tol = mf_kgamma.conv_tol = 1e-12
    mf_gamma.kernel()
    mf_kgamma.kernel()

    gradient_gamma = mf_gamma.nuc_grad_method().kernel()
    gradient_kgamma = mf_kgamma.nuc_grad_method().kernel()
    np.testing.assert_allclose(
        gradient_kgamma, gradient_gamma, atol=1e-10, rtol=0)


def test_gradient_obeys_kpoint_permutation_and_time_reversal(h2_result):
    cell, mf, _, reference = h2_result
    kpts = np.asarray(mf.kpts)
    for transformed in (kpts[::-1], -kpts):
        candidate = KRHFDFT(
            cell, kpts=transformed, xc='PBE',
            kpts_wrap_around=True)
        candidate.conv_tol = 1e-12
        candidate.kernel()
        gradient = candidate.nuc_grad_method().kernel()
        assert candidate.e_dft == pytest.approx(mf.e_dft, abs=1e-10)
        np.testing.assert_allclose(
            gradient, reference, atol=1e-10, rtol=0)


def test_grid_response_false_is_valid_for_fixed_uniform_grid(h2_result):
    _, mf, _, reference = h2_result
    mf_grad = mf.nuc_grad_method().set(grid_response=False)
    gradient = mf_grad.kernel()

    np.testing.assert_allclose(gradient, reference, atol=2e-14, rtol=0)
    assert np.max(np.abs(mf_grad.de_grid)) == 0


def test_default_matrix_free_complete_gradient_matches_dense(h2_result):
    _, mf, _, dense_gradient = h2_result
    mf_grad = mf.nuc_grad_method().set(
        cphf_conv_tol=1e-11, cphf_residual_tol=1e-10)
    matrix_free_gradient = mf_grad.kernel()

    np.testing.assert_allclose(
        matrix_free_gradient, dense_gradient, atol=2e-10, rtol=0)
    assert mf_grad.zvec_method == 'matrix_free'
    assert mf_grad.zvec_method_used == 'matrix_free'
    assert mf_grad.z_solver == 'scipy_gmres_matrix_free'
    assert mf_grad.z_residual_max_abs <= 1e-10
    assert mf_grad.z_self_adjoint_probe_error <= 1e-10
    assert mf_grad.z_iterations > 0
    assert mf_grad.z_dense_matrix_bytes == 0
    assert mf_grad.z_workspace_estimate_bytes > 0
