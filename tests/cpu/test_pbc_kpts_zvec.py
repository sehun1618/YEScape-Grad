'''Dense real-linear sampled-k KRHF Z-vector checks.'''

import numpy as np
import pytest
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import KRHFDFT, RHFDFT
from dcforge.cpu.pbc.grad.krhf_zvec import (
    apply_real_adjoint,
    build_real_linear_matrix,
    complex_to_real,
    flatten_vo,
    gen_hf_response,
    get_complex_linearity_error,
    make_fvo,
    make_hessian_operator,
    make_mo_partitions,
    make_z_dm,
    real_to_complex,
    solve_z_vector,
    solve_z_vector_matrix_free,
    unflatten_vo,
    weighted_vo_inner,
)
from dcforge.cpu.pbc.grad.rhf_zvec import solve_z_vector as solve_gamma_z_vector

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


@pytest.fixture(scope='module')
def sampled_response():
    cell = _make_cell()
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    mf = KRHFDFT(
        cell, kpts=kpts, xc='PBE', kpts_wrap_around=True).run()
    Co, Cv, _ = make_mo_partitions(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    fvo = make_fvo(mf, Co, Cv)
    fvind = gen_hf_response(mf.mf_hf)
    vind = make_hessian_operator(
        mf.mo_energy, mf.mo_occ, Co, Cv, fvind)
    zvec, info = solve_z_vector(
        mf.mo_energy, mf.mo_occ, Co, Cv, fvo, fvind,
        kpts_weights=mf.kpts_weights)
    return mf, Co, Cv, fvo, fvind, vind, zvec, info


def _random_blocks(Co, Cv, seed):
    rng = np.random.default_rng(seed)
    return tuple(
        rng.normal(size=(Cv_k.shape[1], Co_k.shape[1]))
        + 1j * rng.normal(size=(Cv_k.shape[1], Co_k.shape[1]))
        for Co_k, Cv_k in zip(Co, Cv, strict=True)
    )


def test_real_expansion_retains_the_conjugate_channel():
    template = (np.zeros((1, 1), dtype=np.complex128),)

    def operator(blocks):
        z = blocks[0]
        return (2 * z + z.conj(),)

    matrix, shapes = build_real_linear_matrix(operator, template)
    vector = np.array([1.5 - 0.25j])
    np.testing.assert_allclose(matrix, np.diag([3.0, 1.0]))
    np.testing.assert_allclose(
        unflatten_vo(real_to_complex(complex_to_real(vector)), shapes)[0],
        vector.reshape(1, 1))


def test_target_pbe_fvo_drives_stationary_krhf(sampled_response):
    mf, Co, Cv, fvo, _, _, _, _ = sampled_response
    fock_hf = np.asarray(mf.mf_hf.get_fock())
    fvo_hf = tuple(
        Cv_k.conj().T @ fock_k @ Co_k
        for fock_k, Co_k, Cv_k in zip(fock_hf, Co, Cv, strict=True))

    assert max(np.max(np.abs(block)) for block in fvo_hf) < 1e-8
    assert max(np.max(np.abs(block)) for block in fvo) > 1e-6


def test_hessian_is_real_linear_and_not_complex_linear(sampled_response):
    _, Co, Cv, _, _, vind, _, _ = sampled_response
    x = _random_blocks(Co, Cv, 11)
    y = _random_blocks(Co, Cv, 29)
    lhs = vind(tuple(.37 * x_k - 1.21 * y_k
                     for x_k, y_k in zip(x, y, strict=True)))
    ax = vind(x)
    ay = vind(y)
    rhs = tuple(.37 * ax_k - 1.21 * ay_k
                for ax_k, ay_k in zip(ax, ay, strict=True))
    for lhs_k, rhs_k in zip(lhs, rhs, strict=True):
        np.testing.assert_allclose(lhs_k, rhs_k, atol=2e-13, rtol=0)
    assert get_complex_linearity_error(vind, x) > 1e-4


def test_dense_adjoint_has_a_fresh_physical_residual(sampled_response):
    mf, Co, Cv, fvo, _, vind, zvec, info = sampled_response
    shapes = tuple(block.shape for block in fvo)
    x = _random_blocks(Co, Cv, 53)
    y = _random_blocks(Co, Cv, 71)
    at_x = unflatten_vo(real_to_complex(apply_real_adjoint(
        vind, complex_to_real(flatten_vo(x)), shapes)), shapes)

    assert weighted_vo_inner(
        x, vind(y), mf.kpts_weights) == pytest.approx(
            weighted_vo_inner(at_x, y, mf.kpts_weights), abs=2e-13)
    assert len(zvec) == 2
    assert info['solver'] == 'dense_real_linear'
    assert info['equation'] == 'adjoint'
    assert info['dimension'] == 4
    assert info['residual_max_abs'] <= 1e-10
    assert info['matrix_asymmetry_max_abs'] <= 1e-12
    assert info['direct_adjoint_max_abs'] <= 1e-12
    assert np.isfinite(info['condition_number'])
    assert make_z_dm(Co, Cv, zvec).shape == (
        2, mf.cell.nao_nr(), mf.cell.nao_nr())


def test_one_gamma_zvector_collapses_to_existing_real_solver():
    cell_k = _make_cell()
    mf_k = KRHFDFT(cell_k, kpts=np.zeros((1, 3)), xc='PBE').run()
    Co_k, Cv_k, _ = make_mo_partitions(
        mf_k.mo_energy, mf_k.mo_coeff, mf_k.mo_occ)
    fvo_k = make_fvo(mf_k, Co_k, Cv_k)
    z_k, info_k = solve_z_vector(
        mf_k.mo_energy, mf_k.mo_occ, Co_k, Cv_k, fvo_k,
        gen_hf_response(mf_k.mf_hf))

    cell_g = _make_cell()
    mf_g = RHFDFT(cell_g, xc='PBE').run()
    mo_energy = np.asarray(mf_g.mo_energy)
    mo_coeff = np.asarray(mf_g.mo_coeff)
    mo_occ = np.asarray(mf_g.mo_occ)
    so = mo_occ > 0
    sv = mo_occ == 0
    Co_g = mo_coeff[:, so]
    Cv_g = mo_coeff[:, sv]
    fvo_g = Cv_g.conj().T @ np.asarray(mf_g.target_fock) @ Co_g
    fvind_g = mf_g.mf_hf.gen_response(
        mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=1)
    z_g, info_g = solve_gamma_z_vector(
        mo_energy, mo_occ, Co_g, Cv_g, fvo_g, fvind_g,
        residual_tol=1e-10)

    np.testing.assert_allclose(z_k[0], z_g, atol=2e-13, rtol=0)
    assert info_k['residual_max_abs'] <= 1e-10
    assert info_g['residual_max_abs'] <= 1e-10


def test_dense_reference_rejects_production_size(sampled_response):
    mf, Co, Cv, fvo, fvind, _, _, _ = sampled_response
    with pytest.raises(RuntimeError, match='matrix-free'):
        solve_z_vector(
            mf.mo_energy, mf.mo_occ, Co, Cv, fvo, fvind,
            dense_reference_size=3)


def test_matrix_free_gmres_matches_dense_adjoint(sampled_response):
    mf, Co, Cv, fvo, fvind, _, z_dense, _ = sampled_response
    z_matrix_free, info = solve_z_vector_matrix_free(
        mf.mo_energy, mf.mo_occ, Co, Cv, fvo, fvind,
        conv_tol=1e-11, residual_tol=1e-10, max_cycle=200,
        kpts_weights=mf.kpts_weights)

    for actual, reference in zip(z_matrix_free, z_dense, strict=True):
        np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=0)
    assert info['solver'] == 'scipy_gmres_matrix_free'
    assert info['residual_max_abs'] <= 1e-10
    assert info['self_adjoint_probe_error'] <= 1e-10
    assert info['iterations'] > 0
    assert info['dense_matrix_bytes'] == 0
    assert info['workspace_estimate_bytes'] > 0
