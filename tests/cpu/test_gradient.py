"""CPU analytic-gradient route, ledger, and capability tests."""

import numpy as np
import pytest

from dcforge.cpu.energy import RHFDFT, UHFDFT
from tests.common import make_mol, run_workflow


def assert_gradient_ledger(mf_grad, gradient):
    assert np.all(np.isfinite(gradient))
    assert gradient.shape == (mf_grad.mol.natm, 3)
    assert np.allclose(
        gradient,
        mf_grad.de_skel + mf_grad.de_nvar + mf_grad.de_d2c,
        atol=2e-10,
        rtol=0,
    )
    assert np.allclose(
        mf_grad.de_skel_elec,
        mf_grad.de_hcore + mf_grad.de_veff + mf_grad.de_ovlp
        + mf_grad.de_grid + mf_grad.de_aux,
        atol=2e-10,
        rtol=0,
    )
    assert np.isfinite(mf_grad.z_residual_max_abs)
    assert mf_grad.z_residual_warning == (
        mf_grad.z_residual_max_abs > mf_grad.cphf_residual_warn_tol)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "workflow,name",
    [(RHFDFT, "water1.xyz"), (UHFDFT, "o2_triplet.xyz")],
)
def test_conventional_gradient_has_complete_term_ledger(workflow, name):
    mol = make_mol(name, basis="sto-3g")
    mf, _ = run_workflow(workflow, mol)
    mf_grad = mf.nuc_grad_method()
    gradient = mf_grad.kernel()
    assert_gradient_ledger(mf_grad, gradient)
    assert np.max(np.abs(mf_grad.de_aux)) == 0
    assert mf_grad.fock_deriv_method_used == "direct"


@pytest.mark.cpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "workflow,name",
    [(RHFDFT, "water1.xyz"), (UHFDFT, "o2_triplet.xyz")],
)
def test_df_gradient_has_auxiliary_response_and_shared_provider(workflow, name):
    mol = make_mol(name, basis="sto-3g")
    mf, _ = run_workflow(workflow, mol, density_fit=True)
    mf_grad = mf.nuc_grad_method()
    gradient = mf_grad.kernel()
    assert_gradient_ledger(mf_grad, gradient)
    assert mf.with_df is mf.mf_hf.with_df
    assert mf_grad.auxbasis_response is True


@pytest.mark.cpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "workflow,name",
    [(RHFDFT, "water1.xyz"), (UHFDFT, "o2_triplet.xyz")],
)
@pytest.mark.parametrize("density_fit", [False, True], ids=["conventional", "df"])
def test_direct_and_make_h1_fock_derivative_routes_agree(
        workflow, name, density_fit):
    mol = make_mol(name, basis="sto-3g")
    mf, _ = run_workflow(workflow, mol, density_fit=density_fit)
    grad_direct = mf.nuc_grad_method()
    grad_direct.fock_deriv_method = "direct"
    de_direct = grad_direct.kernel()
    grad_make_h1 = mf.nuc_grad_method()
    grad_make_h1.fock_deriv_method = "make_h1"
    de_make_h1 = grad_make_h1.kernel()
    assert np.allclose(de_direct, de_make_h1, atol=2e-9, rtol=0)


@pytest.mark.cpu
def test_cpu_atom_subset_matches_full_gradient():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf, _ = run_workflow(RHFDFT, mol)
    full = mf.nuc_grad_method().kernel()
    subset = mf.nuc_grad_method().kernel(atmlst=[1])
    assert np.allclose(subset, full[[1]], atol=2e-10, rtol=0)


@pytest.mark.cpu
def test_cpu_conventional_gradient_can_omit_grid_response():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf, _ = run_workflow(RHFDFT, mol)
    mf_grad = mf.nuc_grad_method()
    mf_grad.grid_response = False
    gradient = mf_grad.kernel()
    assert np.all(np.isfinite(gradient))
    assert np.max(np.abs(mf_grad.de_grid)) == 0


@pytest.mark.cpu
@pytest.mark.parametrize("option", ["grid_response", "auxbasis_response"])
def test_df_partial_representation_response_is_rejected(option):
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf, _ = run_workflow(RHFDFT, mol, density_fit=True)
    mf_grad = mf.nuc_grad_method()
    setattr(mf_grad, option, False)
    with pytest.raises(NotImplementedError, match=option):
        mf_grad.kernel()


@pytest.mark.cpu
def test_d2c_is_an_explicit_additive_gradient_term():
    pytest.importorskip('dftd4')
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf, _ = run_workflow(RHFDFT, mol, xc="r2SCAN", dispersion="d2c")
    mf_grad = mf.nuc_grad_method()
    gradient = mf_grad.kernel()
    assert_gradient_ledger(mf_grad, gradient)
    assert mf_grad.d2c_result is not None
    assert np.allclose(mf_grad.de_d2c, mf_grad.d2c_result.gradient)
