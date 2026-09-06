"""GPU analytic-gradient ledger, route, and CPU parity tests."""

import numpy as np
import pytest

from dcforge.cpu.energy import RHFDFT as CPU_RHFDFT
from dcforge.cpu.energy import UHFDFT as CPU_UHFDFT
from dcforge.gpu.energy import RHFDFT as GPU_RHFDFT
from dcforge.gpu.energy import UHFDFT as GPU_UHFDFT
from tests.common import as_numpy, make_mol, run_workflow


pytestmark = pytest.mark.gpu


def assert_gradient_ledger(mf_grad, gradient):
    gradient = as_numpy(gradient)
    assert np.all(np.isfinite(gradient))
    assert gradient.shape == (mf_grad.mol.natm, 3)
    assert np.allclose(
        gradient,
        mf_grad.de_skel + mf_grad.de_nvar + mf_grad.de_d2c,
        atol=3e-9,
        rtol=0,
    )
    assert np.isfinite(mf_grad.z_residual_max_abs)
    assert mf_grad.z_residual_warning == (
        mf_grad.z_residual_max_abs > mf_grad.cphf_residual_warn_tol)


@pytest.mark.parametrize(
    "cpu_workflow,gpu_workflow,name",
    [
        (CPU_RHFDFT, GPU_RHFDFT, "water1.xyz"),
        (CPU_UHFDFT, GPU_UHFDFT, "o2_triplet.xyz"),
    ],
)
@pytest.mark.parametrize("density_fit", [False, True], ids=["conventional", "df"])
def test_gpu_gradient_matches_cpu(
        cpu_workflow, gpu_workflow, name, density_fit):
    mol = make_mol(name, basis="sto-3g")
    mf_cpu, _ = run_workflow(cpu_workflow, mol, density_fit=density_fit)
    de_cpu = mf_cpu.nuc_grad_method().kernel()

    mf_gpu, _ = run_workflow(gpu_workflow, mol, density_fit=density_fit)
    mf_grad = mf_gpu.nuc_grad_method()
    de_gpu = as_numpy(mf_grad.kernel())

    assert_gradient_ledger(mf_grad, de_gpu)
    assert np.allclose(de_gpu, de_cpu, atol=2e-7, rtol=0)
    assert mf_grad.fock_deriv_method_used == "direct"
    expected_backend = "mo_direct" if density_fit else "ao"
    assert mf_grad.response_backend_used == expected_backend


@pytest.mark.slow
@pytest.mark.parametrize(
    "workflow,name",
    [(GPU_RHFDFT, "water1.xyz"), (GPU_UHFDFT, "o2_triplet.xyz")],
)
@pytest.mark.parametrize("density_fit", [False, True], ids=["conventional", "df"])
def test_gpu_direct_and_make_h1_routes_agree(workflow, name, density_fit):
    mol = make_mol(name, basis="sto-3g")
    mf, _ = run_workflow(workflow, mol, density_fit=density_fit)
    grad_direct = mf.nuc_grad_method()
    grad_direct.fock_deriv_method = "direct"
    direct = as_numpy(grad_direct.kernel())
    grad_make_h1 = mf.nuc_grad_method()
    grad_make_h1.fock_deriv_method = "make_h1"
    make_h1 = as_numpy(grad_make_h1.kernel())
    assert np.allclose(direct, make_h1, atol=3e-7, rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "workflow,name",
    [(GPU_RHFDFT, "water1.xyz"), (GPU_UHFDFT, "o2_triplet.xyz")],
)
def test_gpu_df_mo_direct_response_matches_ao_reference(workflow, name):
    mol = make_mol(name, basis="sto-3g")
    mf, _ = run_workflow(workflow, mol, density_fit=True)
    grad_mo = mf.nuc_grad_method()
    grad_mo.response_backend = "mo_direct"
    de_mo = as_numpy(grad_mo.kernel())
    grad_ao = mf.nuc_grad_method()
    grad_ao.response_backend = "ao"
    de_ao = as_numpy(grad_ao.kernel())
    assert np.allclose(de_mo, de_ao, atol=3e-7, rtol=0)
