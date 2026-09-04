"""Finite-difference validation of complete GPU HF-DFT gradients."""

import pytest

from dcforge.gpu.energy import RHFDFT, UHFDFT
from tests.common import as_numpy, finite_difference, make_mol, run_workflow


pytestmark = [pytest.mark.gpu, pytest.mark.fd]


@pytest.mark.parametrize("density_fit", [False, True], ids=["conventional", "df"])
def test_gpu_rhf_gradient_matches_relaxed_source_finite_difference(density_fit):
    mol = make_mol("water1.xyz", basis="def2-svp")
    mf, _ = run_workflow(RHFDFT, mol, density_fit=density_fit)
    analytic = as_numpy(mf.nuc_grad_method().kernel())
    fd = finite_difference(
        RHFDFT, mol, 1, 1, density_fit=density_fit)
    assert abs(analytic[1, 1] - fd) < 3e-6


@pytest.mark.parametrize("density_fit", [False, True], ids=["conventional", "df"])
def test_gpu_uhf_gradient_matches_relaxed_source_finite_difference(density_fit):
    mol = make_mol("nh2_doublet.xyz", basis="def2-svp")
    mf, _ = run_workflow(UHFDFT, mol, density_fit=density_fit)
    analytic = as_numpy(mf.nuc_grad_method().kernel())
    fd = finite_difference(
        UHFDFT, mol, 1, 1, density_fit=density_fit)
    assert abs(analytic[1, 1] - fd) < 4e-6
