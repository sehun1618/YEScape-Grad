"""Finite-difference validation of complete GPU HF-DFT gradients."""

import numpy as np
import pytest
from pyscf.data import nist

from yescape_grad.gpu.energy import RNSC, UNSC
from tests.common import as_numpy, finite_difference_gradient, make_mol, run_workflow


pytestmark = [pytest.mark.gpu, pytest.mark.fd]


@pytest.mark.parametrize("density_fit", [False, True], ids=["conventional", "df"])
def test_gpu_rhf_gradient_matches_relaxed_source_finite_difference(density_fit):
    mol = make_mol("water1.xyz", basis="def2-svp")
    mf, _ = run_workflow(RNSC, mol, density_fit=density_fit)
    analytic = as_numpy(mf.nuc_grad_method().kernel())
    fd = finite_difference_gradient(RNSC, mol, density_fit=density_fit)
    error = np.max(np.abs(analytic - fd))
    print(f'GPU RHF DF={density_fit}: max |analytic-FD| = {error:.6e} '
          f'Ha/Bohr = {error * nist.HARTREE2EV / nist.BOHR:.6e} eV/Angstrom')
    np.testing.assert_allclose(analytic, fd, atol=3e-6, rtol=0)


@pytest.mark.parametrize("density_fit", [False, True], ids=["conventional", "df"])
def test_gpu_uhf_gradient_matches_relaxed_source_finite_difference(density_fit):
    mol = make_mol("nh2_doublet.xyz", basis="def2-svp")
    mf, _ = run_workflow(UNSC, mol, density_fit=density_fit)
    analytic = as_numpy(mf.nuc_grad_method().kernel())
    fd = finite_difference_gradient(UNSC, mol, density_fit=density_fit)
    error = np.max(np.abs(analytic - fd))
    print(f'GPU UHF DF={density_fit}: max |analytic-FD| = {error:.6e} '
          f'Ha/Bohr = {error * nist.HARTREE2EV / nist.BOHR:.6e} eV/Angstrom')
    np.testing.assert_allclose(analytic, fd, atol=4e-6, rtol=0)
