"""Finite-difference validation of complete CPU HF-DFT gradients."""

import numpy as np
import pytest
from pyscf.data import nist

from dcforge.cpu.energy import RHFDFT, UHFDFT
from tests.common import (
    finite_difference,
    finite_difference_gradient,
    make_mol,
    run_workflow,
)


@pytest.mark.cpu
@pytest.mark.fd
@pytest.mark.parametrize("density_fit", [False, True], ids=["conventional", "df"])
def test_rhf_gradient_matches_relaxed_source_finite_difference(density_fit):
    mol = make_mol("water1.xyz", basis="def2-svp")
    mf, _ = run_workflow(RHFDFT, mol, density_fit=density_fit)
    analytic = mf.nuc_grad_method().kernel()
    fd = finite_difference_gradient(RHFDFT, mol, density_fit=density_fit)
    error = np.max(np.abs(analytic - fd))
    print(f'CPU RHF DF={density_fit}: max |analytic-FD| = {error:.6e} '
          f'Ha/Bohr = {error * nist.HARTREE2EV / nist.BOHR:.6e} eV/Angstrom')
    np.testing.assert_allclose(analytic, fd, atol=2e-6, rtol=0)


@pytest.mark.cpu
@pytest.mark.fd
@pytest.mark.parametrize("density_fit", [False, True], ids=["conventional", "df"])
def test_uhf_gradient_matches_relaxed_source_finite_difference(density_fit):
    mol = make_mol("nh2_doublet.xyz", basis="def2-svp")
    mf, _ = run_workflow(UHFDFT, mol, density_fit=density_fit)
    analytic = mf.nuc_grad_method().kernel()
    fd = finite_difference_gradient(UHFDFT, mol, density_fit=density_fit)
    error = np.max(np.abs(analytic - fd))
    print(f'CPU UHF DF={density_fit}: max |analytic-FD| = {error:.6e} '
          f'Ha/Bohr = {error * nist.HARTREE2EV / nist.BOHR:.6e} eV/Angstrom')
    np.testing.assert_allclose(analytic, fd, atol=3e-6, rtol=0)


@pytest.mark.cpu
@pytest.mark.fd
@pytest.mark.slow
def test_rhf_d2c_total_gradient_matches_total_energy_finite_difference():
    pytest.importorskip('dftd4')
    mol = make_mol("water1_asymmetric_stretch.xyz", basis="sto-3g")
    mf, _ = run_workflow(RHFDFT, mol, xc="r2SCAN", dispersion="d2c")
    analytic = mf.nuc_grad_method().kernel()
    fd = finite_difference(
        RHFDFT, mol, 1, 2, xc="r2SCAN", dispersion="d2c")
    assert abs(analytic[1, 2] - fd) < 3e-6


@pytest.mark.cpu
@pytest.mark.fd
@pytest.mark.slow
def test_rhf_water_all_cartesian_components_match_finite_difference():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf, _ = run_workflow(RHFDFT, mol)
    analytic = mf.nuc_grad_method().kernel()
    fd = finite_difference_gradient(RHFDFT, mol)
    assert np.max(np.abs(analytic - fd)) < 2e-6
