"""CPU fixed-density RHF/UHF energy integration tests."""

from unittest import mock

import numpy as np
import pytest
from pyscf import dft, scf

from dcforge.cpu.energy import RHFDFT, UHFDFT, energy_tot
from tests.common import configure_workflow, make_mol, run_workflow


@pytest.mark.cpu
@pytest.mark.parametrize(
    "workflow,reference,name",
    [
        (RHFDFT, dft.RKS, "water1.xyz"),
        (UHFDFT, dft.UKS, "o2_triplet.xyz"),
    ],
)
@pytest.mark.parametrize("density_fit", [False, True], ids=["conventional", "df"])
def test_fixed_density_energy_matches_independent_pyscf_evaluator(
        workflow, reference, name, density_fit):
    mol = make_mol(name, basis="def2-svp")
    mf, energy = run_workflow(
        workflow, mol, xc="PBE", density_fit=density_fit)

    mf_ref = reference(mol, xc="PBE")
    if density_fit:
        mf_ref = mf_ref.density_fit(auxbasis=mf.with_df.auxbasis)
        assert mf.with_df is mf.mf_hf.with_df
    configure_workflow(mf_ref)
    expected = float(mf_ref.energy_tot(dm=mf.mf_hf.make_rdm1()))

    assert mf.mf_hf.converged
    assert np.shares_memory(np.asarray(mf.mo_occ), np.asarray(mf.mf_hf.mo_occ))
    assert abs(energy - expected) < 1e-10
    assert abs(mf.e_tot - mf.e_dft) < 1e-14


@pytest.mark.cpu
def test_module_energy_tot_uses_supplied_density_without_target_scf():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf_hf = scf.RHF(mol).run(conv_tol=1e-11)
    mf_dft = configure_workflow(dft.RKS(mol, xc="PBE"))
    expected = mf_dft.energy_tot(dm=mf_hf.make_rdm1())
    assert abs(energy_tot(mf_hf, mf_dft) - expected) < 1e-12


@pytest.mark.cpu
@pytest.mark.parametrize(
    "workflow,name,method",
    [
        (RHFDFT, "water1.xyz", "run_hf"),
        (UHFDFT, "o2_triplet.xyz", "run_uhf"),
    ],
)
def test_run_hf_false_reuses_converged_source(workflow, name, method):
    mol = make_mol(name, basis="sto-3g")
    mf, first = run_workflow(workflow, mol, xc="PBE")
    with mock.patch.object(
            mf.mf_hf, "kernel", side_effect=AssertionError("source reran")):
        second = mf.kernel(run_hf=False)
    assert abs(first - second) < 1e-12
    assert callable(getattr(mf, method))


@pytest.mark.cpu
@pytest.mark.slow
@pytest.mark.parametrize("xc", ["LDA,VWN", "PBE", "TPSS", "B3LYP", "CAM-B3LYP"])
def test_representative_nlc_free_xc_families(xc):
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf_hf = scf.RHF(mol).run(conv_tol=1e-11)
    mf = configure_workflow(RHFDFT(mol, xc=xc, mf_hf=mf_hf))
    energy = mf.kernel(run_hf=False)
    assert np.isfinite(energy)


@pytest.mark.cpu
def test_only_dfj_energy_is_allowed_but_gradient_is_not():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf = configure_workflow(RHFDFT(mol, xc="PBE").density_fit(only_dfj=True))
    assert np.isfinite(mf.kernel())
    with pytest.raises(NotImplementedError, match="only_dfj=True"):
        mf.nuc_grad_method()
