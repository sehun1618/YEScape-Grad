"""CPU fixed-density RHF/UHF energy integration tests."""

from unittest import mock

import numpy as np
import pytest
from pyscf import df, dft, scf

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


@pytest.mark.cpu
@pytest.mark.parametrize("workflow", [RHFDFT, UHFDFT])
def test_density_fit_rejects_provider_owned_by_another_molecule(workflow):
    mol = make_mol("water1.xyz", basis="sto-3g")
    other_mol = make_mol("water1.xyz", basis="sto-3g")
    foreign_provider = df.DF(other_mol)

    with pytest.raises(ValueError, match="same Mole object"):
        workflow(mol, xc="PBE").density_fit(with_df=foreign_provider)


@pytest.mark.cpu
def test_energy_rechecks_df_owner_after_direct_provider_mutation():
    mol = make_mol("water1.xyz", basis="sto-3g")
    other_mol = make_mol("water1.xyz", basis="sto-3g")
    mf = RHFDFT(mol, xc="PBE").density_fit()
    foreign_provider = df.DF(other_mol)
    mf.with_df = mf.mf_hf.with_df = foreign_provider

    with pytest.raises(ValueError, match="same Mole object"):
        mf.kernel()


@pytest.mark.cpu
@pytest.mark.parametrize(
    "workflow,name",
    [(RHFDFT, "water1.xyz"), (UHFDFT, "o2_triplet.xyz")],
)
def test_explicit_reuse_allows_in_place_geometry_change(workflow, name):
    mol = make_mol(name, basis="sto-3g")
    mf, _ = run_workflow(workflow, mol, xc="PBE")

    coords = mol.atom_coords()
    coords[0, 0] += 0.1
    mol.set_geom_(coords, unit="Bohr")

    assert np.all(np.isfinite(mf.kernel(run_hf=False)))
    assert np.all(np.isfinite(mf.nuc_grad_method().kernel()))


@pytest.mark.cpu
@pytest.mark.parametrize(
    "workflow,name",
    [(RHFDFT, "water1.xyz"), (UHFDFT, "o2_triplet.xyz")],
)
def test_density_fit_and_undo_df_allow_explicit_source_reuse(workflow, name):
    mol = make_mol(name, basis="sto-3g")
    mf, _ = run_workflow(workflow, mol, xc="PBE")

    mf_df = mf.density_fit()
    assert mf_df.e_dft is None
    assert np.all(np.isfinite(mf_df.kernel(run_hf=False)))

    mf_df.kernel()
    mf_conventional = mf_df.undo_df()
    assert mf_conventional.e_dft is None
    assert np.all(np.isfinite(mf_conventional.kernel(run_hf=False)))


@pytest.mark.cpu
def test_target_xc_change_keeps_source_state_reusable():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf, _ = run_workflow(RHFDFT, mol, xc="PBE")
    mf.xc = "LDA,VWN"

    with mock.patch.object(
            mf.mf_hf, "kernel", side_effect=AssertionError("source reran")):
        energy = mf.kernel(run_hf=False)
    assert np.isfinite(energy)


@pytest.mark.cpu
def test_initial_df_workflow_accepts_independently_validated_source_state():
    mol = make_mol("water1.xyz", basis="sto-3g")
    restored = scf.RHF(mol).density_fit()
    restored.conv_tol = 1e-11
    restored.kernel()

    mf = configure_workflow(RHFDFT(mol, xc="PBE").density_fit())
    mf.mf_hf.e_tot = restored.e_tot
    mf.mf_hf.mo_energy = restored.mo_energy
    mf.mf_hf.mo_coeff = restored.mo_coeff
    mf.mf_hf.mo_occ = restored.mo_occ
    mf.mf_hf.converged = True

    assert np.isfinite(mf.kernel(run_hf=False))


@pytest.mark.cpu
@pytest.mark.parametrize(
    "workflow,name",
    [(RHFDFT, "water1.xyz"), (UHFDFT, "o2_triplet.xyz")],
)
@pytest.mark.parametrize("attribute", ["mo_energy", "mo_coeff", "mo_occ"])
def test_explicit_reuse_accepts_canonical_array_changes(
        workflow, name, attribute):
    mol = make_mol(name, basis="sto-3g")
    mf, _ = run_workflow(workflow, mol, xc="PBE")

    value = getattr(mf.mf_hf, attribute)
    array = value[0] if isinstance(value, (tuple, list)) else value
    array[(0,) * array.ndim] += 1e-6

    assert np.isfinite(mf.kernel(run_hf=False))


@pytest.mark.cpu
@pytest.mark.parametrize(
    "workflow,name",
    [(RHFDFT, "water1.xyz"), (UHFDFT, "o2_triplet.xyz")],
)
def test_default_energy_accepts_reused_source_density(
        workflow, name):
    mol = make_mol(name, basis="sto-3g")
    mf, _ = run_workflow(workflow, mol, xc="PBE")
    dm = mf.mf_hf.make_rdm1().copy()

    coords = mol.atom_coords()
    coords[0, 0] += 0.1
    mol.set_geom_(coords, unit="Bohr")

    assert np.all(np.isfinite(mf.energy_tot()))
    assert np.isfinite(mf.energy_tot(dm=dm))


@pytest.mark.cpu
def test_reuse_synchronizes_workflow_orbitals_from_source():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf, _ = run_workflow(RHFDFT, mol, xc="PBE")
    mf.mo_coeff = mf.mo_coeff.copy()
    mf.mo_coeff[0, 0] += 1e-6

    assert np.isfinite(mf.kernel(run_hf=False))
    assert mf.mo_coeff is mf.mf_hf.mo_coeff
