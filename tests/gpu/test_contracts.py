"""Molecular GPU API and capability-boundary tests."""

import pytest

from dcforge.gpu._df import default_auxbasis
from dcforge.gpu.energy import RHFDFT, UHFDFT
from tests.common import make_mol

pytestmark = pytest.mark.gpu


def test_default_auxbasis_follows_pyscf_hf_match():
    mol = make_mol("water1.xyz", basis="def2-svp")
    assert default_auxbasis(mol) == "def2-svp-jkfit"


def test_gpu_density_fit_rejects_only_dfj():
    mol = make_mol("water1.xyz", basis="sto-3g")
    with pytest.raises(NotImplementedError, match="only_dfj=False"):
        RHFDFT(mol, xc="PBE").density_fit(only_dfj=True)


def test_gpu_gradient_requires_all_atoms():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf_grad = RHFDFT(mol, xc="PBE").nuc_grad_method()
    with pytest.raises(NotImplementedError, match="all atoms"):
        mf_grad.kernel(atmlst=[0])


def test_gpu_df_gradient_requires_auxiliary_response():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf_grad = RHFDFT(mol, xc="PBE").density_fit().nuc_grad_method()
    mf_grad.auxbasis_response = False
    with pytest.raises(NotImplementedError, match="auxbasis_response=True"):
        mf_grad.kernel()


def test_gpu_gradient_requires_grid_response():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf_grad = RHFDFT(mol, xc="PBE").nuc_grad_method()
    mf_grad.grid_response = False
    with pytest.raises(NotImplementedError, match="grid_response=True"):
        mf_grad.kernel()


@pytest.mark.parametrize("workflow", [RHFDFT, UHFDFT])
def test_gpu_density_fit_rejects_provider_owned_by_another_molecule(workflow):
    mol = make_mol("water1.xyz", basis="sto-3g")
    other_mol = make_mol("water1.xyz", basis="sto-3g")
    foreign_provider = RHFDFT(other_mol, xc="PBE").density_fit().with_df

    with pytest.raises(ValueError, match="same Mole object"):
        workflow(mol, xc="PBE").density_fit(with_df=foreign_provider)


def test_gpu_energy_rechecks_df_owner_after_direct_provider_mutation():
    mol = make_mol("water1.xyz", basis="sto-3g")
    other_mol = make_mol("water1.xyz", basis="sto-3g")
    mf = RHFDFT(mol, xc="PBE").density_fit()
    foreign_provider = RHFDFT(other_mol, xc="PBE").density_fit().with_df
    mf.with_df = mf.mf_hf.with_df = foreign_provider

    with pytest.raises(ValueError, match="same Mole object"):
        mf.kernel()


@pytest.mark.parametrize("workflow", [RHFDFT, UHFDFT])
def test_gpu_source_state_rejects_in_place_geometry_change(workflow):
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf = workflow(mol, xc="PBE")
    mf.mf_hf.converged = True

    coords = mol.atom_coords()
    coords[0, 0] += 0.1
    mol.set_geom_(coords, unit="Bohr")

    with pytest.raises(RuntimeError, match="current molecule or DF representation"):
        mf.kernel(run_hf=False)
    with pytest.raises(RuntimeError, match="current molecule or DF representation"):
        mf.nuc_grad_method().kernel()


@pytest.mark.parametrize("workflow", [RHFDFT, UHFDFT])
def test_gpu_density_fit_requires_source_reconvergence(workflow):
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf = workflow(mol, xc="PBE")
    mf.mf_hf.converged = True

    mf_df = mf.density_fit()
    assert mf_df.e_dft is None
    with pytest.raises(RuntimeError, match="current molecule or DF representation"):
        mf_df.kernel(run_hf=False)


@pytest.mark.parametrize("workflow", [RHFDFT, UHFDFT])
def test_gpu_source_state_rejects_canonical_orbital_mutation(workflow):
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf = workflow(mol, xc="PBE")
    mf.kernel()
    coefficient = mf.mf_hf.mo_coeff
    coefficient[(0,) * coefficient.ndim] += 1e-6

    with pytest.raises(RuntimeError, match="canonical orbitals changed"):
        mf.kernel(run_hf=False)
