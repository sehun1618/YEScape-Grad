"""Molecular GPU API and capability-boundary tests."""

import pytest

from yescape_grad.gpu._df import default_auxbasis
from yescape_grad.gpu.energy import RNSC, UNSC
from tests.common import make_mol

pytestmark = pytest.mark.gpu


def test_default_auxbasis_follows_pyscf_hf_match():
    mol = make_mol("water1.xyz", basis="def2-svp")
    assert default_auxbasis(mol) == "def2-svp-jkfit"


def test_gpu_density_fit_rejects_only_dfj():
    mol = make_mol("water1.xyz", basis="sto-3g")
    with pytest.raises(NotImplementedError, match="only_dfj=False"):
        RNSC(mol, xc="PBE").density_fit(only_dfj=True)


def test_gpu_gradient_requires_all_atoms():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf_grad = RNSC(mol, xc="PBE").nuc_grad_method()
    with pytest.raises(NotImplementedError, match="all atoms"):
        mf_grad.kernel(atmlst=[0])


def test_gpu_df_gradient_requires_auxiliary_response():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf_grad = RNSC(mol, xc="PBE").density_fit().nuc_grad_method()
    mf_grad.auxbasis_response = False
    with pytest.raises(NotImplementedError, match="auxbasis_response=True"):
        mf_grad.kernel()


def test_gpu_gradient_requires_grid_response():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf_grad = RNSC(mol, xc="PBE").nuc_grad_method()
    mf_grad.grid_response = False
    with pytest.raises(NotImplementedError, match="grid_response=True"):
        mf_grad.kernel()


@pytest.mark.parametrize("workflow", [RNSC, UNSC])
def test_gpu_density_fit_rejects_provider_owned_by_another_molecule(workflow):
    mol = make_mol("water1.xyz", basis="sto-3g")
    other_mol = make_mol("water1.xyz", basis="sto-3g")
    foreign_provider = RNSC(other_mol, xc="PBE").density_fit().with_df

    with pytest.raises(ValueError, match="same Mole object"):
        workflow(mol, xc="PBE").density_fit(with_df=foreign_provider)


def test_gpu_energy_rechecks_df_owner_after_direct_provider_mutation():
    mol = make_mol("water1.xyz", basis="sto-3g")
    other_mol = make_mol("water1.xyz", basis="sto-3g")
    mf = RNSC(mol, xc="PBE").density_fit()
    foreign_provider = RNSC(other_mol, xc="PBE").density_fit().with_df
    mf.with_df = mf.mf_source.with_df = foreign_provider

    with pytest.raises(ValueError, match="same Mole object"):
        mf.kernel()


@pytest.mark.parametrize("workflow", [RNSC, UNSC])
def test_gpu_geometry_change_without_orbitals_is_rejected(workflow):
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf = workflow(mol, xc="PBE")
    mf.mf_source.converged = True

    coords = mol.atom_coords()
    coords[0, 0] += 0.1
    mol.set_geom_(coords, unit="Bohr")

    with pytest.raises(RuntimeError, match="no canonical orbital state"):
        mf.kernel(run_source=False)
    with pytest.raises(RuntimeError, match="no canonical orbital state"):
        mf.nuc_grad_method().kernel()


@pytest.mark.parametrize("workflow", [RNSC, UNSC])
def test_gpu_density_fit_without_orbitals_is_rejected(workflow):
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf = workflow(mol, xc="PBE")
    mf.mf_source.converged = True

    mf_df = mf.density_fit()
    assert mf_df.e_dft is None
    with pytest.raises(RuntimeError, match="no canonical orbital state"):
        mf_df.kernel(run_source=False)


@pytest.mark.parametrize("workflow", [RNSC, UNSC])
def test_gpu_explicit_reuse_accepts_canonical_orbital_changes(workflow):
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf = workflow(mol, xc="PBE")
    mf.kernel()
    coefficient = mf.mf_source.mo_coeff
    coefficient[(0,) * coefficient.ndim] += 1e-6

    assert mf.kernel(run_source=False) == pytest.approx(mf.e_tot)
