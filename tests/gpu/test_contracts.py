"""Molecular GPU API and capability-boundary tests."""

import pytest

from dcforge.gpu._df import default_auxbasis
from dcforge.gpu.energy import RHFDFT
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
