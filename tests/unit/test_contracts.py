"""Public import and capability-boundary tests."""

import subprocess
import sys

import pytest
from pyscf import gto, scf

from dcforge.cpu.energy import RHFDFT, UHFDFT
from tests.common import ROOT, make_mol


def test_top_level_import_does_not_import_gpu4pyscf():
    code = "import dcforge, sys; assert 'gpu4pyscf' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_source_and_target_must_share_one_molecule():
    mol_a = make_mol("water1.xyz", basis="sto-3g")
    mol_b = make_mol("water1.xyz", basis="sto-3g")
    with pytest.raises(ValueError, match="same Mole"):
        RHFDFT(mol_b, mf_hf=scf.RHF(mol_a))


def test_run_hf_false_requires_converged_source():
    mf = RHFDFT(make_mol("water1.xyz", basis="sto-3g"), xc="PBE")
    with pytest.raises(RuntimeError, match="source HF calculation must be converged"):
        mf.kernel(run_hf=False)


@pytest.mark.parametrize("workflow", [RHFDFT, UHFDFT])
def test_nlc_is_rejected_before_source_scf(workflow):
    name = "water1.xyz" if workflow is RHFDFT else "o2_triplet.xyz"
    mf = workflow(make_mol(name, basis="sto-3g"), xc="B97M-V")
    with pytest.raises(NotImplementedError, match="nonlocal correlation"):
        mf.kernel()


def test_refactor_slice_uses_molecular_pyscf_objects():
    mol = make_mol("water1.xyz", basis="sto-3g")
    assert isinstance(mol, gto.Mole)
    assert not hasattr(mol, "lattice_vectors")
