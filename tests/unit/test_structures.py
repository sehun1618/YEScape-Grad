"""Repository XYZ and electronic-state metadata checks."""

import pytest

from tests.common import STRUCTURES, STRUCTURE_STATES, make_mol, read_xyz


@pytest.mark.parametrize("name,state", STRUCTURE_STATES.items())
def test_structure_parses_and_electron_spin_parity_is_valid(name, state):
    charge, spin = state
    atoms = read_xyz(STRUCTURES / name)
    mol = make_mol(name, basis="sto-3g")
    assert len(atoms) == mol.natm
    assert mol.charge == charge
    assert mol.spin == spin
    assert (mol.nelectron - spin) % 2 == 0
