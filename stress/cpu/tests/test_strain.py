import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft_stress.strain import deform_cell, deform_kpoints


def make_cell():
    return gto.Cell(
        atom="H 0.1 0.2 0.3; H 1.0 0.7 0.8",
        a=np.asarray(((5.0, 0.2, 0.0), (0.0, 5.5, 0.1), (0.1, 0.0, 6.0))),
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((9, 11, 13)), verbose=0,
    ).build()


class HomogeneousStrainTest(unittest.TestCase):
    def test_fractional_atoms_and_mesh_are_fixed(self):
        cell = make_cell()
        lattice = np.asarray(cell.a).copy()
        coordinates = np.asarray(cell.atom_coords()).copy()
        fractional = np.asarray(cell.get_scaled_atom_coords()).copy()

        strained = deform_cell(cell, "xy", 2.0e-3)

        np.testing.assert_allclose(strained.get_scaled_atom_coords(), fractional)
        np.testing.assert_array_equal(strained.mesh, cell.mesh)
        np.testing.assert_allclose(cell.a, lattice)
        np.testing.assert_allclose(cell.atom_coords(), coordinates)
        self.assertFalse(np.allclose(strained.a, lattice))

    def test_fractional_kpoints_are_fixed(self):
        cell = make_cell()
        kpts = cell.make_kpts((2, 1, 1))
        strained = deform_cell(cell, "zz", 1.0e-3)
        moved = deform_kpoints(cell, strained, kpts)
        np.testing.assert_allclose(
            strained.get_scaled_kpts(moved), cell.get_scaled_kpts(kpts),
            atol=1e-12, rtol=0,
        )

    def test_cutoff_derived_mesh_change_is_explicit(self):
        cell = gto.Cell(
            atom="H 0 0 0; H 0 0 1.4", a=np.eye(3) * 5.0,
            unit="Bohr", basis="gth-szv", pseudo="gth-hf",
            ke_cutoff=20.0, verbose=0,
        ).build()
        plus = deform_cell(cell, "xx", 0.02, preserve_mesh=False)
        minus = deform_cell(cell, "xx", -0.02, preserve_mesh=False)
        self.assertNotEqual(tuple(plus.mesh), tuple(minus.mesh))
        np.testing.assert_array_equal(cell.mesh, (13, 13, 13))


if __name__ == "__main__":
    unittest.main()
