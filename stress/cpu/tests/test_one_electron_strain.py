import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft_stress import (
    gamma_kinetic_strain,
    gamma_overlap_strain,
)
from pyscf_pbc_dcdft_stress.strain import deform_cell


def make_cell():
    return gto.Cell(
        atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
        a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3), (0.1, 0.0, 6.0))),
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((7, 9, 11)), verbose=0,
    ).build()


class OneElectronStrainTest(unittest.TestCase):
    def test_overlap_and_kinetic_match_matrix_finite_difference(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        step = 1e-5
        for integral, response in (
            ("int1e_ovlp", gamma_overlap_strain(cell)),
            ("int1e_kin", gamma_kinetic_strain(cell)),
        ):
            self.assertLess(float(np.max(np.abs(
                response.value - cell.pbc_intor(integral, hermi=0)
            ))), 2e-13)
            for index in range(6):
                plus = deform_cell(cell, index, step, mesh=mesh)
                minus = deform_cell(cell, index, -step, mesh=mesh)
                reference = (
                    plus.pbc_intor(integral, hermi=0)
                    - minus.pbc_intor(integral, hermi=0)
                ) / (2.0 * step)
                error = np.max(np.abs(
                    response.voigt_derivative[index] - reference
                ))
                self.assertLess(float(error), 3e-10, msg=(integral, index))

    def test_raw_response_has_periodic_hermitian_pair_symmetry(self):
        for response in (
            gamma_overlap_strain(make_cell()),
            gamma_kinetic_strain(make_cell()),
        ):
            paired = response.raw_derivative.transpose(1, 0, 3, 2).conj()
            self.assertLess(float(np.max(np.abs(
                response.raw_derivative - paired
            ))), 3e-13)


if __name__ == "__main__":
    unittest.main()
