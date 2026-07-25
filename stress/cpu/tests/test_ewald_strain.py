import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft_stress import analytic_ewald_strain
from pyscf_pbc_dcdft_stress.strain import deform_cell


def make_cell():
    return gto.Cell(
        atom="O 0.3 0.4 0.5; H 1.5 0.7 0.9; H 0.6 1.6 1.1",
        a=np.asarray(((6.0, 0.3, 0.2), (0.1, 6.5, 0.4), (0.2, 0.1, 7.0))),
        unit="Bohr", basis="gth-szv", pseudo="gth-pbe",
        mesh=np.asarray((13, 15, 17)), precision=1e-12, verbose=0,
    ).build()


class EwaldStrainTest(unittest.TestCase):
    def test_energy_and_strain_match_fixed_partition_finite_difference(self):
        cell = make_cell()
        analytic = analytic_ewald_strain(cell)
        self.assertLess(abs(analytic.value - cell.ewald(
            ew_eta=analytic.eta, ew_cut=analytic.cutoff
        )), 2e-12)

        step = 1e-5
        for index in range(6):
            plus = deform_cell(cell, index, step, mesh=analytic.mesh)
            minus = deform_cell(cell, index, -step, mesh=analytic.mesh)
            plus_value = plus.ewald(
                ew_eta=analytic.eta, ew_cut=analytic.cutoff
            )
            minus_value = minus.ewald(
                ew_eta=analytic.eta, ew_cut=analytic.cutoff
            )
            reference = (plus_value - minus_value) / (2.0 * step)
            self.assertLess(
                abs(analytic.voigt_derivative[index] - reference), 2e-8,
                msg=index,
            )

    def test_component_sum_and_symmetry(self):
        analytic = analytic_ewald_strain(make_cell(), reciprocal_block_size=31)
        self.assertLess(abs(
            analytic.value - analytic.real_value - analytic.self_value
            - analytic.reciprocal_value
        ), 2e-13)
        component_sum = (
            analytic.real_derivative + analytic.self_derivative
            + analytic.reciprocal_derivative
        )
        self.assertLess(float(np.max(np.abs(
            analytic.raw_derivative - component_sum
        ))), 2e-13)
        self.assertLess(float(np.max(np.abs(
            analytic.raw_derivative - analytic.raw_derivative.T
        ))), 2e-13)

    def test_reciprocal_blocking_is_invariant(self):
        cell = make_cell()
        large = analytic_ewald_strain(cell, reciprocal_block_size=10**9)
        small = analytic_ewald_strain(cell, reciprocal_block_size=17)
        self.assertLess(abs(large.value - small.value), 2e-12)
        self.assertLess(float(np.max(np.abs(
            large.raw_derivative - small.raw_derivative
        ))), 2e-12)


if __name__ == "__main__":
    unittest.main()
