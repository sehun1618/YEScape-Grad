import unittest

import numpy as np

from pyscf_pbc_dcdft_stress.conventions import (
    HARTREE_PER_BOHR3_TO_GPA,
    pressure_from_stress,
    strain_basis,
    stress_tensor_to_voigt,
    stress_voigt_to_tensor,
    tensor_strain_to_voigt,
    virial_from_stress,
    voigt_strain_to_tensor,
)


class StressConventionTest(unittest.TestCase):
    def test_stress_and_strain_voigt_round_trips(self):
        stress_voigt = np.asarray([1.0, 2.0, 3.0, 0.4, -0.2, 0.7])
        stress = stress_voigt_to_tensor(stress_voigt)
        np.testing.assert_allclose(stress_tensor_to_voigt(stress), stress_voigt)

        strain_voigt = np.asarray([0.01, -0.02, 0.03, 0.04, -0.05, 0.06])
        strain = voigt_strain_to_tensor(strain_voigt)
        np.testing.assert_allclose(tensor_strain_to_voigt(strain), strain_voigt)
        self.assertAlmostEqual(strain[1, 2], 0.5 * strain_voigt[3])

    def test_shear_basis_and_pressure_virial_signs(self):
        yz = strain_basis("yz")
        self.assertAlmostEqual(yz[1, 2], 0.5)
        self.assertAlmostEqual(yz[2, 1], 0.5)

        stress = np.diag([2.0, 2.0, 2.0])
        self.assertAlmostEqual(pressure_from_stress(stress), -2.0)
        np.testing.assert_allclose(virial_from_stress(stress, 3.0), -6.0 * np.eye(3))
        self.assertGreater(HARTREE_PER_BOHR3_TO_GPA, 29000.0)


if __name__ == "__main__":
    unittest.main()
