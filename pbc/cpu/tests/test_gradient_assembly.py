import unittest

import numpy as np

from pyscf_pbc_dcdft.grad import GradientContributions


class GradientAssemblyTest(unittest.TestCase):
    def test_assembly_preserves_sign_and_units(self) -> None:
        explicit = np.array([[1.0, 2.0, 3.0]])
        response = np.array([[-0.25, 0.5, -1.0]])
        result = GradientContributions.assemble(explicit, response)
        self.assertTrue(np.allclose(result.total, [[0.75, 2.5, 2.0]]))
        self.assertEqual(result.metadata()["units"], "Ha/Bohr")

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GradientContributions.assemble(np.zeros((2, 3)), np.zeros((1, 3)))

    def test_coupled_channel_is_included_without_repartitioning(self) -> None:
        explicit = np.array([[1.0, 2.0, 3.0]])
        response = np.array([[-0.25, 0.5, -1.0]])
        coupled = np.array([[0.1, -0.2, 0.3]])
        result = GradientContributions.assemble(
            explicit, response, coupled=coupled
        )
        self.assertTrue(np.allclose(result.coupled, coupled))
        self.assertTrue(np.allclose(result.total, [[0.85, 2.3, 2.3]]))


if __name__ == "__main__":
    unittest.main()
