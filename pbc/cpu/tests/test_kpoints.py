import unittest

import numpy as np

from pyscf_pbc_dcdft.lib.kpoints import is_gamma, normalize_kpts


class KPointTest(unittest.TestCase):
    def test_normalize_single_kpoint(self) -> None:
        result = normalize_kpts([0.1, 0.0, 0.0])
        self.assertEqual(result.shape, (1, 3))
        self.assertTrue(np.allclose(result[0], [0.1, 0.0, 0.0]))

    def test_gamma_detection(self) -> None:
        self.assertTrue(is_gamma(None))
        self.assertFalse(is_gamma([[0.0, 0.0, 0.1]]))

    def test_invalid_kpoint_shape(self) -> None:
        with self.assertRaises(ValueError):
            normalize_kpts([0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
