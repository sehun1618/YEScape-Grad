import io
import unittest

import numpy as np
from pyscf.lib import logger

from pyscf_dcdft_grad.grad.rhf import solve_z_vector


class RHFZVectorTest(unittest.TestCase):
    def test_mo_e1_is_not_interpreted_as_a_convergence_flag(self):
        mo_energy = np.array([-1.0, 0.5])
        mo_occ = np.array([2.0, 0.0])
        mo_coeff = np.eye(2)
        occupied = mo_coeff[:, :1]
        virtual = mo_coeff[:, 1:]
        source = np.array([[0.3]])

        output = io.StringIO()
        log = logger.Logger(output, logger.NOTE)
        zvector = solve_z_vector(
            mo_energy,
            mo_occ,
            occupied,
            virtual,
            source,
            lambda density: np.zeros_like(density),
            log=log,
        )

        np.testing.assert_allclose(zvector, [[-0.2]], atol=1e-12)
        self.assertNotIn("not converged", output.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
