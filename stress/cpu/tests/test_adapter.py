import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft
from pyscf_pbc_dcdft_stress.adapters import GammaGDFEnergyEvaluator


class AdapterIsolationTest(unittest.TestCase):
    def test_from_method_is_read_only_and_strict(self):
        cell = gto.Cell(
            atom="H 0 0 0; H 0 0 1.4", a=np.eye(3) * 6.0,
            unit="Bohr", basis="gth-szv", pseudo="gth-hf",
            mesh=np.asarray((9, 9, 9)), verbose=0,
        ).build()
        method = dft.DCRKS(cell, xc="PBE").density_fit("gth-szv")
        provider_id = id(method.with_df)
        self.assertFalse(hasattr(dft.DCRKS, "Stress"))

        evaluator = GammaGDFEnergyEvaluator.from_method(method)

        self.assertEqual(evaluator.xc, "PBE")
        self.assertEqual(evaluator.auxbasis, "gth-szv")
        self.assertEqual(id(method.with_df), provider_id)
        self.assertFalse(method.converged)
        self.assertFalse(hasattr(dft.DCRKS, "Stress"))


if __name__ == "__main__":
    unittest.main()
