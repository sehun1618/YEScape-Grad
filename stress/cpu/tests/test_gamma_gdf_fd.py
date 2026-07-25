import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft_stress import (
    FDStress,
    FDStressConfig,
    GammaGDFEnergyEvaluator,
    step_convergence,
)


def make_h2_cell():
    return gto.Cell(
        atom="H 0 0 0; H 0 0 1.4", a=np.eye(3) * 5.0,
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((9, 9, 9)), verbose=0,
    ).build()


def make_distorted_h2_cell():
    return gto.Cell(
        atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
        a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3), (0.1, 0.0, 6.0))),
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((15, 15, 15)), verbose=0,
    ).build()


class GammaGDFFiniteDifferenceTest(unittest.TestCase):
    @staticmethod
    def evaluator():
        return GammaGDFEnergyEvaluator(
            xc="PBE", auxbasis="gth-szv", conv_tol=1e-10,
            max_cycle=60, verbose=0,
        )

    def test_full_tensor_obeys_cubic_h2_symmetry(self):
        cell = make_h2_cell()
        result = FDStress(
            self.evaluator(), FDStressConfig(step=5e-4)
        ).kernel(cell)

        self.assertTrue(result.complete)
        self.assertTrue(np.isfinite(result.stress_tensor_ha_per_bohr3).all())
        self.assertAlmostEqual(
            result.stress_tensor_ha_per_bohr3[0, 0],
            result.stress_tensor_ha_per_bohr3[1, 1], places=10,
        )
        shear = result.stress_tensor_ha_per_bohr3 - np.diag(
            np.diag(result.stress_tensor_ha_per_bohr3)
        )
        self.assertLess(float(np.max(np.abs(shear))), 1e-10)
        for pair in result.pairs:
            self.assertEqual(pair.plus_mesh, (9, 9, 9))
            self.assertEqual(pair.minus_mesh, (9, 9, 9))

    def test_diagonal_and_shear_step_convergence_reach_plateau(self):
        cell = make_distorted_h2_cell()
        for component, index in (("xx", 0), ("xy", 5)):
            values = step_convergence(
                self.evaluator(), cell, (1e-3, 5e-4, 2.5e-4),
                components=(component,),
            )
            stresses = [result.stress_voigt_ha_per_bohr3[index] for result in values]
            self.assertLess(
                abs(stresses[2] - stresses[1]), abs(stresses[1] - stresses[0])
            )
            self.assertLess(abs(stresses[2] - stresses[1]), 1e-8)

    def test_rigid_translation_error_converges_with_mesh(self):
        base = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.4)))
        shift = np.asarray((0.23, 0.17, 0.11))
        differences = []
        for mesh in (9, 15, 21):
            stresses = []
            for coordinates in (base, base + shift):
                cell = gto.Cell(
                    atom=[("H", coordinates[0]), ("H", coordinates[1])],
                    a=np.eye(3) * 5.0, unit="Bohr", basis="gth-szv",
                    pseudo="gth-hf", mesh=np.asarray((mesh, mesh, mesh)),
                    verbose=0,
                ).build()
                result = FDStress(
                    self.evaluator(),
                    FDStressConfig(step=5e-4, components=("xx",)),
                ).kernel(cell)
                stresses.append(result.stress_voigt_ha_per_bohr3[0])
            differences.append(abs(stresses[1] - stresses[0]))

        self.assertGreater(differences[0], differences[1])
        self.assertGreater(differences[1], differences[2])
        self.assertLess(differences[2], 5e-8)


if __name__ == "__main__":
    unittest.main()
