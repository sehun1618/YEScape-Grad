import time
import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft_stress import FDStress, FDStressConfig
from pyscf_pbc_dcdft_stress.results import EnergyEvaluation


class LinearStrainEnergy:
    def __init__(self, reference_cell, stress):
        self.reference_lattice = np.asarray(reference_cell.a, dtype=float)
        self.volume = float(reference_cell.vol)
        self.stress = np.asarray(stress, dtype=float)

    def __call__(self, cell, *, dm0=None, kpts=None):
        started = time.perf_counter()
        deformation_t = np.linalg.solve(self.reference_lattice, np.asarray(cell.a))
        deformation = deformation_t.T
        strain = 0.5 * (deformation + deformation.T) - np.eye(3)
        energy = -1.25 + self.volume * np.einsum("ij,ij->", self.stress, strain)
        return EnergyEvaluation(
            energy_ha=energy, wall_seconds=time.perf_counter() - started,
            density=np.eye(2), metadata={"synthetic": True},
        )


class NonConvergedEnergy(LinearStrainEnergy):
    def __call__(self, cell, *, dm0=None, kpts=None):
        result = super().__call__(cell, dm0=dm0, kpts=kpts)
        return EnergyEvaluation(
            energy_ha=result.energy_ha, wall_seconds=result.wall_seconds,
            converged=False, density=result.density,
        )


def make_cell():
    return gto.Cell(
        atom="H 0 0 0; H 0 0 1.4", a=np.eye(3) * 6.0,
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((9, 9, 9)), verbose=0,
    ).build()


class FiniteDifferenceStressTest(unittest.TestCase):
    def test_recovers_known_symmetric_stress(self):
        cell = make_cell()
        expected = np.asarray(
            ((0.003, -0.0004, 0.0002),
             (-0.0004, -0.001, 0.0007),
             (0.0002, 0.0007, 0.002))
        )
        result = FDStress(
            LinearStrainEnergy(cell, expected), FDStressConfig(step=1.0e-5)
        ).kernel(cell)

        self.assertTrue(result.complete)
        np.testing.assert_allclose(
            result.stress_tensor_ha_per_bohr3, expected, atol=2e-12, rtol=0
        )
        self.assertEqual(len(result.pairs), 6)
        for pair in result.pairs:
            self.assertEqual(pair.plus_mesh, tuple(cell.mesh))
            self.assertEqual(pair.minus_mesh, tuple(cell.mesh))

    def test_partial_component_is_explicitly_incomplete(self):
        cell = make_cell()
        expected = np.diag((0.001, 0.002, 0.003))
        result = FDStress(
            LinearStrainEnergy(cell, expected),
            FDStressConfig(step=1.0e-5, components=("xx",)),
        ).kernel(cell)
        self.assertFalse(result.complete)
        self.assertAlmostEqual(result.stress_voigt_ha_per_bohr3[0], 0.001)
        self.assertTrue(np.isnan(result.stress_tensor_ha_per_bohr3).all())

    def test_nonconverged_energy_is_not_silently_accepted(self):
        cell = make_cell()
        with self.assertRaisesRegex(RuntimeError, "central.*did not converge"):
            FDStress(
                NonConvergedEnergy(cell, np.eye(3)),
                FDStressConfig(components=("xx",)),
            ).kernel(cell)


if __name__ == "__main__":
    unittest.main()
