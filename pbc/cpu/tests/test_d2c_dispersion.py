"""D2C-D4 parameter, periodic derivative, and assembly tests."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft.dispersion import (
    apply_d2c_energy,
    apply_d2c_gradient,
    evaluate_d2c,
    get_d2c_parameters,
    normalize_dispersion,
)
from pyscf_pbc_dcdft.grad.assembly import GradientContributions
from pyscf_pbc_dcdft.lib.types import DCEnergyComponents, DCEnergyResult


def _water_cell(coordinates=None):
    if coordinates is None:
        coordinates = np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.7, 0.0]]
        )
    return gto.Cell(
        atom=list(zip(("O", "H", "H"), coordinates)),
        a=np.eye(3) * 12.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-hf",
        verbose=0,
    ).build()


class D2CParameterTest(unittest.TestCase):
    def test_published_pbe_and_r2scan_parameters(self):
        pbe = get_d2c_parameters("PBE")
        r2scan = get_d2c_parameters("r2SCAN")

        self.assertAlmostEqual(pbe["s8"], 1.78595387)
        self.assertAlmostEqual(pbe["a1"], 0.88511469)
        self.assertAlmostEqual(pbe["a2"], 2.32863362)
        self.assertAlmostEqual(r2scan["s8"], 0.02734375)
        self.assertEqual(pbe["s6"], 1.0)
        self.assertEqual(pbe["s9"], 1.0)

    def test_option_normalization_and_unknown_functional(self):
        self.assertIsNone(normalize_dispersion(None))
        self.assertIsNone(normalize_dispersion("off"))
        self.assertEqual(normalize_dispersion(True), "d2c")
        self.assertEqual(normalize_dispersion("D2C-D4"), "d2c")
        with self.assertRaisesRegex(ValueError, "dispersion must"):
            normalize_dispersion("d3")
        with self.assertRaisesRegex(ValueError, "no published"):
            get_d2c_parameters("made-up-functional")


@unittest.skipUnless(
    importlib.util.find_spec("dftd4") is not None,
    "requires optional dftd4",
)
class D2CPeriodicDerivativeTest(unittest.TestCase):
    def test_periodic_analytic_gradient_matches_central_difference(self):
        cell = _water_cell()
        result = evaluate_d2c(cell, "PBE", gradient=True)
        step = 1.0e-4
        coordinates = cell.atom_coords()
        energies = []
        for sign in (-1.0, 1.0):
            displaced = coordinates.copy()
            displaced[1, 2] += sign * step
            energies.append(
                evaluate_d2c(
                    _water_cell(displaced), "PBE", gradient=False
                ).energy
            )
        finite_difference = (energies[1] - energies[0]) / (2.0 * step)

        self.assertAlmostEqual(
            result.gradient[1, 2], finite_difference, places=10
        )
        np.testing.assert_allclose(
            result.gradient.sum(axis=0), np.zeros(3), atol=1.0e-12
        )
        self.assertEqual(result.virial.shape, (3, 3))

    def test_energy_and_gradient_assembly_keep_d2c_explicit(self):
        correction = evaluate_d2c(_water_cell(), "PBE", gradient=True)
        components = DCEnergyComponents(
            one_electron=-2.0,
            coulomb=0.5,
            xc=-0.2,
            nuclear=0.6,
            total=-1.1,
        )
        energy_result = DCEnergyResult(
            energy=-1.1,
            reference_energy=-1.0,
            target_xc="PBE",
            nkpts=1,
            components=components,
            metadata={"formula": "E_DFA[n_HF]"},
        )
        corrected_energy = apply_d2c_energy(energy_result, correction)
        self.assertAlmostEqual(
            corrected_energy.energy, -1.1 + correction.energy
        )
        self.assertAlmostEqual(
            corrected_energy.energy,
            corrected_energy.components.reconstructed_total,
        )

        class ElectronicResult:
            def __init__(self):
                zeros = np.zeros((3, 3))
                self.contributions = GradientContributions.assemble(
                    np.ones((3, 3)), zeros
                )

            @property
            def gradient(self):
                return self.contributions.total

            def metadata(self):
                return {"device": "cpu", "component_max_abs": {}}

        corrected_gradient = apply_d2c_gradient(
            ElectronicResult(), correction
        )
        np.testing.assert_allclose(
            corrected_gradient.gradient,
            np.ones((3, 3)) + correction.gradient,
        )
        np.testing.assert_allclose(
            corrected_gradient.contributions.explicit,
            np.ones((3, 3)) + correction.gradient,
        )
        self.assertEqual(
            corrected_gradient.metadata()["dispersion"]["name"],
            "D2C-D4",
        )


if __name__ == "__main__":
    unittest.main()
