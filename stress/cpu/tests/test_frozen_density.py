import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft
from pyscf_pbc_dcdft_stress import (
    FDStress,
    FDStressConfig,
    GammaGDFEnergyEvaluator,
    GammaGDFFrozenDensityEvaluator,
    component_finite_difference,
    frozen_density_analytic_components,
    gamma_gdf_analytic_stress,
    gamma_gdf_semi_analytic_stress,
    gdf_matrix_finite_difference,
    stress_tensor_to_voigt,
)


def make_method():
    cell = gto.Cell(
        atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
        a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3), (0.1, 0.0, 6.0))),
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((15, 15, 15)), verbose=0,
    ).build()
    method = dft.DCRKS(cell, xc="PBE").density_fit(auxbasis="gth-szv")
    method.conv_tol = 1e-11
    method.kernel()
    return method


class FrozenDensityComponentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.method = make_method()
        cls.evaluator = GammaGDFFrozenDensityEvaluator.from_method(cls.method)
        cls.analytic = frozen_density_analytic_components(cls.method)
        cls.finite_difference = component_finite_difference(
            cls.evaluator, cls.method.cell, step=2.5e-4
        )

    def test_central_component_energy_reproduces_target_energy(self):
        evaluated = self.evaluator(self.method.cell)
        expected = self.method.dc_energy_result.components

        self.assertAlmostEqual(evaluated.one_electron, expected.one_electron, places=10)
        self.assertAlmostEqual(evaluated.coulomb, expected.coulomb, places=10)
        self.assertAlmostEqual(evaluated.xc, expected.xc, places=10)
        self.assertAlmostEqual(evaluated.nuclear, expected.nuclear, places=10)
        self.assertAlmostEqual(evaluated.total, self.method.e_tot, places=10)

    def test_partial_analytic_terms_match_fixed_density_component_fd(self):
        tolerances = {"kinetic": 2e-9, "xc": 2e-9, "nuclear": 2e-9}
        for name, tolerance in tolerances.items():
            error = np.max(
                np.abs(
                    self.analytic.stress_ha_per_bohr3[name]
                    - self.finite_difference.stress_ha_per_bohr3[name]
                )
            )
            self.assertLess(float(error), tolerance, msg=name)

        self.assertFalse(self.analytic.complete)
        self.assertIn("gdf_pseudopotential", self.analytic.missing_components)
        self.assertIn("gdf_coulomb", self.analytic.missing_components)
        self.assertIn(
            "fft_pseudopotential_not_gdf",
            self.analytic.diagnostic_derivatives_ha,
        )

    def test_gdf_matrix_fd_matches_component_energy_fd(self):
        matrix_fd = gdf_matrix_finite_difference(
            self.method, step=2.5e-4, components=("xx", "xy"),
            with_exchange=True,
        )
        for name in ("pseudopotential", "coulomb"):
            observed = matrix_fd.energy_derivatives_ha[name][[0, 5]]
            expected = stress_tensor_to_voigt(
                self.finite_difference.derivatives_ha[name]
            )[[0, 5]]
            self.assertLess(float(np.max(np.abs(observed - expected))), 2e-9)
        self.assertIsNotNone(matrix_fd.exchange_ha)
        self.assertFalse(matrix_fd.metadata["production"])
        self.assertFalse(matrix_fd.metadata["source_hf_rerun"])
        for pair in matrix_fd.metadata["resolved_parameter_pairs"].values():
            self.assertEqual(
                pair["plus"]["cderi"]["source_mesh"],
                pair["minus"]["cderi"]["source_mesh"],
            )
            self.assertEqual(
                pair["plus"]["cderi"]["metric_mesh"],
                pair["minus"]["cderi"]["metric_mesh"],
            )
            self.assertEqual(
                pair["plus"]["pseudopotential"]["mesh"],
                pair["minus"]["pseudopotential"]["mesh"],
            )

    def test_semi_analytic_lagrangian_matches_full_energy_fd(self):
        semi_analytic = gamma_gdf_analytic_stress(self.method)
        full_fd = FDStress(
            GammaGDFEnergyEvaluator(
                xc="PBE", auxbasis="gth-szv", conv_tol=1e-11,
                max_cycle=80, verbose=0,
            ),
            FDStressConfig(step=2.5e-4),
        ).kernel(self.method.cell)
        error = np.max(
            np.abs(
                semi_analytic.stress_voigt_ha_per_bohr3
                - full_fd.stress_voigt_ha_per_bohr3
            )
        )
        self.assertLess(float(error), 2e-8)
        self.assertLess(semi_analytic.z_vector.residual_max_abs, 1e-8)
        self.assertTrue(semi_analytic.metadata["production"])
        self.assertEqual(
            semi_analytic.metadata["low_level_numerical_components"], []
        )
        self.assertFalse(semi_analytic.metadata["source_hf_rerun"])

    def test_historical_semi_analytic_name_is_an_fd_free_alias(self):
        result = gamma_gdf_semi_analytic_stress(
            self.method, matrix_fd_step=1.0
        )
        self.assertTrue(result.metadata["production"])
        self.assertNotIn("matrix_fd_step", result.metadata)

    def test_r2scan_tau_xc_matches_fixed_density_component_fd(self):
        cell = self.method.cell
        method = dft.DCRKS(cell, xc="r2SCAN").density_fit(auxbasis="gth-szv")
        method.conv_tol = 1e-11
        method.kernel()
        analytic = frozen_density_analytic_components(method)
        reference = component_finite_difference(
            GammaGDFFrozenDensityEvaluator.from_method(method), cell,
            step=2.5e-4, components=("xx", "xy"),
        )
        observed = stress_tensor_to_voigt(
            analytic.stress_ha_per_bohr3["xc"]
        )[[0, 5]]
        expected = reference.stress_ha_per_bohr3["xc"][[0, 5]]
        self.assertLess(float(np.max(np.abs(observed - expected))), 2e-9)


if __name__ == "__main__":
    unittest.main()
