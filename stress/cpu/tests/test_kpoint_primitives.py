import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft
from pyscf_pbc_dcdft_stress import (
    FDStress,
    FDStressConfig,
    KPointGaussianDFEnergyEvaluator,
    kpoint_frozen_density_analytic_components,
    kpoint_gdf_analytic_stress,
    kpoint_rsdf_analytic_stress,
    kpoint_one_electron_strain,
)
from pyscf_pbc_dcdft_stress.strain import deform_cell, deform_kpoints


def make_cell():
    return gto.Cell(
        atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
        a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3), (0.1, 0.0, 6.0))),
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((11, 11, 11)), verbose=0,
    ).build()


class KPointPrimitiveTest(unittest.TestCase):
    def test_sampled_k_gdf_pbe_full_stress_matches_relaxed_energy_fd(self):
        cell = make_cell()
        kpts = cell.make_kpts((2, 1, 1), wrap_around=True)
        method = dft.DCKRKS(cell, kpts=kpts, xc="PBE").density_fit(
            auxbasis="gth-szv"
        )
        method.conv_tol = 1e-11
        method.kernel()
        analytic = kpoint_gdf_analytic_stress(method)
        reference = FDStress(
            KPointGaussianDFEnergyEvaluator.from_method(method),
            FDStressConfig(step=2.5e-4, components=("xx", "xy")),
        ).kernel(cell)
        observed = analytic.stress_voigt_ha_per_bohr3[[0, 5]]
        expected = reference.stress_voigt_ha_per_bohr3[[0, 5]]
        self.assertLess(float(np.max(np.abs(observed-expected))), 3e-8)
        self.assertEqual(analytic.metadata["low_level_numerical_components"], [])

    def test_sampled_k_rsdf_pbe_full_stress_matches_relaxed_energy_fd(self):
        cell = make_cell()
        kpts = cell.make_kpts((2, 1, 1), wrap_around=True)
        method = dft.DCKRKS(cell, kpts=kpts, xc="PBE").rs_density_fit(
            auxbasis="gth-szv"
        )
        method.conv_tol = 1e-11
        method.kernel()
        analytic = kpoint_rsdf_analytic_stress(method)
        reference = FDStress(
            KPointGaussianDFEnergyEvaluator.from_method(method),
            FDStressConfig(step=2.5e-4, components=("xx", "xy")),
        ).kernel(cell)
        observed = analytic.stress_voigt_ha_per_bohr3[[0, 5]]
        expected = reference.stress_voigt_ha_per_bohr3[[0, 5]]
        self.assertLess(float(np.max(np.abs(observed-expected))), 3e-8)
        self.assertEqual(analytic.metadata["low_level_numerical_components"], [])

    def test_sampled_k_gdf_r2scan_full_stress_matches_relaxed_energy_fd(self):
        cell = make_cell()
        kpts = cell.make_kpts((2, 1, 1), wrap_around=True)
        method = dft.DCKRKS(cell, kpts=kpts, xc="R2SCAN").density_fit(
            auxbasis="gth-szv"
        )
        method.conv_tol = 1e-11
        method.kernel()
        analytic = kpoint_gdf_analytic_stress(method)
        reference = FDStress(
            KPointGaussianDFEnergyEvaluator.from_method(method),
            FDStressConfig(step=2.5e-4, components=("xx",)),
        ).kernel(cell)
        self.assertLess(
            abs(
                analytic.stress_voigt_ha_per_bohr3[0]
                - reference.stress_voigt_ha_per_bohr3[0]
            ),
            3e-8,
        )
        self.assertTrue(analytic.metadata["tau_channel"])

    def test_complex_overlap_and_kinetic_match_fixed_fractional_k_fd(self):
        cell = make_cell()
        kpts = cell.make_kpts((2, 1, 1), wrap_around=True)
        step = 1e-5
        for integral in ("int1e_ovlp", "int1e_kin"):
            response = kpoint_one_electron_strain(cell, kpts, integral)
            for index in (0, 5):
                plus = deform_cell(cell, index, step, mesh=cell.mesh)
                minus = deform_cell(cell, index, -step, mesh=cell.mesh)
                plus_kpts = deform_kpoints(cell, plus, kpts)
                minus_kpts = deform_kpoints(cell, minus, kpts)
                reference = (
                    np.asarray(plus.pbc_intor(integral, kpts=plus_kpts))
                    - np.asarray(minus.pbc_intor(integral, kpts=minus_kpts))
                ) / (2.0 * step)
                error = np.max(np.abs(
                    response.voigt_derivative[index] - reference
                ))
                self.assertLess(float(error), 5e-9, msg=(integral, index))
            self.assertLess(float(np.max(np.abs(
                response.raw_derivative.imag
            ))), 1.0)

    def test_sampled_k_fd_oracle_preserves_fractional_points(self):
        cell = make_cell()
        kpts = cell.make_kpts((2, 1, 1), wrap_around=True)
        method = dft.DCKRKS(cell, kpts=kpts, xc="PBE").density_fit(
            auxbasis="gth-szv"
        )
        method.conv_tol = 1e-10
        method.kernel()
        evaluator = KPointGaussianDFEnergyEvaluator.from_method(method)
        result = FDStress(
            evaluator, FDStressConfig(step=2.5e-4, components=("xx",))
        ).kernel(cell)
        self.assertTrue(np.isfinite(result.stress_voigt_ha_per_bohr3[0]))
        for pair in result.pairs:
            self.assertEqual(pair.plus.metadata["scaled_kpts"], pair.minus.metadata["scaled_kpts"])

    def test_sampled_k_pbe_xc_and_kinetic_match_fixed_density_fd(self):
        from pyscf.pbc import dft as pbc_dft

        cell = make_cell()
        kpts = cell.make_kpts((2, 1, 1), wrap_around=True)
        method = dft.DCKRKS(cell, kpts=kpts, xc="PBE").density_fit(
            auxbasis="gth-szv"
        )
        method.conv_tol = 1e-10
        method.kernel()
        analytic = kpoint_frozen_density_analytic_components(method)
        density = np.asarray(method.source_scf.make_rdm1())
        step = 1e-5
        for index in (0, 5):
            values = []
            for sign in (1.0, -1.0):
                strained = deform_cell(
                    cell, index, sign * step, mesh=cell.mesh
                )
                points = deform_kpoints(cell, strained, kpts)
                target = pbc_dft.KRKS(strained, kpts=points)
                target.xc = "PBE"
                target.build()
                kinetic = np.einsum(
                    "kij,kji->",
                    np.asarray(strained.pbc_intor("int1e_kin", kpts=points)),
                    density, optimize=True,
                ).real / len(points)
                veff = target.get_veff(strained, density)
                values.append((kinetic, float(veff.exc)))
            kinetic_fd = (values[0][0] - values[1][0]) / (2.0 * step)
            xc_fd = (values[0][1] - values[1][1]) / (2.0 * step)
            kinetic = analytic.derivatives_ha["kinetic"]
            xc = analytic.derivatives_ha["xc"]
            x, y = ((0, 0), (0, 1))[index == 5]
            self.assertLess(abs(kinetic[x, y] - kinetic_fd), 2e-8)
            self.assertLess(abs(xc[x, y] - xc_fd), 2e-8)


if __name__ == "__main__":
    unittest.main()
