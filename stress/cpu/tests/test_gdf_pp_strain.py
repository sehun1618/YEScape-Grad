import unittest

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.df import rsdf_builder
from pyscf.pbc.df import aft
from pyscf.pbc.gto.pseudo import pp_int

from pyscf_pbc_dcdft import dft
from pyscf_pbc_dcdft_stress import (
    gamma_gdf_local_part1_strain,
    gamma_gdf_local_part2_strain,
    gamma_gdf_nonlocal_strain,
    gamma_gdf_pseudopotential_strain,
    gdf_pseudopotential_finite_difference,
    gdf_residual_pseudopotential_finite_difference,
    kpoint_gdf_pseudopotential_strain,
)
from pyscf_pbc_dcdft_stress.strain import deform_cell, deform_kpoints


def make_cell():
    return gto.Cell(
        atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
        a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3), (0.1, 0.0, 6.0))),
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((15, 15, 15)), verbose=0,
    ).build()


def local_part1_value(cell):
    builder = rsdf_builder._RSNucBuilder(
        cell, np.zeros((1, 3))
    ).build()
    return np.asarray(builder.get_pp_loc_part1())[0]


def make_water_cell():
    return gto.Cell(
        atom="O 1.0 1.0 1.0; H 1.0 1.0 2.0; H 1.8 1.0 0.7",
        a=np.eye(3) * 7.0, unit="Bohr", basis="gth-szv",
        pseudo="gth-pbe", mesh=np.asarray((15, 15, 15)), verbose=0,
    ).build()


class GDFPseudopotentialStrainTest(unittest.TestCase):
    def test_sampled_k_h2_pseudopotential_matches_builder_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        kpts = cell.make_kpts((2, 1, 1))
        analytic = kpoint_gdf_pseudopotential_strain(
            cell, kpts, block_size=4096
        )

        def value(current_cell, current_kpts):
            builder = rsdf_builder._RSNucBuilder(
                current_cell, current_kpts
            ).build()
            return np.asarray(builder.get_pp())

        self.assertLess(
            float(np.max(np.abs(analytic.value-value(cell, kpts)))), 3e-8
        )
        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_kpts = deform_kpoints(cell, plus_cell, kpts)
            minus_kpts = deform_kpoints(cell, minus_cell, kpts)
            reference = (
                value(plus_cell, plus_kpts)-value(minus_cell, minus_kpts)
            )/(2.0*step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index]-reference)
            )
            self.assertLess(float(error), 3e-7, msg=index)

    def test_complete_h2_pseudopotential_matches_provider_fd(self):
        cell = make_cell()
        method = dft.DCRKS(cell, xc="PBE").density_fit(auxbasis="gth-szv")
        method.conv_tol = 1e-11
        method.kernel()
        analytic = gamma_gdf_pseudopotential_strain(
            cell, block_size=4096
        )
        reference = gdf_pseudopotential_finite_difference(
            method, step=2e-5, components=("xx", "xy")
        )
        error = np.nanmax(
            np.abs(
                analytic.voigt_derivative[[0, 5]]
                - reference.derivative_ha[[0, 5]]
            )
        )
        self.assertLess(float(error), 2e-7)

    def test_complete_water_pseudopotential_matches_provider_fd(self):
        cell = make_water_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        analytic = gamma_gdf_pseudopotential_strain(
            cell, block_size=4096
        )

        def value(current_cell):
            builder = rsdf_builder._RSNucBuilder(
                current_cell, np.zeros((1, 3))
            ).build()
            return np.asarray(builder.get_pp())[0]

        self.assertLess(
            float(np.max(np.abs(analytic.value - value(cell)))), 2e-8
        )
        step = 2e-5
        plus_cell = deform_cell(cell, "xx", step, mesh=mesh)
        minus_cell = deform_cell(cell, "xx", -step, mesh=mesh)
        reference = (value(plus_cell) - value(minus_cell)) / (2.0 * step)
        error = np.max(np.abs(analytic.voigt_derivative[0] - reference))
        self.assertLess(float(error), 2e-7)

    def test_nonlocal_projector_matches_integral_builder_strain_fd(self):
        cell = make_water_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        reciprocal_mesh = np.asarray((41, 41, 41))
        analytic = gamma_gdf_nonlocal_strain(
            cell, mesh=reciprocal_mesh, block_size=4096
        )
        central = np.asarray(
            pp_int.get_pp_nl(cell, np.zeros((1, 3)))
        )[0]
        self.assertLess(
            float(np.max(np.abs(analytic.value - central))), 1e-8
        )
        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus = np.asarray(
                pp_int.get_pp_nl(plus_cell, np.zeros((1, 3)))
            )[0]
            minus = np.asarray(
                pp_int.get_pp_nl(minus_cell, np.zeros((1, 3)))
            )[0]
            reference = (plus - minus) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 1e-7, msg=index)

    def test_local_part2_matches_integral_builder_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        analytic = gamma_gdf_local_part2_strain(cell, block_size=4096)
        central = np.asarray(
            aft._IntPPBuilder(cell, np.zeros((1, 3))).get_pp_loc_part2()
        )[0]
        self.assertLess(
            float(np.max(np.abs(analytic.value - central))), 2e-8
        )
        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus = np.asarray(
                aft._IntPPBuilder(
                    plus_cell, np.zeros((1, 3))
                ).get_pp_loc_part2()
            )[0]
            minus = np.asarray(
                aft._IntPPBuilder(
                    minus_cell, np.zeros((1, 3))
                ).get_pp_loc_part2()
            )[0]
            reference = (plus - minus) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 5e-8, msg=index)

    def test_full_pp_splits_into_analytic_local1_and_residual_fd(self):
        cell = make_cell()
        method = dft.DCRKS(cell, xc="PBE").density_fit(auxbasis="gth-szv")
        method.conv_tol = 1e-11
        method.kernel()
        full = gdf_pseudopotential_finite_difference(
            method, step=2e-5, components=("xx", "xy")
        )
        residual = gdf_residual_pseudopotential_finite_difference(
            method, step=2e-5, components=("xx", "xy")
        )
        local1 = gamma_gdf_local_part1_strain(cell)
        reconstructed = local1.voigt_derivative + residual.derivative_ha
        error = np.nanmax(
            np.abs(reconstructed[[0, 5]] - full.derivative_ha[[0, 5]])
        )
        self.assertLess(float(error), 8e-8)

    def test_local_part1_matches_builder_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        analytic = gamma_gdf_local_part1_strain(cell)
        self.assertLess(
            float(np.max(np.abs(analytic.value - local_part1_value(cell)))),
            1e-8,
        )
        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            reference = (
                local_part1_value(plus_cell)
                - local_part1_value(minus_cell)
            ) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 6e-8, msg=index)


if __name__ == "__main__":
    unittest.main()
