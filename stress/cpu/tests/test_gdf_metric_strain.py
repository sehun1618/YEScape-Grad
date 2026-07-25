import unittest

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.df import ft_ao, incore, rsdf

from pyscf_pbc_dcdft_stress import (
    auxiliary_fourier_strain,
    gamma_auxiliary_metric_strain,
    g0_metric_strain,
    long_range_metric_strain,
    short_range_metric_strain,
    kpoint_auxiliary_metric_strain,
    weighted_coulomb_strain,
)
from pyscf_pbc_dcdft_stress.strain import deform_cell, deform_kpoints
from pyscf_pbc_dcdft.df.gdf.metric import auxiliary_metric, short_range_metric


def make_cell():
    return gto.Cell(
        atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
        a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3), (0.1, 0.0, 6.0))),
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((7, 9, 11)), verbose=0,
    ).build()


class GDFMetricStrainTest(unittest.TestCase):
    def test_long_range_metric_blocking_preserves_result(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        grid_points = int(np.prod(mesh))
        full = long_range_metric_strain(
            cell, auxcell, 0.9, mesh,
            reciprocal_block_size=grid_points,
        )
        blocked = long_range_metric_strain(
            cell, auxcell, 0.9, mesh, reciprocal_block_size=19,
        )
        self.assertEqual(blocked.reciprocal_block_size, 19)
        self.assertEqual(blocked.grid_points, grid_points)
        self.assertLess(
            float(np.max(np.abs(blocked.value-full.value))), 2e-11
        )
        self.assertLess(
            float(np.max(np.abs(
                blocked.raw_derivative-full.raw_derivative
            ))),
            2e-10,
        )

    def test_auxiliary_fourier_response_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        analytic = auxiliary_fourier_strain(auxcell, cell.get_Gv(mesh))
        step = 1e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus = ft_ao.ft_ao(plus_aux, plus_cell.get_Gv(mesh)).T
            minus = ft_ao.ft_ao(minus_aux, minus_cell.get_Gv(mesh)).T
            reference = (plus - minus) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 2e-9, msg=index)

    def test_weighted_coulomb_kernel_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        grid_vectors = cell.get_Gv(mesh)
        step = 1e-5
        for omega in (0.9, -0.9, 0.0):
            analytic = weighted_coulomb_strain(
                cell, grid_vectors, omega=omega
            )
            central = rsdf.RSGDF(cell).weighted_coulG(
                np.zeros(3), mesh=mesh, omega=omega
            )
            self.assertLess(float(np.max(np.abs(analytic.value - central))), 1e-13)
            for index in range(6):
                plus_cell = deform_cell(cell, index, step, mesh=mesh)
                minus_cell = deform_cell(cell, index, -step, mesh=mesh)
                plus = rsdf.RSGDF(plus_cell).weighted_coulG(
                    np.zeros(3), mesh=mesh, omega=omega
                )
                minus = rsdf.RSGDF(minus_cell).weighted_coulG(
                    np.zeros(3), mesh=mesh, omega=omega
                )
                reference = (plus - minus) / (2.0 * step)
                error = np.max(
                    np.abs(analytic.voigt_derivative[index] - reference)
                )
                self.assertLess(float(error), 1e-10, msg=(omega, index))

    def test_g0_metric_volume_response_matches_strain_fd(self):
        cell = make_cell()
        auxcell = incore.make_auxcell(cell, "gth-szv")
        analytic = g0_metric_strain(cell, auxcell, 0.9)
        mesh = np.asarray(cell.mesh, dtype=int)
        step = 1e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus = g0_metric_strain(plus_cell, plus_aux, 0.9).value
            minus = g0_metric_strain(minus_cell, minus_aux, 0.9).value
            reference = (plus - minus) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 1e-8, msg=index)

    def test_complete_long_range_metric_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        analytic = long_range_metric_strain(
            cell, auxcell, omega, mesh
        )

        def metric_value(current_cell, current_auxcell):
            grid_vectors = current_cell.get_Gv(mesh)
            auxiliary = ft_ao.ft_ao(current_auxcell, grid_vectors).T
            kernel = rsdf.RSGDF(current_cell).weighted_coulG(
                np.zeros(3), mesh=mesh, omega=omega
            )
            return (auxiliary.conj() * kernel) @ auxiliary.T

        self.assertLess(
            float(np.max(np.abs(analytic.value - metric_value(cell, auxcell)))),
            1e-12,
        )
        step = 1e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            reference = (
                metric_value(plus_cell, plus_aux)
                - metric_value(minus_cell, minus_aux)
            ) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 2e-9, msg=index)

    def test_image_resolved_short_range_metric_matches_strain_fd(self):
        cell = make_cell()
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        analytic = short_range_metric_strain(auxcell, omega)
        central = short_range_metric(auxcell, omega, q=np.zeros(3))
        self.assertLess(
            float(np.max(np.abs(analytic.value - central))), 1e-12
        )
        step = 2e-5
        mesh = np.asarray(cell.mesh, dtype=int)
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus = short_range_metric(plus_aux, omega, q=np.zeros(3))
            minus = short_range_metric(minus_aux, omega, q=np.zeros(3))
            reference = (plus - minus) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 2e-8, msg=index)

    def test_complete_gamma_metric_matches_strain_fd(self):
        cell = make_cell()
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        mesh = np.asarray(cell.mesh, dtype=int)
        analytic = gamma_auxiliary_metric_strain(
            cell, auxcell, omega, mesh
        )
        central = auxiliary_metric(
            cell, auxcell, omega, mesh=mesh, q=np.zeros(3)
        )
        self.assertLess(
            float(np.max(np.abs(analytic.value - central))), 1e-12
        )
        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus = auxiliary_metric(
                plus_cell, plus_aux, omega, mesh=mesh, q=np.zeros(3)
            )
            minus = auxiliary_metric(
                minus_cell, minus_aux, omega, mesh=mesh, q=np.zeros(3)
            )
            reference = (plus - minus) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 3e-8, msg=index)

    def test_complete_complex_transfer_metric_matches_strain_fd(self):
        cell = make_cell()
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        mesh = np.asarray(cell.mesh, dtype=int)
        transfer = cell.get_abs_kpts(np.asarray(((0.21, -0.08, 0.04),)))[0]
        analytic = kpoint_auxiliary_metric_strain(
            cell, auxcell, omega, mesh, transfer
        )
        central = auxiliary_metric(
            cell, auxcell, omega, mesh=mesh, q=transfer
        )
        self.assertLess(
            float(np.max(np.abs(analytic.value-central))), 3e-9
        )
        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus_q = deform_kpoints(cell, plus_cell, (transfer,))[0]
            minus_q = deform_kpoints(cell, minus_cell, (transfer,))[0]
            plus = auxiliary_metric(
                plus_cell, plus_aux, omega, mesh=mesh, q=plus_q
            )
            minus = auxiliary_metric(
                minus_cell, minus_aux, omega, mesh=mesh, q=minus_q
            )
            reference = (plus-minus) / (2.0*step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index]-reference)
            )
            self.assertLess(float(error), 6e-8, msg=index)


if __name__ == "__main__":
    unittest.main()
