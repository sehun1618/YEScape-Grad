import unittest

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.df import incore

from pyscf_pbc_dcdft import dft
from pyscf_pbc_dcdft.df.rsdf import parameters_from_provider
from pyscf_pbc_dcdft.df.rsdf.metric import rsdf_metric
from pyscf_pbc_dcdft.df.rsdf.source import rsdf_source
from pyscf_pbc_dcdft.df.gdf.cholesky import solve_cderi
from pyscf_pbc_dcdft_stress import (
    FDStress,
    FDStressConfig,
    GammaRSDFEnergyEvaluator,
    gamma_gdf_jk_strain,
    gamma_rsdf_analytic_stress,
    kpoint_rsdf_metric_strain,
    kpoint_rsdf_cderi_strain,
)
from pyscf_pbc_dcdft_stress.gdf_source_strain import (
    gamma_three_center_source_strain,
)
from pyscf_pbc_dcdft_stress.rsdf_cderi_strain import gamma_rsdf_cderi_strain
from pyscf_pbc_dcdft_stress.rsdf_metric_strain import gamma_rsdf_metric_strain
from pyscf_pbc_dcdft_stress.strain import deform_cell, deform_kpoints


def make_method():
    cell = gto.Cell(
        atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
        a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3), (0.1, 0.0, 6.0))),
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((15, 15, 15)), verbose=0,
    ).build()
    method = dft.DCRKS(cell, xc="PBE").rs_density_fit(auxbasis="gth-szv")
    method.conv_tol = 1e-11
    method.kernel()
    return method


class RSDFStrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.method = make_method()
        cls.parameters = parameters_from_provider(cls.method.with_df)

    def test_source_and_metric_match_fixed_parameter_strain_fd(self):
        method = self.method
        cell = method.cell
        provider = method.with_df
        parameters = self.parameters
        source = gamma_three_center_source_strain(
            cell, provider.auxcell, parameters.source_omega,
            parameters.source_mesh, precision=parameters.precision_r,
        )
        metric = gamma_rsdf_metric_strain(
            cell, provider.auxcell, parameters
        )
        point = np.zeros(3)
        step = 1e-5
        for index in (0, 5):
            plus = deform_cell(cell, index, step, mesh=cell.mesh)
            minus = deform_cell(cell, index, -step, mesh=cell.mesh)
            plus_aux = incore.make_auxcell(plus, "gth-szv")
            minus_aux = incore.make_auxcell(minus, "gth-szv")
            plus_aux.precision = parameters.precision_j2c
            minus_aux.precision = parameters.precision_j2c
            metric_fd = (
                rsdf_metric(plus, plus_aux, parameters, q=point)
                - rsdf_metric(minus, minus_aux, parameters, q=point)
            ) / (2.0 * step)
            source_fd = (
                rsdf_source(
                    plus, plus_aux, parameters, kpti=point, kptj=point
                )
                - rsdf_source(
                    minus, minus_aux, parameters, kpti=point, kptj=point
                )
            ) / (2.0 * step)
            self.assertLess(float(np.max(np.abs(
                metric.voigt_derivative[index] - metric_fd
            ))), 5e-8, msg=("metric", index))
            self.assertLess(float(np.max(np.abs(
                source.voigt_derivative[index] - source_fd
            ))), 5e-8, msg=("source", index))

    def test_complex_transfer_metric_matches_fixed_parameter_strain_fd(self):
        cell = self.method.cell
        parameters = self.parameters
        auxcell = self.method.with_df.auxcell
        transfer = cell.get_abs_kpts(
            np.asarray(((0.21, -0.08, 0.04),))
        )[0]
        analytic = kpoint_rsdf_metric_strain(
            cell, auxcell, parameters, transfer
        )
        central = rsdf_metric(
            cell, auxcell, parameters, q=transfer
        )
        self.assertLess(
            float(np.max(np.abs(analytic.value-central))), 3e-9
        )
        step = 1e-5
        for index in range(6):
            plus = deform_cell(cell, index, step, mesh=cell.mesh)
            minus = deform_cell(cell, index, -step, mesh=cell.mesh)
            plus_aux = incore.make_auxcell(plus, "gth-szv")
            minus_aux = incore.make_auxcell(minus, "gth-szv")
            plus_aux.precision = parameters.precision_j2c
            minus_aux.precision = parameters.precision_j2c
            plus_q = deform_kpoints(cell, plus, (transfer,))[0]
            minus_q = deform_kpoints(cell, minus, (transfer,))[0]
            reference = (
                rsdf_metric(plus, plus_aux, parameters, q=plus_q)
                - rsdf_metric(minus, minus_aux, parameters, q=minus_q)
            ) / (2.0*step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index]-reference)
            )
            self.assertLess(float(error), 8e-8, msg=index)

    def test_complex_kpair_cderi_matches_storage_and_strain_fd(self):
        cell = self.method.cell
        kpts = cell.make_kpts((2, 1, 1))
        method = dft.DCKRKS(cell, kpts=kpts, xc="PBE").rs_density_fit(
            auxbasis="gth-szv"
        )
        method.with_df.build(j_only=False)
        response = kpoint_rsdf_cderi_strain(method.with_df, 1, 0)
        self.assertLess(response.reconstructed_value_parity_max_abs, 1e-7)
        parameters = response.parameters
        kpti, kptj = kpts[[1, 0]]

        def fixed_parameter_value(current_cell, current_aux, current_kpts):
            source = rsdf_source(
                current_cell, current_aux, parameters,
                kpti=current_kpts[0], kptj=current_kpts[1],
            )
            metric = rsdf_metric(
                current_cell, current_aux, parameters,
                q=current_kpts[1]-current_kpts[0],
            )
            return solve_cderi(source, metric, current_cell.nao_nr())

        step = 1e-5
        for index in (0, 5):
            plus = deform_cell(cell, index, step, mesh=cell.mesh)
            minus = deform_cell(cell, index, -step, mesh=cell.mesh)
            plus_aux = incore.make_auxcell(plus, "gth-szv")
            minus_aux = incore.make_auxcell(minus, "gth-szv")
            plus_aux.precision = parameters.precision_j2c
            minus_aux.precision = parameters.precision_j2c
            plus_kpts = deform_kpoints(cell, plus, (kpti, kptj))
            minus_kpts = deform_kpoints(cell, minus, (kpti, kptj))
            reference = (
                fixed_parameter_value(plus, plus_aux, plus_kpts)
                - fixed_parameter_value(minus, minus_aux, minus_kpts)
            ) / (2.0*step)
            error = np.max(
                np.abs(response.voigt_derivative[index]-reference)
            )
            self.assertLess(float(error), 1e-7, msg=index)

    def test_cderi_matches_storage_and_produces_finite_jk_response(self):
        method = self.method
        response = gamma_rsdf_cderi_strain(method.with_df)
        self.assertLess(response.reconstructed_value_parity_max_abs, 1e-7)
        density = np.asarray(method.source_scf.make_rdm1())
        jk = gamma_gdf_jk_strain(response, density)
        self.assertTrue(np.isfinite(jk.restricted_hf).all())

    def test_complete_pbe_stress_matches_relaxed_energy_fd(self):
        analytic = gamma_rsdf_analytic_stress(self.method)
        reference = FDStress(
            GammaRSDFEnergyEvaluator.from_method(self.method),
            FDStressConfig(step=2.5e-4, components=("xx", "xy")),
        ).kernel(self.method.cell)
        observed = analytic.stress_voigt_ha_per_bohr3[[0, 5]]
        expected = reference.stress_voigt_ha_per_bohr3[[0, 5]]
        self.assertLess(float(np.max(np.abs(observed - expected))), 2e-8)
        self.assertEqual(analytic.metadata["low_level_numerical_components"], [])


if __name__ == "__main__":
    unittest.main()
