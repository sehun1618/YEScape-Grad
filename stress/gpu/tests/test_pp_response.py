import unittest

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.grad import rks_stress
from pyscf.pbc import gto

from gpu4pyscf_pbc_dcdft_stress.pp_response import (
    _fused_pp_nonlocal_strain,
    _vpploc_g_strain,
)


def _water_cell():
    cell = gto.Cell()
    cell.atom = (
        "O 0 0 0; O 2 0 0; "
        "H 0 1.4 1.1; H 0 -1.4 1.1"
    )
    cell.a = np.asarray((
        (14.0, 0.1, 0.0),
        (0.0, 13.5, 0.2),
        (0.1, 0.0, 14.5),
    ))
    cell.unit = "Bohr"
    cell.basis = "SZV-GTH"
    cell.pseudo = {"H": "GTH-SCAN-q1", "O": "GTH-SCAN-q6"}
    cell.mesh = [9, 11, 13]
    cell.precision = 1e-9
    cell.verbose = 0
    cell.build()
    return cell


def _hydrogen_cell():
    cell = gto.Cell()
    cell.atom = "H 0 0 0; H 0 0 1.4"
    cell.a = np.eye(3)*8.0
    cell.unit = "Bohr"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf"
    cell.mesh = [9, 9, 9]
    cell.precision = 1e-10
    cell.verbose = 0
    cell.build()
    return cell


def _water_dzvp_cell():
    cell = _water_cell()
    cell.basis = "DZVP-MOLOPT-SR-GTH"
    cell.mesh = [7, 9, 11]
    cell.build(False, False)
    return cell


class ContractedPPResponseTest(unittest.TestCase):
    def test_analytic_local_reciprocal_strain_matches_upstream_fd(self):
        cell = _water_cell()
        expected_value, expected_derivative = (
            rks_stress._get_vpplocG_strain_derivatives(cell, cell.mesh)
        )
        value, _, factors = _vpploc_g_strain(
            rks_stress, cell, np.asarray(cell.mesh)
        )
        gvectors, slope = factors
        derivative = cp.empty_like(expected_derivative)
        for row in range(3):
            for column in range(3):
                derivative[row, column] = (
                    2.0*gvectors[:, row]*gvectors[:, column]*slope
                )
        self.assertLess(
            float(cp.max(cp.abs(value-expected_value)).get()), 1e-10
        )
        self.assertLess(
            float(cp.max(cp.abs(
                derivative-expected_derivative
            )).get()),
            2e-7,
        )

    def test_batched_voigt_nonlocal_matches_symmetrized_upstream(self):
        cell = _water_cell()
        rng = np.random.default_rng(7)
        densities = []
        for _ in range(2):
            value = rng.normal(size=(cell.nao_nr(), cell.nao_nr()))
            densities.append(value+value.T)
        density = cp.asarray(np.stack(densities))
        observed = _fused_pp_nonlocal_strain(
            rks_stress, cell, np.asarray(cell.mesh), density
        )
        expected = []
        for value in density:
            tensor = cp.asarray(
                rks_stress._get_pp_nonloc_strain_derivatives(
                    cell, cell.mesh, value
                )
            )
            expected.append(0.5*(tensor+tensor.T))
        self.assertLess(
            float(cp.max(cp.abs(
                observed-cp.stack(expected)
            )).get()),
            2e-7,
        )

    def test_nonlocal_profile_reports_real_work(self):
        cell = _water_cell()
        density = cp.zeros((2, cell.nao_nr(), cell.nao_nr()))
        _, profile = _fused_pp_nonlocal_strain(
            rks_stress, cell, np.asarray(cell.mesh), density,
            return_profile=True,
        )
        self.assertEqual(profile["strain_component_count"], 6)
        self.assertEqual(profile["displaced_cell_evaluation_count"], 12)
        self.assertEqual(profile["ft_ao_call_count"], 12)
        self.assertGreater(profile["projector_template_build_count"], 0)
        self.assertGreater(profile["explicit_host_to_device_bytes"], 0)
        self.assertGreater(profile["structure_factor_device_cache_bytes"], 0)
        self.assertGreater(
            profile["estimated_explicit_live_workspace_bytes"], 0
        )

    def test_host_analytic_nonlocal_matches_centered_displacement(self):
        cell = _water_cell()
        rng = np.random.default_rng(17)
        densities = []
        for _ in range(2):
            value = rng.normal(size=(cell.nao_nr(), cell.nao_nr()))
            densities.append(value+value.T)
        density = cp.asarray(np.stack(densities))
        numerical = _fused_pp_nonlocal_strain(
            rks_stress, cell, np.asarray(cell.mesh), density,
        )
        analytic, profile = _fused_pp_nonlocal_strain(
            rks_stress, cell, np.asarray(cell.mesh), density,
            response_backend="host_analytic_matrix", host_block_size=17,
            return_profile=True,
        )
        self.assertLess(
            float(cp.max(cp.abs(analytic-numerical)).get()), 3e-7
        )
        self.assertEqual(profile["displaced_cell_evaluation_count"], 0)
        self.assertGreater(profile["host_ft_ao_block_count"], 0)
        self.assertGreater(profile["explicit_host_to_device_bytes"], 0)
        self.assertEqual(
            profile["owner"],
            "host_analytic_fourier_projector_strain_device_contracted",
        )

    def test_gpu_analytic_nonlocal_matches_host_analytic(self):
        cell = _water_cell()
        rng = np.random.default_rng(23)
        densities = []
        for _ in range(2):
            value = rng.normal(size=(cell.nao_nr(), cell.nao_nr()))
            densities.append(value+value.T)
        density = cp.asarray(np.stack(densities))
        expected = _fused_pp_nonlocal_strain(
            rks_stress, cell, np.asarray(cell.mesh), density,
            response_backend="host_analytic_matrix", host_block_size=17,
        )
        observed, profile = _fused_pp_nonlocal_strain(
            rks_stress, cell, np.asarray(cell.mesh), density,
            response_backend="gpu_analytic", gpu_block_size=17,
            return_profile=True,
        )
        self.assertLess(
            float(cp.max(cp.abs(observed-expected)).get()), 2e-10
        )
        self.assertEqual(profile["displaced_cell_evaluation_count"], 0)
        self.assertGreater(profile["ft_ao_call_count"], 0)
        self.assertEqual(
            profile["ft_ao_call_count"], profile["moment_ft_ao_call_count"]
        )
        self.assertEqual(
            profile["owner"],
            "gpu_analytic_augmented_shell_fourier_moment_contracted",
        )

    def test_gpu_analytic_nonlocal_matches_host_for_sampled_k(self):
        cell = _water_cell()
        kpts = np.asarray(cell.make_kpts([2, 1, 1]), dtype=float)
        rng = np.random.default_rng(29)
        densities = np.empty(
            (2, len(kpts), cell.nao_nr(), cell.nao_nr()),
            dtype=np.complex128,
        )
        for density_id in range(2):
            for kpoint_id in range(len(kpts)):
                value = rng.normal(size=densities.shape[-2:])
                value = value+1j*rng.normal(size=densities.shape[-2:])
                densities[density_id, kpoint_id] = value+value.conj().T
        density = cp.asarray(densities)
        expected = _fused_pp_nonlocal_strain(
            rks_stress, cell, np.asarray(cell.mesh), density, kpts=kpts,
            response_backend="host_analytic_matrix", host_block_size=17,
        )
        observed, profile = _fused_pp_nonlocal_strain(
            rks_stress, cell, np.asarray(cell.mesh), density, kpts=kpts,
            response_backend="gpu_analytic", gpu_block_size=17,
            return_profile=True,
        )
        self.assertLess(
            float(cp.max(cp.abs(observed-expected)).get()), 2e-10
        )
        self.assertEqual(profile["kpoint_count"], 2)
        self.assertEqual(profile["displaced_cell_evaluation_count"], 0)

    def test_gpu_analytic_nonlocal_matches_host_with_d_polarization(self):
        cell = _water_dzvp_cell()
        rng = np.random.default_rng(31)
        value = rng.normal(size=(cell.nao_nr(), cell.nao_nr()))
        density = cp.asarray((value+value.T)[None])
        expected = _fused_pp_nonlocal_strain(
            rks_stress, cell, np.asarray(cell.mesh), density,
            response_backend="host_analytic_matrix", host_block_size=19,
        )
        observed = _fused_pp_nonlocal_strain(
            rks_stress, cell, np.asarray(cell.mesh), density,
            response_backend="gpu_analytic", gpu_block_size=19,
        )
        self.assertLess(
            float(cp.max(cp.abs(observed-expected)).get()), 3e-10
        )

    def test_empty_nonlocal_projector_is_an_explicit_zero(self):
        cell = _hydrogen_cell()
        density = cp.zeros((2, cell.nao_nr(), cell.nao_nr()))
        value, profile = _fused_pp_nonlocal_strain(
            rks_stress, cell, np.asarray(cell.mesh), density,
            return_profile=True,
        )
        self.assertEqual(float(cp.max(cp.abs(value)).get()), 0.0)
        self.assertTrue(profile["empty_projector_fast_path"])
        self.assertEqual(profile["displaced_cell_evaluation_count"], 0)


if __name__ == "__main__":
    unittest.main()
