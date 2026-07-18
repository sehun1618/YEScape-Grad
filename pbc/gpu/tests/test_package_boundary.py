import unittest

import numpy as np
import gpu4pyscf_pbc_dcdft
from gpu4pyscf_pbc_dcdft import dft
from pyscf.pbc import gto
from pyscf_pbc_dcdft.dft.state import validate_restricted_input


def _cell():
    cell = gto.Cell()
    cell.atom = "H 0 0 0; H 0 0 0.74"
    cell.a = np.eye(3) * 6.0
    cell.unit = "Angstrom"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf"
    cell.ke_cutoff = 30.0
    cell.verbose = 0
    return cell.build()


class GPUPackageBoundaryTest(unittest.TestCase):
    def test_gpu_package_reports_all_completed_df_routes(self) -> None:
        report = gpu4pyscf_pbc_dcdft.capability()
        self.assertIs(report["silent_cpu_fallback"], False)
        self.assertIs(report["fftdf_energy"], True)
        self.assertIs(report["fftdf_analytic_gradient"], True)
        self.assertIs(report["gdf_energy"], True)
        self.assertIs(report["gdf_analytic_gradient"], True)
        self.assertIs(report["rsdf_energy"], True)
        self.assertIs(report["rsdf_analytic_gradient"], True)
        self.assertIs(report["scan_analytic_gradient"], True)
        self.assertIs(report["r2scan_analytic_gradient"], True)
        self.assertEqual(
            report["validated_target_functionals"],
            ["PBE", "SCAN", "R2SCAN"],
        )
        self.assertEqual(
            report["route_ownership"]["fftdf_full_jk_derivative"],
            "gpu4pyscf_pbc_dcdft.df.fft",
        )
        self.assertEqual(
            report["route_ownership"]["gdf_full_jk_derivative"],
            "gpu4pyscf_pbc_dcdft.df.gdf",
        )
        self.assertEqual(
            report["route_ownership"]["rsdf_full_jk_derivative"],
            "gpu4pyscf_pbc_dcdft.df.rsdf",
        )

    def test_invalid_conversion_fails_loudly(self) -> None:
        with self.assertRaises(AttributeError):
            gpu4pyscf_pbc_dcdft.to_gpu(object())

    def test_gpu_provider_modifier_invalidates_state(self) -> None:
        method = dft.DCRKS(_cell(), xc="PBE")
        method.converged = True
        method.source_scf = object()
        method.e_tot = -1.0
        method._dcdft_input_signature = "old"

        method.fft_density_fit()
        self.assertFalse(method.converged)
        self.assertIsNone(method.source_scf)
        self.assertIsNone(method.e_tot)
        self.assertIsNone(method._dcdft_input_signature)

    def test_gpu_fftdf_custom_mesh_mismatch_is_rejected(self) -> None:
        method = dft.DCRKS(_cell(), xc="PBE")
        method.fft_density_fit(mesh=[9, 11, 13])
        with self.assertRaisesRegex(ValueError, "with_df.mesh to match"):
            validate_restricted_input(method)

    def test_target_integrator_modifier_is_explicit_and_invalidates_state(self):
        method = dft.DCRKS(_cell(), xc="R2SCAN")
        method.converged = True
        method.e_tot = -1.0
        returned = method.set_target_integrator(
            "multigrid_xc", mesh=[9, 11, 13]
        )
        self.assertIs(returned, method)
        self.assertEqual(method.target_integrator, "multigrid_xc")
        np.testing.assert_array_equal(
            method.target_multigrid_mesh, [9, 11, 13]
        )
        self.assertFalse(method.converged)
        self.assertIsNone(method.e_tot)

    def test_uniform_target_rejects_multigrid_mesh(self):
        method = dft.DCRKS(_cell(), xc="PBE")
        with self.assertRaisesRegex(ValueError, "requires a multigrid"):
            method.set_target_integrator("uniform", mesh=[9, 9, 9])


if __name__ == "__main__":
    unittest.main()
