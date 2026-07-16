import unittest

import gpu4pyscf_pbc_dcdft


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


if __name__ == "__main__":
    unittest.main()
