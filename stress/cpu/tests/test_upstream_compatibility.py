import unittest

from pyscf_pbc_dcdft_stress.upstream import (
    PySCFKPointStressPrimitives,
    PySCFStressPrimitives,
    supports_pyscf_version,
)


class UpstreamCompatibilityTest(unittest.TestCase):
    def test_supported_release_series_are_explicit(self):
        self.assertTrue(supports_pyscf_version("2.13.0"))
        self.assertTrue(supports_pyscf_version("2.13.1"))
        self.assertTrue(supports_pyscf_version("2.14.0"))
        self.assertFalse(supports_pyscf_version("2.12.1"))
        self.assertFalse(supports_pyscf_version("2.15.0"))

    def test_installed_gamma_primitives_match_validated_api(self):
        primitives = PySCFStressPrimitives.load()

        self.assertTrue(supports_pyscf_version(primitives.version))
        self.assertTrue(callable(primitives.module.get_vxc))

    def test_installed_kpoint_primitives_match_validated_api(self):
        primitives = PySCFKPointStressPrimitives.load()

        self.assertTrue(supports_pyscf_version(primitives.version))
        self.assertTrue(callable(primitives.module.get_vxc))


if __name__ == "__main__":
    unittest.main()
