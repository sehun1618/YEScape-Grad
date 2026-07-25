import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft_stress import (
    mlip_record_from_results,
    stress_metadata_for_mlip,
    write_mlip_extxyz,
    write_mlip_json,
)


class MLIPRecordTest(unittest.TestCase):
    def setUp(self):
        self.cell = gto.Cell(
            atom="H 0 0 0; H 0 0 1.4", a=np.eye(3)*5.0,
            unit="Bohr", basis="gth-szv", pseudo="gth-hf",
            mesh=np.asarray((7, 7, 7)), verbose=0,
        ).build()

    def test_force_and_virial_conventions(self):
        gradient = np.asarray(((0.1, 0.2, 0.3), (-0.1, -0.2, -0.3)))
        stress = np.asarray(((1.0, 0.2, 0.3), (0.2, 2.0, 0.4), (0.3, 0.4, 3.0)))*1e-4
        result = SimpleNamespace(
            volume_bohr3=self.cell.vol,
            stress_tensor_ha_per_bohr3=stress,
        )
        record = mlip_record_from_results(
            self.cell, -1.0, gradient, result,
            metadata={"state_fingerprint": "same-state-test"},
        )
        self.assertTrue(np.allclose(record.forces_ha_per_bohr, -gradient))
        self.assertTrue(np.allclose(record.virial_ha, -self.cell.vol*stress))

        with tempfile.TemporaryDirectory() as directory:
            json_path = write_mlip_json(record, Path(directory)/"frame.json")
            xyz_path = write_mlip_extxyz(record, Path(directory)/"frame.xyz")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["conventions"]["stress"], "tensile_positive")
            lines = xyz_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "2")
            self.assertIn("dcdft_stress_tensile_eV_per_A3", lines[1])
            self.assertIn('virial_convention="W=-V*stress"', lines[1])
            self.assertIn(
                "dcdft_state_fingerprint=same-state-test", lines[1]
            )

    def test_rejects_mismatched_volume(self):
        result = SimpleNamespace(
            volume_bohr3=self.cell.vol+1.0,
            stress_tensor_ha_per_bohr3=np.zeros((3, 3)),
        )
        with self.assertRaisesRegex(ValueError, "volume"):
            mlip_record_from_results(
                self.cell, 0.0, np.zeros((2, 3)), result
            )

    def test_stress_owner_metadata_is_exported(self):
        result = SimpleNamespace(metadata={
            "stage": "S7_rsdf_mgga_analytic",
            "production": True,
            "low_level_numerical_components": [],
            "primitive_backend": "host_integral_device_response",
            "pp_response_backend":
                "gpu4pyscf_native_contracted_analytic_nonlocal_device",
            "df_response_backend": "host_analytic_source_adjoint_gpu_ft",
            "source_hf_owner": "gpu",
            "z_vector_owner": "gpu",
        })
        metadata = stress_metadata_for_mlip(result)
        self.assertTrue(metadata["stress_production"])
        self.assertEqual(
            metadata["stress_df_response_backend"],
            "host_analytic_source_adjoint_gpu_ft",
        )
        self.assertEqual(metadata["stress_low_level_numerical_components"], [])

        record = mlip_record_from_results(
            self.cell, -1.0, np.zeros((2, 3)),
            SimpleNamespace(
                volume_bohr3=self.cell.vol,
                stress_tensor_ha_per_bohr3=np.zeros((3, 3)),
            ),
            metadata={"device": "gpu", **metadata},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_mlip_extxyz(record, Path(directory)/"owner.xyz")
            header = path.read_text(encoding="utf-8").splitlines()[1]
        self.assertIn("dcdft_stress_production=T", header)
        self.assertIn(
            "dcdft_stress_df_backend=host_analytic_source_adjoint_gpu_ft",
            header,
        )


if __name__ == "__main__":
    unittest.main()
