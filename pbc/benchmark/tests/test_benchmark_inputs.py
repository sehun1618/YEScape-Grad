"""Fast regression tests for benchmark queue/input normalization."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

import dcdft_benchmark as benchmark  # noqa: E402
import run_benchmark_queue as queue  # noqa: E402
import show_progress  # noqa: E402


class BenchmarkInputTest(unittest.TestCase):
    def test_zvector_controls_are_applied_to_public_gradient_object(self):
        class Gradient:
            z_solver = "auto"
            z_tolerance = 1.0e-12
            z_residual_tolerance = 1.0e-8
            z_max_cycle = 80
            z_dense_max_dimension = 256
            z_nuclear_batch_memory_fraction = 0.05
            z_nuclear_batch_max_components = None
            gdf_bilinear_mode = "two_pass"

        gradient = Gradient()
        benchmark._configure_gradient(
            {
                "z_solver": "matrix_free",
                "z_tolerance": "1e-11",
                "z_residual_tolerance": "2e-8",
                "z_max_cycle": "120",
                "z_dense_max_dimension": "128",
                "z_nuclear_batch_memory_fraction": "0.025",
                "z_nuclear_batch_max_components": "4",
                "gdf_bilinear_mode": "one_pass",
            },
            gradient,
        )

        self.assertEqual(gradient.z_solver, "matrix_free")
        self.assertEqual(gradient.z_tolerance, 1.0e-11)
        self.assertEqual(gradient.z_residual_tolerance, 2.0e-8)
        self.assertEqual(gradient.z_max_cycle, 120)
        self.assertEqual(gradient.z_dense_max_dimension, 128)
        self.assertEqual(gradient.z_nuclear_batch_memory_fraction, 0.025)
        self.assertEqual(gradient.z_nuclear_batch_max_components, 4)
        self.assertEqual(gradient.gdf_bilinear_mode, "one_pass")

        benchmark._configure_gradient(
            {"z_nuclear_batch_max_components": None}, gradient
        )
        self.assertIsNone(gradient.z_nuclear_batch_max_components)

    def test_all_fd_selection_accepts_string_and_singleton_list(self):
        self.assertEqual(benchmark._selected_atoms("all", 3), [0, 1, 2])
        self.assertEqual(benchmark._selected_atoms(["all"], 3), [0, 1, 2])
        self.assertEqual(benchmark._selected_axes("all"), [0, 1, 2])
        self.assertEqual(benchmark._selected_axes(["all"]), [0, 1, 2])

    def test_result_directory_uses_xyz_stem_after_numeric_prefix(self):
        run = {
            "atom_file": "/tmp/Water Cluster.xyz",
            "name": "selected_fd_check_cpu_fftdf",
        }
        run_id = f"006_{queue._structure_slug(run)}_{queue._slug(run['name'])}"
        self.assertEqual(
            run_id,
            "006_Water_Cluster_selected_fd_check_cpu_fftdf",
        )

    def test_progress_reporter_writes_atomic_stage_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            reporter = benchmark.ProgressReporter(
                path,
                context={"run_name": "test", "device": "gpu"},
                minimum_write_interval_s=0.0,
            )
            reporter.update(
                "zvector_matrix",
                force=True,
                completed=25,
                total=100,
                matrix_gib=1.5,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "running")
            self.assertEqual(payload["stage"], "zvector_matrix")
            self.assertEqual(payload["stage_fraction"], 0.25)
            self.assertEqual(payload["context"]["run_name"], "test")
            self.assertEqual(payload["timezone"], "Asia/Seoul")
            self.assertTrue(payload["updated_at_kst"].endswith("+09:00"))
            self.assertNotIn("updated_at_utc", payload)
            self.assertFalse(any(path.parent.glob("*.tmp")))

            line = show_progress.format_progress(path, payload)
            self.assertIn("25/100", line)
            self.assertIn("25.00%", line)
            self.assertIn("matrix=1.50 GiB", line)

            reporter.complete(report_path="report.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["stage"], "complete")

    def test_memory_summary_keeps_stage_samples_and_high_water(self):
        report = benchmark.summarize_memory(
            [
                {
                    "stage": "start",
                    "host_peak_rss_bytes": 100,
                    "gpu_free_device_bytes": 900,
                    "gpu_total_device_bytes": 1000,
                    "cupy_pool_used_bytes": 10,
                    "cupy_pool_reserved_bytes": 20,
                },
                {
                    "stage": "gradient",
                    "host_peak_rss_bytes": 180,
                    "gpu_free_device_bytes": 700,
                    "gpu_total_device_bytes": 1000,
                    "cupy_pool_used_bytes": 80,
                    "cupy_pool_reserved_bytes": 120,
                },
            ]
        )
        self.assertEqual(report["host_peak_rss_bytes"], 180)
        self.assertEqual(
            report["minimum_observed_gpu_free_device_bytes"], 700
        )
        self.assertEqual(
            report["peak_observed_cupy_pool_reserved_bytes"], 120
        )
        self.assertEqual(len(report["samples"]), 2)


if __name__ == "__main__":
    unittest.main()
