"""Fast regression tests for benchmark queue/input normalization."""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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
                context={
                    "run_name": "test",
                    "device": "gpu",
                    "run_identity": {
                        "case_run_id": "000_test",
                        "attempt_id": "attempt-1",
                    },
                },
                minimum_write_interval_s=0.0,
                memory_provider=lambda: {
                    "status": "running",
                    "current_device_used_bytes": 1024**3,
                    "peak_observed_device_used_bytes": 2 * 1024**3,
                },
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
            self.assertEqual(
                payload["context"]["run_identity"]["attempt_id"],
                "attempt-1",
            )
            self.assertEqual(payload["timezone"], "Asia/Seoul")
            self.assertTrue(payload["updated_at_kst"].endswith("+09:00"))
            self.assertEqual(
                payload["vram_peak_tracking"][
                    "peak_observed_device_used_bytes"
                ],
                2 * 1024**3,
            )
            self.assertNotIn("updated_at_utc", payload)
            self.assertFalse(any(path.parent.glob("*.tmp")))

            line = show_progress.format_progress(path, payload)
            self.assertIn("25/100", line)
            self.assertIn("25.00%", line)
            self.assertIn("matrix=1.50 GiB", line)
            self.assertIn("vram=1.00/2.00 GiB", line)

            reporter.complete(report_path="report.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["stage"], "complete")

    def test_dead_running_attempt_is_reconciled_as_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory)
            identity = {
                "case_run_id": "000_test",
                "attempt_id": "dead-attempt",
                "started_at_kst": "2026-07-17T12:00:00+09:00",
                "hostname": socket.gethostname(),
            }
            queue._write_json(
                case_dir / "attempt.json",
                {
                    "status": "running",
                    "run_identity": identity,
                    "child_pid": 999_999_999,
                    "child_process_start_ticks": 1,
                },
            )
            queue._write_json(
                case_dir / "progress.json",
                {
                    "status": "running",
                    "stage": "hf_scf",
                    "elapsed_s": 12.5,
                    "context": {"run_identity": identity},
                    "history": [],
                },
            )

            self.assertEqual(
                queue._reconcile_previous_attempt(case_dir), "interrupted"
            )
            attempt = queue._read_json(case_dir / "attempt.json")
            progress = queue._read_json(case_dir / "progress.json")
            self.assertEqual(attempt["status"], "interrupted")
            self.assertEqual(progress["status"], "interrupted")
            self.assertEqual(progress["details"]["previous_stage"], "hf_scf")

    def test_live_attempt_requires_matching_process_start_time(self):
        identity = {"hostname": socket.gethostname()}
        pid = queue.os.getpid()
        start_ticks = queue._process_start_ticks(pid)
        self.assertIsNotNone(start_ticks)
        self.assertTrue(
            queue._attempt_is_active(
                {
                    "run_identity": identity,
                    "child_pid": pid,
                    "child_process_start_ticks": start_ticks,
                }
            )
        )
        self.assertFalse(
            queue._attempt_is_active(
                {
                    "run_identity": identity,
                    "child_pid": pid,
                    "child_process_start_ticks": int(start_ticks) + 1,
                }
            )
        )

    def test_rerun_archives_prior_attempt_as_one_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory)
            arrays = case_dir / "arrays"
            arrays.mkdir()
            (arrays / "gradient.npy").write_bytes(b"array")
            for name in ("report.json", "progress.json"):
                queue._write_json(case_dir / name, {"status": "failed"})
            queue._write_json(
                case_dir / "attempt.json",
                {
                    "run_identity": {"attempt_id": "old-attempt"},
                    "status": "failed",
                },
            )

            archive = queue._archive_case_artifacts(case_dir)
            self.assertIsNotNone(archive)
            self.assertTrue((archive / "report.json").is_file())
            self.assertTrue((archive / "progress.json").is_file())
            self.assertTrue((archive / "attempt.json").is_file())
            self.assertTrue((archive / "arrays" / "gradient.npy").is_file())
            self.assertFalse((case_dir / "report.json").exists())
            self.assertIn("old-attempt", archive.name)

    def test_case_runner_rejects_report_from_another_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_dir = root / "case"
            fake_runner = root / "fake_runner.py"
            fake_runner.write_text(
                "from pathlib import Path\n"
                "import argparse, json\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--input')\n"
                "p.add_argument('--output-dir')\n"
                "a = p.parse_args()\n"
                "Path(a.output_dir, 'report.json').write_text(\n"
                "    json.dumps({'status': 'complete', "
                "'run_identity': {'attempt_id': 'stale'}}))\n",
                encoding="utf-8",
            )

            with mock.patch.object(queue, "CASE_RUNNER", fake_runner):
                outcome = queue._run_case(
                    {"device": "cpu", "num_threads": 1},
                    case_dir,
                    "000_test",
                )

            self.assertEqual(outcome["returncode"], 0)
            self.assertEqual(outcome["status"], "failed")
            self.assertEqual(outcome["report"], {})
            attempt = queue._read_json(case_dir / "attempt.json")
            self.assertFalse(attempt["report_identity_matches"])

    def test_case_runner_accepts_only_its_current_complete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_dir = root / "case"
            fake_runner = root / "fake_runner.py"
            fake_runner.write_text(
                "from pathlib import Path\n"
                "import argparse, json\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--input')\n"
                "p.add_argument('--output-dir')\n"
                "a = p.parse_args()\n"
                "config = json.loads(Path(a.input).read_text())\n"
                "Path(a.output_dir, 'report.json').write_text(\n"
                "    json.dumps({'status': 'complete', "
                "'run_identity': config['run_identity']}))\n",
                encoding="utf-8",
            )

            with mock.patch.object(queue, "CASE_RUNNER", fake_runner):
                outcome = queue._run_case(
                    {"device": "cpu", "num_threads": 1},
                    case_dir,
                    "000_test",
                )

            self.assertEqual(outcome["status"], "complete")
            self.assertEqual(
                outcome["report"]["run_identity"], outcome["run_identity"]
            )
            attempt = queue._read_json(case_dir / "attempt.json")
            self.assertEqual(attempt["status"], "complete")
            self.assertTrue(attempt["report_identity_matches"])

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

    def test_vram_tracker_aggregates_overall_and_stage_high_water(self):
        tracker = benchmark.VRAMPeakTracker(
            "gpu", sample_interval_s=0.1
        )
        tracker._record_sample(
            {
                "stage": "energy",
                "elapsed_s": 0.0,
                "gpu_free_device_bytes": 900,
                "gpu_total_device_bytes": 1000,
                "gpu_device_used_bytes": 100,
                "cupy_pool_used_bytes": 40,
                "cupy_pool_reserved_bytes": 60,
            }
        )
        tracker._record_sample(
            {
                "stage": "gradient",
                "elapsed_s": 0.1,
                "gpu_free_device_bytes": 650,
                "gpu_total_device_bytes": 1000,
                "gpu_device_used_bytes": 350,
                "cupy_pool_used_bytes": 180,
                "cupy_pool_reserved_bytes": 240,
            }
        )
        tracker._record_sample(
            {
                "stage": "gradient",
                "elapsed_s": 0.2,
                "gpu_free_device_bytes": 700,
                "gpu_total_device_bytes": 1000,
                "gpu_device_used_bytes": 300,
                "cupy_pool_used_bytes": 100,
                "cupy_pool_reserved_bytes": 220,
            }
        )

        report = tracker.report()
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(report["baseline_gpu_device_used_bytes"], 100)
        self.assertEqual(report["peak_observed_gpu_device_used_bytes"], 350)
        self.assertEqual(
            report[
                "peak_observed_gpu_device_used_delta_from_baseline_bytes"
            ],
            250,
        )
        self.assertEqual(
            report["peak_observed_cupy_pool_reserved_bytes"], 240
        )
        self.assertEqual(report["stage_peaks"]["gradient"]["sample_count"], 2)
        self.assertEqual(
            report["stage_peaks"]["gradient"][
                "peak_observed_gpu_device_used_bytes"
            ],
            350,
        )

    def test_cpu_vram_tracker_is_explicitly_disabled(self):
        report = benchmark.VRAMPeakTracker("cpu").stop()
        self.assertEqual(report["status"], "disabled")
        self.assertEqual(report["reason"], "cpu_run")


if __name__ == "__main__":
    unittest.main()
