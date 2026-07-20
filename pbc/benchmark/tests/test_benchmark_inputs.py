"""Fast regression tests for benchmark queue/input normalization."""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(BENCHMARK_DIR / "development"))
sys.path.insert(
    0,
    str(BENCHMARK_DIR / "development" / "diagnostics"),
)

import dcdft_benchmark as benchmark  # noqa: E402
import diagnose_rigid_translation_eggbox as translation  # noqa: E402
import run_water8_eggbox_stage12 as eggbox_stage12  # noqa: E402
import summarize_optimization_baseline as optimization_baseline  # noqa: E402
import evidence  # noqa: E402
import run_benchmark_queue as queue  # noqa: E402
import show_progress  # noqa: E402


def _case_evidence(value="fingerprint-1"):
    return {
        "schema": "pbc_dcdft_case_evidence_v1",
        "fingerprint_sha256": value,
    }


class BenchmarkInputTest(unittest.TestCase):
    def test_repository_json_layout_and_queue_paths(self):
        user_queues_dir = BENCHMARK_DIR / "user" / "queues"
        development_queues_dir = (
            BENCHMARK_DIR / "development" / "queues"
        )
        backup_dir = BENCHMARK_DIR / "backup"
        backup_queues_dir = backup_dir / "queues"
        backup_evidence_dir = backup_dir / "evidence"
        user_results_dir = (
            BENCHMARK_DIR / "user" / "results"
        ).resolve()
        development_results_dir = (
            BENCHMARK_DIR / "development" / "results"
        ).resolve()
        backup_results_dir = (backup_dir / "results").resolve()
        structures_dir = (BENCHMARK_DIR / "structures").resolve()

        self.assertEqual(list(BENCHMARK_DIR.glob("*.json")), [])
        self.assertFalse((BENCHMARK_DIR / "queues").exists())
        user_queue_paths = sorted(user_queues_dir.glob("*.json"))
        development_queue_paths = sorted(
            development_queues_dir.glob("*.json")
        )
        backup_queue_paths = sorted(backup_queues_dir.glob("*.json"))
        evidence_paths = sorted(backup_evidence_dir.glob("*.json"))
        self.assertGreater(len(user_queue_paths), 0)
        self.assertGreater(len(development_queue_paths), 0)
        self.assertGreater(len(backup_queue_paths), 0)
        self.assertGreater(len(evidence_paths), 0)

        for path, expected_results_dir, expected_scope in [
            *((path, user_results_dir, "user") for path in user_queue_paths),
            *(
                (path, development_results_dir, "development")
                for path in development_queue_paths
            ),
            *(
                (path, backup_results_dir, None)
                for path in backup_queue_paths
            ),
        ]:
            payload = json.loads(path.read_text())
            self.assertIn("defaults", payload, path.name)
            self.assertIn("jobs", payload, path.name)
            if expected_scope is not None:
                self.assertEqual(
                    payload.get("benchmark_scope"),
                    expected_scope,
                    path.name,
                )
            runs = queue.expand_jobs(payload)
            self.assertGreater(len(runs), 0, path.name)

            output_root = (path.parent / payload["output_root"]).resolve()
            self.assertTrue(
                output_root.is_relative_to(expected_results_dir), path.name
            )
            for run in runs:
                if run.get("atom_file") is None:
                    self.assertIsNotNone(run.get("atom"), path.name)
                    continue
                queue._resolve_atom_file(run, path.parent)
                atom_path = Path(run["atom_file"])
                self.assertTrue(atom_path.is_file(), path.name)
                self.assertTrue(
                    atom_path.is_relative_to(structures_dir), path.name
                )

        for path in evidence_paths:
            payload = json.loads(path.read_text())
            self.assertIn("schema", payload, path.name)
            self.assertNotIn("jobs", payload, path.name)

    def test_managed_queue_scope_rejects_cross_scope_output(self):
        queue_path = (
            BENCHMARK_DIR / "user" / "queues" / "current_correctness.json"
        )
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
        user_output = BENCHMARK_DIR / "user" / "results" / "correctness"
        development_output = (
            BENCHMARK_DIR / "development" / "results" / "correctness"
        )

        self.assertEqual(
            queue._validate_managed_queue(
                queue_path,
                payload,
                user_output,
            ),
            "user",
        )
        with self.assertRaisesRegex(ValueError, "output must stay"):
            queue._validate_managed_queue(
                queue_path,
                payload,
                development_output,
            )

    def test_user_queues_follow_current_production_profile(self):
        queues_dir = BENCHMARK_DIR / "user" / "queues"
        forbidden_name_parts = ("probe", "release", "baseline", "eggbox")
        for path in sorted(queues_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            defaults = payload["defaults"]
            self.assertFalse(
                any(part in path.stem for part in forbidden_name_parts),
                path.name,
            )
            self.assertEqual(defaults["gdf_setup_mode"], "native")
            self.assertEqual(
                defaults["rsdf_central_setup"],
                "exact_gpu_source",
            )
            self.assertEqual(
                defaults["rsdf_compressed_device_cache"],
                "host",
            )
            self.assertEqual(defaults["gdf_derivative_omega_scale"], 1.0)
            self.assertEqual(defaults["rsdf_derivative_omega_scale"], 1.0)
            for job in payload["jobs"]:
                if job.get("df_backend") == "gdf":
                    self.assertEqual(
                        job.get("gdf_bilinear_mode"),
                        "optimized_fused",
                        path.name,
                    )

        water64_dz = json.loads(
            (queues_dir / "current_water64_dz.json").read_text(
                encoding="utf-8"
            )
        )
        for job in water64_dz["jobs"]:
            self.assertEqual(
                bool(job.get("enabled")),
                job["df_backend"] == "gdf",
            )

    def test_array_identity_is_compact_and_byte_sensitive(self):
        first = benchmark._array_identity(np.asarray([[1.0, 2.0]]))
        second = benchmark._array_identity(np.asarray([[1.0, 3.0]]))

        self.assertEqual(first["shape"], [1, 2])
        self.assertEqual(len(first["sha256"]), 64)
        self.assertEqual(len(first["rounded_1e10_sha256"]), 64)
        self.assertAlmostEqual(first["frobenius_norm"], np.sqrt(5.0))
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_queue_expands_target_integrators_without_renaming_uniform(self):
        runs = queue.expand_jobs(
            {
                "defaults": {
                    "device": "gpu",
                    "df_backend": "gdf",
                },
                "jobs": [
                    {
                        "name": "target_modes",
                        "target_integrators": [
                            "uniform",
                            "multigrid_xc",
                            "multigrid_j_xc",
                        ],
                    }
                ],
            }
        )
        self.assertEqual(
            [run["name"] for run in runs],
            [
                "target_modes_gpu_gdf",
                "target_modes_gpu_gdf_multigrid_xc",
                "target_modes_gpu_gdf_multigrid_j_xc",
            ],
        )
        self.assertEqual(
            [run["target_integrator"] for run in runs],
            ["uniform", "multigrid_xc", "multigrid_j_xc"],
        )

    def test_cpu_benchmark_rejects_multigrid_target(self):
        with self.assertRaisesRegex(NotImplementedError, "CPU benchmark"):
            benchmark._validate_config(
                {
                    "device": "cpu",
                    "df_backend": "gdf",
                    "target_integrator": "multigrid_xc",
                }
            )

    def test_translation_diagnostic_decomposes_component_sums(self):
        result = SimpleNamespace(
            one_electron=np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]),
            target_coulomb=np.zeros((2, 3)),
            target_xc=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            target_overlap_pulay=None,
            nuclear=np.zeros((2, 3)),
            fused_target_j_z_bilinear=None,
            contributions=SimpleNamespace(
                explicit=np.ones((2, 3)),
                response=np.full((2, 3), 2.0),
                coupled=np.full((2, 3), 3.0),
                total=np.asarray([[1.5, 2.7, 4.3], [-0.6, 0.5, 1.6]]),
            ),
        )

        sums = translation._component_translation_sums(result)

        np.testing.assert_allclose(sums["one_electron"], [0.0, 2.0, 4.0])
        np.testing.assert_allclose(sums["target_xc"], [0.5, 0.7, 0.9])
        np.testing.assert_allclose(sums["explicit_channel"], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(sums["response_channel"], [4.0, 4.0, 4.0])
        np.testing.assert_allclose(sums["coupled_channel"], [6.0, 6.0, 6.0])
        np.testing.assert_allclose(sums["total"], [0.9, 3.2, 5.9])
        self.assertIsNone(sums["target_overlap_pulay"])

    def test_translation_diagnostic_samples_eight_periodic_slopes(self):
        records = [
            {
                "phase": phase,
                "energy_ha": np.sin(2.0 * np.pi * phase),
            }
            for phase in translation.DEFAULT_PHASES
        ]

        slopes = translation._sampled_periodic_slopes(records, 1.0)

        self.assertEqual(len(slopes), 8)
        self.assertAlmostEqual(slopes[0]["phase"], 0.0)
        self.assertAlmostEqual(slopes[-1]["phase"], 0.875)
        self.assertAlmostEqual(
            max(abs(value["slope_ha_per_bohr"]) for value in slopes),
            4.0 * np.sqrt(2.0),
        )

    def test_water8_eggbox_stage12_case_matrix(self):
        cases = eggbox_stage12.build_study_cases()

        self.assertEqual(len(cases), 18)
        self.assertEqual(
            sorted({case["mesh"] for case in cases if case["xc"] == "r2SCAN"}),
            [123, 173, 211, 240, 271],
        )
        self.assertEqual(
            [case["axis"] for case in cases if case["xc"] == "PBE"],
            ["x", "y", "z"],
        )

    def test_optimization_baseline_statistics_report_spread(self):
        result = optimization_baseline._stats([10.0, 11.0, 12.0])

        self.assertEqual(result["median"], 11.0)
        self.assertEqual(result["minimum"], 10.0)
        self.assertEqual(result["maximum"], 12.0)
        self.assertAlmostEqual(result["relative_spread"], 2.0 / 11.0)

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
            gdf_derivative_omega_scale = 1.5
            rsdf_derivative_omega_scale = 1.0

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
                "gdf_derivative_omega_scale": "2.0",
                "rsdf_derivative_omega_scale": "1.2",
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
        self.assertEqual(gradient.gdf_derivative_omega_scale, 2.0)
        self.assertEqual(gradient.rsdf_derivative_omega_scale, 1.2)

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
                "config = json.loads(Path(a.input).read_text())\n"
                "Path(a.output_dir, 'report.json').write_text(\n"
                "    json.dumps({'status': 'complete', "
                "'run_identity': {'attempt_id': 'stale'}, "
                "'evidence': config['evidence']}))\n",
                encoding="utf-8",
            )

            with mock.patch.object(queue, "CASE_RUNNER", fake_runner):
                outcome = queue._run_case(
                    {
                        "device": "cpu",
                        "num_threads": 1,
                        "evidence": _case_evidence(),
                    },
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
                "'execution_status': 'complete', "
                "'run_identity': config['run_identity'], "
                "'evidence': config['evidence']}))\n",
                encoding="utf-8",
            )

            with mock.patch.object(queue, "CASE_RUNNER", fake_runner):
                outcome = queue._run_case(
                    {
                        "device": "cpu",
                        "num_threads": 1,
                        "evidence": _case_evidence(),
                    },
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
            self.assertTrue(attempt["report_fingerprint_matches"])

    def test_accuracy_failure_does_not_relabel_execution_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_dir = root / "case"
            fake_runner = root / "fake_runner.py"
            fake_runner.write_text(
                "from pathlib import Path\n"
                "import argparse, json, sys\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--input')\n"
                "p.add_argument('--output-dir')\n"
                "a = p.parse_args()\n"
                "config = json.loads(Path(a.input).read_text())\n"
                "Path(a.output_dir, 'report.json').write_text(\n"
                "    json.dumps({'status': 'complete', "
                "'execution_status': 'complete', "
                "'accuracy_status': 'fail', "
                "'run_identity': config['run_identity'], "
                "'evidence': config['evidence']}))\n"
                "sys.exit(2)\n",
                encoding="utf-8",
            )

            with mock.patch.object(queue, "CASE_RUNNER", fake_runner):
                outcome = queue._run_case(
                    {
                        "device": "cpu",
                        "num_threads": 1,
                        "evidence": _case_evidence(),
                    },
                    case_dir,
                    "000_test",
                )

            self.assertEqual(outcome["returncode"], 2)
            self.assertEqual(outcome["status"], "complete")
            self.assertEqual(outcome["report"]["accuracy_status"], "fail")

    def test_case_fingerprint_changes_with_input_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            atom_file = Path(directory) / "h2.xyz"
            atom_file.write_text(
                "2\nH2\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8"
            )
            config = {
                "atom_file": str(atom_file),
                "basis": "gth-szv",
                "lattice_vectors": [[6, 0, 0], [0, 6, 0], [0, 0, 6]],
            }
            source = {
                "git_commit": "abc",
                "source_tree_sha256": "tree-1",
            }
            first = evidence.build_case_evidence(config, source=source)
            same = evidence.build_case_evidence(config, source=source)
            self.assertEqual(
                first["fingerprint_sha256"], same["fingerprint_sha256"]
            )

            changed_config = evidence.build_case_evidence(
                {**config, "basis": "gth-dzvp"}, source=source
            )
            self.assertNotEqual(
                first["fingerprint_sha256"],
                changed_config["fingerprint_sha256"],
            )
            atom_file.write_text(
                "2\nH2\nH 0 0 0\nH 0 0 0.75\n", encoding="utf-8"
            )
            changed_geometry = evidence.build_case_evidence(
                config, source=source
            )
            self.assertNotEqual(
                first["fingerprint_sha256"],
                changed_geometry["fingerprint_sha256"],
            )
            changed_source = evidence.build_case_evidence(
                config,
                source={**source, "source_tree_sha256": "tree-2"},
            )
            self.assertNotEqual(
                changed_geometry["fingerprint_sha256"],
                changed_source["fingerprint_sha256"],
            )

    def test_completed_report_requires_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            queue._write_json(
                path,
                {
                    "status": "complete",
                    "execution_status": "complete",
                    "evidence": _case_evidence("expected"),
                },
            )
            self.assertTrue(queue._completed_report(path, "expected"))
            self.assertFalse(queue._completed_report(path, "different"))

    def test_case_lock_rejects_a_second_queue_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory)
            with queue._case_lock(case_dir) as first:
                self.assertTrue(first)
                with queue._case_lock(case_dir) as second:
                    self.assertFalse(second)

    def test_accuracy_gate_separates_execution_from_thresholds(self):
        gradient = np.asarray([[1.0, 0.0, 0.0], [-0.9, 0.0, 0.0]])
        summary = benchmark._gradient_summary(gradient)
        self.assertAlmostEqual(
            summary["translation_sum_norm_ha_per_bohr"], 0.1
        )
        status, gate = benchmark._accuracy_gate({}, None, summary)
        self.assertEqual(status, "not_requested")
        self.assertFalse(gate["requested"])

        accuracy = {"max_abs_error_ha_per_bohr": 2.0e-6}
        status, gate = benchmark._accuracy_gate(
            {
                "fd_max_abs_tolerance_ha_per_bohr": 1.0e-5,
                "net_gradient_tolerance_ha_per_bohr": 0.2,
            },
            accuracy,
            summary,
        )
        self.assertEqual(status, "pass")
        self.assertTrue(gate["passed"])

        status, gate = benchmark._accuracy_gate(
            {"net_gradient_tolerance_ha_per_bohr": 0.01},
            None,
            summary,
        )
        self.assertEqual(status, "fail")
        self.assertFalse(gate["passed"])

    def test_fd_energy_result_does_not_retain_method(self):
        self.assertEqual(
            [field.name for field in fields(benchmark.EnergyRun)],
            ["energy_ha", "setup_wall_time_s", "wall_time_s"],
        )

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
