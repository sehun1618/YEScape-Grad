import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
for path in (HERE, ROOT/"pbc"/"benchmark"):
    sys.path.insert(0, str(path))

import run_stress_queue
import show_stress_progress
import stress_benchmark
import validate_scaling_gate
from benchmark_checkpoint import CheckpointStore


class StressQueueTest(unittest.TestCase):
    @staticmethod
    def _scaling_report(backend):
        return {
            "status": "complete",
            "execution_status": "complete",
            "device": "gpu",
            "calculation_mode": "stress_only",
            "df_backend": backend,
            "input_fingerprint": "input",
            "state_fingerprint": "state",
            "system": {"natoms": 60},
            "stress": {"metadata": {
                "production": True,
                "pp_response_backend": validate_scaling_gate.EXPECTED_PP,
                "df_response_backend": validate_scaling_gate.EXPECTED_DF,
                "low_level_numerical_components": [],
                "silent_cpu_fallback": False,
            }},
            "timing_s": {"analytic_stress_wall_s": 12.5},
            "memory": {
                "host_peak_rss_bytes": 1024,
                "vram_peak_tracking": {
                    "status": "complete",
                    "peak_observed_cupy_pool_reserved_bytes": 2048,
                },
            },
        }

    def test_checkpoint_accepts_tagged_cuda_array(self):
        class TaggedCUDAArray:
            __cuda_array_interface__ = {
                "shape": (2,),
                "typestr": "<f8",
                "data": (0, False),
                "version": 3,
            }

            def get(self):
                return np.asarray((1.25, -0.5))

            def __array__(self, *args, **kwargs):
                raise TypeError("implicit host conversion is forbidden")

        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory), "fingerprint")
            store.save_array("tagged", TaggedCUDAArray())
            np.testing.assert_allclose(
                store.load_array("tagged"), (1.25, -0.5)
            )

    def test_h2_matrix_expands_all_device_backend_modes(self):
        path = HERE/"user"/"queues"/"current_h2_matrix.json"
        runs = run_stress_queue.expand_jobs(json.loads(path.read_text()))
        self.assertEqual(len(runs), 8)
        combinations = {
            (run["device"], run["df_backend"], run["calculation_mode"])
            for run in runs
        }
        self.assertEqual(len(combinations), 8)

    def test_all_user_queues_reference_existing_structures(self):
        for path in sorted((HERE/"user"/"queues").glob("*.json")):
            with self.subTest(queue=path.name):
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["benchmark_scope"], "user")
                self.assertTrue(payload["defaults"]["checkpoint_resume"])
                runs = run_stress_queue.expand_jobs(payload)
                self.assertTrue(runs)
                for run in runs:
                    run_stress_queue._resolve_atom_file(run, path.parent)
                    self.assertTrue(Path(run["atom_file"]).is_file())
                    self.assertIn(run["df_backend"], {"gdf", "rsdf"})
                    self.assertIn(
                        run["calculation_mode"],
                        {"stress_only", "gradient_and_stress"},
                    )

    def test_gpu_user_queues_select_current_exact_response_owners(self):
        queues = HERE/"user"/"queues"
        for path in sorted(queues.glob("*.json")):
            if path.name == "current_h2_matrix.json":
                continue
            with self.subTest(queue=path.name):
                defaults = json.loads(
                    path.read_text(encoding="utf-8")
                )["defaults"]
                self.assertEqual(
                    defaults["stress_pp_response_backend"],
                    "gpu4pyscf_native_contracted_analytic_nonlocal_device",
                )
                self.assertEqual(
                    defaults["stress_df_response_backend"],
                    "host_analytic_source_adjoint_gpu_ft",
                )

    def test_unsupported_routes_are_rejected(self):
        base = {
            "device": "cpu",
            "df_backend": "gdf",
            "calculation_mode": "stress_only",
            "target_integrator": "uniform",
        }
        self.assertEqual(
            stress_benchmark._validate(base),
            ("cpu", "gdf", "stress_only"),
        )
        with self.assertRaises(NotImplementedError):
            stress_benchmark._validate({**base, "df_backend": "fftdf"})
        with self.assertRaises(NotImplementedError):
            stress_benchmark._validate({
                **base, "device": "gpu", "target_integrator": "multigrid_xc"
            })
        with self.assertRaises(NotImplementedError):
            stress_benchmark._validate({**base, "dispersion": "d2c"})

    def test_scaling_gate_requires_both_exact_production_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for backend in ("gdf", "rsdf"):
                case = root/backend
                case.mkdir()
                (case/"report.json").write_text(
                    json.dumps(self._scaling_report(backend)), encoding="utf-8"
                )
            result = validate_scaling_gate.validate_result_root(root)
            self.assertTrue(result["pass"])
            self.assertEqual(result["missing_backends"], [])

            report = self._scaling_report("rsdf")
            report["stress"]["metadata"]["production"] = False
            (root/"rsdf"/"report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            result = validate_scaling_gate.validate_result_root(root)
            self.assertFalse(result["pass"])
            self.assertIn("stress is not production", result["cases"][1]["errors"])

    def test_live_child_attempt_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            payload = {
                "status": "running",
                "child_pid": os.getpid(),
                "run_identity": {"hostname": socket.gethostname()},
            }
            (case/"attempt.json").write_text(json.dumps(payload))
            self.assertTrue(run_stress_queue._attempt_is_active(case))
            payload["child_pid"] = 999999999
            (case/"attempt.json").write_text(json.dumps(payload))
            self.assertFalse(run_stress_queue._attempt_is_active(case))

    def test_fd_energy_evaluator_reuses_geometry_checkpoint(self):
        class FakeCell:
            mesh = np.asarray([9, 9, 9])

            @staticmethod
            def lattice_vectors():
                return np.eye(3)*8.0

            @staticmethod
            def atom_coords():
                return np.asarray([[0.0, 0.0, 0.0]])

        method = SimpleNamespace(
            converged=True,
            kernel=mock.Mock(return_value=-1.25),
            make_rdm1=mock.Mock(return_value=np.eye(2)),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = CheckpointStore(
                Path(directory)/"checkpoints", "stress-fingerprint"
            )
            evaluator = stress_benchmark.BenchmarkEnergyEvaluator(
                {}, "cpu", "gdf", checkpoint=checkpoint, expected_calls=2
            )
            with mock.patch.object(
                stress_benchmark.gradient_benchmark,
                "build_method",
                return_value=(method, None),
            ) as build_method:
                first = evaluator(FakeCell())
                second = evaluator(FakeCell())

            self.assertEqual(build_method.call_count, 1)
            self.assertAlmostEqual(first.energy_ha, second.energy_ha)
            self.assertFalse(first.metadata["checkpoint_reused"])
            self.assertTrue(second.metadata["checkpoint_reused"])
            self.assertEqual(
                evaluator.checkpoint_summary(),
                {
                    "evaluation_count": 2,
                    "reused_evaluations": 1,
                    "computed_evaluations": 1,
                },
            )

    def test_attempt_archive_preserves_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            checkpoint = case/"checkpoints"/"marker.txt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("keep", encoding="utf-8")
            (case/"report.json").write_text("{}", encoding="utf-8")
            run_stress_queue._archive(case)
            self.assertTrue(checkpoint.is_file())
            self.assertFalse((case/"report.json").exists())

    def test_progress_view_reports_checkpoint_work(self):
        payload = {
            "stage": "stress_finite_difference",
            "details": {
                "checkpoint_reused_evaluations": 3,
                "computed_evaluations": 1,
            },
        }
        self.assertEqual(
            show_stress_progress._work_summary(payload),
            "reuse=3 compute=1",
        )
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            progress = case/"progress.json"
            progress.write_text("{}", encoding="utf-8")
            self.assertEqual(
                show_stress_progress._progress_paths(case), [progress]
            )


if __name__ == "__main__":
    unittest.main()
