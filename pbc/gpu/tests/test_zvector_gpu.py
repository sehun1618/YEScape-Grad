"""Small algebra and policy gates for the GPU Z-vector solvers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import cupy as cp
import numpy as np

from gpu4pyscf_pbc_dcdft.grad import zvector


def _gpu_available() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


class _FakeHessian:
    """A = gap + coupling = 1 + 0.5 in one complex VO block."""

    def coupling(self, blocks):
        return [0.5 * cp.asarray(blocks[0])]

    def __call__(self, blocks):
        return [1.5 * cp.asarray(blocks[0])]


class ZVectorPolicyTest(unittest.TestCase):
    def test_auto_route_reserves_dense_solver_for_small_reference(self):
        self.assertEqual(zvector.select_solver_route("auto", 32)[0], "dense")
        self.assertEqual(
            zvector.select_solver_route("auto", 1024)[0], "matrix_free"
        )
        self.assertEqual(
            zvector.select_solver_route("dense", 65536)[0], "dense"
        )

    def test_large_legacy_svd_is_rejected_before_allocation(self):
        evaluate, reason = zvector.exact_condition_diagnostic_policy(
            65536,
            free_device_bytes=256 * 1024**3,
        )
        self.assertFalse(evaluate)
        self.assertIn("int32", reason)


@unittest.skipUnless(_gpu_available(), "CUDA device is required")
class MatrixFreeZVectorTest(unittest.TestCase):
    def test_matrix_free_matches_dense_reference(self):
        partition = SimpleNamespace(energy_gap=cp.ones((1, 1)))
        rhs = [cp.asarray([[3.0 + 0.25j]])]
        replacement = mock.patch.object(
            zvector,
            "HFOrbitalHessian",
            return_value=_FakeHessian(),
        )
        with replacement:
            dense = zvector.solve_z_vector(
                None,
                [partition],
                rhs,
                solver="dense",
            )
            matrix_free = zvector.solve_z_vector(
                None,
                [partition],
                rhs,
                solver="matrix_free",
                tolerance=1.0e-12,
                residual_tolerance=1.0e-10,
            )

        self.assertLess(matrix_free.residual_max_abs, 1.0e-10)
        self.assertIsNone(matrix_free.condition_number)
        self.assertEqual(
            matrix_free.condition_number_status,
            "not_evaluated_matrix_free",
        )
        self.assertGreater(matrix_free.iterations, 0)
        self.assertGreater(matrix_free.solver_wall_time_s, 0.0)
        self.assertAlmostEqual(
            matrix_free.mean_iteration_wall_time_s,
            matrix_free.solver_wall_time_s / matrix_free.iterations,
        )
        self.assertTrue(
            np.allclose(
                cp.asnumpy(matrix_free.blocks[0]),
                cp.asnumpy(dense.blocks[0]),
                atol=1.0e-11,
            )
        )


if __name__ == "__main__":
    unittest.main()
