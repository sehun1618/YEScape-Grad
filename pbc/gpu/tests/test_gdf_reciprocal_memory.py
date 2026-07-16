"""Unit tests for bounded native-GDF reciprocal-space scheduling."""

from __future__ import annotations

import unittest

from gpu4pyscf_pbc_dcdft.df.gdf.reciprocal_memory import (
    execute_adaptive_reciprocal_blocks,
    plan_gamma_reciprocal_blocks,
)
from gpu4pyscf_pbc_dcdft.df.gdf.signed_gamma_native import (
    _is_gpu_out_of_memory,
)


class _SyntheticOOM(RuntimeError):
    pass


class CUDARuntimeError(RuntimeError):
    """Stand-in for CuPy's CUDA runtime exception in a pure unit test."""


class GammaReciprocalMemoryPlanTest(unittest.TestCase):
    def test_plan_uses_larger_derivative_live_set(self):
        plan = plan_gamma_reciprocal_blocks(
            100,
            nao=8,
            naux=5,
            nocc_max=3,
            free_device_bytes=10 * 1024**3,
            memory_fraction=1.0,
            minimum_reserve_bytes=0,
            block_alignment=1,
        )

        expected_build = (8 * 8 + 3 * 8 + 3 * 3 + 2 * 5) * 16
        expected_derivative = (
            2 * 8 * 8 + 3 * 8 + 3 * 3 + 4 * 5 + 2
        ) * 16
        self.assertEqual(plan.build_bytes_per_grid, expected_build)
        self.assertEqual(
            plan.derivative_bytes_per_grid, expected_derivative
        )
        self.assertEqual(plan.bytes_per_grid, expected_derivative)
        self.assertEqual(plan.initial_block_size, 100)

    def test_plan_retains_fraction_and_minimum_reserve(self):
        free = 1024**3
        plan = plan_gamma_reciprocal_blocks(
            100_000,
            nao=16,
            naux=20,
            nocc_max=4,
            free_device_bytes=free,
            memory_fraction=0.75,
            minimum_reserve_bytes=512 * 1024**2,
            block_alignment=32,
        )

        self.assertEqual(plan.budget_bytes, free // 2)
        self.assertEqual(plan.reserve_bytes, free // 2)
        self.assertEqual(plan.initial_block_size % 32, 0)
        self.assertLessEqual(
            plan.estimated_block_working_bytes, plan.budget_bytes
        )

    def test_plan_rejects_memory_smaller_than_one_grid_vector(self):
        with self.assertRaisesRegex(
            MemoryError, "one Gamma GDF reciprocal vector"
        ):
            plan_gamma_reciprocal_blocks(
                10,
                nao=128,
                naux=128,
                nocc_max=32,
                free_device_bytes=1024,
                memory_fraction=1.0,
                minimum_reserve_bytes=0,
                block_alignment=1,
            )

    def test_water64_dz_dimensions_do_not_consume_all_free_memory(self):
        free = 100 * 1024**3
        plan = plan_gamma_reciprocal_blocks(
            100_000,
            nao=1472,
            naux=1472,
            nocc_max=256,
            free_device_bytes=free,
        )
        previous_block_estimate = free // (
            (1472 + 256) * 1472 * 16
        )

        self.assertLess(plan.initial_block_size, previous_block_estimate)
        self.assertGreaterEqual(plan.reserve_bytes, int(free * 0.35))
        self.assertLessEqual(
            plan.estimated_block_working_bytes, plan.budget_bytes
        )
        self.assertGreater(
            plan.derivative_bytes_per_grid,
            plan.build_bytes_per_grid,
        )


class AdaptiveReciprocalBlockTest(unittest.TestCase):
    def test_cuda_runtime_memory_allocation_error_is_retryable(self):
        self.assertTrue(
            _is_gpu_out_of_memory(
                CUDARuntimeError("cudaErrorMemoryAllocation: out of memory")
            )
        )
        self.assertFalse(
            _is_gpu_out_of_memory(CUDARuntimeError("invalid device value"))
        )

    def test_failed_range_is_retried_without_dropping_vectors(self):
        attempts = []
        accepted = []
        failed_once = False

        def worker(start, stop):
            nonlocal failed_once
            attempts.append((start, stop))
            if not failed_once:
                failed_once = True
                raise _SyntheticOOM("forced allocation failure")
            return tuple(range(start, stop))

        def accept(start, stop, value):
            self.assertEqual(value, tuple(range(start, stop)))
            accepted.extend(value)

        execution = execute_adaptive_reciprocal_blocks(
            10,
            8,
            worker,
            accept,
            phase="synthetic",
            is_oom=lambda error: isinstance(error, _SyntheticOOM),
            block_alignment=1,
        )

        self.assertEqual(
            attempts,
            [(0, 8), (0, 4), (4, 8), (8, 10)],
        )
        self.assertEqual(accepted, list(range(10)))
        self.assertEqual(execution.completed_count, 10)
        self.assertEqual(execution.block_count, 3)
        self.assertEqual(execution.rebatch_count, 1)
        self.assertEqual(execution.final_block_size, 4)

    def test_minimum_block_oom_is_explicit(self):
        with self.assertRaisesRegex(MemoryError, "minimum block size 1"):
            execute_adaptive_reciprocal_blocks(
                2,
                1,
                lambda _start, _stop: (_ for _ in ()).throw(
                    _SyntheticOOM("forced allocation failure")
                ),
                lambda _start, _stop, _value: None,
                phase="synthetic",
                is_oom=lambda error: isinstance(error, _SyntheticOOM),
                block_alignment=1,
            )


if __name__ == "__main__":
    unittest.main()
