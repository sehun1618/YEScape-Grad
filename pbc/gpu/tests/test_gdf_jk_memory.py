"""Bounded-memory tests for canonical GPU GDF/RSDF J/K contractions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cupy as cp
import numpy as np

from gpu4pyscf_pbc_dcdft.df.gdf import gdf as gdf_module


def _gpu_available() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


@unittest.skipUnless(_gpu_available(), "CUDA device is required")
class BoundedExchangeContractionTest(unittest.TestCase):
    def test_auxiliary_blocks_match_direct_complex_contraction(self):
        rng = np.random.default_rng(211)
        left = cp.asarray(
            rng.normal(size=(7, 4, 4))
            + 1j * rng.normal(size=(7, 4, 4))
        )
        right = cp.asarray(
            rng.normal(size=(7, 4, 4))
            + 1j * rng.normal(size=(7, 4, 4))
        )
        density = cp.asarray(
            rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        )
        expected = cp.einsum(
            "pij,jk,pkl->il",
            left,
            density,
            right,
            optimize=True,
        )

        with patch.object(
            gdf_module,
            "get_avail_mem",
            return_value=1024,
        ):
            actual, metadata = gdf_module._blocked_exchange_pair(
                left,
                density,
                right,
            )

        np.testing.assert_allclose(
            cp.asnumpy(actual),
            cp.asnumpy(expected),
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        self.assertEqual(metadata["initial_block_size"], 1)
        self.assertEqual(metadata["completed_auxiliary_count"], 7)
        self.assertEqual(metadata["block_count"], 7)

    def test_allocation_failure_retries_same_auxiliary_range(self):
        left = cp.arange(6 * 3 * 3, dtype=cp.float64).reshape(6, 3, 3)
        right = left.transpose(0, 2, 1).copy()
        density = cp.eye(3)
        original_matmul = cp.matmul
        failed = False

        def fail_first_large_block(first, second):
            nonlocal failed
            if not failed and first.ndim == 3 and first.shape[0] > 1:
                failed = True
                raise RuntimeError("cudaErrorMemoryAllocation: out of memory")
            return original_matmul(first, second)

        plan = {
            "naux": 6,
            "nao": 3,
            "dtype": "float64",
            "free_device_bytes_at_plan": 1024**3,
            "memory_fraction": 0.2,
            "reserve_bytes": 0,
            "budget_bytes": 1024**3,
            "bytes_per_auxiliary": 288,
            "estimated_block_working_bytes": 1728,
            "initial_block_size": 6,
            "maximum_block_size": 128,
        }
        expected = cp.einsum(
            "pij,jk,pkl->il", left, density, right, optimize=True
        )
        with (
            patch.object(
                gdf_module,
                "_plan_exchange_auxiliary_blocks",
                return_value=plan,
            ),
            patch.object(gdf_module.cp, "matmul", fail_first_large_block),
        ):
            actual, metadata = gdf_module._blocked_exchange_pair(
                left, density, right
            )

        np.testing.assert_allclose(cp.asnumpy(actual), cp.asnumpy(expected))
        self.assertEqual(metadata["rebatch_count"], 1)
        self.assertEqual(metadata["final_block_size"], 3)
        self.assertEqual(metadata["completed_auxiliary_count"], 6)

    def test_water64_dz_exchange_plan_is_bounded(self):
        free = int(22.5 * 1024**3)
        plan = gdf_module._plan_exchange_auxiliary_blocks(
            1472,
            1472,
            np.complex128,
            free_device_bytes=free,
        )

        self.assertLess(plan["initial_block_size"], 1472)
        self.assertLessEqual(
            plan["estimated_block_working_bytes"], plan["budget_bytes"]
        )
        self.assertGreaterEqual(plan["reserve_bytes"], 2 * 1024**3)


if __name__ == "__main__":
    unittest.main()
