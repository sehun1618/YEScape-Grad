import unittest

import cupy as cp

from gpu4pyscf_pbc_dcdft_stress.profiling import CUDAEventProfiler


class CUDAEventProfilerTest(unittest.TestCase):
    def test_disabled_profiler_only_counts_calls(self):
        profiler = CUDAEventProfiler(False)
        value = profiler.call("add", lambda: cp.arange(4)+1)
        cp.testing.assert_array_equal(value, cp.asarray((1, 2, 3, 4)))
        result = profiler.result()
        self.assertFalse(result["enabled"])
        self.assertEqual(result["call_counts"], {"add": 1})

    def test_enabled_profiler_reports_stream_elapsed(self):
        profiler = CUDAEventProfiler(True)
        value = profiler.call("square", lambda: cp.arange(1024.0)**2)
        self.assertEqual(float(value[-1].get()), 1023.0**2)
        result = profiler.result()
        self.assertTrue(result["enabled"])
        self.assertEqual(result["call_counts"], {"square": 1})
        self.assertGreaterEqual(result["stream_elapsed_s"]["square"], 0.0)


if __name__ == "__main__":
    unittest.main()
