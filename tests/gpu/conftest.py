"""CUDA availability and cleanup for molecular GPU tests."""

import pytest


cp = pytest.importorskip("cupy")
pytest.importorskip("gpu4pyscf")

try:
    _DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    _DEVICE_COUNT = 0

if _DEVICE_COUNT < 1:
    pytest.skip("no CUDA device is available", allow_module_level=True)


@pytest.fixture(autouse=True)
def release_gpu_pool():
    """Keep independent tests from retaining CuPy pool blocks."""
    yield
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
