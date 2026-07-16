"""Lightweight GPU-memory observations for bounded gradient kernels."""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    """One device and CuPy-pool memory observation in bytes."""

    free_device_bytes: int
    total_device_bytes: int
    pool_used_bytes: int
    pool_reserved_bytes: int

    @classmethod
    def capture(cls) -> "DeviceMemorySnapshot":
        """Sample the active CUDA device and default CuPy memory pool."""

        free_bytes, total_bytes = cp.cuda.Device().mem_info
        pool = cp.get_default_memory_pool()
        return cls(
            free_device_bytes=int(free_bytes),
            total_device_bytes=int(total_bytes),
            pool_used_bytes=int(pool.used_bytes()),
            pool_reserved_bytes=int(pool.total_bytes()),
        )


@dataclass(frozen=True)
class DeviceMemoryUsage:
    """Observed high-water diagnostics for one bounded kernel section."""

    start_free_device_bytes: int
    minimum_free_device_bytes: int
    total_device_bytes: int
    start_pool_used_bytes: int
    peak_pool_used_bytes: int
    start_pool_reserved_bytes: int
    peak_pool_reserved_bytes: int

    def metadata(self) -> dict[str, int]:
        """Return JSON-safe byte counters without claiming exact kernel peaks."""

        return {
            "start_free_device_bytes": self.start_free_device_bytes,
            "minimum_observed_free_device_bytes": self.minimum_free_device_bytes,
            "total_device_bytes": self.total_device_bytes,
            "start_pool_used_bytes": self.start_pool_used_bytes,
            "peak_observed_pool_used_bytes": self.peak_pool_used_bytes,
            "start_pool_reserved_bytes": self.start_pool_reserved_bytes,
            "peak_observed_pool_reserved_bytes": self.peak_pool_reserved_bytes,
            "observed_device_use_increase_bytes": max(
                0,
                self.start_free_device_bytes - self.minimum_free_device_bytes,
            ),
            "observed_pool_used_increase_bytes": max(
                0,
                self.peak_pool_used_bytes - self.start_pool_used_bytes,
            ),
            "observed_pool_reserved_increase_bytes": max(
                0,
                self.peak_pool_reserved_bytes
                - self.start_pool_reserved_bytes,
            ),
        }


class DeviceMemoryTracker:
    """Track sampled high-water values without resetting the global pool."""

    def __init__(self, snapshot: DeviceMemorySnapshot | None = None):
        self._start = snapshot or DeviceMemorySnapshot.capture()
        self._minimum_free = self._start.free_device_bytes
        self._peak_pool_used = self._start.pool_used_bytes
        self._peak_pool_reserved = self._start.pool_reserved_bytes

    def observe(self) -> DeviceMemorySnapshot:
        """Record and return one observation."""

        snapshot = DeviceMemorySnapshot.capture()
        self._minimum_free = min(
            self._minimum_free, snapshot.free_device_bytes
        )
        self._peak_pool_used = max(
            self._peak_pool_used, snapshot.pool_used_bytes
        )
        self._peak_pool_reserved = max(
            self._peak_pool_reserved, snapshot.pool_reserved_bytes
        )
        return snapshot

    def result(self) -> DeviceMemoryUsage:
        """Freeze the observations into JSON-reportable diagnostics."""

        return DeviceMemoryUsage(
            start_free_device_bytes=self._start.free_device_bytes,
            minimum_free_device_bytes=self._minimum_free,
            total_device_bytes=self._start.total_device_bytes,
            start_pool_used_bytes=self._start.pool_used_bytes,
            peak_pool_used_bytes=self._peak_pool_used,
            start_pool_reserved_bytes=self._start.pool_reserved_bytes,
            peak_pool_reserved_bytes=self._peak_pool_reserved,
        )
