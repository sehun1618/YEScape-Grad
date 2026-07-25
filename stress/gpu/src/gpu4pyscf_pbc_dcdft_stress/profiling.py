"""Opt-in CUDA event timing for stress kernels."""

from __future__ import annotations

from collections import defaultdict

import cupy as cp


class CUDAEventProfiler:
    """Accumulate stream elapsed time without changing the default route.

    CUDA events measure elapsed time between two points on the current stream.
    The value can include stream idle time caused by host preparation between
    the two records; it is therefore reported as ``stream_elapsed`` rather than
    as pure kernel time.  Python wall timings remain the authoritative
    end-to-end measurement.
    """

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self._events = defaultdict(list)
        self.call_counts = defaultdict(int)

    def call(self, label, function):
        """Evaluate ``function`` and record its current-stream interval."""

        self.call_counts[str(label)] += 1
        if not self.enabled:
            return function()
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        value = function()
        end.record()
        self._events[str(label)].append((start, end))
        return value

    def result(self):
        """Synchronize recorded events and return JSON-ready diagnostics."""

        if not self.enabled:
            return {
                "enabled": False,
                "measurement": "disabled",
                "stream_elapsed_s": {},
                "call_counts": dict(self.call_counts),
            }
        cp.cuda.Stream.null.synchronize()
        elapsed = {
            label: float(sum(
                cp.cuda.get_elapsed_time(start, end)
                for start, end in pairs
            )/1000.0)
            for label, pairs in self._events.items()
        }
        return {
            "enabled": True,
            "measurement": "cuda_current_stream_event_elapsed",
            "limitations": [
                "stream elapsed is not a hardware-kernel activity trace",
                "host delay between event records can appear as stream idle time",
                "Python wall time remains authoritative for end-to-end timing",
            ],
            "stream_elapsed_s": elapsed,
            "stream_elapsed_sum_s": float(sum(elapsed.values())),
            "call_counts": dict(self.call_counts),
        }


__all__ = ["CUDAEventProfiler"]
