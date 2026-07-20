"""Low-overhead phase profiler for the source-HF SCF calculation."""

from __future__ import annotations

from time import perf_counter

import cupy as cp


class SourceHFSCFProfiler:
    """Measure SCF phases with CUDA events without per-call synchronization."""

    _PHASE_METHODS = {
        "eig": "diagonalization",
        "get_occ": "occupation",
        "make_rdm1": "density_update",
    }

    def __init__(self, reference, provider):
        self.reference = reference
        self.provider = provider
        self._original_methods = {}
        self._phase_records = []

    def start(self):
        start_profile = getattr(self.provider, "start_jk_profile", None)
        if callable(start_profile):
            start_profile(label="source_hf_scf")
        for method_name, phase_name in self._PHASE_METHODS.items():
            method = getattr(self.reference, method_name, None)
            if callable(method):
                self._original_methods[method_name] = (
                    method_name in vars(self.reference),
                    method,
                )
                setattr(
                    self.reference,
                    method_name,
                    self._timed_method(method, phase_name),
                )
        return self

    def finish(self, *, scf_wall_s):
        self._restore_methods()
        finish_profile = getattr(self.provider, "finish_jk_profile", None)
        jk_profile = (
            finish_profile(scf_wall_s=scf_wall_s)
            if callable(finish_profile)
            else None
        )
        phases = self._resolve_phase_records()
        central_s = 0.0 if jk_profile is None else jk_profile["central_build_s"]
        jk_s = 0.0 if jk_profile is None else jk_profile["device_total_s"]
        diagonalization_s = phases.get("diagonalization", {}).get(
            "device_total_s", 0.0
        )
        density_update_s = phases.get("density_update", {}).get(
            "device_total_s", 0.0
        )
        result = {
            "scf_wall_s": float(scf_wall_s),
            "central_build_s": float(central_s),
            "jk": jk_profile,
            "phases": phases,
            "accounted_device_and_build_s": float(
                central_s + jk_s + diagonalization_s + density_update_s
            ),
            "unattributed_scf_wall_s": float(
                max(
                    0.0,
                    scf_wall_s
                    - central_s
                    - jk_s
                    - diagonalization_s
                    - density_update_s,
                )
            ),
            "timing_note": (
                "CUDA phase times exclude central host work; unattributed wall "
                "contains Fock assembly, DIIS, occupations, dispatch, and sync."
            ),
        }
        return result

    def abort(self):
        """Restore wrapped methods after an SCF exception."""

        self._restore_methods()
        finish_profile = getattr(self.provider, "finish_jk_profile", None)
        if callable(finish_profile):
            finish_profile()

    def _timed_method(self, method, phase_name):
        def timed(*args, **kwargs):
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            start_event.record()
            host_start = perf_counter()
            try:
                return method(*args, **kwargs)
            finally:
                host_dispatch_s = perf_counter() - host_start
                end_event.record()
                self._phase_records.append(
                    {
                        "phase": phase_name,
                        "start_event": start_event,
                        "end_event": end_event,
                        "host_dispatch_s": float(host_dispatch_s),
                    }
                )

        return timed

    def _restore_methods(self):
        for method_name, (was_instance_attribute, method) in (
            self._original_methods.items()
        ):
            if was_instance_attribute:
                setattr(self.reference, method_name, method)
            else:
                delattr(self.reference, method_name)
        self._original_methods.clear()

    def _resolve_phase_records(self):
        phases = {}
        for record in self._phase_records:
            phase = phases.setdefault(
                record["phase"],
                {
                    "call_count": 0,
                    "device_total_s": 0.0,
                    "host_dispatch_total_s": 0.0,
                },
            )
            phase["call_count"] += 1
            phase["device_total_s"] += float(
                cp.cuda.get_elapsed_time(
                    record["start_event"],
                    record["end_event"],
                )
                / 1000.0
            )
            phase["host_dispatch_total_s"] += record["host_dispatch_s"]
        return phases
