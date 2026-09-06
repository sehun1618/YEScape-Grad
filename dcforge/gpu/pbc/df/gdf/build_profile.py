'''Opt-in exclusive substage profiling for GPU4PySCF Gamma GDF builds.'''

import threading
import time
from contextlib import contextmanager
from pathlib import Path

import cupy as cp
import gpu4pyscf
from gpu4pyscf.pbc.df import rsdf_builder
from pyscf.pbc.df import df as cpu_df

_PROFILE_LOCK = threading.RLock()


def _array_bytes(value):
    '''Return recursively visible array bytes without materializing values.'''
    if value is None:
        return 0
    nbytes = getattr(value, 'nbytes', None)
    if nbytes is not None:
        return int(nbytes)
    if isinstance(value, dict):
        return sum(_array_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_bytes(item) for item in value)
    return 0


def _host_rss_bytes():
    '''Return current Linux process RSS, or ``None`` outside procfs.'''
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


class BuildSubstageProfiler:
    '''Aggregate exclusive synchronized time and sampled memory by substage.'''

    def __init__(self, sample_interval=.02):
        self.sample_interval = float(sample_interval)
        if self.sample_interval <= 0.:
            raise ValueError('build profile sample_interval must be positive')
        self._records = {}
        self._stack = []
        self._active = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._global = {
            'host_rss_peak_bytes': 0,
            'device_used_peak_bytes': 0,
            'sample_count': 0,
        }

    def start(self):
        '''Start the opt-in memory sampler.'''
        self._thread = threading.Thread(
            target=self._sample_loop, name='gdf-build-profiler', daemon=True)
        self._thread.start()

    def stop(self):
        '''Stop sampling and return a JSON-safe report.'''
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1., 5 * self.sample_interval))
        return self.report()

    @contextmanager
    def stage(self, name, *, input_bytes=0):
        '''Measure one exclusive nested stage with CUDA completion boundaries.'''
        cp.cuda.Stream.null.synchronize()
        now = time.perf_counter()
        with self._lock:
            if self._stack:
                parent = self._stack[-1]
                self._records[parent['name']]['wall_seconds'] += (
                    now - parent['started'])
            record = self._records.setdefault(str(name), {
                'name': str(name),
                'call_count': 0,
                'wall_seconds': 0.,
                'input_bytes': 0,
                'output_bytes': 0,
                'host_rss_peak_bytes': 0,
                'device_used_peak_bytes': 0,
                'sample_count': 0,
            })
            record['call_count'] += 1
            record['input_bytes'] += int(input_bytes)
            frame = {'name': str(name), 'started': now}
            self._stack.append(frame)
            self._active = str(name)
        output = {'value': None}
        try:
            yield output
        finally:
            cp.cuda.Stream.null.synchronize()
            now = time.perf_counter()
            with self._lock:
                frame = self._stack.pop()
                record = self._records[frame['name']]
                record['wall_seconds'] += now - frame['started']
                record['output_bytes'] += _array_bytes(output['value'])
                if self._stack:
                    self._stack[-1]['started'] = now
                    self._active = self._stack[-1]['name']
                else:
                    self._active = None

    def call(self, name, function, *args, **kwargs):
        '''Call one function inside an exclusive stage.'''
        with self.stage(name, input_bytes=_array_bytes(args)) as output:
            value = function(*args, **kwargs)
            output['value'] = value
            return value

    def report(self):
        '''Return aggregated exclusive stage and memory telemetry.'''
        with self._lock:
            records = {
                name: dict(record) for name, record in self._records.items()}
            global_record = dict(self._global)
        return {
            'enabled': True,
            'exclusive_cuda_synchronized': True,
            'sample_interval_seconds': self.sample_interval,
            'stages': records,
            'stage_wall_sum_seconds': sum(
                record['wall_seconds'] for record in records.values()),
            **global_record,
        }

    @property
    def active_name(self):
        '''Return the currently active profiled stage, if any.

        The read-only value lets a provider-specific adapter distinguish a
        shared upstream primitive used by two mathematical owners.  It does
        not start, stop, or synchronize a stage.
        '''
        with self._lock:
            return self._active

    def _sample_loop(self):
        while not self._stop.wait(self.sample_interval):
            host = _host_rss_bytes()
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                device = int(total - free)
            except cp.cuda.runtime.CUDARuntimeError:
                device = None
            with self._lock:
                self._global['sample_count'] += 1
                if host is not None:
                    self._global['host_rss_peak_bytes'] = max(
                        self._global['host_rss_peak_bytes'], host)
                if device is not None:
                    self._global['device_used_peak_bytes'] = max(
                        self._global['device_used_peak_bytes'], device)
                active = self._active
                if active is not None:
                    record = self._records[active]
                    record['sample_count'] += 1
                    if host is not None:
                        record['host_rss_peak_bytes'] = max(
                            record['host_rss_peak_bytes'], host)
                    if device is not None:
                        record['device_used_peak_bytes'] = max(
                            record['device_used_peak_bytes'], device)


@contextmanager
def profile_native_gamma_build(provider):
    r'''Instrument one synchronous GPU4PySCF 1.8.1 real-Gamma GDF build.

    The adapter changes no integral, metric, batch, or storage operation.  It
    replaces selected Python call boundaries only while the inherited build
    is active and restores every object in ``finally``.  CUDA synchronization
    is intentional and therefore this route is diagnostics-only.
    '''
    if str(gpu4pyscf.__version__) != '1.8.1':
        raise RuntimeError(
            'native GDF substage profiler requires GPU4PySCF 1.8.1')
    if not bool(provider.is_gamma_point):
        raise NotImplementedError(
            'native GDF substage profiler currently requires real Gamma')

    profiler = BuildSubstageProfiler(
        provider.native_build_profile_sample_interval)
    originals = {}

    def replace(owner, name, wrapper):
        originals[(owner, name)] = getattr(owner, name)
        setattr(owner, name, wrapper)

    with _PROFILE_LOCK:
        original_make_auxcell = cpu_df.make_auxcell
        replace(
            cpu_df, 'make_auxcell',
            lambda *args, **kwargs: profiler.call(
                'auxcell_build', original_make_auxcell, *args, **kwargs))

        original_metric = rsdf_builder._get_2c2e
        replace(
            rsdf_builder, '_get_2c2e',
            lambda *args, **kwargs: profiler.call(
                'metric_2c', original_metric, *args, **kwargs))

        original_decompose = rsdf_builder.decompose_j2c
        replace(
            rsdf_builder, 'decompose_j2c',
            lambda *args, **kwargs: profiler.call(
                'metric_factorization', original_decompose,
                *args, **kwargs))

        original_precontract = rsdf_builder._precontract_j2c_aux_coeff
        replace(
            rsdf_builder, '_precontract_j2c_aux_coeff',
            lambda *args, **kwargs: profiler.call(
                'metric_application', original_precontract,
                *args, **kwargs))

        original_opt_build = rsdf_builder.SRInt3c2eOpt.build

        def opt_build(instance, *args, **kwargs):
            return profiler.call(
                'cderi_compression', original_opt_build, instance,
                *args, **kwargs)

        replace(rsdf_builder.SRInt3c2eOpt, 'build', opt_build)

        original_evaluator = rsdf_builder.SRInt3c2eOpt.int3c2e_evaluator

        def int3c2e_evaluator(instance, *args, **kwargs):
            values = profiler.call(
                'cderi_compression', original_evaluator, instance,
                *args, **kwargs)
            evaluate = values[0]

            def evaluate_profiled(*eval_args, **eval_kwargs):
                return profiler.call(
                    'three_center_sr', evaluate,
                    *eval_args, **eval_kwargs)

            return (evaluate_profiled, *values[1:])

        replace(
            rsdf_builder.SRInt3c2eOpt, 'int3c2e_evaluator',
            int3c2e_evaluator)

        original_store = rsdf_builder.libvhf_rys.store_col_segment

        def store_profiled(*args, **kwargs):
            return profiler.call(
                'cderi_storage_write', original_store, *args, **kwargs)

        replace(
            rsdf_builder.libvhf_rys, 'store_col_segment', store_profiled)

        original_compressed = rsdf_builder.compressed_cderi_j_only

        def compressed_profiled(*args, **kwargs):
            return profiler.call(
                'three_center_lr', original_compressed, *args, **kwargs)

        replace(
            rsdf_builder, 'compressed_cderi_j_only', compressed_profiled)

        profiler.start()
        try:
            yield profiler
        finally:
            provider.native_build_substage_profile = profiler.stop()
            for (owner, name), value in reversed(list(originals.items())):
                setattr(owner, name, value)


__all__ = ['BuildSubstageProfiler', 'profile_native_gamma_build']
