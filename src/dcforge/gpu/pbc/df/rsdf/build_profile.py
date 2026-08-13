'''Diagnostics-only substage profiling for GPU real-Gamma RSDF builds.

The GPU4PySCF 1.8.1 RSDF builder is a synchronous Python orchestration of
CUDA short-range integrals, reciprocal Fourier contractions, metric
factorization, and mapped-host storage.  This module wraps only those Python
call boundaries while one DC-Forge build is active.  Every wrapper is
restored in ``finally`` and every measured boundary synchronizes CUDA, so the
result is an explanatory profile rather than a production wall time.
'''

import threading
from contextlib import contextmanager

import gpu4pyscf
from gpu4pyscf.pbc.df import rsdf_builder

from dcforge.gpu.pbc.df.gdf.build_profile import BuildSubstageProfiler

_PROFILE_LOCK = threading.RLock()


@contextmanager
def profile_native_gamma_build(provider):
    r'''Profile one GPU4PySCF 1.8.1 real-Gamma RSDF central build.

    Args:
        provider : dcforge.gpu.pbc.df.rsdf.RSDF
            Provider whose diagnostics option enabled this context.

    Yields:
        BuildSubstageProfiler
            Active exclusive timer and host/device memory sampler.  The
            provider receives its JSON-safe final report on context exit.

    Notes:
        ``source_short_range`` times the screened three-center evaluator.
        ``source_long_range`` times pair Fourier evaluation and its immediate
        reciprocal contraction.  The one-time auxiliary Fourier transform is
        recorded separately.  Metric SR/LR timing is supplied by the
        provider's representation-exact metric adapter, because the upstream
        full-Coulomb metric hook is deliberately replaced during RSDF build.
    '''
    if str(gpu4pyscf.__version__) != '1.8.1':
        raise RuntimeError(
            'native RSDF substage profiler requires GPU4PySCF 1.8.1')
    if not bool(provider.is_gamma_point):
        raise NotImplementedError(
            'native RSDF substage profiler currently requires real Gamma')

    profiler = BuildSubstageProfiler(
        provider.native_build_profile_sample_interval)
    originals = {}

    def replace(owner, name, wrapper):
        originals[(owner, name)] = getattr(owner, name)
        setattr(owner, name, wrapper)

    with _PROFILE_LOCK:
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

        def optimizer_build(instance, *args, **kwargs):
            return profiler.call(
                'source_optimizer', original_opt_build, instance,
                *args, **kwargs)

        replace(rsdf_builder.SRInt3c2eOpt, 'build', optimizer_build)

        original_int3c_evaluator = (
            rsdf_builder.SRInt3c2eOpt.int3c2e_evaluator)

        def int3c_evaluator(instance, *args, **kwargs):
            values = profiler.call(
                'source_evaluator_setup', original_int3c_evaluator,
                instance, *args, **kwargs)
            evaluate = values[0]

            def evaluate_profiled(*eval_args, **eval_kwargs):
                return profiler.call(
                    'source_short_range', evaluate,
                    *eval_args, **eval_kwargs)

            return (evaluate_profiled, *values[1:])

        replace(
            rsdf_builder.SRInt3c2eOpt, 'int3c2e_evaluator',
            int3c_evaluator)

        original_ft_evaluator = rsdf_builder.ft_ao.FTOpt.ft_evaluator

        def ft_evaluator(instance, *args, **kwargs):
            values = profiler.call(
                'source_fourier_setup', original_ft_evaluator,
                instance, *args, **kwargs)
            evaluate = values[0]

            def evaluate_profiled(*eval_args, **eval_kwargs):
                return profiler.call(
                    'source_long_range', evaluate,
                    *eval_args, **eval_kwargs)

            return (evaluate_profiled, *values[1:])

        replace(rsdf_builder.ft_ao.FTOpt, 'ft_evaluator', ft_evaluator)

        original_ft_ao = rsdf_builder.ft_ao.ft_ao

        def ft_ao_profiled(*args, **kwargs):
            # The exact metric adapter already encloses its reciprocal
            # Fourier transform in ``metric_long_range``.  Only an otherwise
            # unowned call is the source auxiliary Fourier cache.
            active = profiler.active_name
            if active is not None and active.startswith('metric_'):
                return original_ft_ao(*args, **kwargs)
            return profiler.call(
                'source_long_range_auxiliary_fourier', original_ft_ao,
                *args, **kwargs)

        replace(rsdf_builder.ft_ao, 'ft_ao', ft_ao_profiled)

        original_store = rsdf_builder.libvhf_rys.store_col_segment

        def store_profiled(*args, **kwargs):
            return profiler.call(
                'cderi_storage_write', original_store, *args, **kwargs)

        replace(
            rsdf_builder.libvhf_rys, 'store_col_segment', store_profiled)

        profiler.start()
        try:
            yield profiler
        finally:
            provider.native_build_substage_profile = profiler.stop()
            for (owner, name), value in reversed(list(originals.items())):
                setattr(owner, name, value)


__all__ = ['profile_native_gamma_build']
