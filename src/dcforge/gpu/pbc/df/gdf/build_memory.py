'''Adaptive GPU-memory policy for the native GDF central-factor build.'''

from contextlib import contextmanager
from threading import RLock

from gpu4pyscf.pbc.df import rsdf_builder

DEFAULT_NATIVE_BUILD_MEMORY_FRACTION = .25
DEFAULT_NATIVE_BUILD_MINIMUM_RESERVE_BYTES = 2 * 1024**3
DEFAULT_NATIVE_BUILD_MAX_WORKING_BYTES = 4 * 1024**3

_NATIVE_BUILD_MEMORY_LOCK = RLock()


def plan_native_build_memory(available_bytes, memory_fraction,
                             minimum_reserve_bytes, max_working_bytes):
    r'''Resolve the free-memory value reported to GPU4PySCF's GDF builder.

    Args:
        available_bytes : int
            Device bytes reported free by GPU4PySCF before the central GDF
            build allocates its source and Fourier work arrays.
        memory_fraction : float
            Fraction of ``available_bytes`` that the native builder may use
            as its planning budget.  It must satisfy ``0 < value <= 1``.
        minimum_reserve_bytes : int
            Number of free device bytes kept outside this build budget.
        max_working_bytes : int or None
            Hard upper bound on the planning budget.  ``None`` removes only
            this bound; the fraction and reserve constraints remain active.

    Returns:
        dict
            Resolved byte limits, the effective budget, and the constraint
            that selected it.

    Notes:
        ``effective_bytes`` is not an exact CuPy allocation or peak VRAM.
        GPU4PySCF subsequently subtracts resident metric/Fourier caches and
        divides the remainder among several native work arrays.  Bounding
        this input prevents a large-memory device from requesting one giant
        AO-pair batch while still allowing larger batches than a fixed
        512-MiB policy when VRAM is available.
    '''
    available_bytes = int(available_bytes)
    memory_fraction = float(memory_fraction)
    minimum_reserve_bytes = int(minimum_reserve_bytes)
    if available_bytes < 0:
        raise ValueError('available_bytes must be non-negative')
    if not 0. < memory_fraction <= 1.:
        raise ValueError('native_build_memory_fraction must be in (0, 1]')
    if minimum_reserve_bytes < 0:
        raise ValueError(
            'native_build_minimum_reserve_bytes must be non-negative')
    if max_working_bytes is not None:
        max_working_bytes = int(max_working_bytes)
        if max_working_bytes <= 0:
            raise ValueError(
                'native_build_max_working_bytes must be positive or None')

    limits = {
        'memory_fraction': int(available_bytes * memory_fraction),
        'minimum_reserve': max(
            0, available_bytes - minimum_reserve_bytes),
    }
    if max_working_bytes is not None:
        limits['max_working_bytes'] = max_working_bytes
    limiting_constraint = min(limits, key=limits.get)
    effective_bytes = limits[limiting_constraint]
    if effective_bytes <= 0:
        raise MemoryError(
            'native GDF build has no usable VRAM after applying '
            'native_build_memory_fraction, '
            'native_build_minimum_reserve_bytes, and '
            'native_build_max_working_bytes')

    return {
        'available_bytes': available_bytes,
        'memory_fraction': memory_fraction,
        'minimum_reserve_bytes': minimum_reserve_bytes,
        'max_working_bytes': max_working_bytes,
        'fraction_limit_bytes': limits['memory_fraction'],
        'reserve_limit_bytes': limits['minimum_reserve'],
        'effective_bytes': effective_bytes,
        'limiting_constraint': limiting_constraint,
    }


@contextmanager
def bounded_native_build_memory(provider):
    r'''Apply one provider's adaptive budget to a synchronous native build.

    GPU4PySCF 1.8.0 obtains its central GDF batch size through the module-level
    :func:`rsdf_builder.get_avail_mem` function and exposes no per-provider
    memory option.  This narrow adapter replaces that query only while the
    inherited synchronous ``GDF.build`` call is active.  A process-wide lock
    prevents concurrent providers from observing each other's policy, and
    ``finally`` always restores the upstream function.

    Args:
        provider : dcforge.gpu.pbc.df.gdf.GDF
            Provider carrying the three ``native_build_*`` policy options.

    Yields:
        dict
            Mutable build metadata retained as
            ``provider.native_build_memory_plan`` after the build.
    '''
    metadata = {
        'policy': 'adaptive_free_vram',
        'memory_fraction': float(provider.native_build_memory_fraction),
        'minimum_reserve_bytes': int(
            provider.native_build_minimum_reserve_bytes),
        'max_working_bytes': (
            None if provider.native_build_max_working_bytes is None else
            int(provider.native_build_max_working_bytes)),
        'requests': [],
        'completed': False,
    }
    provider.native_build_memory_plan = metadata

    with _NATIVE_BUILD_MEMORY_LOCK:
        original_get_avail_mem = rsdf_builder.get_avail_mem

        def get_avail_mem(*args, **kwargs):
            available_bytes = original_get_avail_mem(*args, **kwargs)
            plan = plan_native_build_memory(
                available_bytes,
                provider.native_build_memory_fraction,
                provider.native_build_minimum_reserve_bytes,
                provider.native_build_max_working_bytes)
            metadata['requests'].append(plan)
            return plan['effective_bytes']

        rsdf_builder.get_avail_mem = get_avail_mem
        try:
            yield metadata
        except BaseException:
            metadata['completed'] = False
            raise
        else:
            metadata['completed'] = True
        finally:
            rsdf_builder.get_avail_mem = original_get_avail_mem
