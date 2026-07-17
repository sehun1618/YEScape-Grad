"""CPU-equivalent central RSDF factors with a GPU three-center source.

PySCF and GPU4PySCF use slightly different screened two-center integral
drivers in their RSDF builders.  Even after matching the range separator and
reciprocal meshes, that difference changes the metric by roughly ``1e-4`` in
small test cells and the resulting CDERI by roughly ``1e-5``.  The much larger
three-center source, however, agrees to machine precision when GPU4PySCF is
given PySCF's resolved source settings.

This module therefore keeps the expensive source construction on the GPU and
uses the PySCF screened two-center primitive only for the small metric.  The
long-range metric correction, Cholesky factorization, three-center integral
generation, and CDERI transformation remain on the device.  For a typical
problem the host primitive is ``O(naux**2)`` rather than
``O(naux*nao**2)``.

GPU4PySCF 1.7.4 does not expose the source and metric RSDF parameters as
independent arguments.  A narrow, process-local adapter is consequently
installed while its synchronous builder runs.  The adapter is protected by a
reentrant lock and restored in ``finally``; no patched state escapes this
function.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from threading import RLock
from time import perf_counter

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import get_avail_mem
from gpu4pyscf.pbc.df import rsdf_builder
from pyscf.pbc.df import rsdf_helper
from pyscf.pbc.lib.kpts_helper import is_zero


DEFAULT_METRIC_MEMORY_FRACTION = 0.25
DEFAULT_METRIC_MINIMUM_RESERVE_BYTES = 512 * 1024**2
DEFAULT_METRIC_MAX_GRID_BLOCK_SIZE = 4096
DEFAULT_SOURCE_BUILD_MAX_WORKING_BYTES = 512 * 1024**2
DEFAULT_SOURCE_BUILD_MINIMUM_DYNAMIC_BYTES = 64 * 1024**2

_BUILDER_ADAPTER_LOCK = RLock()


@dataclass(frozen=True)
class ExactCentralCDERI:
    """Canonical device CDERI blocks and auditable build metadata."""

    blocks: MappingProxyType
    negative_blocks: MappingProxyType | None
    metadata: dict[str, object]


@dataclass
class _BuildState:
    """Mutable counters private to one synchronous central build."""

    host_metric_seconds: float = 0.0
    device_metric_seconds: float = 0.0
    metric_calls: int = 0
    metric_plans: list[dict[str, int | float]] | None = None
    source_memory_plan: dict[str, int | float] | None = None

    def __post_init__(self):
        if self.metric_plans is None:
            self.metric_plans = []


def build_exact_cderi(cell, auxcell, kpts, parameters) -> ExactCentralCDERI:
    """Build CPU-equivalent regular-Coulomb RSDF CDERI factors.

    Parameters
    ----------
    cell, auxcell
        Primary and auxiliary periodic cells.  Three-dimensional cells are
        currently supported; low-dimensional negative-metric factors remain
        outside the package contract.
    kpts
        Gamma or a complete Monkhorst-Pack mesh in reciprocal Bohr.
    parameters
        Resolved :class:`pyscf_pbc_dcdft.df.rsdf.RSDFParameters` from a PySCF
        RSDF parameter witness.
    """

    if int(cell.dimension) != 3:
        raise NotImplementedError(
            "exact GPU-source RSDF central setup currently requires a 3D cell"
        )
    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    if len(points) < 1:
        raise ValueError("at least one k-point is required")
    kmesh = _resolved_kmesh(points, parameters)

    working_cell = cell.copy()
    working_auxcell = auxcell.copy()
    working_cell.precision = float(parameters.precision_r)
    working_auxcell.precision = float(parameters.precision_r)
    state = _BuildState()

    start = perf_counter()
    with _cpu_equivalent_builder(
        parameters,
        state,
        reference_auxcell=working_auxcell,
    ):
        gamma = len(points) == 1 and is_zero(points[0])
        raw_blocks, raw_negative = rsdf_builder.build_cderi(
            working_cell,
            working_auxcell,
            kpts=None if gamma else points,
            kmesh=kmesh,
            j_only=False,
            omega=None,
            linear_dep_threshold=float(parameters.linear_dep_threshold),
        )
    cp.cuda.Stream.null.synchronize()
    build_seconds = perf_counter() - start

    blocks = _device_pair_mapping(raw_blocks)
    negative = (
        None
        if raw_negative is None
        else _device_pair_mapping(raw_negative)
    )
    central_bytes = sum(int(value.nbytes) for value in blocks.values())
    return ExactCentralCDERI(
        blocks=blocks,
        negative_blocks=negative,
        metadata={
            "builder": "project-local exact RSDF central adapter",
            "source_owner": "GPU4PySCF CUDA three-center builder",
            "short_range_metric_owner": "PySCF screened two-center primitive",
            "long_range_metric_owner": "GPU4PySCF reciprocal-space kernels",
            "factorization_owner": "GPU4PySCF CuPy",
            "source_omega": float(parameters.source_omega),
            "source_mesh": [int(value) for value in parameters.source_mesh],
            "source_precision": float(parameters.precision_r),
            "metric_omega": float(parameters.metric_omega),
            "metric_mesh": [int(value) for value in parameters.metric_mesh],
            "metric_precision": float(parameters.precision_j2c),
            "kmesh": [int(value) for value in kmesh],
            "pair_count": len(blocks),
            "central_device_bytes": central_bytes,
            "total_build_seconds": build_seconds,
            "host_short_range_metric_seconds": state.host_metric_seconds,
            "device_long_range_metric_seconds": state.device_metric_seconds,
            "metric_calls": state.metric_calls,
            "metric_plans": list(state.metric_plans),
            "source_builder_memory": state.source_memory_plan,
        },
    )


@contextmanager
def _cpu_equivalent_builder(parameters, state, *, reference_auxcell):
    """Temporarily supply independent source and metric settings upstream."""

    with _BUILDER_ADAPTER_LOCK:
        original_optimizer = rsdf_builder.SRInt3c2eOpt
        original_guess = rsdf_builder._guess_omega
        original_metric = rsdf_builder._get_2c2e
        original_precontract = rsdf_builder._precontract_j2c_aux_coeff
        original_available_memory = rsdf_builder.get_avail_mem
        source_mesh = np.asarray(parameters.source_mesh, dtype=int)

        class SourceOptimizer(original_optimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.mesh = source_mesh.copy()

        def resolved_source_omega(_cell, kmesh=None):
            del kmesh
            return float(parameters.source_omega)

        def exact_metric(
            auxcell,
            unique_kpts,
            omega,
            metric_omega,
            bvk_kmesh=None,
        ):
            del bvk_kmesh
            return _build_metric(
                auxcell,
                unique_kpts,
                physical_omega=float(omega),
                metric_omega=float(metric_omega),
                metric_mesh=parameters.metric_mesh,
                precision=float(parameters.precision_j2c),
                reference_auxcell=reference_auxcell,
                state=state,
            )

        def precontract_with_metric_settings(
            auxcell,
            kpts,
            omega,
            _source_omega,
            linear_dep_threshold,
            kmesh=None,
        ):
            return original_precontract(
                auxcell,
                kpts,
                omega,
                float(parameters.metric_omega),
                linear_dep_threshold,
                kmesh,
            )

        def bounded_available_memory(exclude_memory_pool=True):
            actual = int(
                original_available_memory(
                    exclude_memory_pool=exclude_memory_pool
                )
            )
            reported, plan = _source_builder_memory_plan(
                actual,
                source_mesh=parameters.source_mesh,
                naux=int(reference_auxcell.nao_nr()),
            )
            state.source_memory_plan = plan
            return reported

        rsdf_builder.SRInt3c2eOpt = SourceOptimizer
        rsdf_builder._guess_omega = resolved_source_omega
        rsdf_builder._get_2c2e = exact_metric
        rsdf_builder._precontract_j2c_aux_coeff = (
            precontract_with_metric_settings
        )
        rsdf_builder.get_avail_mem = bounded_available_memory
        try:
            yield
        finally:
            rsdf_builder.SRInt3c2eOpt = original_optimizer
            rsdf_builder._guess_omega = original_guess
            rsdf_builder._get_2c2e = original_metric
            rsdf_builder._precontract_j2c_aux_coeff = original_precontract
            rsdf_builder.get_avail_mem = original_available_memory


def _build_metric(
    auxcell,
    unique_kpts,
    *,
    physical_omega,
    metric_omega,
    metric_mesh,
    precision,
    reference_auxcell,
    state,
):
    """Return the exact screened metric plus a bounded GPU LR correction."""

    working_auxcell = auxcell.copy()
    working_auxcell.precision = float(precision)
    host_auxcell = reference_auxcell.copy()
    host_auxcell.precision = float(precision)
    if unique_kpts is None:
        points = np.zeros((1, 3))
    else:
        points = np.asarray(unique_kpts, dtype=float).reshape(-1, 3)

    host_start = perf_counter()
    metric = cp.asarray(
        rsdf_helper.intor_j2c(
            host_auxcell,
            float(metric_omega),
            kpts=points,
        )
    )
    state.host_metric_seconds += perf_counter() - host_start

    mesh = (
        np.asarray(metric_mesh, dtype=int)
        if physical_omega < metric_omega
        else np.ones(3, dtype=int)
    )
    vectors = working_auxcell.get_Gv(mesh)
    coulomb = rsdf_builder._weighted_coulG_kpts(
        working_auxcell,
        mesh,
        physical_omega,
        metric_omega,
        None if unique_kpts is None else points,
    )
    block_size, plan = _metric_block_plan(
        len(vectors),
        int(working_auxcell.nao),
    )
    state.metric_calls += 1
    state.metric_plans.append(plan)

    device_start = cp.cuda.Event()
    device_end = cp.cuda.Event()
    device_start.record()
    for point_id, point in enumerate(points):
        for start in range(0, len(vectors), block_size):
            stop = min(len(vectors), start + block_size)
            auxiliary_g = rsdf_builder.ft_ao.ft_ao(
                working_auxcell,
                vectors[start:stop],
                kpt=point,
                sort_output=True,
            )
            weighted = auxiliary_g.conj()
            weighted *= coulomb[point_id, start:stop, None]
            value = weighted.T.dot(auxiliary_g)
            if is_zero(point):
                value = value.real
            metric[point_id] += value
            del auxiliary_g, weighted, value
    device_end.record()
    device_end.synchronize()
    state.device_metric_seconds += (
        cp.cuda.get_elapsed_time(device_start, device_end) / 1000.0
    )
    return metric[0] if unique_kpts is None else metric


def _metric_block_plan(ngrids, naux):
    free_bytes = int(get_avail_mem(exclude_memory_pool=True))
    fraction_budget = int(free_bytes * DEFAULT_METRIC_MEMORY_FRACTION)
    reserve_budget = max(
        0,
        free_bytes - DEFAULT_METRIC_MINIMUM_RESERVE_BYTES,
    )
    budget = min(fraction_budget, reserve_budget)
    bytes_per_grid = 2 * int(naux) * np.dtype(np.complex128).itemsize
    block_size = min(
        int(ngrids),
        DEFAULT_METRIC_MAX_GRID_BLOCK_SIZE,
        budget // max(1, bytes_per_grid),
    )
    if block_size < 1:
        raise MemoryError(
            "insufficient GPU memory for one RSDF metric reciprocal vector: "
            f"budget={budget} bytes, required={bytes_per_grid} bytes"
        )
    return int(block_size), {
        "ngrids": int(ngrids),
        "naux": int(naux),
        "free_device_bytes_at_plan": free_bytes,
        "memory_fraction": DEFAULT_METRIC_MEMORY_FRACTION,
        "minimum_reserve_bytes": DEFAULT_METRIC_MINIMUM_RESERVE_BYTES,
        "budget_bytes": budget,
        "bytes_per_grid": bytes_per_grid,
        "block_size": int(block_size),
    }


def _source_builder_memory_plan(actual_free_bytes, *, source_mesh, naux):
    """Limit upstream's eager source buffers without changing its algebra."""

    reciprocal_cache_bytes = (
        int(np.prod(np.asarray(source_mesh, dtype=int)))
        * int(naux)
        * np.dtype(np.complex128).itemsize
    )
    minimum_viable = (
        reciprocal_cache_bytes
        + DEFAULT_SOURCE_BUILD_MINIMUM_DYNAMIC_BYTES
    )
    requested = max(
        minimum_viable,
        DEFAULT_SOURCE_BUILD_MAX_WORKING_BYTES,
    )
    reported = min(int(actual_free_bytes), int(requested))
    if reported < minimum_viable:
        raise MemoryError(
            "insufficient GPU memory for bounded RSDF central source setup: "
            f"available={actual_free_bytes} bytes, "
            f"minimum={minimum_viable} bytes"
        )
    return reported, {
        "actual_free_device_bytes": int(actual_free_bytes),
        "reported_free_device_bytes": int(reported),
        "max_working_bytes": DEFAULT_SOURCE_BUILD_MAX_WORKING_BYTES,
        "minimum_dynamic_bytes": DEFAULT_SOURCE_BUILD_MINIMUM_DYNAMIC_BYTES,
        "estimated_full_auxiliary_grid_cache_bytes": reciprocal_cache_bytes,
        "source_mesh_points": int(np.prod(source_mesh)),
        "naux": int(naux),
    }


def _device_pair_mapping(raw):
    blocks = {
        (int(pair[0]), int(pair[1])): cp.asarray(value)
        for pair, value in raw.items()
        if (
            isinstance(pair, tuple)
            and len(pair) == 2
            and int(pair[0]) >= int(pair[1])
        )
    }
    if not blocks:
        raise RuntimeError("GPU4PySCF RSDF builder returned no k-point pairs")
    return MappingProxyType(blocks)


def _resolved_kmesh(points, parameters):
    configured = parameters.bvk_kmesh
    if configured is None:
        if len(points) != 1 or not is_zero(points[0]):
            raise ValueError(
                "sampled-k exact RSDF setup requires a resolved BvK k-mesh"
            )
        configured = (1, 1, 1)
    mesh = np.asarray(configured, dtype=int)
    if mesh.shape != (3,) or np.any(mesh < 1):
        raise ValueError(f"invalid RSDF BvK k-mesh: {mesh}")
    if int(np.prod(mesh)) != len(points):
        raise ValueError(
            f"k-mesh {tuple(mesh)} contains {int(np.prod(mesh))} points, "
            f"but {len(points)} were supplied"
        )
    return mesh
