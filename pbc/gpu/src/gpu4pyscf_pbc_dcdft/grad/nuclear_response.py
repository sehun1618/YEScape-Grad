"""GPU nuclear CPHF source, bounded batching, and Z contraction."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import cupy as cp
import numpy as np

from .memory import DeviceMemoryUsage
from .response import apply_hf_response, apply_hf_response_sets


DEFAULT_BATCH_MEMORY_FRACTION = 0.05
WORK_ARRAYS_PER_RESPONSE_COMPONENT = 12


@dataclass(frozen=True)
class NuclearResponseBatchPlan:
    """Preflight decision for native nuclear-response density batching."""

    component_count: int
    batch_size: int
    batch_count: int
    nkpts: int
    nao: int
    memory_fraction: float
    free_device_bytes: int
    total_device_bytes: int | None
    estimated_bytes_per_component: int
    estimated_batch_working_bytes: int
    max_batch_size: int | None

    def metadata(self) -> dict[str, int | float | None]:
        """Return JSON-safe preflight metadata."""

        return {
            "component_count": self.component_count,
            "batch_size": self.batch_size,
            "batch_count": self.batch_count,
            "nkpts": self.nkpts,
            "nao": self.nao,
            "memory_fraction": self.memory_fraction,
            "free_device_bytes_at_plan": self.free_device_bytes,
            "total_device_bytes": self.total_device_bytes,
            "estimated_bytes_per_component": (
                self.estimated_bytes_per_component
            ),
            "estimated_batch_working_bytes": (
                self.estimated_batch_working_bytes
            ),
            "max_batch_size": self.max_batch_size,
        }


@dataclass(frozen=True)
class NuclearResponseBatchExecution:
    """Actual bounded execution, including any OOM-triggered rebatching."""

    initial_batch_size: int
    final_batch_size: int
    minimum_batch_size: int
    completed_count: int
    rebatch_count: int
    device_memory: DeviceMemoryUsage

    def metadata(self) -> dict[str, object]:
        """Return JSON-safe completion and observed-memory diagnostics."""

        return {
            "initial_batch_size": self.initial_batch_size,
            "final_batch_size": self.final_batch_size,
            "minimum_batch_size": self.minimum_batch_size,
            "completed_count": self.completed_count,
            "rebatch_count": self.rebatch_count,
            "device_memory": self.device_memory.metadata(),
        }


@dataclass(frozen=True)
class AdjointNuclearResponseState:
    """Geometry-independent AO sources for adjoint nuclear contractions."""

    transition_density: tuple[cp.ndarray, ...]
    energy_transition_density: tuple[cp.ndarray, ...]
    response_oo: tuple[cp.ndarray, ...]


@dataclass
class AdaptiveBatchScheduler:
    """Monotone scheduler that can halve a failed batch without dropping work."""

    component_count: int
    batch_size: int
    completed_count: int = 0
    rebatch_count: int = 0
    minimum_batch_size: int | None = None

    def __post_init__(self):
        if self.component_count < 1 or self.batch_size < 1:
            raise ValueError("component_count and batch_size must be positive")
        self.batch_size = min(self.component_count, self.batch_size)
        self.minimum_batch_size = self.batch_size

    @property
    def done(self) -> bool:
        return self.completed_count == self.component_count

    def next_range(self) -> tuple[int, int]:
        """Return the next uncompleted half-open component interval."""

        if self.done:
            raise StopIteration("all response components are complete")
        start = self.completed_count
        return start, min(self.component_count, start + self.batch_size)

    def accept(self, stop: int) -> None:
        """Commit exactly the interval ending at ``stop``."""

        start, expected_stop = self.next_range()
        if int(stop) != expected_stop:
            raise ValueError(
                f"batch ending at {stop} does not match pending "
                f"interval [{start}, {expected_stop})"
            )
        self.completed_count = expected_stop

    def shrink_after_oom(self) -> int:
        """Halve future work and keep the failed interval uncommitted."""

        if self.batch_size == 1:
            start, _stop = self.next_range()
            raise RuntimeError(
                "nuclear response ran out of GPU memory at minimum batch "
                f"size 1 (component {start}/{self.component_count})"
            )
        self.batch_size = max(1, self.batch_size // 2)
        self.minimum_batch_size = min(
            int(self.minimum_batch_size), self.batch_size
        )
        self.rebatch_count += 1
        return self.batch_size


def execute_adaptive_batches(
    component_count,
    initial_batch_size,
    worker,
    accept_result,
    *,
    on_oom=None,
    on_rebatch=None,
    on_success=None,
) -> AdaptiveBatchScheduler:
    """Execute bounded ranges and retry an OOM range with half-size batches.

    ``worker(start, stop)`` must have no externally visible partial result.
    ``accept_result(start, stop, value)`` commits the successful value before
    the scheduler advances.  Consequently a failed range is retried from the
    same component and a successful range cannot be silently truncated.
    """

    scheduler = AdaptiveBatchScheduler(component_count, initial_batch_size)
    while not scheduler.done:
        start, stop = scheduler.next_range()
        try:
            value = worker(start, stop)
        except cp.cuda.memory.OutOfMemoryError as error:
            if on_oom is not None:
                on_oom(start, stop, error)
            reduced_size = scheduler.shrink_after_oom()
            if on_rebatch is not None:
                on_rebatch(scheduler, start, reduced_size)
            continue
        accept_result(start, stop, value)
        scheduler.accept(stop)
        if on_success is not None:
            on_success(scheduler, start, stop)
    return scheduler


def nuclear_response_source_vo(context, partitions, h1_kpts, s1_kpts):
    """Build ``H1_vo-S1_vo eps_occ-AX[S1]_vo/2`` on device."""

    metric_densities = []
    for partition, overlap in zip(partitions, s1_kpts, strict=True):
        occupied = partition.occupied_coeff
        overlap_oo = occupied.conj().T @ overlap @ occupied
        one_sided = occupied @ overlap_oo @ occupied.conj().T
        metric_densities.append(one_sided + one_sided.conj().T)
    ax_metric = [
        2.0 * value
        for value in apply_hf_response(context.source_mf, metric_densities)
    ]
    output = []
    for k_index, (partition, h1, overlap, ax) in enumerate(
        zip(partitions, h1_kpts, s1_kpts, ax_metric, strict=True)
    ):
        occupied = partition.occupied_coeff
        virtual = partition.virtual_coeff
        h1_vo = virtual.conj().T @ h1 @ occupied
        overlap_vo = virtual.conj().T @ overlap @ occupied
        ax_vo = virtual.conj().T @ ax @ occupied
        occupied_energy = context.mo_energy[k_index][partition.occupied_mask]
        output.append(
            h1_vo
            - overlap_vo * occupied_energy[None, :]
            - 0.5 * ax_vo
        )
    return output


def contract_z_vector(z_blocks, source_blocks, *, nkpts: int):
    """Return device scalar ``4/Nk Re sum <Z|B>``."""

    raw = cp.asarray(0.0)
    for z_block, source in zip(z_blocks, source_blocks, strict=True):
        raw += cp.vdot(z_block, source).real
    return 4.0 * raw / nkpts


def nuclear_response_batch_size(
    component_count: int,
    nkpts: int,
    nao: int,
    *,
    free_device_bytes: int | None = None,
    memory_fraction: float = DEFAULT_BATCH_MEMORY_FRACTION,
    max_batch_size: int | None = None,
) -> int:
    """Choose a bounded component batch from device memory and tensor shape.

    The working-set estimate reserves twelve complex AO matrices per
    component for overlap, metric density, J/K output, response and temporary
    contractions.  This intentionally overestimates arrays owned directly by
    this module; GPU4PySCF still manages its independent CDERI work buffers.
    """

    return plan_nuclear_response_batches(
        component_count,
        nkpts,
        nao,
        free_device_bytes=free_device_bytes,
        memory_fraction=memory_fraction,
        max_batch_size=max_batch_size,
    ).batch_size


def plan_nuclear_response_batches(
    component_count: int,
    nkpts: int,
    nao: int,
    *,
    free_device_bytes: int | None = None,
    total_device_bytes: int | None = None,
    memory_fraction: float = DEFAULT_BATCH_MEMORY_FRACTION,
    max_batch_size: int | None = None,
) -> NuclearResponseBatchPlan:
    """Build a reproducible memory-derived batching preflight plan."""

    if component_count < 1 or nkpts < 1 or nao < 1:
        raise ValueError("component_count, nkpts, and nao must be positive")
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("memory_fraction must lie in (0, 1]")
    if max_batch_size is not None and int(max_batch_size) < 1:
        raise ValueError("max_batch_size must be positive when provided")
    if free_device_bytes is None:
        free_device_bytes, detected_total_bytes = cp.cuda.Device().mem_info
        free_device_bytes = int(free_device_bytes)
        if total_device_bytes is None:
            total_device_bytes = int(detected_total_bytes)
    matrix_bytes = nkpts * nao * nao * np.dtype(np.complex128).itemsize
    working_bytes = WORK_ARRAYS_PER_RESPONSE_COMPONENT * matrix_bytes
    budget = int(free_device_bytes * memory_fraction)
    batch_size = max(
        1,
        min(component_count, budget // max(1, working_bytes)),
    )
    if max_batch_size is not None:
        batch_size = min(batch_size, int(max_batch_size))
    return NuclearResponseBatchPlan(
        component_count=int(component_count),
        batch_size=int(batch_size),
        batch_count=ceil(component_count / batch_size),
        nkpts=int(nkpts),
        nao=int(nao),
        memory_fraction=float(memory_fraction),
        free_device_bytes=int(free_device_bytes),
        total_device_bytes=(
            None if total_device_bytes is None else int(total_device_bytes)
        ),
        estimated_bytes_per_component=int(working_bytes),
        estimated_batch_working_bytes=int(batch_size * working_bytes),
        max_batch_size=(
            None if max_batch_size is None else int(max_batch_size)
        ),
    )


def nuclear_response_source_vo_sets(
    context,
    partitions,
    h1_sets,
    overlap_sets,
):
    """Build nuclear CPHF source blocks for multiple Cartesian components."""

    h1 = cp.asarray(h1_sets)
    overlap = cp.asarray(overlap_sets)
    if h1.ndim != 4 or h1.shape != overlap.shape:
        raise ValueError(
            "h1_sets and overlap_sets must share (nset,nkpts,nao,nao) shape"
        )
    if h1.shape[1] != len(partitions):
        raise ValueError("nuclear response k-point and partition counts differ")

    metric_density_sets = []
    for k_index, partition in enumerate(partitions):
        occupied = partition.occupied_coeff
        overlap_k = overlap[:, k_index]
        overlap_oo = cp.einsum(
            "pi,xpq,qj->xij",
            occupied.conj(),
            overlap_k,
            occupied,
            optimize=True,
        )
        one_sided = cp.einsum(
            "pi,xij,qj->xpq",
            occupied,
            overlap_oo,
            occupied.conj(),
            optimize=True,
        )
        metric_density_sets.append(
            one_sided + one_sided.transpose(0, 2, 1).conj()
        )

    ax_metric_sets = [
        2.0 * value
        for value in apply_hf_response_sets(
            context.source_mf, metric_density_sets
        )
    ]
    output = []
    for k_index, (partition, ax_sets) in enumerate(
        zip(partitions, ax_metric_sets, strict=True)
    ):
        occupied = partition.occupied_coeff
        virtual = partition.virtual_coeff
        h1_vo = cp.einsum(
            "pv,xpq,qi->xvi",
            virtual.conj(),
            h1[:, k_index],
            occupied,
            optimize=True,
        )
        overlap_vo = cp.einsum(
            "pv,xpq,qi->xvi",
            virtual.conj(),
            overlap[:, k_index],
            occupied,
            optimize=True,
        )
        ax_vo = cp.einsum(
            "pv,xpq,qi->xvi",
            virtual.conj(),
            ax_sets,
            occupied,
            optimize=True,
        )
        occupied_energy = cp.asarray(context.mo_energy[k_index])[
            partition.occupied_mask
        ]
        output.append(
            h1_vo
            - overlap_vo * occupied_energy[None, None, :]
            - 0.5 * ax_vo
        )
    return output


def contract_z_vector_sets(z_blocks, source_sets, *, nkpts: int):
    """Return ``4/Nk Re <Z|B_x>`` for a batch of nuclear sources."""

    if not source_sets:
        raise ValueError("source_sets cannot be empty")
    nsets = int(cp.asarray(source_sets[0]).shape[0])
    raw = cp.zeros(nsets, dtype=cp.float64)
    for z_block, source in zip(z_blocks, source_sets, strict=True):
        raw += cp.einsum(
            "vi,xvi->x",
            cp.asarray(z_block).conj(),
            cp.asarray(source),
            optimize=True,
        ).real
    return 4.0 * raw / nkpts


def build_adjoint_nuclear_response_state(context, partitions, z_blocks):
    """Apply the self-adjoint HF kernel to the Z density exactly once."""

    transition = []
    energy_transition = []
    for k_index, (partition, z_block) in enumerate(
        zip(partitions, z_blocks, strict=True)
    ):
        z = cp.asarray(z_block)
        occupied = partition.occupied_coeff
        virtual = partition.virtual_coeff
        one_sided = virtual @ z @ occupied.conj().T
        transition.append(one_sided + one_sided.conj().T)

        occupied_energy = cp.asarray(context.mo_energy[k_index])[
            partition.occupied_mask
        ]
        energy_one_sided = (
            virtual @ (z * occupied_energy[None, :]) @ occupied.conj().T
        )
        energy_transition.append(
            energy_one_sided + energy_one_sided.conj().T
        )

    response = apply_hf_response(context.source_mf, transition)
    response_oo = []
    for value, partition in zip(response, partitions, strict=True):
        occupied = partition.occupied_coeff
        response_oo.append(occupied.conj().T @ value @ occupied)
    return AdjointNuclearResponseState(
        transition_density=tuple(transition),
        energy_transition_density=tuple(energy_transition),
        response_oo=tuple(response_oo),
    )


def contract_adjoint_z_nuclear_sources(
    state,
    partitions,
    h1_sets,
    overlap_sets,
    *,
    nkpts,
):
    """Contract all CPHF source terms without per-component HF responses.

    The expensive metric-response term uses
    ``<T_Z,G[M_R]> = <G[T_Z],M_R>``.  This is the same restricted-HF
    bilinear form as the conventional source builder, with the self-adjoint
    response moved from every nuclear perturbation onto the single Z density.
    """

    h1 = cp.asarray(h1_sets)
    overlap = cp.asarray(overlap_sets)
    if h1.ndim != 4 or h1.shape != overlap.shape:
        raise ValueError(
            "h1_sets and overlap_sets must share (nset,nkpts,nao,nao) shape"
        )
    if h1.shape[1] != len(partitions):
        raise ValueError("nuclear response k-point and partition counts differ")

    values = cp.zeros(h1.shape[0], dtype=cp.float64)
    for k_index, partition in enumerate(partitions):
        h1_trace = cp.einsum(
            "pq,xqp->x",
            state.transition_density[k_index],
            h1[:, k_index],
            optimize=True,
        ).real
        energy_overlap_trace = cp.einsum(
            "pq,xqp->x",
            state.energy_transition_density[k_index],
            overlap[:, k_index],
            optimize=True,
        ).real

        occupied = partition.occupied_coeff
        overlap_oo = cp.einsum(
            "pi,xpq,qj->xij",
            occupied.conj(),
            overlap[:, k_index],
            occupied,
            optimize=True,
        )
        metric_trace = cp.einsum(
            "ij,xji->x",
            state.response_oo[k_index],
            overlap_oo,
            optimize=True,
        ).real
        values += 2.0 * (h1_trace - energy_overlap_trace)
        values -= 4.0 * metric_trace
    return values / nkpts
