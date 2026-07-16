"""Bounded-memory planning for native reciprocal-space GDF derivatives."""

from __future__ import annotations

from dataclasses import dataclass


COMPLEX128_BYTES = 16
DEFAULT_MEMORY_FRACTION = 0.65
DEFAULT_MINIMUM_RESERVE_BYTES = 512 * 1024**2
DEFAULT_BLOCK_ALIGNMENT = 32


@dataclass(frozen=True)
class GammaReciprocalBlockPlan:
    """Preflight memory decision for Gamma reciprocal-space blocks."""

    ngrids: int
    nao: int
    naux: int
    nocc_max: int
    free_device_bytes: int
    memory_fraction: float
    minimum_reserve_bytes: int
    reserve_bytes: int
    budget_bytes: int
    build_bytes_per_grid: int
    derivative_bytes_per_grid: int
    bytes_per_grid: int
    block_alignment: int
    initial_block_size: int
    estimated_block_working_bytes: int

    def metadata(self) -> dict[str, int | float]:
        """Return JSON-safe planning metadata."""

        return {
            "ngrids": self.ngrids,
            "nao": self.nao,
            "naux": self.naux,
            "nocc_max": self.nocc_max,
            "free_device_bytes_at_plan": self.free_device_bytes,
            "memory_fraction": self.memory_fraction,
            "minimum_reserve_bytes": self.minimum_reserve_bytes,
            "reserve_bytes": self.reserve_bytes,
            "budget_bytes": self.budget_bytes,
            "build_bytes_per_grid": self.build_bytes_per_grid,
            "derivative_bytes_per_grid": self.derivative_bytes_per_grid,
            "bytes_per_grid": self.bytes_per_grid,
            "block_alignment": self.block_alignment,
            "initial_block_size": self.initial_block_size,
            "estimated_block_working_bytes": (
                self.estimated_block_working_bytes
            ),
        }


@dataclass(frozen=True)
class ReciprocalBlockExecution:
    """Completed adaptive traversal for one reciprocal-space phase."""

    phase: str
    initial_block_size: int
    final_block_size: int
    minimum_block_size: int
    completed_count: int
    block_count: int
    rebatch_count: int

    def metadata(self) -> dict[str, int | str]:
        """Return JSON-safe execution metadata."""

        return {
            "phase": self.phase,
            "initial_block_size": self.initial_block_size,
            "final_block_size": self.final_block_size,
            "minimum_block_size": self.minimum_block_size,
            "completed_count": self.completed_count,
            "block_count": self.block_count,
            "rebatch_count": self.rebatch_count,
        }


@dataclass
class AdaptiveReciprocalBlockScheduler:
    """Monotone block scheduler that never commits a failed interval."""

    total_count: int
    block_size: int
    completed_count: int = 0
    block_count: int = 0
    rebatch_count: int = 0
    minimum_block_size: int | None = None

    def __post_init__(self):
        if self.total_count < 1 or self.block_size < 1:
            raise ValueError("total_count and block_size must be positive")
        self.block_size = min(self.total_count, self.block_size)
        self.minimum_block_size = self.block_size

    @property
    def done(self) -> bool:
        return self.completed_count == self.total_count

    def next_range(self) -> tuple[int, int]:
        if self.done:
            raise StopIteration("all reciprocal vectors are complete")
        start = self.completed_count
        return start, min(self.total_count, start + self.block_size)

    def accept(self, stop: int) -> None:
        start, expected_stop = self.next_range()
        if int(stop) != expected_stop:
            raise ValueError(
                f"block ending at {stop} does not match pending interval "
                f"[{start}, {expected_stop})"
            )
        self.completed_count = expected_stop
        self.block_count += 1

    def shrink_after_oom(self, *, alignment: int = 1) -> int:
        """Halve the failed block while retaining its starting vector."""

        if self.block_size == 1:
            start, _stop = self.next_range()
            raise MemoryError(
                "reciprocal GDF traversal ran out of GPU memory at minimum "
                f"block size 1 (vector {start}/{self.total_count})"
            )
        reduced = max(1, self.block_size // 2)
        if reduced >= alignment:
            reduced = max(alignment, reduced // alignment * alignment)
        self.block_size = reduced
        self.minimum_block_size = min(
            int(self.minimum_block_size), self.block_size
        )
        self.rebatch_count += 1
        return self.block_size

    def result(self, phase: str, initial_block_size: int):
        return ReciprocalBlockExecution(
            phase=str(phase),
            initial_block_size=int(initial_block_size),
            final_block_size=int(self.block_size),
            minimum_block_size=int(self.minimum_block_size),
            completed_count=int(self.completed_count),
            block_count=int(self.block_count),
            rebatch_count=int(self.rebatch_count),
        )


def plan_gamma_reciprocal_blocks(
    ngrids: int,
    nao: int,
    naux: int,
    nocc_max: int,
    *,
    free_device_bytes: int,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    minimum_reserve_bytes: int = DEFAULT_MINIMUM_RESERVE_BYTES,
    block_alignment: int = DEFAULT_BLOCK_ALIGNMENT,
    max_block_size: int | None = None,
) -> GammaReciprocalBlockPlan:
    """Plan a block from the largest explicit live tensor set.

    The zero-order build and the long-range derivative have different peak
    allocations.  Both are counted so a block that fits the first phase does
    not fail when reused by the second.  Unknown FFT/einsum workspaces are
    covered by the retained memory fraction and minimum reserve.
    """

    values = (ngrids, nao, naux, nocc_max, free_device_bytes)
    if any(int(value) < 1 for value in values):
        raise ValueError("grid, orbital, and memory dimensions must be positive")
    if not 0.0 < float(memory_fraction) <= 1.0:
        raise ValueError("memory_fraction must lie in (0, 1]")
    if int(minimum_reserve_bytes) < 0:
        raise ValueError("minimum_reserve_bytes cannot be negative")
    if int(block_alignment) < 1:
        raise ValueError("block_alignment must be positive")
    if max_block_size is not None and int(max_block_size) < 1:
        raise ValueError("max_block_size must be positive when provided")

    # Zero-order live arrays: pair_g, tmp, occupied_g, aux_g, aux_weighted.
    build_values_per_grid = (
        nao * nao
        + nocc_max * nao
        + nocc_max * nocc_max
        + 2 * naux
    )
    # Derivative live arrays after lifetime shortening: pair_g, ao_source,
    # tmp, one occupied-space buffer, and four auxiliary-grid buffers.
    derivative_values_per_grid = (
        2 * nao * nao
        + nocc_max * nao
        + nocc_max * nocc_max
        + 4 * naux
        + 2
    )
    build_bytes = int(build_values_per_grid * COMPLEX128_BYTES)
    derivative_bytes = int(
        derivative_values_per_grid * COMPLEX128_BYTES
    )
    bytes_per_grid = max(build_bytes, derivative_bytes)

    fraction_budget = int(free_device_bytes * float(memory_fraction))
    reserve_budget = max(0, int(free_device_bytes) - minimum_reserve_bytes)
    budget = min(fraction_budget, reserve_budget)
    reserve = int(free_device_bytes) - budget
    raw_block = budget // bytes_per_grid
    if max_block_size is not None:
        raw_block = min(raw_block, int(max_block_size))
    raw_block = min(raw_block, int(ngrids))
    if raw_block < 1:
        raise MemoryError(
            "insufficient GPU memory for one Gamma GDF reciprocal vector: "
            f"budget={budget} bytes, required={bytes_per_grid} bytes, "
            f"free={free_device_bytes} bytes"
        )
    block = int(raw_block)
    if block >= block_alignment:
        block = max(
            int(block_alignment),
            block // int(block_alignment) * int(block_alignment),
        )
    return GammaReciprocalBlockPlan(
        ngrids=int(ngrids),
        nao=int(nao),
        naux=int(naux),
        nocc_max=int(nocc_max),
        free_device_bytes=int(free_device_bytes),
        memory_fraction=float(memory_fraction),
        minimum_reserve_bytes=int(minimum_reserve_bytes),
        reserve_bytes=int(reserve),
        budget_bytes=int(budget),
        build_bytes_per_grid=build_bytes,
        derivative_bytes_per_grid=derivative_bytes,
        bytes_per_grid=int(bytes_per_grid),
        block_alignment=int(block_alignment),
        initial_block_size=block,
        estimated_block_working_bytes=int(block * bytes_per_grid),
    )


def execute_adaptive_reciprocal_blocks(
    total_count,
    initial_block_size,
    worker,
    accept_result,
    *,
    phase,
    is_oom,
    block_alignment=DEFAULT_BLOCK_ALIGNMENT,
    on_oom=None,
    on_rebatch=None,
    on_success=None,
) -> ReciprocalBlockExecution:
    """Run transactional blocks and retry failed allocations at half size."""

    scheduler = AdaptiveReciprocalBlockScheduler(
        int(total_count), int(initial_block_size)
    )
    while not scheduler.done:
        start, stop = scheduler.next_range()
        try:
            value = worker(start, stop)
        except Exception as error:
            if not is_oom(error):
                raise
            if on_oom is not None:
                on_oom(start, stop, error)
            try:
                reduced_size = scheduler.shrink_after_oom(
                    alignment=int(block_alignment)
                )
            except MemoryError as minimum_error:
                raise minimum_error from error
            if on_rebatch is not None:
                on_rebatch(scheduler, start, reduced_size)
            continue
        accept_result(start, stop, value)
        scheduler.accept(stop)
        if on_success is not None:
            on_success(scheduler, start, stop)
    return scheduler.result(str(phase), int(initial_block_size))
