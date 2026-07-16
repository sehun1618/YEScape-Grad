"""Restricted PBC DC-DFT Z-vector solvers on the active GPU.

The HF orbital Hessian is real-linear because a Hermitian transition density
contains both ``X`` and ``X*``.  Small reference problems can materialize this
real-linear matrix explicitly.  Production problems use an energy-gap
preconditioned Krylov solver and never allocate the full Hessian.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cupy as cp
import numpy as np
from cupy_backends.cuda.libs.cusolver import CUSOLVERError
from cupyx.scipy.sparse.linalg import LinearOperator, minres

from .response import HFOrbitalHessian


DEFAULT_DENSE_MAX_DIMENSION = 256
DEFAULT_KRYLOV_TOLERANCE = 1.0e-12
DEFAULT_KRYLOV_MAX_CYCLE = 80
DEFAULT_RESIDUAL_TOLERANCE = 1.0e-8
DEFAULT_REFINEMENT_STEPS = 3
INT32_MAX = int(np.iinfo(np.int32).max)


@dataclass(frozen=True)
class ZVectorResult:
    blocks: tuple[cp.ndarray, ...]
    residual_max_abs: float
    condition_number: float | None
    condition_number_status: str
    matrix_asymmetry_max_abs: float | None
    equation: str
    dimension: int
    solver: str
    solver_reason: str
    iterations: int
    solver_wall_time_s: float
    mean_iteration_wall_time_s: float | None
    dense_matrix_gib: float


@dataclass(frozen=True)
class BlockLayout:
    shapes: tuple[tuple[int, ...], ...]
    offsets: tuple[int, ...]

    @classmethod
    def from_blocks(cls, blocks):
        shapes = tuple(tuple(cp.asarray(block).shape) for block in blocks)
        offsets = [0]
        for shape in shapes:
            offsets.append(offsets[-1] + int(np.prod(shape)))
        return cls(shapes, tuple(offsets))

    @property
    def complex_size(self):
        return self.offsets[-1]

    def flatten(self, blocks):
        pieces = []
        for block, shape in zip(blocks, self.shapes, strict=True):
            array = cp.asarray(block)
            if array.shape != shape:
                raise ValueError(f"block shape {array.shape}; expected {shape}")
            pieces.append(array.reshape(-1))
        return cp.concatenate(pieces).astype(cp.complex128, copy=False)

    def split(self, vector):
        array = cp.asarray(vector)
        return [
            array[self.offsets[i] : self.offsets[i + 1]].reshape(shape)
            for i, shape in enumerate(self.shapes)
        ]


def complex_to_real(vector):
    array = cp.asarray(vector, dtype=cp.complex128).reshape(-1)
    return cp.concatenate((array.real, array.imag))


def real_to_complex(vector):
    array = cp.asarray(vector, dtype=cp.float64).reshape(-1)
    if array.size % 2:
        raise ValueError("real-linear vector size must be even")
    half = array.size // 2
    return array[:half] + 1j * array[half:]


def dense_matrix_gib(dimension: int) -> float:
    """Storage of one square float64 real-linear Hessian in GiB."""

    return int(dimension) ** 2 * 8 / 1024**3


def select_solver_route(
    requested: str,
    dimension: int,
    *,
    dense_max_dimension: int = DEFAULT_DENSE_MAX_DIMENSION,
) -> tuple[str, str]:
    """Resolve the public solver policy without inspecting a specific system."""

    normalized = str(requested).lower().replace("-", "_")
    if normalized not in {"auto", "dense", "matrix_free"}:
        raise ValueError("z_solver must be 'auto', 'dense', or 'matrix_free'")
    if dense_max_dimension < 1:
        raise ValueError("z_dense_max_dimension must be positive")
    if normalized == "dense":
        return "dense", "explicit_dense_reference_requested"
    if normalized == "matrix_free":
        return "matrix_free", "matrix_free_requested"
    if dimension <= dense_max_dimension:
        return "dense", "auto_small_reference_dimension"
    return "matrix_free", "auto_dimension_exceeds_dense_reference_limit"


def exact_condition_diagnostic_policy(
    dimension: int,
    *,
    free_device_bytes: int,
) -> tuple[bool, str]:
    """Preflight the legacy cuSOLVER exact-SVD diagnostic.

    CuPy's ``cond(..., p=None)`` copies the matrix and calls the legacy
    cuSOLVER ``gesvd`` API.  Its workspace length is represented by a signed
    32-bit integer.  We also require the matrix copy estimate to consume less
    than half of currently free device memory.
    """

    elements = int(dimension) ** 2
    if elements > INT32_MAX:
        return False, "skipped_legacy_gesvd_int32_workspace_risk"
    copy_bytes = elements * 8
    if copy_bytes * 2 > int(free_device_bytes):
        return False, "skipped_insufficient_free_memory_for_svd_copy"
    return True, "exact_svd"


def build_real_linear_matrix(
    operator,
    template_blocks,
    *,
    progress_callback=None,
):
    """Materialize the small-system Hessian with GPU J/K actions."""

    layout = BlockLayout.from_blocks(template_blocks)
    dimension = 2 * layout.complex_size
    matrix = cp.empty((dimension, dimension), dtype=cp.float64)
    update_interval = max(1, dimension // 2000)
    if progress_callback is not None:
        progress_callback(
            "zvector_matrix",
            completed=0,
            total=dimension,
            dimension=dimension,
            matrix_gib=dense_matrix_gib(dimension),
        )
    for column in range(dimension):
        trial = cp.zeros(dimension, dtype=cp.float64)
        trial[column] = 1.0
        action = operator(layout.split(real_to_complex(trial)))
        matrix[:, column] = complex_to_real(layout.flatten(action))
        completed = column + 1
        if progress_callback is not None and (
            completed == dimension or completed % update_interval == 0
        ):
            progress_callback(
                "zvector_matrix",
                completed=completed,
                total=dimension,
                dimension=dimension,
                matrix_gib=dense_matrix_gib(dimension),
            )
    return matrix, layout


def _real_hessian_action(hessian, layout, vector):
    blocks = layout.split(real_to_complex(vector))
    return complex_to_real(layout.flatten(hessian(blocks)))


def _energy_gap_diagonal(partitions, layout):
    gap_complex = layout.flatten([part.energy_gap for part in partitions])
    max_imaginary = float(cp.max(cp.abs(gap_complex.imag)).item())
    if max_imaginary > 1.0e-12:
        raise ValueError("orbital energy gaps must be real")
    gap = gap_complex.real
    real_gap = cp.concatenate((gap, gap))
    if bool(cp.any(real_gap <= 0.0).item()):
        raise ValueError("matrix-free Z-vector requires positive VO energy gaps")
    return real_gap


def _matrix_free_solution(
    hessian,
    partitions,
    layout,
    rhs,
    *,
    tolerance,
    max_cycle,
    residual_tolerance,
    progress_callback,
    refinement_steps=DEFAULT_REFINEMENT_STEPS,
):
    gap = _energy_gap_diagonal(partitions, layout)
    inverse_gap = 1.0 / gap
    iteration = 0
    total_iteration_budget = max_cycle * (refinement_steps + 1)

    operator = LinearOperator(
        (rhs.size, rhs.size),
        matvec=lambda vector: _real_hessian_action(
            hessian, layout, vector
        ),
        dtype=cp.float64,
    )
    preconditioner = LinearOperator(
        (rhs.size, rhs.size),
        matvec=lambda vector: cp.asarray(vector) * inverse_gap,
        dtype=cp.float64,
    )

    def report_iteration(_solution):
        nonlocal iteration
        iteration += 1
        if progress_callback is not None:
            progress_callback(
                "zvector_matrix_free",
                completed=iteration,
                total=total_iteration_budget,
                tolerance=tolerance,
            )

    def solve_source(source):
        solution, info = minres(
            operator,
            source,
            tol=tolerance,
            maxiter=max_cycle,
            M=preconditioner,
            callback=report_iteration,
        )
        if info != 0:
            raise RuntimeError(
                f"matrix-free MINRES did not converge in {info} iterations"
            )
        return solution

    residual_scale = max(1.0, float(cp.max(cp.abs(rhs)).item()))
    allowed = float(residual_tolerance) * residual_scale
    if progress_callback is not None:
        progress_callback(
            "zvector_matrix_free",
            completed=0,
            total=total_iteration_budget,
            tolerance=tolerance,
        )
    solution = solve_source(-rhs)
    for refinement in range(refinement_steps + 1):
        residual = _real_hessian_action(hessian, layout, solution) + rhs
        residual_max_abs = float(cp.max(cp.abs(residual)).item())
        if residual_max_abs <= allowed:
            return solution, residual_max_abs, iteration
        if refinement == refinement_steps:
            break
        if progress_callback is not None:
            progress_callback(
                "zvector_refinement",
                completed=refinement + 1,
                total=refinement_steps,
                residual_max_abs=residual_max_abs,
                residual_tolerance=allowed,
            )
        correction = solve_source(-residual)
        solution += correction

    raise RuntimeError(
        "matrix-free Z-vector residual exceeds tolerance after iterative "
        f"refinement: {residual_max_abs:.3e} > {allowed:.3e}"
    )


def _dense_diagnostics(matrix, operator):
    dimension = int(matrix.shape[0])
    free_bytes, _total_bytes = cp.cuda.Device().mem_info
    evaluate_condition, condition_status = exact_condition_diagnostic_policy(
        dimension,
        free_device_bytes=int(free_bytes),
    )
    condition_number = None
    if evaluate_condition:
        try:
            condition_number = float(cp.linalg.cond(operator).item())
        except CUSOLVERError as exc:
            condition_status = f"cusolver_unavailable:{type(exc).__name__}"

    matrix_asymmetry = None
    matrix_bytes = dimension**2 * 8
    free_bytes, _total_bytes = cp.cuda.Device().mem_info
    if matrix_bytes < int(free_bytes) // 2:
        matrix_asymmetry = float(cp.max(cp.abs(matrix - matrix.T)).item())
    return condition_number, condition_status, matrix_asymmetry


def solve_z_vector(
    source_mf,
    partitions,
    target_fock_vo,
    *,
    equation="adjoint",
    solver="auto",
    tolerance=DEFAULT_KRYLOV_TOLERANCE,
    max_cycle=DEFAULT_KRYLOV_MAX_CYCLE,
    residual_tolerance=DEFAULT_RESIDUAL_TOLERANCE,
    dense_max_dimension=DEFAULT_DENSE_MAX_DIMENSION,
    progress_callback=None,
):
    """Solve the adjoint HF response with dense-reference or matrix-free AX."""

    if equation not in {"adjoint", "direct"}:
        raise ValueError("equation must be 'adjoint' or 'direct'")
    if len(partitions) != len(target_fock_vo):
        raise ValueError("partition and target-Fock block counts differ")
    if tolerance <= 0.0 or residual_tolerance <= 0.0:
        raise ValueError("Z-vector tolerances must be positive")
    if max_cycle < 1:
        raise ValueError("z_max_cycle must be positive")

    layout = BlockLayout.from_blocks(target_fock_vo)
    dimension = 2 * layout.complex_size
    selected_solver, solver_reason = select_solver_route(
        solver,
        dimension,
        dense_max_dimension=dense_max_dimension,
    )
    if progress_callback is not None:
        progress_callback(
            "zvector_preflight",
            solver=selected_solver,
            solver_reason=solver_reason,
            dimension=dimension,
            dense_matrix_gib=dense_matrix_gib(dimension),
        )

    hessian = HFOrbitalHessian(source_mf, partitions)
    rhs = complex_to_real(layout.flatten(target_fock_vo))
    condition_number = None
    condition_status = "not_evaluated_matrix_free"
    matrix_asymmetry = None
    iterations = 0
    solver_start = perf_counter()

    if selected_solver == "dense":
        matrix, layout = build_real_linear_matrix(
            hessian,
            target_fock_vo,
            progress_callback=progress_callback,
        )
        operator = matrix.T if equation == "adjoint" else matrix
        if progress_callback is not None:
            progress_callback(
                "zvector_dense_solve",
                state="started",
                dimension=dimension,
            )
        solution = cp.linalg.solve(operator, -rhs)
        residual = operator @ solution + rhs
        residual_max_abs = float(cp.max(cp.abs(residual)).item())
        if progress_callback is not None:
            progress_callback(
                "zvector_diagnostics",
                state="started",
                dimension=dimension,
            )
        condition_number, condition_status, matrix_asymmetry = (
            _dense_diagnostics(matrix, operator)
        )
    else:
        # A restricted-HF orbital Hessian is self-adjoint in the real orbital
        # rotation coordinates. Therefore its matrix-free action represents
        # both the direct and adjoint equations; small dense parity tests guard
        # this convention for Gamma and sampled-k-point inputs.
        solution, residual_max_abs, iterations = _matrix_free_solution(
            hessian,
            partitions,
            layout,
            rhs,
            tolerance=float(tolerance),
            max_cycle=int(max_cycle),
            residual_tolerance=float(residual_tolerance),
            progress_callback=progress_callback,
        )
    solver_wall_time_s = perf_counter() - solver_start
    mean_iteration_wall_time_s = (
        solver_wall_time_s / iterations if iterations else None
    )

    residual_scale = max(1.0, float(cp.max(cp.abs(rhs)).item()))
    allowed_residual = float(residual_tolerance) * residual_scale
    if residual_max_abs > allowed_residual:
        raise RuntimeError(
            "Z-vector residual exceeds tolerance: "
            f"{residual_max_abs:.3e} > {allowed_residual:.3e}"
        )

    blocks = tuple(layout.split(real_to_complex(solution)))
    if progress_callback is not None:
        progress_callback(
            "zvector_complete",
            state="complete",
            solver=selected_solver,
            dimension=dimension,
            iterations=iterations,
            residual_max_abs=residual_max_abs,
            condition_number_status=condition_status,
        )
    return ZVectorResult(
        blocks=blocks,
        residual_max_abs=residual_max_abs,
        condition_number=condition_number,
        condition_number_status=condition_status,
        matrix_asymmetry_max_abs=matrix_asymmetry,
        equation=equation,
        dimension=dimension,
        solver=selected_solver,
        solver_reason=solver_reason,
        iterations=iterations,
        solver_wall_time_s=solver_wall_time_s,
        mean_iteration_wall_time_s=mean_iteration_wall_time_s,
        dense_matrix_gib=dense_matrix_gib(dimension),
    )
