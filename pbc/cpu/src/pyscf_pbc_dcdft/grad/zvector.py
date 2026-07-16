"""Dense real-linear Z-vector solver for restricted PBC DC-DFT.

The Hermitian density response contains both ``X`` and ``X*``.  The CPHF map
is therefore real-linear but not generally complex-linear.  We represent each
complex VO vector by concatenating its real and imaginary parts, construct the
real response matrix ``A_R``, and solve the adjoint Lagrange-multiplier equation

``A_R.T @ z_R = -g_R``.

The direct equation is available as an explicit diagnostic because the two
forms coincide when the correctly assembled HF Hessian is symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .orbitals import MOPartition
from .response import HFOrbitalHessian


Equation = Literal["adjoint", "direct"]


@dataclass(frozen=True)
class ZVectorResult:
    """Solution and numerical diagnostics of one Z-vector equation."""

    blocks: tuple[np.ndarray, ...]
    residual_max_abs: float
    condition_number: float
    matrix_asymmetry_max_abs: float
    equation: Equation
    dimension: int


@dataclass(frozen=True)
class BlockLayout:
    """Locations required to flatten and restore k-resolved VO blocks."""

    shapes: tuple[tuple[int, ...], ...]
    offsets: tuple[int, ...]

    @classmethod
    def from_blocks(cls, blocks) -> "BlockLayout":
        shapes = tuple(np.asarray(block).shape for block in blocks)
        offsets = [0]
        for shape in shapes:
            offsets.append(offsets[-1] + int(np.prod(shape)))
        return cls(shapes=shapes, offsets=tuple(offsets))

    @property
    def complex_size(self) -> int:
        return self.offsets[-1]

    def flatten(self, blocks) -> np.ndarray:
        if len(blocks) != len(self.shapes):
            raise ValueError("block count differs from layout")
        pieces = []
        for index, (block, shape) in enumerate(zip(blocks, self.shapes, strict=True)):
            array = np.asarray(block)
            if array.shape != shape:
                raise ValueError(
                    f"block {index} has shape {array.shape}; expected {shape}"
                )
            pieces.append(array.reshape(-1))
        return np.concatenate(pieces).astype(complex, copy=False)

    def split(self, vector: np.ndarray) -> list[np.ndarray]:
        array = np.asarray(vector)
        if array.size != self.complex_size:
            raise ValueError("flat complex vector size differs from layout")
        return [
            array[self.offsets[index] : self.offsets[index + 1]].reshape(shape)
            for index, shape in enumerate(self.shapes)
        ]


def complex_to_real(vector: np.ndarray) -> np.ndarray:
    """Map ``z`` to ``[Re(z), Im(z)]`` without losing complex directions."""

    array = np.asarray(vector, dtype=complex).reshape(-1)
    return np.concatenate((array.real, array.imag))


def real_to_complex(vector: np.ndarray) -> np.ndarray:
    """Inverse of :func:`complex_to_real`."""

    array = np.asarray(vector, dtype=float).reshape(-1)
    if array.size % 2:
        raise ValueError("real-linear vector must have even length")
    half = array.size // 2
    return array[:half] + 1j * array[half:]


def build_real_linear_matrix(operator, template_blocks) -> tuple[np.ndarray, BlockLayout]:
    """Materialize a small-system real-linear response matrix.

    This dense construction is transparent and ideal for correctness gates.
    A matrix-free iterative solver can later use the same ``operator`` without
    changing any physical convention.
    """

    layout = BlockLayout.from_blocks(template_blocks)
    dimension = 2 * layout.complex_size
    matrix = np.empty((dimension, dimension), dtype=float)
    for column in range(dimension):
        trial_real = np.zeros(dimension, dtype=float)
        trial_real[column] = 1.0
        trial_blocks = layout.split(real_to_complex(trial_real))
        action_flat = layout.flatten(operator(trial_blocks))
        matrix[:, column] = complex_to_real(action_flat)
    return matrix, layout


def solve_z_vector(
    source_mf,
    partitions: list[MOPartition],
    target_fock_vo: list[np.ndarray],
    *,
    equation: Equation = "adjoint",
) -> ZVectorResult:
    """Solve the restricted PBC DC-DFT Z-vector equation.

    Parameters
    ----------
    source_mf
        Converged RHF/KRHF source calculation.
    partitions
        Occupied/virtual blocks for every sampled k-point.
    target_fock_vo
        Target DFA Fock ``(virtual, occupied)`` blocks.  These form the
        nonvariational orbital gradient ``g``.
    equation
        ``"adjoint"`` solves the Lagrange-multiplier equation. ``"direct"``
        is retained for Hessian-symmetry diagnostics.
    """

    if equation not in {"adjoint", "direct"}:
        raise ValueError("equation must be 'adjoint' or 'direct'")
    if len(partitions) != len(target_fock_vo):
        raise ValueError("partition and target-Fock block counts differ")

    hessian = HFOrbitalHessian(source_mf, partitions)
    matrix, layout = build_real_linear_matrix(hessian, target_fock_vo)
    rhs = complex_to_real(layout.flatten(target_fock_vo))
    linear_operator = matrix.T if equation == "adjoint" else matrix
    solution = np.linalg.solve(linear_operator, -rhs)
    residual = linear_operator @ solution + rhs
    blocks = tuple(layout.split(real_to_complex(solution)))
    return ZVectorResult(
        blocks=blocks,
        residual_max_abs=float(np.max(np.abs(residual))) if residual.size else 0.0,
        condition_number=float(np.linalg.cond(linear_operator)),
        matrix_asymmetry_max_abs=float(np.max(np.abs(matrix - matrix.T))),
        equation=equation,
        dimension=int(matrix.shape[0]),
    )
