"""Backend-independent assembly of explicit and response gradients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def contract_one_electron(context, hcore_by_atom: np.ndarray) -> np.ndarray:
    """Contract complete one-electron derivatives with the HF density."""

    # Ordering is (atom, x, k, AO, AO).  These are complete matrix
    # derivatives, so no extra AO-row bra/ket factor is needed.
    return np.einsum(
        "axkij,kji->ax",
        hcore_by_atom,
        context.density,
        optimize=True,
    ).real / context.nkpts


def full_overlap_derivatives(one_electron_sources) -> np.ndarray:
    """Collect full Hermitian overlap derivatives in matrix-response order."""

    cell = one_electron_sources.cell
    kpts = one_electron_sources.kpts
    output = np.empty(
        (cell.natm, len(kpts), 3, cell.nao_nr(), cell.nao_nr()),
        dtype=np.complex128,
    )
    for atom_id in range(cell.natm):
        for axis in range(3):
            output[atom_id, :, axis] = one_electron_sources.overlap(
                atom_id, axis
            )
    return output


def _gradient_array(value, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{label} must have shape (natm, 3); got {array.shape}")
    if np.iscomplexobj(array):
        imag_max = float(np.max(np.abs(array.imag))) if array.size else 0.0
        if imag_max > 1.0e-9:
            raise ValueError(f"{label} imaginary residual is {imag_max:.3e}")
        array = array.real
    return np.asarray(array, dtype=float)


@dataclass(frozen=True)
class GradientContributions:
    """Two Lagrangian pieces of a nonvariational DC-DFT gradient.

    ``explicit`` is the nuclear derivative at fixed HF MO coefficients, with
    AO overlap/Pulay terms treated according to the chosen Lagrangian.
    ``response`` is the Z-vector contraction with the nuclear CPHF source.
    ``coupled`` holds an optional exactly evaluated contribution whose
    computationally fused form cannot be separated into those two channels.
    All arrays and their sum use Hartree/Bohr.
    """

    explicit: np.ndarray
    response: np.ndarray
    coupled: np.ndarray
    total: np.ndarray

    @classmethod
    def assemble(
        cls, explicit, response, coupled=None
    ) -> "GradientContributions":
        explicit_array = _gradient_array(explicit, label="explicit gradient")
        response_array = _gradient_array(response, label="response gradient")
        if explicit_array.shape != response_array.shape:
            raise ValueError("explicit and response gradient shapes differ")
        coupled_array = (
            np.zeros_like(explicit_array)
            if coupled is None
            else _gradient_array(coupled, label="coupled gradient")
        )
        if explicit_array.shape != coupled_array.shape:
            raise ValueError("explicit and coupled gradient shapes differ")
        return cls(
            explicit=explicit_array,
            response=response_array,
            coupled=coupled_array,
            total=explicit_array + response_array + coupled_array,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "units": "Ha/Bohr",
            "shape": list(self.total.shape),
            "explicit_max_abs": float(np.max(np.abs(self.explicit))),
            "response_max_abs": float(np.max(np.abs(self.response))),
            "coupled_max_abs": float(np.max(np.abs(self.coupled))),
            "total_max_abs": float(np.max(np.abs(self.total))),
        }
