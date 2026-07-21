"""Device contractions for full periodic GDF J/K nuclear derivatives.

Canonical CDERI and ``dCDERI`` blocks are CuPy arrays.  This module applies
the full product rule to obtain AO matrices ``dJ[D]/dR`` and ``dK[D]/dR``.
The closed-shell density already contains occupation factor two; each matrix
build contains the explicit periodic ``1/Nk`` normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cupy as cp
import numpy as np
from pyscf.pbc.tools import pbc as pbc_tools
from pyscf_pbc_dcdft.df.gdf.storage import Pair


DEFAULT_AUXILIARY_DENSITY_MEMORY_FRACTION = 0.10
DEFAULT_AUXILIARY_DENSITY_RESERVE_BYTES = 2 * 1024**3
DEFAULT_AUXILIARY_DENSITY_MAX_BLOCK = 32
DEFAULT_AUXILIARY_DENSITY_IMAGINARY_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class GDFJKDerivatives:
    """Device J/K derivatives with ``(atom,k,xyz,AO,AO)`` ordering."""

    coulomb: cp.ndarray
    exchange: cp.ndarray

    @property
    def restricted_hf(self) -> cp.ndarray:
        """Return ``d(J-K/2)/dR`` for a restricted spatial density."""

        return self.coulomb - 0.5 * self.exchange


@dataclass(frozen=True)
class ContractedPairDerivative:
    """Scalar contractions produced by one canonical ``dCDERI`` block."""

    target_coulomb: cp.ndarray
    restricted_hf_bilinear: cp.ndarray


def gdf_jk_derivative(
    cderi: Mapping[Pair, cp.ndarray],
    dcderi: Mapping[Pair, cp.ndarray],
    density,
) -> GDFJKDerivatives:
    """Contract device ``C``/``dC`` tensors into full fixed-density J/K."""

    dm = _density_array(density)
    natm, ncart, _naux, nao, nao2 = _derivative_shape(dcderi)
    if (nao, nao2) != tuple(dm.shape[1:]):
        raise ValueError(
            f"dCDERI AO shape {(nao, nao2)} does not match {dm.shape[1:]}"
        )
    return GDFJKDerivatives(
        coulomb=_coulomb_derivative(cderi, dcderi, dm, natm, ncart),
        exchange=_exchange_derivative(cderi, dcderi, dm, natm, ncart),
    )


def contract_coulomb_energy_derivative(density, derivative):
    """Return ``0.5/Nk sum_k Tr[D_k dJ_k]`` on the device."""

    dm = _density_array(density)
    matrix_derivative = cp.asarray(derivative)
    _validate_matrix_derivative_shape(matrix_derivative, dm)
    value = 0.5 * cp.einsum(
        "akxij,kji->ax", matrix_derivative, dm, optimize=True
    )
    return value.real / len(dm)


def contract_restricted_hf_energy_derivative(density, derivative):
    """Contract a full ``d(J-K/2)[D]/dR`` matrix derivative.

    The restricted fixed-density two-electron energy is

    ``Q(D) = 1/2 Nk sum_k Tr[D_k (J[D]-K[D]/2)_k]``.

    ``derivative`` must therefore have been built from the same ``density``.
    This full-matrix helper is a reference contraction for validating the
    source-contracted native GDF kernels; production native mode does not
    materialize the five-index ``dCDERI`` tensors consumed here.
    """

    dm = _density_array(density)
    matrix_derivative = _restricted_hf_matrix_derivative(derivative)
    _validate_matrix_derivative_shape(matrix_derivative, dm)
    value = 0.5 * cp.einsum(
        "akxij,kji->ax", matrix_derivative, dm, optimize=True
    )
    return value.real / len(dm)


def contract_restricted_hf_bilinear_energy_derivative(
    reference_density,
    response_density,
    reference_derivative,
    response_derivative,
):
    """Return the exact cross-density RHF GDF derivative from full matrices.

    Let ``L_R[D] = d(J[D]-K[D]/2)/dR``.  Because ``L_R`` is linear in the
    density, the polarization difference obeys

    ``Q(D+T)-Q(D-T) = 1/Nk sum_k Re[Tr(D L_R[T]) + Tr(T L_R[D])]``.

    This form makes the two density roles explicit.  It is the equation-level
    witness for a future signed two-density native CUDA kernel and avoids
    relying on cancellation between two quadratic reference evaluations.
    """

    reference = _density_array(reference_density)
    response = _density_array(response_density)
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    derivative_at_reference = _restricted_hf_matrix_derivative(
        reference_derivative
    )
    derivative_at_response = _restricted_hf_matrix_derivative(
        response_derivative
    )
    _validate_matrix_derivative_shape(derivative_at_reference, reference)
    _validate_matrix_derivative_shape(derivative_at_response, response)
    value = cp.einsum(
        "kji,akxij->ax",
        reference,
        derivative_at_response,
        optimize=True,
    )
    value += cp.einsum(
        "kji,akxij->ax",
        response,
        derivative_at_reference,
        optimize=True,
    )
    return value.real / len(reference)


def auxiliary_density_contraction(cderi, density, **kwargs):
    """Return ``sum_k Tr[C_kk D_k]`` for a periodic CDERI mapping.

    The auxiliary vector is shared by every nuclear component.  Building it
    once lets a source-contracted derivative consume one canonical k-pair at
    a time without retaining ``dCDERI`` blocks from earlier pairs.
    """

    return auxiliary_density_contractions(cderi, (density,), **kwargs)[0]


def auxiliary_density_contractions(
    cderi,
    densities,
    *,
    memory_fraction=DEFAULT_AUXILIARY_DENSITY_MEMORY_FRACTION,
    minimum_reserve_bytes=DEFAULT_AUXILIARY_DENSITY_RESERVE_BYTES,
    max_auxiliary_block=DEFAULT_AUXILIARY_DENSITY_MAX_BLOCK,
    memory_diagnostics=None,
):
    """Contract several densities through one bounded CDERI traversal.

    GPU4PySCF may expose canonical CDERI with auxiliary-fast or otherwise
    strided storage.  A whole-array ``einsum`` can then request a contiguous
    copy as large as the complete CDERI.  This route copies at most one
    planned auxiliary tile and reuses it for every supplied density.
    """

    density_arrays = tuple(_density_array(value) for value in densities)
    if not density_arrays:
        raise ValueError("at least one density is required")
    shape = density_arrays[0].shape
    if any(value.shape != shape for value in density_arrays[1:]):
        raise ValueError("all densities must have the same shape")

    first = pair_block(cderi, 0, 0)
    naux, nao, nao2 = first.shape
    if (nao, nao2) != tuple(shape[1:]):
        raise ValueError(
            f"CDERI AO shape {(nao, nao2)} does not match {shape[1:]}"
        )
    result = cp.zeros((len(density_arrays), naux), dtype=cp.complex128)
    plan = plan_auxiliary_density_blocks(
        naux,
        nao,
        first.dtype,
        memory_fraction=memory_fraction,
        minimum_reserve_bytes=minimum_reserve_bytes,
        maximum_block_size=max_auxiliary_block,
    )
    execution = _execute_auxiliary_density_blocks(
        result,
        cderi,
        density_arrays,
        initial_block_size=plan["initial_block_size"],
    )
    if memory_diagnostics is not None:
        memory_diagnostics.clear()
        memory_diagnostics.update(
            {
                "density_count": len(density_arrays),
                "plan": plan,
                "execution": execution,
            }
        )
    return tuple(result[index] for index in range(len(result)))


def plan_auxiliary_density_blocks(
    naux,
    nao,
    dtype,
    *,
    memory_fraction=DEFAULT_AUXILIARY_DENSITY_MEMORY_FRACTION,
    minimum_reserve_bytes=DEFAULT_AUXILIARY_DENSITY_RESERVE_BYTES,
    maximum_block_size=DEFAULT_AUXILIARY_DENSITY_MAX_BLOCK,
    free_device_bytes=None,
):
    """Plan the largest bounded auxiliary tile allowed by current VRAM."""

    naux = int(naux)
    nao = int(nao)
    fraction = float(memory_fraction)
    if naux < 1 or nao < 1:
        raise ValueError("naux and nao must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    if free_device_bytes is None:
        free_device_bytes = int(cp.cuda.runtime.memGetInfo()[0])
    reserve = min(int(minimum_reserve_bytes), int(free_device_bytes))
    budget = max(1, int((int(free_device_bytes) - reserve) * fraction))
    bytes_per_auxiliary = nao**2 * cp.dtype(dtype).itemsize
    # A contiguous CDERI tile plus contraction-engine workspace.  The output
    # vectors are only O(naux) and do not affect this bound.
    working_bytes_per_auxiliary = 2 * bytes_per_auxiliary
    block_size = max(1, budget // working_bytes_per_auxiliary)
    block_size = min(naux, max(1, int(maximum_block_size)), int(block_size))
    return {
        "naux": naux,
        "nao": nao,
        "dtype": cp.dtype(dtype).str,
        "free_device_bytes_at_plan": int(free_device_bytes),
        "memory_fraction": fraction,
        "reserve_bytes": reserve,
        "budget_bytes": budget,
        "bytes_per_auxiliary": int(bytes_per_auxiliary),
        "working_bytes_per_auxiliary": int(working_bytes_per_auxiliary),
        "initial_block_size": int(block_size),
        "maximum_block_size": max(1, int(maximum_block_size)),
    }


def _execute_auxiliary_density_blocks(
    output,
    cderi,
    densities,
    *,
    initial_block_size,
):
    """Contract complete auxiliary tiles and retry OOMs before committing."""

    naux = int(output.shape[1])
    block_size = max(1, min(naux, int(initial_block_size)))
    minimum_block_size = block_size
    block_count = 0
    rebatch_count = 0
    for kpoint_id in range(densities[0].shape[0]):
        factor = pair_block(cderi, kpoint_id, kpoint_id)
        vectors = tuple(
            _auxiliary_density_vector(density[kpoint_id])
            for density in densities
        )
        start = 0
        while start < naux:
            stop = min(naux, start + block_size)
            block = None
            values = None
            try:
                block = cp.ascontiguousarray(factor[start:stop]).reshape(
                    stop - start,
                    -1,
                )
                values = cp.stack(
                    [_plain_matrix_vector(block, vector) for vector in vectors]
                )
            except Exception as error:
                if not _is_gpu_out_of_memory(error) or stop - start <= 1:
                    raise
                del block, values
                block_size = max(1, (stop - start) // 2)
                minimum_block_size = min(minimum_block_size, block_size)
                rebatch_count += 1
                _release_unused_gpu_memory()
                continue
            output[:, start:stop] += values
            del block, values
            minimum_block_size = min(minimum_block_size, stop - start)
            block_count += 1
            start = stop
    return {
        "completed_auxiliary_count": naux,
        "kpoint_count": int(densities[0].shape[0]),
        "initial_block_size": int(initial_block_size),
        "final_block_size": int(block_size),
        "minimum_block_size": int(minimum_block_size),
        "block_count": int(block_count),
        "rebatch_count": int(rebatch_count),
    }


def _auxiliary_density_vector(density):
    """Flatten one transposed density, discarding only Gamma roundoff."""

    value = cp.asarray(density)
    if value.dtype.kind == "c" and value.size:
        imaginary = float(cp.max(cp.abs(value.imag)).get())
        if imaginary <= DEFAULT_AUXILIARY_DENSITY_IMAGINARY_TOLERANCE:
            value = value.real
    return cp.ascontiguousarray(value.T).reshape(-1)


def _plain_matrix_vector(matrix, vector):
    """Multiply mixed real/complex operands without promoting the matrix."""

    matrix = cp.asarray(matrix)
    vector = cp.asarray(vector)
    if matrix.dtype.kind != "c" and vector.dtype.kind != "c":
        return matrix @ vector
    real = matrix.real @ vector.real
    if matrix.dtype.kind == "c" and vector.dtype.kind == "c":
        real -= matrix.imag @ vector.imag
        imaginary = matrix.real @ vector.imag
        imaginary += matrix.imag @ vector.real
    elif matrix.dtype.kind == "c":
        imaginary = matrix.imag @ vector
    else:
        imaginary = matrix @ vector.imag
    return real + 1j * imaginary


def _is_gpu_out_of_memory(error):
    if isinstance(error, cp.cuda.memory.OutOfMemoryError):
        return True
    message = str(error).lower()
    return "out of memory" in message or "memoryallocation" in message


def _release_unused_gpu_memory():
    cp.get_default_memory_pool().free_all_blocks()
    pinned = cp.get_default_pinned_memory_pool()
    if hasattr(pinned, "free_all_blocks"):
        pinned.free_all_blocks()


def contract_canonical_pair_derivative(
    cderi,
    pair,
    cderi_derivative,
    reference_density,
    response_density,
    *,
    auxiliary_reference=None,
    auxiliary_response=None,
) -> ContractedPairDerivative:
    """Immediately contract one canonical ``dC`` with target and Z sources.

    ``pair`` owns both ordered orientations.  For a non-diagonal pair the
    reverse block is the conjugate AO transpose, exactly as in
    :func:`derivative_pair_block`.  The routine returns only two real scalars:
    the target Coulomb derivative and the restricted-HF cross-density
    derivative.  It never constructs AO ``dJ``/``dK`` matrices.
    """

    reference = _density_array(reference_density)
    response = _density_array(response_density)
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    nkpts = len(reference)
    canonical = (int(pair[0]), int(pair[1]))
    derivative = cp.asarray(cderi_derivative)
    expected = pair_block(cderi, *canonical).shape
    if derivative.shape != expected:
        raise ValueError(
            f"dCDERI block for pair {canonical} must have shape {expected}, "
            f"got {derivative.shape}"
        )
    if auxiliary_reference is None:
        auxiliary_reference = auxiliary_density_contraction(
            cderi, reference
        )
    if auxiliary_response is None:
        auxiliary_response = auxiliary_density_contraction(cderi, response)
    auxiliary_reference = cp.asarray(auxiliary_reference)
    auxiliary_response = cp.asarray(auxiliary_response)

    target_coulomb = cp.asarray(0.0, dtype=cp.float64)
    bilinear = cp.asarray(0.0, dtype=cp.float64)
    first, second = canonical
    if first == second:
        derivative_reference = cp.einsum(
            "pij,ji->p", derivative, reference[first], optimize=True
        )
        derivative_response = cp.einsum(
            "pij,ji->p", derivative, response[first], optimize=True
        )
        target_coulomb = cp.real(
            cp.einsum(
                "p,p->", auxiliary_reference, derivative_reference
            )
        ) / nkpts**2
        bilinear = 2.0 * cp.real(
            cp.einsum(
                "p,p->", auxiliary_reference, derivative_response
            )
            + cp.einsum(
                "p,p->", auxiliary_response, derivative_reference
            )
        ) / nkpts**2

    exchange = cp.asarray(0.0, dtype=cp.complex128)
    ordered_pairs = (
        (canonical,)
        if first == second
        else (canonical, (second, first))
    )
    for ki, kj in ordered_pairs:
        factor_ij = pair_block(cderi, ki, kj)
        factor_ji = pair_block(cderi, kj, ki)
        derivative_ij = _oriented_pair_derivative(
            derivative, canonical, ki, kj
        )
        derivative_ji = _oriented_pair_derivative(
            derivative, canonical, kj, ki
        )
        exchange += cp.einsum(
            "li,pij,jk,pkl->",
            reference[ki],
            derivative_ij,
            response[kj],
            factor_ji,
            optimize=True,
        )
        exchange += cp.einsum(
            "li,pij,jk,pkl->",
            response[ki],
            derivative_ij,
            reference[kj],
            factor_ji,
            optimize=True,
        )
        exchange += cp.einsum(
            "li,pij,jk,pkl->",
            reference[ki],
            factor_ij,
            response[kj],
            derivative_ji,
            optimize=True,
        )
        exchange += cp.einsum(
            "li,pij,jk,pkl->",
            response[ki],
            factor_ij,
            reference[kj],
            derivative_ji,
            optimize=True,
        )
    bilinear -= 0.5 * exchange.real / nkpts**2
    return ContractedPairDerivative(
        target_coulomb=target_coulomb,
        restricted_hf_bilinear=bilinear,
    )


def _oriented_pair_derivative(derivative, canonical, first, second):
    """Orient one canonical derivative for an ordered k-point pair."""

    if (int(first), int(second)) == canonical:
        return derivative
    if (int(second), int(first)) == canonical:
        return derivative.conj().transpose(0, 2, 1)
    raise ValueError(
        f"ordered pair {(first, second)} is unrelated to {canonical}"
    )


def ewald_exchange_derivative(
    cell,
    kpts,
    density,
    overlap_derivative,
) -> cp.ndarray:
    """Differentiate the GDF ``exxdiv='ewald'`` overlap correction.

    PySCF geometry and the Madelung scalar are host metadata.  The two matrix
    products for each atom, k-point, and Cartesian direction are CuPy kernels.
    """

    dm = _density_array(density)
    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    derivative = cp.asarray(overlap_derivative)
    expected = (
        derivative.shape[0], len(points), 3, dm.shape[1], dm.shape[2]
    )
    if derivative.shape != expected:
        raise ValueError(
            f"overlap derivative must have shape {expected}, got "
            f"{derivative.shape}"
        )
    overlap = cp.asarray(
        cell.pbc_intor("int1e_ovlp", hermi=1, kpts=points)
    )
    madelung = float(pbc_tools.madelung(cell, points))
    output = cp.empty_like(derivative, dtype=cp.complex128)
    for atom_id in range(derivative.shape[0]):
        for kpoint_id in range(len(points)):
            for axis in range(3):
                ds = derivative[atom_id, kpoint_id, axis]
                output[atom_id, kpoint_id, axis] = madelung * (
                    ds @ dm[kpoint_id] @ overlap[kpoint_id]
                    + overlap[kpoint_id] @ dm[kpoint_id] @ ds
                )
    return output


def pair_block(blocks, ki, kj):
    """Return one device CDERI block using Hermitian pair reversal."""

    key = (int(ki), int(kj))
    if key in blocks:
        return blocks[key]
    reverse = (key[1], key[0])
    if reverse in blocks:
        return blocks[reverse].conj().transpose(0, 2, 1)
    raise KeyError(f"CDERI block for pair {key} is unavailable")


def derivative_pair_block(blocks, ki, kj, atom_id, axis):
    """Return one device ``dCDERI`` block using pair reversal."""

    key = (int(ki), int(kj))
    if key in blocks:
        return blocks[key][int(atom_id), int(axis)]
    reverse = (key[1], key[0])
    if reverse in blocks:
        return (
            blocks[reverse][int(atom_id), int(axis)]
            .conj()
            .transpose(0, 2, 1)
        )
    raise KeyError(f"dCDERI block for pair {key} is unavailable")


def _coulomb_derivative(cderi, dcderi, dm, natm, ncart):
    nkpts, nao, _ = dm.shape
    first = pair_block(cderi, 0, 0)
    jaux = cp.zeros(first.shape[0], dtype=cp.complex128)
    for k_id in range(nkpts):
        jaux += cp.einsum(
            "pij,ji->p", pair_block(cderi, k_id, k_id), dm[k_id]
        )
    output = cp.zeros(
        (natm, nkpts, ncart, nao, nao), dtype=cp.complex128
    )
    for atom_id in range(natm):
        for axis in range(ncart):
            djaux = cp.zeros_like(jaux)
            for k_id in range(nkpts):
                djaux += cp.einsum(
                    "pij,ji->p",
                    derivative_pair_block(
                        dcderi, k_id, k_id, atom_id, axis
                    ),
                    dm[k_id],
                )
            for k_id in range(nkpts):
                value = pair_block(cderi, k_id, k_id)
                derivative = derivative_pair_block(
                    dcderi, k_id, k_id, atom_id, axis
                )
                output[atom_id, k_id, axis] = (
                    cp.einsum("p,pij->ij", jaux, derivative)
                    + cp.einsum("p,pij->ij", djaux, value)
                ) / nkpts
    return output


def _exchange_derivative(cderi, dcderi, dm, natm, ncart):
    nkpts, nao, _ = dm.shape
    output = cp.zeros(
        (natm, nkpts, ncart, nao, nao), dtype=cp.complex128
    )
    for atom_id in range(natm):
        for axis in range(ncart):
            for ki in range(nkpts):
                matrix = cp.zeros((nao, nao), dtype=cp.complex128)
                for kj in range(nkpts):
                    c_ij = pair_block(cderi, ki, kj)
                    c_ji = pair_block(cderi, kj, ki)
                    dc_ij = derivative_pair_block(
                        dcderi, ki, kj, atom_id, axis
                    )
                    dc_ji = derivative_pair_block(
                        dcderi, kj, ki, atom_id, axis
                    )
                    matrix += cp.einsum(
                        "pij,jk,pkl->il", dc_ij, dm[kj], c_ji,
                        optimize=True,
                    )
                    matrix += cp.einsum(
                        "pij,jk,pkl->il", c_ij, dm[kj], dc_ji,
                        optimize=True,
                    )
                output[atom_id, ki, axis] = matrix / nkpts
    return output


def _density_array(value):
    dm = cp.asarray(value)
    if dm.ndim == 2:
        dm = dm[None]
    if dm.ndim != 3 or dm.shape[1] != dm.shape[2]:
        raise ValueError(
            f"density must have shape (nkpts,nao,nao), got {dm.shape}"
        )
    return dm


def _restricted_hf_matrix_derivative(value):
    if isinstance(value, GDFJKDerivatives):
        return value.restricted_hf
    derivative = cp.asarray(value)
    if derivative.ndim != 5:
        raise ValueError(
            "restricted HF matrix derivative must have five dimensions, "
            f"got {derivative.shape}"
        )
    return derivative


def _derivative_shape(blocks):
    try:
        first = next(iter(blocks.values()))
    except StopIteration as exc:
        raise ValueError("dcderi cannot be empty") from exc
    if first.ndim != 5 or first.shape[1] != 3:
        raise ValueError(
            "dCDERI blocks must have (natm,3,naux,nao,nao), got "
            f"{first.shape}"
        )
    expected = first.shape
    if any(block.shape != expected for block in blocks.values()):
        raise ValueError("all dCDERI blocks must have the same shape")
    return tuple(int(value) for value in expected)


def _validate_matrix_derivative_shape(derivative, density):
    expected_tail = (len(density), 3, density.shape[1], density.shape[2])
    if derivative.ndim != 5 or derivative.shape[1:] != expected_tail:
        raise ValueError(
            "matrix derivative must have (natm,nkpts,3,nao,nao), got "
            f"{derivative.shape}"
        )
