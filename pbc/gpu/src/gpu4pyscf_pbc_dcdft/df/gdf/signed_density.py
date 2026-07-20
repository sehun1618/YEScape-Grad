"""Gamma-point cross-density sources for native GDF derivatives.

The production DC-DFT gradient currently evaluates the exact polarization
difference ``Q(D + T) - Q(D - T)`` with two calls to GPU4PySCF's quadratic
PBC GDF derivative.  This module constructs the equivalent source contracted
with one three-center derivative, without materializing ``dCDERI``.  It is the
algebra boundary for a future pair-aware periodic CUDA entry point.

Only Gamma-point real Hermitian densities are accepted here.  Sampled
k-points require explicit momentum conservation and conjugate-pair handling
and therefore belong in a separate implementation rather than hidden shape
branches in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from cupyx.scipy.linalg import solve_triangular
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
from pyscf.pbc.tools import pbc as pbc_tools

from .jk_derivative import derivative_pair_block, pair_block


DEFAULT_PRIMITIVE_SOURCE_MEMORY_FRACTION = 0.25
DEFAULT_PRIMITIVE_SOURCE_RESERVE_BYTES = 512 * 1024**2
DEFAULT_PRIMITIVE_SOURCE_MAX_AUX_BLOCK = 32
DEFAULT_IMAGINARY_REDUCTION_BLOCK_BYTES = 256 * 1024**2


@dataclass(frozen=True)
class GammaSignedDensitySource:
    """Cross-density source tensors with ``(naux,nao,nao)`` ordering."""

    coulomb: cp.ndarray
    exchange: cp.ndarray

    @property
    def restricted_hf(self) -> cp.ndarray:
        """Return the source for the restricted ``J-K/2`` derivative."""

        return self.coulomb + self.exchange


@dataclass(frozen=True)
class GammaFactorizationAdjoint:
    """Reverse-contracted source and metric weights for ``C=L^-1 B``."""

    source: cp.ndarray
    metric: cp.ndarray

    def contract(self, source_derivative, metric_derivative):
        """Contract primitive ``dB``/``dM`` without constructing ``dC``."""

        d_source = cp.asarray(source_derivative)
        d_metric = cp.asarray(metric_derivative)
        if d_source.shape != self.source.shape:
            raise ValueError(
                f"source derivative shape {d_source.shape} does not match "
                f"adjoint shape {self.source.shape}"
            )
        if d_metric.shape != self.metric.shape:
            raise ValueError(
                f"metric derivative shape {d_metric.shape} does not match "
                f"adjoint shape {self.metric.shape}"
            )
        value = cp.einsum("ip,ip->", self.source, d_source)
        value += cp.einsum("pq,pq->", self.metric, d_metric)
        return value.real


@dataclass(frozen=True)
class GammaPrimitiveAdjoint:
    """Primitive three-center and metric sources for one Gamma response.

    ``source[p]`` contracts with the derivative of the unorthogonalized
    three-center matrix ``B[p]``. ``metric[p, q]`` contracts with ``dM``.
    Both arrays are built from one metric-solved tensor ``W=M^-1 B`` and do
    not require the two full-rank quadratic states ``D+T`` and ``D-T``.
    """

    source: cp.ndarray
    metric: cp.ndarray


def build_gamma_primitive_adjoint(
    metric_solved_source,
    reference_density,
    response_density,
    *,
    include_target_j=False,
    max_auxiliary_block=None,
    memory_diagnostics=None,
) -> GammaPrimitiveAdjoint:
    """Build direct Gamma ``dB``/``dM`` sources for target J and Z response.

    Expanding ``Q(D+T)-Q(D-T)`` gives the same restricted-HF response using
    only cross terms.  ``include_target_j`` additionally includes the
    selected-density Coulomb derivative at ``D``.
    """

    solved = _real_gamma_matrix_stack(
        metric_solved_source,
        name="metric_solved_source",
    )
    reference = _real_gamma_density(
        reference_density,
        name="reference_density",
    )
    response = _real_gamma_density(
        response_density,
        name="response_density",
    )
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    if solved.shape[1:] != reference.shape:
        raise ValueError(
            f"metric-solved AO shape {solved.shape[1:]} does not match "
            f"density shape {reference.shape}"
        )

    auxiliary_reference = cp.einsum(
        "pij,ji->p", solved, reference, optimize=True
    )
    auxiliary_response = cp.einsum(
        "pij,ji->p", solved, response, optimize=True
    )

    source, source_execution = _build_gamma_primitive_source_bounded(
        solved,
        reference,
        response,
        auxiliary_reference,
        auxiliary_response,
        include_target_j=include_target_j,
        max_auxiliary_block=max_auxiliary_block,
    )
    # For the inverse-metric representation, the complete metric source is
    # the contraction of the primitive-source adjoint with W.  This identity
    # avoids two additional ``(naux, nao, nao)`` transformed-density tensors.
    metric = cp.einsum(
        "pij,qij->pq", source, solved, optimize=True
    )

    if memory_diagnostics is not None:
        memory_diagnostics.clear()
        memory_diagnostics.update(source_execution)

    return GammaPrimitiveAdjoint(
        source=source,
        metric=metric,
    )


def plan_gamma_primitive_source_blocks(
    naux,
    nao,
    dtype,
    *,
    free_device_bytes=None,
    memory_fraction=DEFAULT_PRIMITIVE_SOURCE_MEMORY_FRACTION,
    reserve_bytes=DEFAULT_PRIMITIVE_SOURCE_RESERVE_BYTES,
    max_auxiliary_block=DEFAULT_PRIMITIVE_SOURCE_MAX_AUX_BLOCK,
):
    """Plan bounded auxiliary blocks around one persistent source tensor."""

    naux = int(naux)
    nao = int(nao)
    if naux < 1 or nao < 1:
        raise ValueError("naux and nao must be positive")
    fraction = float(memory_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    if free_device_bytes is None:
        free_device_bytes = int(
            get_avail_mem(exclude_memory_pool=True)
        )
    itemsize = cp.dtype(dtype).itemsize
    source_bytes = naux * nao**2 * itemsize
    reserve = min(max(0, int(reserve_bytes)), int(free_device_bytes))
    remaining = int(free_device_bytes) - source_bytes - reserve
    if remaining <= 0:
        raise MemoryError(
            "insufficient device memory for persistent Gamma primitive "
            f"source: free={int(free_device_bytes)} bytes, "
            f"source={source_bytes} bytes, reserve={reserve} bytes"
        )
    # A transactional block and two reusable contraction temporaries are
    # explicit.  A fourth block allowance covers contraction-engine workspace.
    working_bytes_per_auxiliary = 4 * nao**2 * itemsize
    budget = max(1, int(remaining * fraction))
    block_size = max(1, budget // working_bytes_per_auxiliary)
    block_size = min(
        naux,
        max(1, int(max_auxiliary_block)),
        int(block_size),
    )
    return {
        "naux": naux,
        "nao": nao,
        "dtype": cp.dtype(dtype).str,
        "free_device_bytes_at_plan": int(free_device_bytes),
        "persistent_source_bytes": int(source_bytes),
        "reserve_bytes": int(reserve),
        "temporary_budget_bytes": int(budget),
        "working_bytes_per_auxiliary": int(working_bytes_per_auxiliary),
        "initial_block_size": int(block_size),
        "maximum_block_size": int(max_auxiliary_block),
    }


def _build_gamma_primitive_source_bounded(
    solved,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
    *,
    include_target_j,
    max_auxiliary_block,
):
    naux, nao, _ = solved.shape
    maximum = (
        DEFAULT_PRIMITIVE_SOURCE_MAX_AUX_BLOCK
        if max_auxiliary_block is None
        else int(max_auxiliary_block)
    )
    plan = plan_gamma_primitive_source_blocks(
        naux,
        nao,
        solved.dtype,
        max_auxiliary_block=maximum,
    )
    source = cp.empty_like(solved)
    start = 0
    block_size = int(plan["initial_block_size"])
    minimum_block_size = block_size
    block_count = 0
    rebatch_count = 0
    while start < naux:
        stop = min(naux, start + block_size)
        try:
            block = _gamma_primitive_source_block(
                solved[start:stop],
                reference,
                response,
                auxiliary_reference[start:stop],
                auxiliary_response[start:stop],
                include_target_j=include_target_j,
            )
        except Exception as error:
            if not _is_gpu_out_of_memory(error) or stop - start <= 1:
                raise
            cp.get_default_memory_pool().free_all_blocks()
            block_size = max(1, (stop - start) // 2)
            rebatch_count += 1
            continue
        source[start:stop] = block
        del block
        block_count += 1
        minimum_block_size = min(minimum_block_size, stop - start)
        start = stop
    return source, {
        "plan": plan,
        "execution": {
            "final_block_size": int(block_size),
            "minimum_block_size": int(minimum_block_size),
            "block_count": int(block_count),
            "rebatch_count": int(rebatch_count),
            "completed_auxiliary_functions": int(naux),
        },
    }


def _gamma_primitive_source_block(
    solved,
    reference,
    response,
    auxiliary_reference,
    auxiliary_response,
    *,
    include_target_j,
):
    block = cp.empty_like(solved)
    temporary = cp.empty_like(solved)
    exchange = cp.empty_like(solved)
    cp.multiply(
        auxiliary_response[:, None, None],
        reference[None],
        out=block,
    )
    cp.multiply(
        auxiliary_reference[:, None, None],
        response[None],
        out=temporary,
    )
    block += temporary
    block *= 2.0
    contract(
        "ij,pjk->pik",
        reference,
        solved,
        out=temporary,
    )
    contract(
        "pik,kl->pil",
        temporary,
        response,
        out=exchange,
    )
    block -= exchange
    contract(
        "ij,pjk->pik",
        response,
        solved,
        out=temporary,
    )
    contract(
        "pik,kl->pil",
        temporary,
        reference,
        out=exchange,
    )
    block -= exchange
    if include_target_j:
        cp.multiply(
            auxiliary_reference[:, None, None],
            reference[None],
            out=temporary,
        )
        block += temporary
    return block


def build_gamma_signed_density_source(
    cderi,
    reference_density,
    response_density,
) -> GammaSignedDensitySource:
    """Build the direct ``D/T`` source for one Gamma GDF traversal.

    Let ``C[p]`` denote a real symmetric, metric-orthogonalized three-center
    factor and let ``u_p[X] = Tr(C[p] X)``.  The source which contracts with
    ``dC[p]`` in ``Q(D+T)-Q(D-T)`` is

    ``S_J[p] = 2 * (D.T * u_p[T] + T.T * u_p[D])``

    for Coulomb and

    ``S_K[p] = -(D C[p] T + T C[p] D).T``

    for restricted exchange.  The latter already contains the ``-K/2``
    coefficient and both product-rule orientations.

    This function allocates ``O(naux*nao**2)`` source storage but no nuclear
    derivative tensor.  The source is intended to be consumed blockwise by a
    native derivative kernel.
    """

    factors = _real_gamma_matrix_stack(cderi, name="cderi")
    factor_asymmetry = float(
        cp.max(cp.abs(factors - factors.transpose(0, 2, 1))).item()
    )
    if factor_asymmetry > 1.0e-9:
        raise ValueError(
            "Gamma cderi factors must be AO-symmetric; maximum asymmetry is "
            f"{factor_asymmetry:.3e}"
        )
    reference = _real_gamma_density(reference_density, name="reference_density")
    response = _real_gamma_density(response_density, name="response_density")
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    if factors.shape[1:] != reference.shape:
        raise ValueError(
            f"cderi AO shape {factors.shape[1:]} does not match density "
            f"shape {reference.shape}"
        )

    auxiliary_reference = cp.einsum(
        "pij,ji->p", factors, reference, optimize=True
    )
    auxiliary_response = cp.einsum(
        "pij,ji->p", factors, response, optimize=True
    )
    coulomb = 2.0 * (
        reference.T[None] * auxiliary_response[:, None, None]
        + response.T[None] * auxiliary_reference[:, None, None]
    )

    reference_factor_response = cp.einsum(
        "ij,pjk,kl->pil",
        reference,
        factors,
        response,
        optimize=True,
    )
    response_factor_reference = cp.einsum(
        "ij,pjk,kl->pil",
        response,
        factors,
        reference,
        optimize=True,
    )
    exchange = -(
        reference_factor_response + response_factor_reference
    ).transpose(0, 2, 1)
    return GammaSignedDensitySource(coulomb=coulomb, exchange=exchange)


def contract_gamma_signed_density_source(source, cderi_derivative):
    """Contract a prototype source with ``(natm,3,naux,nao,nao)`` dCDERI.

    This helper is test/reference-only.  Production code must pass the source
    to an integral-derivative kernel instead of constructing full ``dCDERI``.
    """

    if isinstance(source, GammaSignedDensitySource):
        source_array = source.restricted_hf
    else:
        source_array = _real_gamma_matrix_stack(source, name="source")
    derivative = cp.asarray(cderi_derivative)
    if derivative.ndim != 5 or derivative.shape[1] != 3:
        raise ValueError(
            "cderi_derivative must have shape (natm,3,naux,nao,nao), "
            f"got {derivative.shape}"
        )
    if derivative.shape[2:] != source_array.shape:
        raise ValueError(
            f"derivative tail {derivative.shape[2:]} does not match source "
            f"shape {source_array.shape}"
        )
    return cp.einsum(
        "axpij,pij->ax", derivative, source_array, optimize=True
    ).real


def contract_kpoint_bilinear_cderi_derivative(
    cderi,
    cderi_derivative,
    reference_density,
    response_density,
):
    """Contract sampled-k ``dCDERI`` directly with the RHF ``D/T`` source.

    Unlike the full-matrix witness, this routine never constructs
    ``dJ_k/dR`` or ``dK_k/dR``.  It keeps all ordered exchange k-pairs and the
    four cross orientations explicit, making it the algebra contract for the
    future sampled-k native integral kernel.  It remains test/reference-only
    because ``cderi_derivative`` is materialized.
    """

    reference = _kpoint_density(
        reference_density,
        _density_kpoint_count(reference_density),
        name="reference_density",
    )
    response = _kpoint_density(
        response_density,
        len(reference),
        name="response_density",
    )
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    first_derivative = next(iter(cderi_derivative.values()))
    if first_derivative.ndim != 5 or first_derivative.shape[1] != 3:
        raise ValueError(
            "cderi derivatives must have shape (natm,3,naux,nao,nao), "
            f"got {first_derivative.shape}"
        )
    natm, ncart = first_derivative.shape[:2]
    nkpts = len(reference)

    auxiliary_reference = cp.zeros(
        pair_block(cderi, 0, 0).shape[0],
        dtype=cp.complex128,
    )
    auxiliary_response = cp.zeros_like(auxiliary_reference)
    derivative_reference = cp.zeros(
        (natm, ncart, len(auxiliary_reference)),
        dtype=cp.complex128,
    )
    derivative_response = cp.zeros_like(derivative_reference)
    for kpoint_id in range(nkpts):
        factor = pair_block(cderi, kpoint_id, kpoint_id)
        derivative = cderi_derivative.get((kpoint_id, kpoint_id))
        if derivative is None:
            derivative = cp.stack(
                [
                    [
                        derivative_pair_block(
                            cderi_derivative,
                            kpoint_id,
                            kpoint_id,
                            atom_id,
                            axis,
                        )
                        for axis in range(ncart)
                    ]
                    for atom_id in range(natm)
                ]
            )
        auxiliary_reference += cp.einsum(
            "pij,ji->p", factor, reference[kpoint_id], optimize=True
        )
        auxiliary_response += cp.einsum(
            "pij,ji->p", factor, response[kpoint_id], optimize=True
        )
        derivative_reference += cp.einsum(
            "axpij,ji->axp",
            derivative,
            reference[kpoint_id],
            optimize=True,
        )
        derivative_response += cp.einsum(
            "axpij,ji->axp",
            derivative,
            response[kpoint_id],
            optimize=True,
        )

    value = 2.0 * cp.einsum(
        "p,axp->ax",
        auxiliary_reference,
        derivative_response,
        optimize=True,
    )
    value += 2.0 * cp.einsum(
        "p,axp->ax",
        auxiliary_response,
        derivative_reference,
        optimize=True,
    )

    exchange = cp.zeros_like(value)
    for ki in range(nkpts):
        for kj in range(nkpts):
            factor_ij = pair_block(cderi, ki, kj)
            factor_ji = pair_block(cderi, kj, ki)
            for atom_id in range(natm):
                for axis in range(ncart):
                    derivative_ij = derivative_pair_block(
                        cderi_derivative,
                        ki,
                        kj,
                        atom_id,
                        axis,
                    )
                    derivative_ji = derivative_pair_block(
                        cderi_derivative,
                        kj,
                        ki,
                        atom_id,
                        axis,
                    )
                    exchange[atom_id, axis] += cp.einsum(
                        "li,pij,jk,pkl->",
                        reference[ki],
                        derivative_ij,
                        response[kj],
                        factor_ji,
                        optimize=True,
                    )
                    exchange[atom_id, axis] += cp.einsum(
                        "li,pij,jk,pkl->",
                        response[ki],
                        derivative_ij,
                        reference[kj],
                        factor_ji,
                        optimize=True,
                    )
                    exchange[atom_id, axis] += cp.einsum(
                        "li,pij,jk,pkl->",
                        reference[ki],
                        factor_ij,
                        response[kj],
                        derivative_ji,
                        optimize=True,
                    )
                    exchange[atom_id, axis] += cp.einsum(
                        "li,pij,jk,pkl->",
                        response[ki],
                        factor_ij,
                        reference[kj],
                        derivative_ji,
                        optimize=True,
                    )
    value -= 0.5 * exchange
    return value.real / nkpts**2


def ewald_restricted_hf_bilinear_derivative(
    cell,
    kpts,
    reference_density,
    response_density,
    overlap_derivative,
):
    """Return the direct cross-density Ewald exchange contribution.

    GPU4PySCF's GDF exchange correction is

    ``K_G0[X] = m * S X S``

    with Madelung scalar ``m``.  After the restricted ``-K/2`` coefficient
    and both density orientations are included, the polarization difference
    contracts ``dS`` with

    ``-m * (D S T + T S D)``.

    The result has shape ``(natm,3)`` and already includes ``1/Nk``.  This
    direct expression avoids constructing separate Ewald derivative matrices
    for ``D+T`` and ``D-T``.
    """

    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    reference = _kpoint_density(reference_density, len(points), name="reference_density")
    response = _kpoint_density(response_density, len(points), name="response_density")
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    derivative = cp.asarray(overlap_derivative)
    expected = (
        derivative.shape[0],
        len(points),
        3,
        reference.shape[1],
        reference.shape[2],
    )
    if derivative.shape != expected:
        raise ValueError(
            f"overlap_derivative must have shape {expected}, got "
            f"{derivative.shape}"
        )

    overlap = cp.asarray(
        cell.pbc_intor("int1e_ovlp", hermi=1, kpts=points)
    )
    reference_overlap_response = cp.einsum(
        "kij,kjl,klm->kim",
        reference,
        overlap,
        response,
        optimize=True,
    )
    response_overlap_reference = cp.einsum(
        "kij,kjl,klm->kim",
        response,
        overlap,
        reference,
        optimize=True,
    )
    cross_product = reference_overlap_response + response_overlap_reference
    madelung = float(pbc_tools.madelung(cell, points))
    value = cp.einsum(
        "akxij,kji->ax",
        derivative,
        cross_product,
        optimize=True,
    )
    return -madelung * value.real / len(points)


def build_gamma_factorization_adjoint(
    factorization,
    signed_source,
) -> GammaFactorizationAdjoint:
    """Reverse ``C=L^-1 B`` for a Gamma signed-density contraction.

    ``CDERIFactorization.differentiate`` computes ``dC`` from primitive
    three-center and metric derivatives.  The production one-pass route needs
    the reverse operation: contract the signed source with ``dC`` first and
    obtain weights for ``dB`` and ``dM``.  This removes the
    ``(natm,3,naux,nao,nao)`` derivative tensor from the production design.

    The Cholesky reverse is evaluated in the same lower-triangular gauge as
    ``CDERIFactorization._differentiate_cholesky``.  This prototype is
    deliberately restricted to the real Gamma case; complex momentum blocks
    need a conjugate-aware adjoint and are handled separately.
    """

    if isinstance(signed_source, GammaSignedDensitySource):
        source = signed_source.restricted_hf
    else:
        source = _real_gamma_matrix_stack(
            signed_source,
            name="signed_source",
        )
    factor = _real_gamma_matrix(factorization.factor, name="factor")
    value = _real_gamma_matrix_stack(factorization.value, name="cderi")
    if source.shape != value.shape:
        raise ValueError(
            f"signed source shape {source.shape} does not match CDERI "
            f"shape {value.shape}"
        )

    source_flat = source.reshape(source.shape[0], -1)
    value_flat = value.reshape(value.shape[0], -1)
    solved_source = solve_triangular(
        factor.T,
        source_flat,
        lower=False,
        check_finite=False,
    )
    source_adjoint = solved_source.T

    factor_adjoint = -(solved_source @ value_flat.T)
    reduced_adjoint = factor.T @ factor_adjoint
    symmetric_gauge = 0.5 * (
        cp.tril(reduced_adjoint)
        + cp.tril(reduced_adjoint, k=-1).T
    )
    left_solved = solve_triangular(
        factor.T,
        symmetric_gauge,
        lower=False,
        check_finite=False,
    )
    metric_adjoint = solve_triangular(
        factor.T,
        left_solved.T,
        lower=False,
        check_finite=False,
    ).T
    return GammaFactorizationAdjoint(
        source=source_adjoint,
        metric=metric_adjoint,
    )


def _real_gamma_density(value, *, name):
    array = cp.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square Gamma matrix, got {array.shape}")
    imaginary_max = _max_abs_imaginary(array)
    if imaginary_max > 1.0e-10:
        raise NotImplementedError(
            f"{name} must be real at Gamma; max imaginary component is "
            f"{imaginary_max:.3e}"
        )
    result = cp.asarray(array.real, dtype=cp.float64)
    hermiticity = float(cp.max(cp.abs(result - result.T)).item())
    if hermiticity > 1.0e-9:
        raise ValueError(
            f"{name} must be Hermitian; max asymmetry is {hermiticity:.3e}"
        )
    return result


def _density_kpoint_count(value):
    array = cp.asarray(value)
    return 1 if array.ndim == 2 else int(array.shape[0])


def _real_gamma_matrix_stack(value, *, name):
    array = cp.asarray(value)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(
            f"{name} must have shape (naux,nao,nao), got {array.shape}"
        )
    imaginary_max = _max_abs_imaginary(array)
    if imaginary_max > 1.0e-10:
        raise NotImplementedError(
            f"{name} must be real at Gamma; max imaginary component is "
            f"{imaginary_max:.3e}"
        )
    return cp.asarray(array.real, dtype=cp.float64)


def _real_gamma_matrix(value, *, name):
    array = cp.asarray(value)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square, got {array.shape}")
    imaginary_max = _max_abs_imaginary(array)
    if imaginary_max > 1.0e-10:
        raise NotImplementedError(
            f"{name} must be real at Gamma; max imaginary component is "
            f"{imaginary_max:.3e}"
        )
    return cp.asarray(array.real, dtype=cp.float64)


def _max_abs_imaginary(
    value,
    *,
    max_block_bytes=DEFAULT_IMAGINARY_REDUCTION_BLOCK_BYTES,
):
    """Return the imaginary maximum without a full-size ``abs`` temporary."""

    array = cp.asarray(value)
    if not cp.iscomplexobj(array):
        return 0.0
    if array.ndim == 0:
        return float(cp.abs(array.imag).item())
    elements_per_slice = int(np.prod(array.shape[1:], dtype=np.int64))
    slices_per_block = max(
        1,
        int(max_block_bytes)
        // max(1, elements_per_slice * array.real.dtype.itemsize),
    )
    maximum = 0.0
    for start in range(0, array.shape[0], slices_per_block):
        stop = min(array.shape[0], start + slices_per_block)
        block_maximum = float(
            cp.max(cp.abs(array[start:stop].imag)).item()
        )
        maximum = max(maximum, block_maximum)
    return maximum


def _is_gpu_out_of_memory(error):
    if isinstance(error, cp.cuda.memory.OutOfMemoryError):
        return True
    message = str(error).lower()
    return (
        error.__class__.__name__ == "CUDARuntimeError"
        and (
            "out of memory" in message
            or "cudaerrormemoryallocation" in message
        )
    )


def _kpoint_density(value, nkpts, *, name):
    array = cp.asarray(value)
    if array.ndim == 2 and int(nkpts) == 1:
        array = array[None]
    expected = (int(nkpts), array.shape[-1], array.shape[-1])
    if array.ndim != 3 or array.shape != expected:
        raise ValueError(
            f"{name} must have shape (nkpts,nao,nao), got {array.shape}"
        )
    return array
