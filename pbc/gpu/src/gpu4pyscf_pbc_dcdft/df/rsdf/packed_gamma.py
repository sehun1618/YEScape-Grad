"""Packed real-symmetric Gamma source adjoints for native RSDF kernels."""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp


DEFAULT_TRANSFORM_AUXILIARY_BLOCK = 32


@dataclass(frozen=True)
class PackedGammaCDERIAdjoint:
    """Physical-basis lower-triangle CDERI adjoint before source solve."""

    values: cp.ndarray
    nao: int

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def shape(self):
        return self.values.shape


@dataclass(frozen=True)
class PackedGammaSourceAdjoint:
    """Physical-basis lower-triangle source with ``(npair,naux)`` storage."""

    values: cp.ndarray
    nao: int

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def shape(self):
        return self.values.shape


@dataclass(frozen=True)
class SortedPackedGammaSources:
    """Sorted Cartesian lower triangles with a leading source axis."""

    values: cp.ndarray
    nao: int

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)


def pack_gamma_cderi_adjoint(cderi_adjoint) -> PackedGammaSourceAdjoint:
    """Pack a real symmetric ``(naux,nao,nao)`` CDERI adjoint."""

    adjoint = cp.asarray(cderi_adjoint)
    if adjoint.ndim != 3 or adjoint.shape[1] != adjoint.shape[2]:
        raise ValueError(
            "Gamma CDERI adjoint must have shape (naux,nao,nao), "
            f"got {adjoint.shape}"
        )
    imaginary = _imaginary_max(adjoint)
    if imaginary > 1.0e-10:
        raise ValueError(
            "Gamma CDERI adjoint must be real before packing; maximum "
            f"imaginary component is {imaginary:.3e}"
        )
    real = adjoint.real
    asymmetry = float(cp.max(cp.abs(real - real.transpose(0, 2, 1))).get())
    if asymmetry > 1.0e-9:
        raise ValueError(
            "Gamma CDERI adjoint must be AO-symmetric; maximum asymmetry is "
            f"{asymmetry:.3e}"
        )
    nao = int(real.shape[1])
    row, column = cp.tril_indices(nao)
    values = cp.ascontiguousarray(real[:, row, column].T)
    return PackedGammaSourceAdjoint(values=values, nao=nao)


def unpack_gamma_source(source) -> cp.ndarray:
    """Return dense ``(nao,nao,naux)`` storage for tests and diagnostics."""

    if not isinstance(source, PackedGammaSourceAdjoint):
        raise TypeError("source must be PackedGammaSourceAdjoint")
    return unpack_lower_triangle(source.values, source.nao)


def unpack_lower_triangle(values, nao) -> cp.ndarray:
    """Expand ``(...,npair,naux)`` lower triangles along the pair axis."""

    packed = cp.asarray(values)
    nao = int(nao)
    expected = nao * (nao + 1) // 2
    if packed.shape[-2] != expected:
        raise ValueError(
            f"packed pair dimension must be {expected}, got {packed.shape}"
        )
    output = cp.empty((*packed.shape[:-2], nao, nao, packed.shape[-1]))
    row, column = cp.tril_indices(nao)
    output[..., row, column, :] = packed
    output[..., column, row, :] = packed
    return output


def transform_gamma_sources_to_sorted_packed(
    optimizer,
    sources,
    *,
    auxiliary_block_size=DEFAULT_TRANSFORM_AUXILIARY_BLOCK,
) -> SortedPackedGammaSources:
    """Transform physical packed sources without a full dense source tensor."""

    packed_sources = tuple(sources)
    if not packed_sources:
        raise ValueError("at least one packed Gamma source is required")
    if not all(
        isinstance(source, PackedGammaSourceAdjoint)
        for source in packed_sources
    ):
        raise TypeError("all Gamma sources must use packed storage")
    physical_nao = packed_sources[0].nao
    physical_naux = int(optimizer.auxcell.cell.nao_nr())
    expected_shape = (
        physical_nao * (physical_nao + 1) // 2,
        physical_naux,
    )
    for source in packed_sources:
        if source.nao != physical_nao or source.values.shape != expected_shape:
            raise ValueError(
                "packed Gamma sources must share shape "
                f"{expected_shape}, got {source.values.shape}"
            )

    sorted_nao = int(optimizer.cell.nao)
    sorted_naux = int(optimizer.auxcell.nao)
    sorted_pair_count = sorted_nao * (sorted_nao + 1) // 2
    dtype = cp.result_type(*(source.dtype for source in packed_sources))
    output = cp.empty(
        (len(packed_sources), sorted_pair_count, sorted_naux),
        dtype=dtype,
    )
    auxiliary_transform = optimizer.auxcell.apply_C_dot(
        cp.eye(physical_naux, dtype=dtype),
        axis=0,
    )
    sorted_row, sorted_column = cp.tril_indices(sorted_nao)
    block_size = max(1, int(auxiliary_block_size))
    for start in range(0, sorted_naux, block_size):
        stop = min(sorted_naux, start + block_size)
        transform_block = auxiliary_transform[start:stop]
        for source_id, source in enumerate(packed_sources):
            physical_packed = source.values @ transform_block.T
            physical_dense = unpack_lower_triangle(
                physical_packed,
                physical_nao,
            ).transpose(2, 0, 1)
            sorted_dense = optimizer.cell.apply_C_mat_CT(physical_dense)
            output[source_id, :, start:stop] = sorted_dense[
                :, sorted_row, sorted_column
            ].T
    return SortedPackedGammaSources(values=output, nao=sorted_nao)


def packed_indices_for_addresses(addresses, nao):
    """Map flattened AO addresses to lower-triangle packed indices."""

    address = cp.asarray(addresses)
    nao = int(nao)
    first, second = divmod(address, nao)
    row = cp.maximum(first, second)
    column = cp.minimum(first, second)
    return row * (row + 1) // 2 + column


def packed_pair_weights(nao, *, dtype=cp.float64):
    """Return one for diagonal and two for off-diagonal AO pairs."""

    row, column = cp.tril_indices(int(nao))
    return cp.where(row == column, 1.0, 2.0).astype(dtype)


def _imaginary_max(value) -> float:
    array = cp.asarray(value)
    if array.dtype.kind != "c" or array.size == 0:
        return 0.0
    return float(cp.max(cp.abs(array.imag)).get())
