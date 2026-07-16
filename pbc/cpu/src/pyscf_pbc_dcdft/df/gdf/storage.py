"""Dense access to PySCF periodic GDF ``CDERI`` storage.

PySCF stores three-index density-fitting factors in an HDF5 file owned by a
``pyscf.pbc.df.GDF`` object.  Diagonal k-point pairs may use packed AO-pair
storage, while off-diagonal pairs use full complex matrices.  The DC response
code needs one uniform tensor contract, so this module exposes canonical dense
blocks with shape ``(naux, nao, nao)``.

Only value tensors are read here.  Nuclear derivatives are owned by the
analytic source/metric modules; finite-difference generation does not belong
in the production package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from pyscf import lib
from pyscf.pbc.df.df import _load3c


Pair = tuple[int, int]


@dataclass(frozen=True)
class CDERIStorage:
    """Dense canonical GDF factors and their storage metadata.

    Attributes
    ----------
    blocks
        Mapping from canonical ``(ki, kj)`` pairs with ``ki >= kj`` to dense
        ``(naux, nao, nao)`` factors.
    nkpts, nao, naux
        Dimensions of the sampled k-point, AO, and auxiliary spaces.
    records
        Per-pair storage descriptions suitable for diagnostics and reports.
    """

    blocks: Mapping[Pair, np.ndarray]
    nkpts: int
    nao: int
    naux: int
    records: tuple[dict[str, object], ...]

    def block(self, ki: int, kj: int) -> np.ndarray:
        """Return ``CDERI[ki,kj]`` using Hermitian pair symmetry."""

        return pair_block(self.blocks, ki, kj)


def canonical_pairs(nkpts: int) -> tuple[Pair, ...]:
    """Return lower-triangular k-point pairs in PySCF storage order."""

    count = int(nkpts)
    if count < 1:
        raise ValueError("nkpts must be positive")
    return tuple((ki, kj) for ki in range(count) for kj in range(ki + 1))


def load_cderi_storage(
    provider,
    *,
    pairs: Iterable[Pair] | None = None,
) -> CDERIStorage:
    """Load dense canonical factors from a built PySCF GDF provider.

    Parameters
    ----------
    provider
        ``pyscf.pbc.df.GDF`` or the package-local ``df.GDF`` instance.
    pairs
        Optional canonical pair subset.  Every pair must satisfy ``ki >= kj``.

    Returns
    -------
    CDERIStorage
        Dense factors.  No view into the HDF5 file is retained.
    """

    _ensure_built(provider)
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    nao = int(provider.cell.nao_nr())
    naux = int(provider.auxcell.nao_nr())
    selected = canonical_pairs(len(kpts)) if pairs is None else _parse_pairs(pairs)
    blocks: dict[Pair, np.ndarray] = {}
    records = []
    for pair in selected:
        raw = _load_raw_pair(provider, kpts, pair)
        dense, storage_kind = _dense_from_raw(raw, naux=naux, nao=nao)
        blocks[pair] = dense
        records.append(
            {
                "pair": [int(pair[0]), int(pair[1])],
                "raw_shape": [int(value) for value in raw.shape],
                "dense_shape": [int(value) for value in dense.shape],
                "storage_kind": storage_kind,
                "dtype": str(dense.dtype),
            }
        )
    return CDERIStorage(
        blocks=blocks,
        nkpts=len(kpts),
        nao=nao,
        naux=naux,
        records=tuple(records),
    )


def pair_block(blocks: Mapping[Pair, np.ndarray], ki: int, kj: int) -> np.ndarray:
    """Return a dense pair block, reconstructing the reverse orientation.

    The auxiliary index is invariant under pair reversal, while the AO pair is
    Hermitian-transposed:

    ``C_P(ki,kj)[mu,nu] = C_P(kj,ki)[nu,mu]*``.
    """

    key = (int(ki), int(kj))
    if key in blocks:
        return np.asarray(blocks[key])
    reverse = (key[1], key[0])
    if reverse in blocks:
        return np.asarray(blocks[reverse]).conj().transpose(0, 2, 1)
    raise KeyError(f"CDERI block for pair {key} is unavailable")


def derivative_pair_block(
    blocks: Mapping[Pair, np.ndarray],
    ki: int,
    kj: int,
    atom_id: int,
    axis: int,
) -> np.ndarray:
    """Return one ``dCDERI`` component with canonical-pair reconstruction.

    Input derivative blocks have shape ``(natm, 3, naux, nao, nao)``.
    """

    key = (int(ki), int(kj))
    atom = int(atom_id)
    cart = int(axis)
    if key in blocks:
        return np.asarray(blocks[key])[atom, cart]
    reverse = (key[1], key[0])
    if reverse in blocks:
        return (
            np.asarray(blocks[reverse])[atom, cart]
            .conj()
            .transpose(0, 2, 1)
        )
    raise KeyError(f"dCDERI block for pair {key} is unavailable")


def _ensure_built(provider) -> None:
    if getattr(provider, "_cderi", None) is None or getattr(
        provider, "auxcell", None
    ) is None:
        provider.build(j_only=False)


def _parse_pairs(pairs: Iterable[Pair]) -> tuple[Pair, ...]:
    parsed = []
    for ki, kj in pairs:
        pair = (int(ki), int(kj))
        if pair[0] < pair[1]:
            raise ValueError(
                f"canonical GDF pair must satisfy ki >= kj, got {pair}"
            )
        parsed.append(pair)
    return tuple(parsed)


def _load_raw_pair(provider, kpts: np.ndarray, pair: Pair) -> np.ndarray:
    dataname = getattr(provider, "_dataname", "j3c")
    with _load3c(provider._cderi, dataname, kpts[list(pair)]) as cderi:
        return np.asarray(cderi[:]).copy()


def _dense_from_raw(
    raw: np.ndarray,
    *,
    naux: int,
    nao: int,
) -> tuple[np.ndarray, str]:
    array = np.asarray(raw)
    if array.shape[0] != naux:
        raise ValueError(
            f"CDERI auxiliary dimension is {array.shape[0]}, expected {naux}"
        )
    if array.shape[-1] == nao * nao:
        return array.reshape(naux, nao, nao), "full_s1"
    ntril = nao * (nao + 1) // 2
    if array.shape[-1] == ntril:
        return np.asarray([lib.unpack_tril(row) for row in array]), "packed_s2"
    raise ValueError(
        f"unsupported CDERI shape {array.shape}; expected AO size {nao**2} "
        f"or packed size {ntril}"
    )
