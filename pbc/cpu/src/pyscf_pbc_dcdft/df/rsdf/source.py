r"""Public-RSDF three-center source and analytic center derivative.

The central short-range source is evaluated with the same ``precision_R`` and
optional Born-von Karman image grouping as PySCF :class:`RSGDF`.  The
long-range and zero-transfer terms share the already validated reciprocal
space formulas used by the GDF implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import lib
from pyscf.pbc.df import rsdf_helper
from pyscf.pbc.lib.kpts_helper import is_zero

from ..gdf.source_long_range import (
    long_range_source,
    long_range_source_derivative,
)
from ..gdf.source_short_range import short_range_source_derivative
from ..gdf.vbar import vbar_source, vbar_source_derivative


@dataclass(frozen=True)
class RSDFSourceDerivativeParts:
    """Short-range, long-range, and zero-transfer source derivatives."""

    short_range: np.ndarray
    long_range: np.ndarray
    vbar: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """Return ``dB_SR + dB_LR + dB_vbar``."""

        return self.short_range + self.long_range + self.vbar


def rsdf_source(
    cell,
    auxcell,
    parameters,
    *,
    kpti: np.ndarray,
    kptj: np.ndarray,
) -> np.ndarray:
    """Return the pre-factorization RSDF source ``B(ki,kj)``.

    The result has shape ``(nao**2, naux)`` and follows PySCF's orientation
    before the lower-triangular metric solve.
    """

    first, second = _kpoint_pair(kpti, kptj)
    source = np.asarray(
        _short_range_source(
            cell,
            auxcell,
            parameters,
            kpti=first,
            kptj=second,
        ),
        dtype=np.complex128,
    )
    source += long_range_source(
        cell,
        auxcell,
        parameters.source_omega,
        mesh=parameters.source_mesh,
        kpti=first,
        kptj=second,
    )
    if is_zero(second - first):
        source += vbar_source(
            cell,
            auxcell,
            parameters.source_omega,
            kpt=first,
        )
    return _real_if_close(source)


def rsdf_source_derivative(
    cell,
    auxcell,
    atom_id: int,
    parameters,
    *,
    kpti: np.ndarray,
    kptj: np.ndarray,
) -> RSDFSourceDerivativeParts:
    """Return all Cartesian derivatives of ``B(ki,kj)`` for one atom.

    The short-range derivative uses the same analytic ``int3c2e_ip1/ip2``
    integrals as GDF but tightens screening to the RSDF ``precision_R``.
    PySCF's BvK grouping changes evaluation order, not the derivative formula;
    the central-value parity guard below remains the authoritative convention
    check.
    """

    first, second = _kpoint_pair(kpti, kptj)
    short_range = short_range_source_derivative(
        cell,
        auxcell,
        atom_id,
        parameters.source_omega,
        kpti=first,
        kptj=second,
        precision=parameters.precision_r,
    )
    long_range = long_range_source_derivative(
        cell,
        auxcell,
        atom_id,
        parameters.source_omega,
        mesh=parameters.source_mesh,
        kpti=first,
        kptj=second,
    )
    if is_zero(second - first):
        zero_mode = vbar_source_derivative(
            cell,
            auxcell,
            atom_id,
            parameters.source_omega,
            kpt=first,
        )
    else:
        zero_mode = np.zeros_like(short_range)
    return RSDFSourceDerivativeParts(
        short_range=np.asarray(short_range),
        long_range=np.asarray(long_range),
        vbar=np.asarray(zero_mode),
    )


def _short_range_source(cell, auxcell, parameters, *, kpti, kptj):
    temporary = lib.H5TmpFile()
    try:
        rsdf_helper._aux_e2_nospltbas(
            cell,
            auxcell,
            parameters.source_omega,
            temporary,
            intor="int3c2e",
            aosym="s2",
            kptij_lst=np.asarray([[kpti, kptj]], dtype=float),
            dataname="j3c",
            max_memory=4000,
            bvk_kmesh=(
                parameters.bvk_kmesh if parameters.use_bvk else None
            ),
            precision=parameters.precision_r,
        )
        group = temporary["j3c/0"]
        keys = sorted(group.keys(), key=lambda key: int(key))
        raw = np.concatenate(
            [np.asarray(group[key]) for key in keys], axis=-1
        )
        if raw.shape[0] == 1:
            raw = raw[0]
        return _dense_source(raw, cell.nao_nr(), auxcell.nao_nr())
    finally:
        temporary.close()


def _dense_source(raw, nao, naux):
    array = np.asarray(raw)
    ntril = int(nao) * (int(nao) + 1) // 2
    if array.shape[-2] == ntril:
        array = lib.unpack_tril(array.T).transpose(1, 2, 0)
    if array.size != int(nao) ** 2 * int(naux):
        raise ValueError(
            f"unexpected RSDF source shape {array.shape} for "
            f"nao={nao}, naux={naux}"
        )
    return array.reshape(int(nao) ** 2, int(naux))


def _kpoint_pair(kpti, kptj):
    first = np.asarray(kpti, dtype=float)
    second = np.asarray(kptj, dtype=float)
    if first.shape != (3,) or second.shape != (3,):
        raise ValueError(
            "kpti and kptj must each have shape (3,), got "
            f"{first.shape} and {second.shape}"
        )
    return first, second


def _real_if_close(array, tolerance=1.0e-10):
    value = np.asarray(array)
    if value.size and np.max(np.abs(value.imag)) <= tolerance:
        return value.real
    return value
