"""Analytic CDERI response for the public PySCF RSDF representation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..gdf.cholesky import differentiate_cderi, solve_cderi
from ..gdf.storage import Pair, canonical_pairs, load_cderi_storage
from .metric import rsdf_metric, rsdf_metric_derivative
from .parameters import RSDFParameters, parameters_from_provider
from .source import rsdf_source, rsdf_source_derivative


@dataclass(frozen=True)
class RSDFResponseTensors:
    """Central CDERI blocks and analytic derivatives for canonical k-pairs."""

    cderi: Mapping[Pair, np.ndarray]
    dcderi: Mapping[Pair, np.ndarray]
    source_dcderi: Mapping[Pair, np.ndarray]
    metric_dcderi: Mapping[Pair, np.ndarray]
    parameters: RSDFParameters
    value_parity_max_abs: float


def build_rsdf_response_tensors(
    provider,
    *,
    value_tolerance: float = 1.0e-7,
) -> RSDFResponseTensors:
    """Build analytic ``dCDERI`` for one built project RSDF provider.

    Central values are always loaded from PySCF's HDF5 storage because those
    are the factors used by SCF.  Independently reconstructed source/metric
    values must reproduce every stored factor before any derivative is
    accepted.  This catches mismatched ``omega``, compact mesh, BvK image,
    auxiliary basis, pair orientation, or metric-factor convention.
    """

    if bool(getattr(provider, "j2c_eig_always", False)):
        raise NotImplementedError(
            "RSDF analytic response currently requires Cholesky metric "
            "factorization; j2c_eig_always=True is unsupported"
        )
    storage = load_cderi_storage(provider)
    parameters = parameters_from_provider(provider)
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    derivatives = {}
    source_parts = {}
    metric_parts = {}
    parity = 0.0
    for pair in canonical_pairs(len(kpts)):
        value, derivative, source_part, metric_part = _pair_response(
            provider.cell,
            provider.auxcell,
            kpts,
            pair,
            parameters,
        )
        parity = max(
            parity,
            float(np.max(np.abs(value - storage.blocks[pair]))),
        )
        derivatives[pair] = derivative
        source_parts[pair] = source_part
        metric_parts[pair] = metric_part
    if parity > float(value_tolerance):
        raise RuntimeError(
            "analytic RSDF source/metric convention does not reproduce "
            f"PySCF CDERI storage: max gap {parity:.3e} > "
            f"{float(value_tolerance):.3e}"
        )
    return RSDFResponseTensors(
        cderi=MappingProxyType(dict(storage.blocks)),
        dcderi=MappingProxyType(derivatives),
        source_dcderi=MappingProxyType(source_parts),
        metric_dcderi=MappingProxyType(metric_parts),
        parameters=parameters,
        value_parity_max_abs=parity,
    )


def _pair_response(cell, auxcell, kpts, pair, parameters):
    ki, kj = pair
    first = kpts[ki]
    second = kpts[kj]
    source = rsdf_source(
        cell,
        auxcell,
        parameters,
        kpti=first,
        kptj=second,
    )
    metric = rsdf_metric(
        cell,
        auxcell,
        parameters,
        q=second - first,
    )
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    shape = (int(cell.natm), 3, naux, nao, nao)
    derivative = np.empty(shape, dtype=np.complex128)
    source_part = np.empty(shape, dtype=np.complex128)
    metric_part = np.empty(shape, dtype=np.complex128)
    for atom_id in range(cell.natm):
        d_source = rsdf_source_derivative(
            cell,
            auxcell,
            atom_id,
            parameters,
            kpti=first,
            kptj=second,
        ).total
        d_metric = rsdf_metric_derivative(
            cell,
            auxcell,
            atom_id,
            parameters,
            q=second - first,
        ).total
        for axis in range(3):
            response = differentiate_cderi(
                source,
                metric,
                d_source[axis],
                d_metric[axis],
                nao,
            )
            derivative[atom_id, axis] = response.total
            source_part[atom_id, axis] = response.source
            metric_part[atom_id, axis] = response.metric
    return (
        _real_if_close(solve_cderi(source, metric, nao)),
        _real_if_close(derivative),
        _real_if_close(source_part),
        _real_if_close(metric_part),
    )


def _real_if_close(array, tolerance=1.0e-10):
    value = np.asarray(array)
    if value.size and np.max(np.abs(value.imag)) <= tolerance:
        return value.real
    return value
