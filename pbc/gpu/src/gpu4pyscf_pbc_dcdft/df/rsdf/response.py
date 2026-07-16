"""RSDF host primitives with device-side metric and CDERI response."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import cupy as cp
import numpy as np
from pyscf_pbc_dcdft.df.gdf.storage import Pair, canonical_pairs
from pyscf_pbc_dcdft.df.rsdf.metric import (
    rsdf_metric,
    rsdf_metric_derivative,
)
from pyscf_pbc_dcdft.df.rsdf.parameters import (
    RSDFParameters,
    parameters_from_provider,
)
from pyscf_pbc_dcdft.df.rsdf.source import (
    rsdf_source,
    rsdf_source_derivative,
)

from ..gdf.response import _device_pair_response


@dataclass(frozen=True)
class RSDFResponseTensors:
    """Device RSDF factors and derivatives for canonical k-point pairs."""

    cderi: Mapping[Pair, cp.ndarray]
    dcderi: Mapping[Pair, cp.ndarray]
    source_dcderi: Mapping[Pair, cp.ndarray]
    metric_dcderi: Mapping[Pair, cp.ndarray]
    parameters: RSDFParameters
    value_parity_max_abs: float
    host_primitive_seconds: float
    device_response_seconds: float


def build_rsdf_response_tensors(
    provider,
    *,
    value_tolerance: float = 1.0e-7,
) -> RSDFResponseTensors:
    """Build analytic RSDF ``dCDERI`` with CuPy factor response.

    Host work is limited to periodic derivative primitive generation and the
    CPU witness used for exact value parity.  Cholesky factorization,
    ``dL``, ``dC``, and all downstream J/K contractions are device operations.
    """

    if getattr(provider, "backend_name", None) != "rsdf":
        raise TypeError("build_rsdf_response_tensors requires GPU RSDF")
    if getattr(provider, "_cderi", None) is None:
        provider.build(j_only=False)
    witness = provider.cpu_witness
    if bool(getattr(witness, "j2c_eig_always", False)):
        raise NotImplementedError(
            "GPU RSDF analytic response requires Cholesky metric "
            "factorization; j2c_eig_always=True is unsupported"
        )
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    parameters = parameters_from_provider(witness)

    host_start = perf_counter()
    primitives = {
        pair: _host_pair_primitives(
            witness.cell,
            witness.auxcell,
            kpts,
            pair,
            parameters,
        )
        for pair in canonical_pairs(len(kpts))
    }
    host_seconds = perf_counter() - host_start

    device_start = perf_counter()
    values = {}
    derivatives = {}
    source_parts = {}
    metric_parts = {}
    parity = 0.0
    central = provider.central_cderi_blocks
    for pair, pair_primitives in primitives.items():
        value, total, source_part, metric_part = _device_pair_response(
            pair_primitives, nao=int(provider.cell.nao_nr())
        )
        values[pair] = value
        derivatives[pair] = total
        source_parts[pair] = source_part
        metric_parts[pair] = metric_part
        parity = max(
            parity,
            float(cp.max(cp.abs(value - central[pair])).get()),
        )
    cp.cuda.Stream.null.synchronize()
    device_seconds = perf_counter() - device_start
    if parity > float(value_tolerance):
        raise RuntimeError(
            "GPU RSDF source/metric convention does not reproduce CPU "
            f"storage: max gap {parity:.3e} > {float(value_tolerance):.3e}"
        )
    return RSDFResponseTensors(
        cderi=MappingProxyType(values),
        dcderi=MappingProxyType(derivatives),
        source_dcderi=MappingProxyType(source_parts),
        metric_dcderi=MappingProxyType(metric_parts),
        parameters=parameters,
        value_parity_max_abs=parity,
        host_primitive_seconds=host_seconds,
        device_response_seconds=device_seconds,
    )


def _host_pair_primitives(cell, auxcell, kpts, pair, parameters):
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
    source_derivatives = []
    metric_derivatives = []
    for atom_id in range(cell.natm):
        source_derivatives.append(
            rsdf_source_derivative(
                cell,
                auxcell,
                atom_id,
                parameters,
                kpti=first,
                kptj=second,
            ).total
        )
        metric_derivatives.append(
            rsdf_metric_derivative(
                cell,
                auxcell,
                atom_id,
                parameters,
                q=second - first,
            ).total
        )
    return (
        np.asarray(source),
        np.asarray(metric),
        np.asarray(source_derivatives),
        np.asarray(metric_derivatives),
    )
