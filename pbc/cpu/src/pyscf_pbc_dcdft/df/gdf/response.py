"""Complete CPU GDF value/derivative tensors for all canonical k-pairs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .pair_response import (
    RSGDFParameters,
    pair_cderi_response,
    rsgdf_parameters,
)
from .storage import Pair, canonical_pairs, load_cderi_storage


@dataclass(frozen=True)
class GDFResponseTensors:
    """Central CDERI and analytic derivatives in canonical pair storage."""

    cderi: Mapping[Pair, np.ndarray]
    dcderi: Mapping[Pair, np.ndarray]
    source_dcderi: Mapping[Pair, np.ndarray]
    metric_dcderi: Mapping[Pair, np.ndarray]
    parameters: RSGDFParameters
    value_parity_max_abs: float


def build_gdf_response_tensors(
    provider,
    *,
    value_tolerance: float = 1.0e-7,
) -> GDFResponseTensors:
    """Build all analytic ``dCDERI`` blocks for a PySCF GDF provider.

    Central values are retained from PySCF's HDF5 storage because those are
    exactly the factors used by the converged HF calculation.  Independently
    reconstructed values are compared against storage before their analytic
    derivatives are accepted.  This guard catches mismatched omega, mesh,
    auxiliary basis, k-pair orientation, or Cholesky convention.
    """

    storage = load_cderi_storage(provider)
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    parameters = rsgdf_parameters(
        provider.cell,
        provider.auxcell,
        kpts,
        source_mesh=getattr(provider, "mesh", None),
        linear_dep_threshold=float(provider.linear_dep_threshold),
    )
    derivatives = {}
    source_parts = {}
    metric_parts = {}
    parity = 0.0
    for pair in canonical_pairs(len(kpts)):
        response = pair_cderi_response(
            provider.cell,
            provider.auxcell,
            kpts,
            pair,
            parameters=parameters,
        )
        gap = float(
            np.max(np.abs(response.value - storage.blocks[pair]))
        )
        parity = max(parity, gap)
        derivatives[pair] = np.asarray(response.derivative)
        source_parts[pair] = np.asarray(response.source_derivative)
        metric_parts[pair] = np.asarray(response.metric_derivative)
    if parity > float(value_tolerance):
        raise RuntimeError(
            "analytic GDF source/metric convention does not reproduce "
            f"PySCF CDERI storage: max gap {parity:.3e} > "
            f"{float(value_tolerance):.3e}"
        )
    return GDFResponseTensors(
        cderi=MappingProxyType(dict(storage.blocks)),
        dcderi=MappingProxyType(derivatives),
        source_dcderi=MappingProxyType(source_parts),
        metric_dcderi=MappingProxyType(metric_parts),
        parameters=parameters,
        value_parity_max_abs=parity,
    )
