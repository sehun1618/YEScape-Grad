"""Stable Gamma metadata for project-local native GDF response kernels."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.df import rsdf_builder

from .build_signature import GDFBuildSignature, make_gdf_build_signature


@dataclass(frozen=True)
class GammaFactorizationMetadata:
    """Native central-factor convention needed by Gamma reverse response.

    The upstream CUDA builder keeps compressed CDERI factors and AO-pair
    indices on the provider, but discards its metric whitening transform.
    This object reconstructs that small transform once and gives later
    project-local derivative code one versioned ownership boundary.
    """

    build_signature: GDFBuildSignature
    compressed_cderi: Mapping[int, object]
    pair_address: cp.ndarray
    diagonal_indices: cp.ndarray
    auxiliary_transform: cp.ndarray
    target_omega: float
    factorization_omega: float
    nao: int
    naux: int

    def summary(self) -> dict[str, object]:
        """Return JSON-safe shape and convention metadata."""

        return {
            "representation": "native_gamma_compressed",
            "nao": int(self.nao),
            "naux": int(self.naux),
            "compressed_block_shapes": {
                str(key): list(np.shape(value))
                for key, value in self.compressed_cderi.items()
            },
            "pair_count": int(self.pair_address.size),
            "diagonal_pair_count": int(self.diagonal_indices.size),
            "auxiliary_transform_shape": list(self.auxiliary_transform.shape),
            "target_omega": float(self.target_omega),
            "factorization_omega": float(self.factorization_omega),
        }


def build_gamma_factorization_metadata(provider) -> GammaFactorizationMetadata:
    """Capture the exact native Gamma factor convention without dense CDERI."""

    if not bool(getattr(provider, "uses_native_integrals", False)):
        raise RuntimeError("Gamma native metadata requires setup_mode='native'")
    provider._ensure_built()
    points = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    if len(points) != 1 or float(np.max(np.abs(points))) > 1.0e-12:
        raise NotImplementedError(
            "native factorization metadata currently supports Gamma only"
        )

    target_omega = abs(float(getattr(provider, "_omega", 0.0) or 0.0))
    factorization_omega = max(
        target_omega,
        float(rsdf_builder._guess_omega(provider.cell)),
    )
    transforms, negative_sizes = rsdf_builder._precontract_j2c_aux_coeff(
        provider.auxcell,
        None,
        target_omega,
        factorization_omega,
        float(provider.linear_dep_threshold),
    )
    if len(transforms) != 1:
        raise RuntimeError(
            "Gamma GDF expected one auxiliary transform, got "
            f"{len(transforms)}"
        )
    if negative_sizes:
        raise NotImplementedError(
            "negative-metric Gamma factors are not supported by the 3D route"
        )
    compressed = MappingProxyType(dict(provider._cderi))
    pair_address, diagonal = provider._cderi_idx
    return GammaFactorizationMetadata(
        build_signature=make_gdf_build_signature(provider),
        compressed_cderi=compressed,
        pair_address=cp.asarray(pair_address),
        diagonal_indices=cp.asarray(diagonal),
        auxiliary_transform=cp.asarray(transforms[0]),
        target_omega=target_omega,
        factorization_omega=factorization_omega,
        nao=int(provider.cell.nao_nr()),
        naux=int(provider.get_naoaux()),
    )
