"""Analytic Gamma RSDF CDERI homogeneous-strain response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gdf_metric_strain import matrix_tensor_to_voigt
from .gdf_source_strain import (
    gamma_three_center_source_strain,
    kpoint_three_center_source_strain,
)
from .rsdf_metric_strain import (
    gamma_rsdf_metric_strain,
    kpoint_rsdf_metric_strain,
)


@dataclass(frozen=True)
class GammaRSDFCDERIStrain:
    """Stored Gamma RSDF factors and exact-representation strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    source_response: np.ndarray
    metric_response: np.ndarray
    source: object
    metric: object
    parameters: object
    reconstructed_value_parity_max_abs: float

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class KPointRSDFCDERIStrain:
    """Stored complex RSDF factor and strain response for one canonical pair."""

    value: np.ndarray
    raw_derivative: np.ndarray
    source_response: np.ndarray
    metric_response: np.ndarray
    source: object
    metric: object
    parameters: object
    pair: tuple[int, int]
    reconstructed_value_parity_max_abs: float

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


def gamma_rsdf_cderi_strain(provider, *, value_tolerance=1e-7):
    """Differentiate the Gamma RSDF factor used by a built provider."""

    from pyscf_pbc_dcdft.df.gdf.cholesky import differentiate_cderi
    from pyscf_pbc_dcdft.df.gdf.storage import load_cderi_storage
    from pyscf_pbc_dcdft.df.rsdf import parameters_from_provider

    if getattr(provider, "backend_name", None) != "rsdf":
        raise TypeError("gamma_rsdf_cderi_strain requires the project RSDF provider")
    if bool(getattr(provider, "j2c_eig_always", False)):
        raise NotImplementedError(
            "Gamma RSDF stress currently requires Cholesky metric factorization"
        )
    parameters = parameters_from_provider(provider)
    cell = provider.cell
    auxcell = provider.auxcell
    source = gamma_three_center_source_strain(
        cell, auxcell, parameters.source_omega, parameters.source_mesh,
        precision=parameters.precision_r,
    )
    metric = gamma_rsdf_metric_strain(cell, auxcell, parameters)
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    shape = (3, 3, naux, nao, nao)
    derivative = np.empty(shape, dtype=np.complex128)
    source_response = np.empty(shape, dtype=np.complex128)
    metric_response = np.empty(shape, dtype=np.complex128)
    reconstructed = None
    for x in range(3):
        for y in range(3):
            response = differentiate_cderi(
                source.value, metric.value, source.raw_derivative[x, y],
                metric.raw_derivative[x, y], nao,
            )
            if reconstructed is None:
                reconstructed = response.value
            derivative[x, y] = response.total
            source_response[x, y] = response.source
            metric_response[x, y] = response.metric

    stored = np.asarray(load_cderi_storage(provider).blocks[(0, 0)])
    parity = float(np.max(np.abs(np.asarray(reconstructed) - stored)))
    if parity > float(value_tolerance):
        raise RuntimeError(
            "reconstructed RSDF factor does not match central storage: "
            f"{parity:.3e} > {float(value_tolerance):.3e}"
        )
    return GammaRSDFCDERIStrain(
        value=_real_if_close(stored),
        raw_derivative=_real_if_close(derivative),
        source_response=_real_if_close(source_response),
        metric_response=_real_if_close(metric_response),
        source=source,
        metric=metric,
        parameters=parameters,
        reconstructed_value_parity_max_abs=parity,
    )


def kpoint_rsdf_cderi_strain(
    provider, ki, kj, *, value_tolerance=1e-7
):
    """Differentiate one canonical sampled-k RSDF factor from a provider."""

    from pyscf_pbc_dcdft.df.gdf.cholesky import differentiate_cderi
    from pyscf_pbc_dcdft.df.gdf.storage import load_cderi_storage
    from pyscf_pbc_dcdft.df.rsdf import parameters_from_provider

    if getattr(provider, "backend_name", None) != "rsdf":
        raise TypeError("kpoint_rsdf_cderi_strain requires the project RSDF provider")
    if bool(getattr(provider, "j2c_eig_always", False)):
        raise NotImplementedError(
            "sampled-k RSDF stress currently requires Cholesky metric factorization"
        )
    pair = (int(ki), int(kj))
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    if pair[0] < pair[1] or pair[0] >= len(kpts) or pair[1] < 0:
        raise ValueError(f"invalid canonical k-point pair {pair}")
    parameters = parameters_from_provider(provider)
    cell = provider.cell
    auxcell = provider.auxcell
    kpti, kptj = kpts[list(pair)]
    source = kpoint_three_center_source_strain(
        cell, auxcell, parameters.source_omega, parameters.source_mesh,
        kpti, kptj, precision=parameters.precision_r,
    )
    metric = kpoint_rsdf_metric_strain(
        cell, auxcell, parameters, kptj-kpti
    )
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    shape = (3, 3, naux, nao, nao)
    derivative = np.empty(shape, dtype=np.complex128)
    source_response = np.empty(shape, dtype=np.complex128)
    metric_response = np.empty(shape, dtype=np.complex128)
    reconstructed = None
    for x in range(3):
        for y in range(3):
            response = differentiate_cderi(
                source.value, metric.value, source.raw_derivative[x, y],
                metric.raw_derivative[x, y], nao,
            )
            if reconstructed is None:
                reconstructed = response.value
            derivative[x, y] = response.total
            source_response[x, y] = response.source
            metric_response[x, y] = response.metric

    stored = np.asarray(load_cderi_storage(provider, pairs=(pair,)).blocks[pair])
    parity = float(np.max(np.abs(np.asarray(reconstructed)-stored)))
    if parity > float(value_tolerance):
        raise RuntimeError(
            "reconstructed sampled-k RSDF factor does not match storage: "
            f"{parity:.3e} > {float(value_tolerance):.3e}"
        )
    return KPointRSDFCDERIStrain(
        value=_real_if_close(stored),
        raw_derivative=_real_if_close(derivative),
        source_response=_real_if_close(source_response),
        metric_response=_real_if_close(metric_response),
        source=source,
        metric=metric,
        parameters=parameters,
        pair=pair,
        reconstructed_value_parity_max_abs=parity,
    )


def _real_if_close(value, tolerance=1e-10):
    array = np.asarray(value)
    if array.size and np.max(np.abs(array.imag)) <= tolerance:
        return array.real
    return array
