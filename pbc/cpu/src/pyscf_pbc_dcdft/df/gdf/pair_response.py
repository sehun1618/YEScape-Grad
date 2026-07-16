"""Analytic response of one periodic GDF CDERI k-point pair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder

from .cholesky import differentiate_cderi
from .metric import (
    auxiliary_metric,
    auxiliary_metric_derivative,
    recommended_metric_mesh,
)
from .source import three_center_source, three_center_source_derivative


@dataclass(frozen=True)
class RSGDFParameters:
    """Numerical parameters which define one PySCF RSGDF factorization."""

    omega: float
    source_mesh: tuple[int, int, int]
    metric_mesh: tuple[int, int, int]


@dataclass(frozen=True)
class PairCDERIResponse:
    """Central CDERI and all atomic Cartesian derivatives for one k-pair."""

    value: np.ndarray
    derivative: np.ndarray
    source_derivative: np.ndarray
    metric_derivative: np.ndarray
    parameters: RSGDFParameters


def rsgdf_parameters(
    cell,
    auxcell,
    kpts: np.ndarray,
    *,
    source_mesh: tuple[int, int, int] | np.ndarray | None = None,
    linear_dep_threshold: float = 1.0e-10,
) -> RSGDFParameters:
    """Resolve the same omega/mesh policy as PySCF's CPU GDF builder."""

    points = _kpoints(kpts)
    builder = _RSGDFBuilder(cell, auxcell, points)
    if source_mesh is not None:
        builder.mesh = np.asarray(source_mesh, dtype=int)
    builder.linear_dep_threshold = float(linear_dep_threshold)
    builder.build()
    metric_mesh = recommended_metric_mesh(
        cell, auxcell, float(builder.omega)
    )
    return RSGDFParameters(
        omega=float(builder.omega),
        source_mesh=tuple(int(value) for value in builder.mesh),
        metric_mesh=tuple(int(value) for value in metric_mesh),
    )


def pair_cderi_response(
    cell,
    auxcell,
    kpts: np.ndarray,
    pair: tuple[int, int],
    *,
    parameters: RSGDFParameters | None = None,
) -> PairCDERIResponse:
    """Return analytic ``C`` and ``dC/dR`` for one canonical k-point pair.

    ``C = L^-1 B`` and

    ``dC = L^-1 dB - L^-1 dL C``.

    The second contribution is retained separately as ``metric_derivative`` so
    callers can audit the auxiliary response instead of treating CDERI as a
    geometry-independent fitting table.
    """

    points = _kpoints(kpts)
    ki, kj = _pair(pair, len(points))
    settings = parameters or rsgdf_parameters(cell, auxcell, points)
    first = points[ki]
    second = points[kj]
    momentum_transfer = second - first
    source = three_center_source(
        cell,
        auxcell,
        settings.omega,
        mesh=settings.source_mesh,
        kpti=first,
        kptj=second,
    )
    metric = auxiliary_metric(
        cell,
        auxcell,
        settings.omega,
        mesh=settings.metric_mesh,
        q=momentum_transfer,
    )

    natm = int(cell.natm)
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    derivative = np.empty(
        (natm, 3, naux, nao, nao), dtype=np.complex128
    )
    source_part = np.empty_like(derivative)
    metric_part = np.empty_like(derivative)
    central_value = None
    for atom_id in range(natm):
        d_source = three_center_source_derivative(
            cell,
            auxcell,
            atom_id,
            settings.omega,
            mesh=settings.source_mesh,
            kpti=first,
            kptj=second,
        ).total
        d_metric = auxiliary_metric_derivative(
            cell,
            auxcell,
            atom_id,
            settings.omega,
            mesh=settings.metric_mesh,
            q=momentum_transfer,
        ).total
        for axis in range(3):
            response = differentiate_cderi(
                source,
                metric,
                d_source[axis],
                d_metric[axis],
                nao,
            )
            if central_value is None:
                central_value = response.value
            derivative[atom_id, axis] = response.total
            source_part[atom_id, axis] = response.source
            metric_part[atom_id, axis] = response.metric

    return PairCDERIResponse(
        value=np.asarray(central_value),
        derivative=_real_if_close(derivative),
        source_derivative=_real_if_close(source_part),
        metric_derivative=_real_if_close(metric_part),
        parameters=settings,
    )


def _kpoints(value):
    points = np.asarray(value, dtype=float).reshape(-1, 3)
    if len(points) < 1:
        raise ValueError("at least one k-point is required")
    return points


def _pair(value, nkpts):
    if len(value) != 2:
        raise ValueError(f"pair must contain two indices, got {value}")
    ki, kj = (int(value[0]), int(value[1]))
    if not (0 <= kj <= ki < nkpts):
        raise ValueError(
            f"pair must be canonical with 0 <= kj <= ki < {nkpts}, got {value}"
        )
    return ki, kj


def _real_if_close(array, tolerance=1.0e-10):
    value = np.asarray(array)
    if value.size and np.max(np.abs(value.imag)) <= tolerance:
        return value.real
    return value
