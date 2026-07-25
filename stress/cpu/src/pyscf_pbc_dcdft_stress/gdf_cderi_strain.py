"""Analytic Gamma GDF CDERI and fixed-density J/K strain response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gdf_metric_strain import (
    GammaAuxiliaryMetricStrain,
    gamma_auxiliary_metric_strain,
    kpoint_auxiliary_metric_strain,
    matrix_tensor_to_voigt,
)
from .gdf_source_strain import (
    GammaThreeCenterSourceStrain,
    gamma_three_center_source_strain,
    kpoint_three_center_source_strain,
)


@dataclass(frozen=True)
class GammaCDERIStrain:
    """Gamma orthonormalized three-index factors and strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    source_response: np.ndarray
    metric_response: np.ndarray
    source: GammaThreeCenterSourceStrain
    metric: GammaAuxiliaryMetricStrain

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class KPointCDERIStrain:
    """Complex orthonormalized three-index response for one k-point pair."""

    value: np.ndarray
    raw_derivative: np.ndarray
    source_response: np.ndarray
    metric_response: np.ndarray
    source: object
    metric: object
    kpti: np.ndarray
    kptj: np.ndarray
    momentum_transfer: np.ndarray

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class GammaGDFJKStrain:
    """Fixed-density Gamma GDF Coulomb and exchange matrix strain."""

    coulomb: np.ndarray
    exchange: np.ndarray

    @property
    def coulomb_voigt(self):
        return matrix_tensor_to_voigt(self.coulomb)

    @property
    def exchange_voigt(self):
        return matrix_tensor_to_voigt(self.exchange)

    @property
    def restricted_hf(self):
        return self.coulomb - 0.5 * self.exchange


@dataclass(frozen=True)
class KPointGDFJKStrain:
    """Fixed-density sampled-k J/K matrix response to cell strain."""

    coulomb: np.ndarray
    exchange: np.ndarray

    @property
    def coulomb_voigt(self):
        return matrix_tensor_to_voigt(self.coulomb)

    @property
    def exchange_voigt(self):
        return matrix_tensor_to_voigt(self.exchange)

    @property
    def restricted_hf(self):
        return self.coulomb-0.5*self.exchange


@dataclass(frozen=True)
class GammaMadelungStrain:
    """Gamma Ewald probe-charge scalar and analytic cell response."""

    value: float
    raw_derivative: np.ndarray

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


def gamma_cderi_strain(
    cell, auxcell, omega, source_mesh, metric_mesh, *, precision=None,
    rcut=None,
):
    """Differentiate ``C = chol(M)^-1 B`` for a Gamma GDF block."""

    from pyscf_pbc_dcdft.df.gdf.cholesky import differentiate_cderi

    source = gamma_three_center_source_strain(
        cell, auxcell, omega, source_mesh, precision=precision, rcut=rcut
    )
    metric = gamma_auxiliary_metric_strain(
        cell, auxcell, omega, metric_mesh
    )
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    derivative = np.empty((3, 3, naux, nao, nao), dtype=np.complex128)
    source_response = np.empty_like(derivative)
    metric_response = np.empty_like(derivative)
    central = None
    for x in range(3):
        for y in range(3):
            response = differentiate_cderi(
                source.value, metric.value, source.raw_derivative[x, y],
                metric.raw_derivative[x, y], nao,
            )
            if central is None:
                central = response.value
            derivative[x, y] = response.total
            source_response[x, y] = response.source
            metric_response[x, y] = response.metric
    return GammaCDERIStrain(
        value=_real_if_close(central),
        raw_derivative=_real_if_close(derivative),
        source_response=_real_if_close(source_response),
        metric_response=_real_if_close(metric_response),
        source=source,
        metric=metric,
    )


def kpoint_cderi_strain(
    cell, auxcell, omega, source_mesh, metric_mesh, kpti, kptj, *,
    precision=None, rcut=None,
):
    """Differentiate one complex ``C(ki,kj)=chol(M(q))^-1 B(ki,kj)``."""

    from pyscf_pbc_dcdft.df.gdf.cholesky import differentiate_cderi

    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    transfer = kj-ki
    source = kpoint_three_center_source_strain(
        cell, auxcell, omega, source_mesh, ki, kj,
        precision=precision, rcut=rcut,
    )
    metric = kpoint_auxiliary_metric_strain(
        cell, auxcell, omega, metric_mesh, transfer
    )
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    derivative = np.empty((3, 3, naux, nao, nao), dtype=np.complex128)
    source_response = np.empty_like(derivative)
    metric_response = np.empty_like(derivative)
    central = None
    for x in range(3):
        for y in range(3):
            response = differentiate_cderi(
                source.value, metric.value, source.raw_derivative[x, y],
                metric.raw_derivative[x, y], nao,
            )
            if central is None:
                central = response.value
            derivative[x, y] = response.total
            source_response[x, y] = response.source
            metric_response[x, y] = response.metric
    return KPointCDERIStrain(
        value=_real_if_close(central),
        raw_derivative=_real_if_close(derivative),
        source_response=_real_if_close(source_response),
        metric_response=_real_if_close(metric_response),
        source=source,
        metric=metric,
        kpti=ki,
        kptj=kj,
        momentum_transfer=transfer,
    )


def gamma_gdf_jk_strain(cderi_response, density):
    """Contract one Gamma ``C/dC`` pair into fixed-density ``dJ`` and ``dK``."""

    cderi = np.asarray(cderi_response.value)
    derivative = np.asarray(cderi_response.raw_derivative)
    dm = np.asarray(density)
    if dm.ndim == 3 and dm.shape[0] == 1:
        dm = dm[0]
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError(f"Gamma density must be square, got {dm.shape}")
    if cderi.ndim != 3 or cderi.shape[1:] != dm.shape:
        raise ValueError(
            f"CDERI AO dimensions {cderi.shape[1:]} do not match {dm.shape}"
        )
    expected = (3, 3) + cderi.shape
    if derivative.shape != expected:
        raise ValueError(
            f"CDERI strain must have shape {expected}, got {derivative.shape}"
        )

    auxiliary_density = np.einsum(
        "pij,ji->p", cderi, dm, optimize=True
    )
    d_auxiliary_density = np.einsum(
        "xypij,ji->xyp", derivative, dm, optimize=True
    )
    coulomb = (
        np.einsum(
            "xyp,pij->xyij", d_auxiliary_density, cderi, optimize=True
        )
        + np.einsum(
            "p,xypij->xyij", auxiliary_density, derivative, optimize=True
        )
    )
    exchange = (
        np.einsum(
            "xypik,kl,plj->xyij", derivative, dm, cderi,
            optimize=True,
        )
        + np.einsum(
            "pik,kl,xyplj->xyij", cderi, dm, derivative,
            optimize=True,
        )
    )
    return GammaGDFJKStrain(
        coulomb=_real_if_close(coulomb),
        exchange=_real_if_close(exchange),
    )


def kpoint_gdf_jk_strain(cderi_responses, density):
    """Contract canonical sampled-k ``C/dC`` blocks into ``dJ`` and ``dK``.

    ``cderi_responses`` maps canonical ``(ki,kj)`` pairs with ``ki >= kj``
    to objects exposing ``value`` and ``raw_derivative``.  The latter has
    shape ``(3,3,naux,nao,nao)``.  Reverse pairs are reconstructed by
    conjugate AO transpose without additional integral evaluation.
    """

    responses = dict(cderi_responses)
    dm = np.asarray(density)
    if dm.ndim == 2:
        dm = dm[None]
    if dm.ndim != 3 or dm.shape[1] != dm.shape[2]:
        raise ValueError(f"density must have shape (nkpts,nao,nao), got {dm.shape}")
    nkpts, nao, _ = dm.shape
    required = {(ki, kj) for ki in range(nkpts) for kj in range(ki+1)}
    if set(responses) != required:
        missing = sorted(required-set(responses))
        extra = sorted(set(responses)-required)
        raise ValueError(f"canonical CDERI pairs mismatch; missing={missing}, extra={extra}")

    jaux = np.sum([
        np.einsum("pij,ji->p", _pair_value(responses, k, k), dm[k], optimize=True)
        for k in range(nkpts)
    ], axis=0)
    coulomb = np.zeros((3, 3, nkpts, nao, nao), dtype=np.complex128)
    exchange = np.zeros_like(coulomb)
    for x in range(3):
        for y in range(3):
            djaux = np.sum([
                np.einsum(
                    "pij,ji->p", _pair_derivative(responses, k, k, x, y),
                    dm[k], optimize=True,
                )
                for k in range(nkpts)
            ], axis=0)
            for ki in range(nkpts):
                c_ii = _pair_value(responses, ki, ki)
                dc_ii = _pair_derivative(responses, ki, ki, x, y)
                coulomb[x, y, ki] = (
                    np.einsum("p,pij->ij", jaux, dc_ii, optimize=True)
                    + np.einsum("p,pij->ij", djaux, c_ii, optimize=True)
                ) / nkpts
                for kj in range(nkpts):
                    c_ij = _pair_value(responses, ki, kj)
                    c_ji = _pair_value(responses, kj, ki)
                    dc_ij = _pair_derivative(responses, ki, kj, x, y)
                    dc_ji = _pair_derivative(responses, kj, ki, x, y)
                    exchange[x, y, ki] += np.einsum(
                        "pij,jk,pkl->il", dc_ij, dm[kj], c_ji,
                        optimize=True,
                    )
                    exchange[x, y, ki] += np.einsum(
                        "pij,jk,pkl->il", c_ij, dm[kj], dc_ji,
                        optimize=True,
                    )
                exchange[x, y, ki] /= nkpts
    return KPointGDFJKStrain(
        coulomb=_real_if_close(coulomb),
        exchange=_real_if_close(exchange),
    )


def gamma_madelung_strain(cell):
    """Differentiate PySCF's full-range Gamma Ewald exchange scalar."""

    from pyscf.pbc.tools import pbc as pbc_tools

    from .ewald_strain import analytic_ewald_strain

    probe = cell.copy(deep=False)
    probe._atm = np.asarray(
        [[1, cell._env.size, 0, 0, 0, 0]], dtype=np.int32
    )
    probe._env = np.append(cell._env, (0.0, 0.0, 0.0))
    probe.unit = "B"
    probe.a = np.asarray(cell.lattice_vectors(), dtype=float)
    derivative = -2.0 * analytic_ewald_strain(probe).raw_derivative
    value = float(pbc_tools.madelung(cell, np.zeros((1, 3))))
    return GammaMadelungStrain(
        value=value, raw_derivative=np.asarray(derivative, dtype=float)
    )


def kpoint_madelung_strain(cell, kpts):
    """Differentiate PySCF's Monkhorst-Pack probe-supercell Madelung scalar."""

    from pyscf.pbc.tools import pbc as pbc_tools

    from .ewald_strain import analytic_ewald_strain

    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    copies = np.asarray(
        pbc_tools.get_monkhorst_pack_size(cell, points), dtype=int
    )
    probe = cell.copy(deep=False)
    probe._atm = np.asarray(
        [[1, cell._env.size, 0, 0, 0, 0]], dtype=np.int32
    )
    probe._env = np.append(cell._env, (0.0, 0.0, 0.0))
    probe.unit = "B"
    probe.a = np.einsum(
        "xi,x->xi", np.asarray(cell.lattice_vectors()), copies
    )
    derivative = -2.0*analytic_ewald_strain(probe).raw_derivative
    value = float(pbc_tools.madelung(cell, points))
    return GammaMadelungStrain(
        value=value, raw_derivative=np.asarray(derivative, dtype=float)
    )


def gamma_ewald_exchange_strain(cell, density):
    """Return the cell derivative of ``madelung * S D S`` at Gamma."""

    from .one_electron_strain import gamma_overlap_strain

    dm = np.asarray(density)
    if dm.ndim == 3 and dm.shape[0] == 1:
        dm = dm[0]
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError(f"Gamma density must be square, got {dm.shape}")
    overlap = np.asarray(cell.pbc_intor("int1e_ovlp", hermi=1))
    overlap_derivative = gamma_overlap_strain(cell).raw_derivative
    madelung = gamma_madelung_strain(cell)
    central = overlap @ dm @ overlap
    derivative = (
        madelung.raw_derivative[:, :, None, None] * central[None, None]
        + madelung.value * (
            np.einsum(
                "xyik,kl,lj->xyij", overlap_derivative, dm, overlap,
                optimize=True,
            )
            + np.einsum(
                "ik,kl,xylj->xyij", overlap, dm, overlap_derivative,
                optimize=True,
            )
        )
    )
    return _real_if_close(derivative)


def kpoint_ewald_exchange_strain(cell, kpts, density):
    """Return ``d[m S_k D_k S_k]/d strain`` for sampled k points."""

    from .one_electron_strain import kpoint_one_electron_strain

    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    dm = np.asarray(density)
    if dm.ndim == 2:
        dm = dm[None]
    nao = int(cell.nao_nr())
    if dm.shape != (len(points), nao, nao):
        raise ValueError(
            f"density must have shape {(len(points), nao, nao)}, got {dm.shape}"
        )
    overlap = np.asarray(
        cell.pbc_intor("int1e_ovlp", hermi=1, kpts=points)
    )
    overlap_derivative = kpoint_one_electron_strain(
        cell, points, "int1e_ovlp"
    ).raw_derivative
    madelung = kpoint_madelung_strain(cell, points)
    output = np.empty(
        (3, 3, len(points), nao, nao), dtype=np.complex128
    )
    for kpoint_id in range(len(points)):
        central = overlap[kpoint_id]@dm[kpoint_id]@overlap[kpoint_id]
        output[:, :, kpoint_id] = (
            madelung.raw_derivative[:, :, None, None]
            * central[None, None]
            + madelung.value*(
                np.einsum(
                    "xyik,kl,lj->xyij", overlap_derivative[:, :, kpoint_id],
                    dm[kpoint_id], overlap[kpoint_id], optimize=True,
                )
                + np.einsum(
                    "ik,kl,xylj->xyij", overlap[kpoint_id], dm[kpoint_id],
                    overlap_derivative[:, :, kpoint_id], optimize=True,
                )
            )
        )
    return _real_if_close(output)


def _real_if_close(value, tolerance=1e-10):
    array = np.asarray(value)
    if array.size and np.max(np.abs(array.imag)) <= tolerance:
        return array.real
    return array


def _kpoint(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"k-point must have shape (3,), got {point.shape}")
    return point


def _pair_value(responses, ki, kj):
    key = (int(ki), int(kj))
    if key in responses:
        return np.asarray(responses[key].value)
    reverse = (key[1], key[0])
    return np.asarray(responses[reverse].value).conj().transpose(0, 2, 1)


def _pair_derivative(responses, ki, kj, x, y):
    key = (int(ki), int(kj))
    if key in responses:
        return np.asarray(responses[key].raw_derivative)[x, y]
    reverse = (key[1], key[0])
    return (
        np.asarray(responses[reverse].raw_derivative)[x, y]
        .conj().transpose(0, 2, 1)
    )
