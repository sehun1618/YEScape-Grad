"""Analytic Gamma AO-pair Fourier strain for the GDF reciprocal source."""

from __future__ import annotations

import ctypes
import importlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, lib
from pyscf.scf import _vhf
from pyscf.gto import ft_ao as molecular_ft_ao
from pyscf.gto import moleintor
from pyscf.gto.moleintor import libcgto
from pyscf.pbc.df import rsdf, rsdf_helper

from .gdf_metric_strain import matrix_tensor_to_voigt
from .gdf_metric_strain import (
    auxiliary_fourier_strain,
    weighted_coulomb_strain,
)


_DEFAULT_RECIPROCAL_BLOCK_BYTES = 8 * (1 << 30)
_FACTORIZED_PAIR_RESPONSE_THRESHOLD_BYTES = 256 * (1 << 20)


def _as_voigt_derivative(value):
    """Return an existing compact derivative or project a Cartesian tensor."""

    derivative = np.asarray(value)
    if derivative.ndim == 3 and derivative.shape[0] == 6:
        return derivative
    return matrix_tensor_to_voigt(derivative)


@dataclass(frozen=True)
class GammaAOPairFourierStrain:
    """Image-resolved periodic AO-pair Fourier value and strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    lattice_vectors: int
    rcut_bohr: float

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class KPointAOPairFourierStrain:
    """Periodic AO-pair Fourier response for one ordered k-point pair."""

    value: np.ndarray
    raw_derivative: np.ndarray
    kpti: np.ndarray
    kptj: np.ndarray
    momentum_transfer: np.ndarray
    lattice_vectors: int
    rcut_bohr: float

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class GammaLongRangeSourceStrain:
    """Resolved analytic long-range three-center source strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    ao_pair_response: np.ndarray | None
    auxiliary_response: np.ndarray | None
    kernel_response: np.ndarray | None
    reciprocal_block_size: int
    grid_points: int
    estimated_block_work_bytes: int
    timings_s: dict[str, float]
    pair_response_mode: str
    pair_response_image_bytes: int

    @property
    def voigt_derivative(self):
        return _as_voigt_derivative(self.raw_derivative)


@dataclass(frozen=True)
class KPointLongRangeSourceStrain:
    """Resolved long-range source strain for one ordered k-point pair."""

    value: np.ndarray
    raw_derivative: np.ndarray
    ao_pair_response: np.ndarray | None
    auxiliary_response: np.ndarray | None
    kernel_response: np.ndarray | None
    kpti: np.ndarray
    kptj: np.ndarray
    momentum_transfer: np.ndarray
    reciprocal_block_size: int
    grid_points: int
    estimated_block_work_bytes: int
    timings_s: dict[str, float]
    pair_response_mode: str
    pair_response_image_bytes: int

    @property
    def voigt_derivative(self):
        return _as_voigt_derivative(self.raw_derivative)


@dataclass(frozen=True)
class KPointLongRangeSourceAdjointStrain:
    """Direct contractions of long-range source strain with adjoints."""

    values: np.ndarray
    kpti: np.ndarray
    kptj: np.ndarray
    reciprocal_block_size: int
    grid_points: int
    estimated_block_work_bytes: int
    wall_seconds: float
    full_response_bytes_avoided: int


@dataclass(frozen=True)
class GammaVBarSourceStrain:
    """Analytic zero-transfer vbar source and resolved strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    overlap_response: np.ndarray
    volume_response: np.ndarray

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class KPointVBarSourceStrain:
    """Zero-transfer vbar source response for one ordered k-point pair."""

    value: np.ndarray
    raw_derivative: np.ndarray
    overlap_response: np.ndarray
    volume_response: np.ndarray
    kpti: np.ndarray
    kptj: np.ndarray

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class GammaShortRangeSourceStrain:
    """Screened image-resolved short-range three-center source strain."""

    value: np.ndarray
    raw_derivative: np.ndarray
    lattice_vectors: int
    precision: float

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class KPointShortRangeSourceStrain:
    """Complex short-range source response for one ordered k-point pair."""

    value: np.ndarray
    raw_derivative: np.ndarray
    kpti: np.ndarray
    kptj: np.ndarray
    lattice_vectors: int
    precision: float

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class GammaThreeCenterSourceStrain:
    """Complete Gamma pre-Cholesky GDF source strain decomposition."""

    value: np.ndarray
    raw_derivative: np.ndarray
    short_range: GammaShortRangeSourceStrain | None
    long_range: GammaLongRangeSourceStrain | None
    vbar: GammaVBarSourceStrain | None
    reciprocal_block_size: int
    reciprocal_grid_points: int
    estimated_block_work_bytes: int
    timings_s: dict[str, float]
    pair_response_mode: str
    pair_response_image_bytes: int

    @property
    def voigt_derivative(self):
        return _as_voigt_derivative(self.raw_derivative)


@dataclass(frozen=True)
class KPointThreeCenterSourceStrain:
    """Complete complex pre-Cholesky source response for one k pair."""

    value: np.ndarray
    raw_derivative: np.ndarray
    short_range: KPointShortRangeSourceStrain | None
    long_range: KPointLongRangeSourceStrain | None
    vbar: KPointVBarSourceStrain | None
    kpti: np.ndarray
    kptj: np.ndarray
    reciprocal_block_size: int
    reciprocal_grid_points: int
    estimated_block_work_bytes: int
    timings_s: dict[str, float]
    pair_response_mode: str
    pair_response_image_bytes: int

    @property
    def voigt_derivative(self):
        return _as_voigt_derivative(self.raw_derivative)


@dataclass(frozen=True)
class KPointThreeCenterSourceAdjointStrain:
    """Exact complete-source strain contracted into scalar observables."""

    values: np.ndarray
    kpti: np.ndarray
    kptj: np.ndarray
    reciprocal_block_size: int
    reciprocal_grid_points: int
    estimated_block_work_bytes: int
    timings_s: dict[str, float]
    memory_estimate: dict[str, int]


@dataclass(frozen=True)
class _AOPairSourceImage:
    """One immutable translated AO-pair integral context."""

    molecule: object
    shell_slice: tuple[int, int, int, int]
    phase: complex
    shifted_centers: np.ndarray


@dataclass(frozen=True)
class _AOPairSourceContext:
    """Geometry and image data reused by every reciprocal block of one pair."""

    transfer: np.ndarray
    centers: np.ndarray
    nao: int
    images: tuple[_AOPairSourceImage, ...]


def gamma_ao_pair_fourier_strain(cell, grid_vectors, *, rcut=None):
    r"""Differentiate the Gamma periodic AO-pair Fourier tensor.

    The raw output is ordered ``(strain_x,strain_y,G,AO_i,AO_j)``.  For each
    lattice image, center derivatives move the two Gaussian centers while the
    Fourier moment accounts for the contravariant reciprocal-vector response.
    """

    general = kpoint_ao_pair_fourier_strain(
        cell, grid_vectors, np.zeros(3), np.zeros(3), rcut=rcut
    )
    return GammaAOPairFourierStrain(
        value=general.value,
        raw_derivative=general.raw_derivative,
        lattice_vectors=general.lattice_vectors,
        rcut_bohr=general.rcut_bohr,
    )


def kpoint_ao_pair_fourier_strain(
    cell, grid_vectors, kpti, kptj, *, rcut=None
):
    r"""Differentiate a periodic AO-pair Fourier tensor at fixed scaled k.

    ``grid_vectors`` contains reciprocal lattice vectors ``G``.  The
    molecular Fourier integrals are evaluated at ``G + q`` with
    ``q = k_j-k_i`` and each ket image carries ``exp(+i k_j L)``.  Both
    ``(G+q).R`` and ``k_j.L`` remain invariant when the lattice, reciprocal
    vectors, and absolute k points are strained together.
    """

    vectors = np.asarray(grid_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"grid_vectors must have shape (nG,3), got {vectors.shape}")
    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    transfer = kj - ki
    momentum = vectors + transfer
    cutoff = float(cell.rcut if rcut is None else rcut)
    if cutoff <= 0:
        raise ValueError("AO-pair lattice cutoff must be positive")
    lattice_vectors = np.asarray(cell.get_lattice_Ls(rcut=cutoff), dtype=float)
    centers = _ao_centers(cell)
    nao = int(cell.nao_nr())
    value = np.zeros((len(vectors), nao, nao), dtype=np.complex128)
    derivative = np.zeros(
        (3, 3, len(vectors), nao, nao), dtype=np.complex128
    )

    for lattice in lattice_vectors:
        shifted = cell.copy(deep=True)
        shifted.set_geom_(cell.atom_coords() + lattice, unit="Bohr")
        combined = gto.conc_mol(cell, shifted)
        shell_slice = (0, cell.nbas, cell.nbas, 2 * cell.nbas)
        pair, center_i, center_j = _molecular_ft_overlap_center_derivatives(
            combined, momentum, shell_slice
        )
        moment = molecular_ft_ao.ft_aopair(
            combined,
            momentum,
            shls_slice=shell_slice,
            aosym="s1",
            intor="GTO_ft_rc",
            comp=3,
        )
        phase = np.exp(1j * np.dot(kj, lattice))
        value += phase * pair
        shifted_centers = centers + lattice
        image_derivative = -np.einsum(
            "xgij,iy->xygij", center_i, centers, optimize=True
        )
        image_derivative -= np.einsum(
            "xgij,jy->xygij", center_j, shifted_centers, optimize=True
        )
        image_derivative += 1j * np.einsum(
            "gx,ygij->xygij", momentum, moment, optimize=True
        )
        derivative += phase * image_derivative
    return KPointAOPairFourierStrain(
        value=value,
        raw_derivative=derivative,
        kpti=ki,
        kptj=kj,
        momentum_transfer=transfer,
        lattice_vectors=int(len(lattice_vectors)),
        rcut_bohr=cutoff,
    )


def gamma_long_range_source_strain(
    cell, auxcell, omega, mesh, *, rcut=None, reciprocal_block_size=None,
    retain_components=True, compact_voigt=False, pair_response_matmul=None,
    pair_response_mode="assembled",
    factorized_projection=None,
):
    """Return the analytic Gamma long-range GDF three-center source strain."""

    general = kpoint_long_range_source_strain(
        cell, auxcell, omega, mesh, np.zeros(3), np.zeros(3), rcut=rcut,
        reciprocal_block_size=reciprocal_block_size,
        retain_components=retain_components, compact_voigt=compact_voigt,
        pair_response_matmul=pair_response_matmul,
        pair_response_mode=pair_response_mode,
        factorized_projection=factorized_projection,
    )
    return GammaLongRangeSourceStrain(
        value=general.value,
        raw_derivative=general.raw_derivative,
        ao_pair_response=general.ao_pair_response,
        auxiliary_response=general.auxiliary_response,
        kernel_response=general.kernel_response,
        reciprocal_block_size=general.reciprocal_block_size,
        grid_points=general.grid_points,
        estimated_block_work_bytes=general.estimated_block_work_bytes,
        timings_s=general.timings_s,
        pair_response_mode=general.pair_response_mode,
        pair_response_image_bytes=general.pair_response_image_bytes,
    )


def kpoint_long_range_source_strain(
    cell, auxcell, omega, mesh, kpti, kptj, *, rcut=None,
    reciprocal_block_size=None, retain_components=True, compact_voigt=False,
    pair_response_matmul=None, pair_response_mode="assembled",
    factorized_projection=None,
):
    """Return the analytic long-range source for one ordered k pair.

    The AO-pair strain has shape ``(3,3,nG,nao,nao)``.  Materializing it for
    the complete reciprocal mesh makes the peak host memory proportional to
    ``9*nG*nao**2`` and is prohibitive for production cells.  This routine
    therefore contracts independent G-vector blocks directly into the final
    three-center source.  ``retain_components=False`` additionally avoids
    three audit-only copies of the final response.
    """

    total_started = perf_counter()
    timings = {
        "auxiliary_kernel_build": 0.0,
        "ao_fourier_integrals": 0.0,
        "ao_pair_response_assembly": 0.0,
        "ao_pair_response_matmul": 0.0,
        "ao_pair_response_contraction": 0.0,
        "central_source_contraction": 0.0,
        "auxiliary_kernel_response_contraction": 0.0,
    }
    if compact_voigt and retain_components:
        raise ValueError(
            "compact_voigt requires retain_components=False because the "
            "three audit tensors use Cartesian strain layout"
        )
    if pair_response_mode not in {"auto", "assembled", "factorized"}:
        raise ValueError(
            "pair_response_mode must be 'auto', 'assembled', or 'factorized'"
        )
    mesh_array = np.asarray(mesh, dtype=int)
    if mesh_array.shape != (3,) or np.any(mesh_array <= 0):
        raise ValueError(f"mesh must contain three positive integers: {mesh_array}")
    grid_vectors = cell.get_Gv(mesh_array)
    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    transfer = kj - ki
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    pair_response_image_bytes = int(
        len(grid_vectors)*nao*nao*np.dtype(np.complex128).itemsize
    )
    if pair_response_mode == "auto":
        pair_response_mode = (
            "factorized"
            if pair_response_image_bytes
            >= _FACTORIZED_PAIR_RESPONSE_THRESHOLD_BYTES
            else "assembled"
        )
    if pair_response_mode == "factorized" and not compact_voigt:
        raise ValueError(
            "factorized pair response currently requires compact_voigt=True"
        )
    if pair_response_mode == "factorized" and pair_response_matmul is not None:
        raise ValueError(
            "factorized pair response owns its contractions and does not "
            "accept pair_response_matmul"
        )
    if pair_response_mode != "factorized" and factorized_projection is not None:
        raise ValueError(
            "factorized_projection requires pair_response_mode='factorized'"
        )
    block_size, estimated_block_bytes = _reciprocal_block_size(
        reciprocal_block_size, grid_points=len(grid_vectors), nao=nao,
        naux=naux,
    )
    value = np.zeros((nao*nao, naux), dtype=np.complex128)
    response_shape = (
        (6, nao*nao, naux) if compact_voigt
        else (3, 3, nao*nao, naux)
    )
    if retain_components:
        pair_response = np.zeros(response_shape, dtype=np.complex128)
        auxiliary_response = np.zeros_like(pair_response)
        kernel_response = np.zeros_like(pair_response)
        total = None
    else:
        pair_response = auxiliary_response = kernel_response = None
        total = np.zeros(response_shape, dtype=np.complex128)
    pair_context = _build_ao_pair_source_context(
        cell, ki, kj, rcut=rcut
    )

    for p0, p1 in lib.prange(0, len(grid_vectors), block_size):
        vectors = grid_vectors[p0:p1]
        momentum = vectors+transfer
        phase_started = perf_counter()
        auxiliary = auxiliary_fourier_strain(auxcell, momentum)
        kernel = weighted_coulomb_strain(cell, momentum, omega=float(omega))
        weighted_auxiliary = (auxiliary.value*kernel.value).T
        timings["auxiliary_kernel_build"] += perf_counter()-phase_started
        pair_target = pair_response if retain_components else total
        if pair_response_mode == "factorized":
            pair_flat = _contract_ao_pair_source_block_factorized(
                pair_context, vectors, weighted_auxiliary, pair_target,
                timings=timings,
                factorized_projection=factorized_projection,
            )
        else:
            pair_flat = _contract_ao_pair_source_block(
                pair_context, vectors, weighted_auxiliary, pair_target,
                compact_voigt=compact_voigt, timings=timings,
                pair_response_matmul=pair_response_matmul,
            )
        phase_started = perf_counter()
        value += pair_flat.T@weighted_auxiliary.conj()
        timings["central_source_contraction"] += perf_counter()-phase_started
        phase_started = perf_counter()
        if compact_voigt:
            auxiliary_derivative = matrix_tensor_to_voigt(
                auxiliary.raw_derivative
            )
            kernel_derivative = matrix_tensor_to_voigt(kernel.raw_derivative)
            for component in range(6):
                auxiliary_term = pair_flat.T@(
                    auxiliary_derivative[component]*kernel.value
                ).T.conj()
                kernel_term = pair_flat.T@(
                    auxiliary.value*kernel_derivative[component]
                ).T.conj()
                total[component] += auxiliary_term+kernel_term
        else:
            for x in range(3):
                for y in range(3):
                    auxiliary_term = pair_flat.T@(
                        auxiliary.raw_derivative[x, y]*kernel.value
                    ).T.conj()
                    kernel_term = pair_flat.T@(
                        auxiliary.value*kernel.raw_derivative[x, y]
                    ).T.conj()
                    if retain_components:
                        auxiliary_response[x, y] += auxiliary_term
                        kernel_response[x, y] += kernel_term
                    else:
                        total[x, y] += auxiliary_term+kernel_term
        timings["auxiliary_kernel_response_contraction"] += (
            perf_counter()-phase_started
        )
    if retain_components:
        total = pair_response+auxiliary_response+kernel_response
    timings["total"] = perf_counter()-total_started
    return KPointLongRangeSourceStrain(
        value=_real_if_close(value),
        raw_derivative=_real_if_close(total),
        ao_pair_response=(
            None if pair_response is None else _real_if_close(pair_response)
        ),
        auxiliary_response=(
            None if auxiliary_response is None
            else _real_if_close(auxiliary_response)
        ),
        kernel_response=(
            None if kernel_response is None else _real_if_close(kernel_response)
        ),
        kpti=ki,
        kptj=kj,
        momentum_transfer=transfer,
        reciprocal_block_size=block_size,
        grid_points=int(len(grid_vectors)),
        estimated_block_work_bytes=estimated_block_bytes,
        timings_s=timings,
        pair_response_mode=pair_response_mode,
        pair_response_image_bytes=pair_response_image_bytes,
    )


def kpoint_long_range_source_adjoint_strain(
    cell, auxcell, omega, mesh, kpti, kptj, source_adjoints, *, rcut=None,
    reciprocal_block_size=None,
):
    """Contract analytic long-range source strain before ``dB`` assembly.

    ``source_adjoints`` uses AO-pair-by-auxiliary ordering.  Reassociating the
    existing reciprocal contractions leaves the exact arithmetic expression
    unchanged while replacing six ``(nao**2,naux)`` outputs with one scalar
    per adjoint and Voigt component.
    """

    started = perf_counter()
    mesh_array = np.asarray(mesh, dtype=int)
    if mesh_array.shape != (3,) or np.any(mesh_array <= 0):
        raise ValueError(f"mesh must contain three positive integers: {mesh_array}")
    grid_vectors = cell.get_Gv(mesh_array)
    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    transfer = kj-ki
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    adjoints = np.asarray(source_adjoints)
    if adjoints.ndim == 2:
        adjoints = adjoints[None]
    expected = (nao*nao, naux)
    if adjoints.ndim != 3 or tuple(adjoints.shape[1:]) != expected:
        raise ValueError(
            "source_adjoints must have shape (nsource,nao**2,naux); "
            f"expected (*,{expected[0]},{expected[1]}), got {adjoints.shape}"
        )
    dtype = np.result_type(adjoints.dtype, np.complex128)
    adjoints = adjoints.astype(dtype, copy=False)
    block_size, estimated_block_bytes = _reciprocal_block_size(
        reciprocal_block_size, grid_points=len(grid_vectors), nao=nao,
        naux=naux,
    )
    result = np.zeros((len(adjoints), 6), dtype=dtype)
    pair_context = _build_ao_pair_source_context(
        cell, ki, kj, rcut=rcut
    )
    peak_projection_bytes = 0

    for p0, p1 in lib.prange(0, len(grid_vectors), block_size):
        vectors = grid_vectors[p0:p1]
        momentum = vectors+transfer
        auxiliary = auxiliary_fourier_strain(auxcell, momentum)
        kernel = weighted_coulomb_strain(cell, momentum, omega=float(omega))
        weighted_auxiliary = (auxiliary.value*kernel.value).T
        pair_flat, pair_values = _contract_ao_pair_source_adjoint_block(
            pair_context, vectors, weighted_auxiliary, adjoints, result
        )
        peak_projection_bytes = max(
            peak_projection_bytes, int(pair_values.nbytes)
        )
        auxiliary_derivative = matrix_tensor_to_voigt(
            auxiliary.raw_derivative
        )
        kernel_derivative = matrix_tensor_to_voigt(kernel.raw_derivative)
        for component in range(6):
            derivative_weight = (
                auxiliary_derivative[component]*kernel.value
                + auxiliary.value*kernel_derivative[component]
            ).T.conj()
            result[:, component] += np.einsum(
                "sgp,gp->s", pair_values, derivative_weight, optimize=True
            )
        del pair_flat, pair_values

    full_response_bytes = int(
        6*nao*nao*naux*np.dtype(np.complex128).itemsize
    )
    return KPointLongRangeSourceAdjointStrain(
        values=np.real(result),
        kpti=ki,
        kptj=kj,
        reciprocal_block_size=block_size,
        grid_points=int(len(grid_vectors)),
        estimated_block_work_bytes=int(
            estimated_block_bytes+peak_projection_bytes
        ),
        wall_seconds=float(perf_counter()-started),
        full_response_bytes_avoided=full_response_bytes,
    )


def gamma_vbar_source_strain(cell, auxcell, omega, *, rcut=None):
    """Return ``-S*pi*q_aux/(omega**2*volume)`` and its strain derivative."""

    general = kpoint_vbar_source_strain(
        cell, auxcell, omega, np.zeros(3), np.zeros(3), rcut=rcut
    )
    return GammaVBarSourceStrain(
        value=general.value,
        raw_derivative=general.raw_derivative,
        overlap_response=general.overlap_response,
        volume_response=general.volume_response,
    )


def kpoint_vbar_source_strain(
    cell, auxcell, omega, kpti, kptj, *, rcut=None, overlap=None,
):
    """Return the vbar source response, which is nonzero only for q=0."""

    range_parameter = abs(float(omega))
    if range_parameter <= 0:
        raise ValueError("vbar source strain requires nonzero omega")
    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    nao_pair = int(cell.nao_nr()) ** 2
    naux = int(auxcell.nao_nr())
    if np.linalg.norm(kj-ki) >= 1e-10:
        zeros = np.zeros((nao_pair, naux), dtype=np.complex128)
        derivative = np.zeros((3, 3, nao_pair, naux), dtype=np.complex128)
        return KPointVBarSourceStrain(
            value=zeros,
            raw_derivative=derivative,
            overlap_response=derivative.copy(),
            volume_response=derivative.copy(),
            kpti=ki,
            kptj=kj,
        )
    if overlap is None:
        overlap = kpoint_ao_pair_fourier_strain(
            cell, np.zeros((1, 3)), ki, kj, rcut=rcut
        )
    elif not isinstance(overlap, KPointAOPairFourierStrain):
        raise TypeError("overlap must be a KPointAOPairFourierStrain")
    overlap_value = overlap.value[0]
    overlap_derivative = overlap.raw_derivative[:, :, 0]
    charges = np.asarray(rsdf.get_aux_chg(auxcell), dtype=float)
    coefficient = np.pi * charges / (
        range_parameter**2 * float(cell.vol)
    )
    value = -overlap_value.reshape(-1, 1) * coefficient[None, :]
    overlap_response = -overlap_derivative.reshape(
        3, 3, -1, 1
    ) * coefficient[None, None, None, :]
    volume_response = np.zeros_like(overlap_response)
    for axis in range(3):
        volume_response[axis, axis] = -value
    total = overlap_response + volume_response
    return KPointVBarSourceStrain(
        value=_real_if_close(value),
        raw_derivative=_real_if_close(total),
        overlap_response=_real_if_close(overlap_response),
        volume_response=_real_if_close(volume_response),
        kpti=ki,
        kptj=kj,
    )


def gamma_short_range_source_strain(cell, auxcell, omega, *, precision=None):
    r"""Return the analytic Gamma short-range three-center source strain.

    For one screened image triplet, translational invariance eliminates a
    separate derivative integral on the second primary AO center:

    .. math::

       B^{(1)}_{xy} = I^{ip1}_x (R_j-R_i)_y
                     + I^{ip2}_x (R_j-R_P)_y.

    A stress-owned C fill callback applies these image-relative weights inside
    PySCF's existing screened lattice traversal.  No full image tensor is
    materialized and no global PySCF callback is replaced.
    """

    range_parameter = abs(float(omega))
    if range_parameter <= 0:
        raise ValueError("short-range source strain requires nonzero omega")
    target_precision = (
        float(cell.precision) if precision is None else float(precision)
    )
    if target_precision <= 0:
        raise ValueError("short-range source precision must be positive")
    context = _short_range_driver_context(
        cell, auxcell, range_parameter, target_precision
    )
    packed = np.zeros(
        (9, cell.nao_nr() * (cell.nao_nr() + 1) // 2, auxcell.nao_nr()),
        dtype=np.double,
    )
    extension = _load_short_range_strain_extension()
    fill_name = (
        "PBCsr3c_strain_gs2_cart" if cell.cart
        else "PBCsr3c_strain_gs2_sph"
    )
    driver = rsdf_helper.libpbc.PBCsr3c_g_drv
    ip1_name = cell._add_suffix("int3c2e_ip1")
    driver(
        getattr(rsdf_helper.libpbc, ip1_name),
        getattr(extension, fill_name),
        packed.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(9),
        ctypes.c_int(len(context["lattice_vectors"])),
        context["lattice_vectors"].ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int * 6)(0, cell.nbas, cell.nbas, 2 * cell.nbas,
                           2 * cell.nbas, 2 * cell.nbas + auxcell.nbas),
        context["ao_locations"].ctypes.data_as(ctypes.c_void_p),
        context["optimizer"],
        context["shell_pair_mask"].ctypes.data_as(ctypes.c_void_p),
        context["reference_unique_map"].ctypes.data_as(ctypes.c_void_p),
        context["auxiliary_unique_map"].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(context["naux_unique"]),
        context["unique_exponents"].ctypes.data_as(ctypes.c_void_p),
        context["distance_cutoffs"].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(context["distance_step"]),
        context["three_center_cutoffs"].ctypes.data_as(ctypes.c_void_p),
        context["distance_locations"].ctypes.data_as(ctypes.c_void_p),
        context["atm"].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(cell.natm),
        context["bas"].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(cell.nbas),
        context["env"].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(context["env"].size),
    )
    nao = int(cell.nao_nr())
    derivative = np.empty((3, 3, nao * nao, auxcell.nao_nr()))
    for component in range(9):
        full = lib.unpack_tril(packed[component].T).transpose(1, 2, 0)
        derivative[component // 3, component % 3] = full.reshape(
            nao * nao, auxcell.nao_nr()
        )

    from pyscf_pbc_dcdft.df.gdf.source_short_range import short_range_source

    value = short_range_source(
        cell, auxcell, range_parameter, kpti=np.zeros(3), kptj=np.zeros(3)
    )
    return GammaShortRangeSourceStrain(
        value=_real_if_close(value),
        raw_derivative=_real_if_close(derivative),
        lattice_vectors=int(len(context["lattice_vectors"])),
        precision=target_precision,
    )


def kpoint_short_range_source_strain(
    cell, auxcell, omega, kpti, kptj, *, precision=None
):
    r"""Return the analytic short-range source for one ordered k pair.

    At fixed fractional k points the Bloch phase is invariant under a
    homogeneous cell strain.  The stress-owned C driver therefore accumulates

    .. math::

       e^{-i k_i\cdot L_i+i k_j\cdot L_j}
       \left[I^{ip1}_x(R_j+L_j-R_i-L_i)_y
             +I^{ip2}_x(R_j+L_j-R_P)_y\right]

    while the screened image indices are still available.  This preserves the
    general complex ``s1`` AO-pair layout and avoids Gamma-only triangular
    symmetry assumptions.
    """

    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    range_parameter = abs(float(omega))
    if range_parameter <= 0:
        raise ValueError("short-range source strain requires nonzero omega")
    target_precision = (
        float(cell.precision) if precision is None else float(precision)
    )
    if target_precision <= 0:
        raise ValueError("short-range source precision must be positive")
    context = _short_range_driver_context(
        cell, auxcell, range_parameter, target_precision
    )
    lattice_vectors = context["lattice_vectors"]
    phase_i = np.asarray(
        np.exp(1j * np.dot(ki, lattice_vectors.T)),
        dtype=np.complex128, order="C",
    )
    phase_j = np.asarray(
        np.exp(1j * np.dot(kj, lattice_vectors.T)),
        dtype=np.complex128, order="C",
    )
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    output = np.zeros((10, nao, nao, naux), dtype=np.complex128)
    extension = _load_short_range_strain_extension()
    driver_name = (
        "PBCsr3c_strain_kpair_cart" if cell.cart
        else "PBCsr3c_strain_kpair_sph"
    )
    driver = getattr(extension, driver_name)
    driver(
        output.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(len(lattice_vectors)),
        lattice_vectors.ctypes.data_as(ctypes.c_void_p),
        phase_i.ctypes.data_as(ctypes.c_void_p),
        phase_j.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int * 6)(0, cell.nbas, cell.nbas, 2 * cell.nbas,
                           2 * cell.nbas, 2 * cell.nbas + auxcell.nbas),
        context["ao_locations"].ctypes.data_as(ctypes.c_void_p),
        context["shell_pair_mask"].ctypes.data_as(ctypes.c_void_p),
        context["reference_unique_map"].ctypes.data_as(ctypes.c_void_p),
        context["auxiliary_unique_map"].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(context["naux_unique"]),
        context["unique_exponents"].ctypes.data_as(ctypes.c_void_p),
        context["distance_cutoffs"].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(context["distance_step"]),
        context["three_center_cutoffs"].ctypes.data_as(ctypes.c_void_p),
        context["distance_locations"].ctypes.data_as(ctypes.c_void_p),
        context["atm"].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(cell.natm),
        context["bas"].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(cell.nbas),
        context["env"].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(context["env"].size),
    )
    return KPointShortRangeSourceStrain(
        value=_real_if_close(output[0].reshape(nao * nao, naux)),
        raw_derivative=_real_if_close(
            output[1:].reshape(3, 3, nao * nao, naux)
        ),
        kpti=ki,
        kptj=kj,
        lattice_vectors=int(len(lattice_vectors)),
        precision=target_precision,
    )


def gamma_three_center_source_strain(
    cell, auxcell, omega, mesh, *, precision=None, rcut=None,
    reciprocal_block_size=None, retain_components=True,
    zero_transfer_overlap=None, compact_voigt=False,
    pair_response_matmul=None,
    pair_response_mode="assembled",
    factorized_projection=None,
):
    """Assemble the complete analytic Gamma GDF pre-Cholesky source strain."""

    general = kpoint_three_center_source_strain(
        cell, auxcell, omega, mesh, np.zeros(3), np.zeros(3),
        precision=precision, rcut=rcut,
        reciprocal_block_size=reciprocal_block_size,
        retain_components=retain_components,
        zero_transfer_overlap=zero_transfer_overlap,
        compact_voigt=compact_voigt,
        pair_response_matmul=pair_response_matmul,
        pair_response_mode=pair_response_mode,
        factorized_projection=factorized_projection,
    )
    return GammaThreeCenterSourceStrain(
        value=general.value,
        raw_derivative=general.raw_derivative,
        short_range=general.short_range,
        long_range=general.long_range,
        vbar=general.vbar,
        reciprocal_block_size=general.reciprocal_block_size,
        reciprocal_grid_points=general.reciprocal_grid_points,
        estimated_block_work_bytes=general.estimated_block_work_bytes,
        timings_s=general.timings_s,
        pair_response_mode=general.pair_response_mode,
        pair_response_image_bytes=general.pair_response_image_bytes,
    )


def kpoint_three_center_source_strain(
    cell, auxcell, omega, mesh, kpti, kptj, *, precision=None, rcut=None,
    reciprocal_block_size=None, retain_components=True,
    zero_transfer_overlap=None, compact_voigt=False,
    pair_response_matmul=None,
    pair_response_mode="assembled",
    factorized_projection=None,
):
    """Assemble the complete complex GDF source strain for one k pair."""

    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    started = perf_counter()
    short_range = kpoint_short_range_source_strain(
        cell, auxcell, omega, ki, kj, precision=precision
    )
    short_seconds = perf_counter()-started
    started = perf_counter()
    long_range = kpoint_long_range_source_strain(
        cell, auxcell, omega, mesh, ki, kj, rcut=rcut,
        reciprocal_block_size=reciprocal_block_size,
        retain_components=retain_components, compact_voigt=compact_voigt,
        pair_response_matmul=pair_response_matmul,
        pair_response_mode=pair_response_mode,
        factorized_projection=factorized_projection,
    )
    long_seconds = perf_counter()-started
    started = perf_counter()
    vbar = kpoint_vbar_source_strain(
        cell, auxcell, omega, ki, kj, rcut=rcut,
        overlap=zero_transfer_overlap,
    )
    vbar_seconds = perf_counter()-started
    value = np.array(short_range.value, dtype=np.complex128, copy=True)
    value += long_range.value
    value += vbar.value
    if compact_voigt:
        derivative = np.array(
            matrix_tensor_to_voigt(short_range.raw_derivative),
            dtype=np.complex128, copy=True,
        )
        derivative += long_range.raw_derivative
        derivative += matrix_tensor_to_voigt(vbar.raw_derivative)
    else:
        derivative = np.array(
            short_range.raw_derivative, dtype=np.complex128, copy=True
        )
        derivative += long_range.raw_derivative
        derivative += vbar.raw_derivative
    return KPointThreeCenterSourceStrain(
        value=_real_if_close(value),
        raw_derivative=_real_if_close(derivative),
        short_range=short_range if retain_components else None,
        long_range=long_range if retain_components else None,
        vbar=vbar if retain_components else None,
        kpti=ki,
        kptj=kj,
        reciprocal_block_size=long_range.reciprocal_block_size,
        reciprocal_grid_points=long_range.grid_points,
        estimated_block_work_bytes=long_range.estimated_block_work_bytes,
        timings_s={
            "short_range": float(short_seconds),
            "long_range": float(long_seconds),
            "vbar": float(vbar_seconds),
            **{
                f"long_range_{key}": float(value)
                for key, value in long_range.timings_s.items()
            },
        },
        pair_response_mode=long_range.pair_response_mode,
        pair_response_image_bytes=long_range.pair_response_image_bytes,
    )


def kpoint_three_center_source_adjoint_strain(
    cell, auxcell, omega, mesh, kpti, kptj, source_adjoints, *,
    precision=None, rcut=None, reciprocal_block_size=None,
):
    """Contract complete GDF/RSDF source strain with fixed adjoints.

    The short-range C integral remains the established exact tensor builder
    in this increment.  Reciprocal and zero-transfer terms are contracted
    before their ``(6,nao**2,naux)`` responses are assembled.
    """

    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    adjoints = np.asarray(source_adjoints)
    if adjoints.ndim == 2:
        adjoints = adjoints[None]
    expected = (nao*nao, naux)
    if adjoints.ndim != 3 or tuple(adjoints.shape[1:]) != expected:
        raise ValueError(
            "source_adjoints must have shape (nsource,nao**2,naux); "
            f"expected (*,{expected[0]},{expected[1]}), got {adjoints.shape}"
        )

    started = perf_counter()
    short_range = kpoint_short_range_source_strain(
        cell, auxcell, omega, ki, kj, precision=precision
    )
    short_seconds = perf_counter()-started
    short_voigt = matrix_tensor_to_voigt(short_range.raw_derivative)
    values = np.real(np.einsum(
        "sqp,xqp->sx", adjoints, short_voigt, optimize=True
    ))

    reciprocal = kpoint_long_range_source_adjoint_strain(
        cell, auxcell, omega, mesh, ki, kj, adjoints, rcut=rcut,
        reciprocal_block_size=reciprocal_block_size,
    )
    values += reciprocal.values

    started = perf_counter()
    values += _contract_vbar_source_adjoint_strain(
        cell, auxcell, omega, ki, kj, adjoints, rcut=rcut
    )
    vbar_seconds = perf_counter()-started
    short_bytes = int(
        np.asarray(short_range.raw_derivative).nbytes
        + np.asarray(short_range.value).nbytes
    )
    return KPointThreeCenterSourceAdjointStrain(
        values=values,
        kpti=ki,
        kptj=kj,
        reciprocal_block_size=reciprocal.reciprocal_block_size,
        reciprocal_grid_points=reciprocal.grid_points,
        estimated_block_work_bytes=reciprocal.estimated_block_work_bytes,
        timings_s={
            "short_range_full_tensor": float(short_seconds),
            "reciprocal_direct_adjoint": float(reciprocal.wall_seconds),
            "vbar_direct_adjoint": float(vbar_seconds),
            "total": float(short_seconds+reciprocal.wall_seconds+vbar_seconds),
        },
        memory_estimate={
            "source_adjoint_bytes": int(adjoints.nbytes),
            "short_range_tensor_bytes": short_bytes,
            "reciprocal_full_response_avoided_bytes": int(
                reciprocal.full_response_bytes_avoided
            ),
            "retained_source_derivative_bytes": 0,
        },
    )


def _contract_vbar_source_adjoint_strain(
    cell, auxcell, omega, kpti, kptj, source_adjoints, *, rcut=None,
):
    """Contract the rank-one zero-transfer source without ``dB_vbar``."""

    range_parameter = abs(float(omega))
    if range_parameter <= 0:
        raise ValueError("vbar source strain requires nonzero omega")
    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    adjoints = np.asarray(source_adjoints)
    if np.linalg.norm(kj-ki) >= 1e-10:
        return np.zeros((len(adjoints), 6), dtype=float)
    overlap = kpoint_ao_pair_fourier_strain(
        cell, np.zeros((1, 3)), ki, kj, rcut=rcut
    )
    charges = np.asarray(rsdf.get_aux_chg(auxcell), dtype=float)
    coefficient = np.pi*charges/(range_parameter**2*float(cell.vol))
    projected = np.einsum(
        "sqp,p->sq", adjoints, coefficient, optimize=True
    )
    overlap_value = overlap.value[0].reshape(-1)
    overlap_derivative = matrix_tensor_to_voigt(
        overlap.raw_derivative[:, :, 0]
    ).reshape(6, -1)
    result = -np.einsum(
        "sq,xq->sx", projected, overlap_derivative, optimize=True
    )
    volume = np.einsum(
        "sq,q->s", projected, overlap_value, optimize=True
    )
    result[:, :3] += volume[:, None]
    return np.real(result)


def _reciprocal_block_size(requested, *, grid_points, nao, naux):
    """Resolve a bounded G-vector block and a conservative work estimate."""

    points = int(grid_points)
    if points < 1:
        raise ValueError("reciprocal grid must contain at least one point")
    primary = int(nao)*int(nao)
    auxiliary = int(naux)
    # The production contraction keeps the AO value, three i/j center
    # derivatives, three moments, and one ordered-strain temporary.  Audit
    # outputs are independent of nG and are not part of this block estimate.
    bytes_per_point = np.dtype(np.complex128).itemsize*(16*primary+16*auxiliary)
    if requested is None:
        block = max(1, _DEFAULT_RECIPROCAL_BLOCK_BYTES//bytes_per_point)
    else:
        block = int(requested)
        if block < 1:
            raise ValueError("reciprocal_block_size must be positive")
    block = min(points, block)
    return block, int(block*bytes_per_point)


def _build_ao_pair_source_context(cell, kpti, kptj, *, rcut=None):
    """Build translated molecular contexts once for all reciprocal blocks."""

    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    cutoff = float(cell.rcut if rcut is None else rcut)
    if cutoff <= 0:
        raise ValueError("AO-pair lattice cutoff must be positive")
    lattice_vectors = np.asarray(
        cell.get_lattice_Ls(rcut=cutoff), dtype=float
    )
    centers = _ao_centers(cell)
    images = []
    for lattice in lattice_vectors:
        shifted = cell.copy(deep=True)
        shifted.set_geom_(cell.atom_coords()+lattice, unit="Bohr")
        combined = gto.conc_mol(cell, shifted)
        images.append(_AOPairSourceImage(
            molecule=combined,
            shell_slice=(0, cell.nbas, cell.nbas, 2*cell.nbas),
            phase=complex(np.exp(1j*np.dot(kj, lattice))),
            shifted_centers=centers+lattice,
        ))
    return _AOPairSourceContext(
        transfer=kj-ki, centers=centers, nao=int(cell.nao_nr()),
        images=tuple(images),
    )


def _contract_ao_pair_source_block(
    context, grid_vectors, weighted_auxiliary, response_accumulator, *,
    compact_voigt=False, timings=None, pair_response_matmul=None,
):
    """Contract one AO-pair Fourier block without a 9-component G tensor."""

    vectors = np.asarray(grid_vectors, dtype=float)
    momentum = vectors+context.transfer
    centers = context.centers
    nao = context.nao
    pair_value = np.zeros((len(vectors), nao, nao), dtype=np.complex128)
    weighted_conjugate = np.asarray(weighted_auxiliary).conj()
    matmul = np.matmul if pair_response_matmul is None else pair_response_matmul

    for image_context in context.images:
        combined = image_context.molecule
        shell_slice = image_context.shell_slice
        phase_started = perf_counter()
        pair, center_i, center_j = _molecular_ft_overlap_center_derivatives(
            combined, momentum, shell_slice
        )
        moment = molecular_ft_ao.ft_aopair(
            combined, momentum, shls_slice=shell_slice, aosym="s1",
            intor="GTO_ft_rc", comp=3,
        )
        if timings is not None:
            timings["ao_fourier_integrals"] += perf_counter()-phase_started
        phase = image_context.phase
        pair_value += phase*pair
        shifted_centers = image_context.shifted_centers

        def image_response(x, y):
            image = -center_i[x]*centers[:, y][None, :, None]
            image -= center_j[x]*shifted_centers[:, y][None, None, :]
            image += 1j*momentum[:, x, None, None]*moment[y]
            return image

        phase_started = perf_counter()
        if compact_voigt:
            indices = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
            for component, (x, y) in enumerate(indices):
                assembly_started = perf_counter()
                image = image_response(x, y)
                if x != y:
                    image += image_response(y, x)
                    image *= 0.5
                if timings is not None:
                    timings["ao_pair_response_assembly"] += (
                        perf_counter()-assembly_started
                    )
                matmul_started = perf_counter()
                response_accumulator[component] += matmul(
                    phase*image.reshape(len(vectors), -1).T,
                    weighted_conjugate,
                )
                if timings is not None:
                    timings["ao_pair_response_matmul"] += (
                        perf_counter()-matmul_started
                    )
        else:
            for x in range(3):
                for y in range(3):
                    assembly_started = perf_counter()
                    image = image_response(x, y)
                    if timings is not None:
                        timings["ao_pair_response_assembly"] += (
                            perf_counter()-assembly_started
                        )
                    matmul_started = perf_counter()
                    response_accumulator[x, y] += matmul(
                        phase*image.reshape(len(vectors), -1).T,
                        weighted_conjugate,
                    )
                    if timings is not None:
                        timings["ao_pair_response_matmul"] += (
                            perf_counter()-matmul_started
                        )
        if timings is not None:
            timings["ao_pair_response_contraction"] += (
                perf_counter()-phase_started
            )
    return pair_value.reshape(len(vectors), nao*nao)


def _contract_ao_pair_source_adjoint_block(
    context, grid_vectors, weighted_auxiliary, adjoints, output,
):
    """Contract AO-pair strain with source adjoints inside one G block."""

    vectors = np.asarray(grid_vectors, dtype=float)
    momentum = vectors+context.transfer
    centers = context.centers
    nao = context.nao
    pair_value = np.zeros((len(vectors), nao, nao), dtype=np.complex128)
    weighted_conjugate = np.asarray(weighted_auxiliary).conj()
    response_weights = np.einsum(
        "gp,sqp->sgq", weighted_conjugate, adjoints, optimize=True
    )
    indices = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

    for image_context in context.images:
        combined = image_context.molecule
        shell_slice = image_context.shell_slice
        pair, center_i, center_j = _molecular_ft_overlap_center_derivatives(
            combined, momentum, shell_slice
        )
        moment = molecular_ft_ao.ft_aopair(
            combined, momentum, shls_slice=shell_slice, aosym="s1",
            intor="GTO_ft_rc", comp=3,
        )
        phase = image_context.phase
        pair_value += phase*pair
        shifted_centers = image_context.shifted_centers

        def image_response(x, y):
            image = -center_i[x]*centers[:, y][None, :, None]
            image -= center_j[x]*shifted_centers[:, y][None, None, :]
            image += 1j*momentum[:, x, None, None]*moment[y]
            return image

        for component, (x, y) in enumerate(indices):
            image = image_response(x, y)
            if x != y:
                image += image_response(y, x)
                image *= 0.5
            output[:, component] += np.einsum(
                "gq,sgq->s", phase*image.reshape(len(vectors), -1),
                response_weights, optimize=True,
            )

    pair_flat = pair_value.reshape(len(vectors), nao*nao)
    pair_values = np.einsum(
        "gq,sqp->sgp", pair_flat, adjoints, optimize=True
    )
    return pair_flat, pair_values


def _contract_ao_pair_source_block_factorized(
    context, grid_vectors, weighted_auxiliary, response_accumulator, *,
    timings=None, factorized_projection=None,
):
    """Project derivative factors before assembling six Voigt responses."""

    vectors = np.asarray(grid_vectors, dtype=float)
    momentum = vectors+context.transfer
    centers = context.centers
    nao = context.nao
    pair_value = np.zeros((len(vectors), nao, nao), dtype=np.complex128)
    weighted_conjugate = np.asarray(weighted_auxiliary).conj()
    indices = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

    for image_context in context.images:
        combined = image_context.molecule
        shell_slice = image_context.shell_slice
        phase_started = perf_counter()
        pair, center_i, center_j = _molecular_ft_overlap_center_derivatives(
            combined, momentum, shell_slice
        )
        moment = molecular_ft_ao.ft_aopair(
            combined, momentum, shls_slice=shell_slice, aosym="s1",
            intor="GTO_ft_rc", comp=3,
        )
        if timings is not None:
            timings["ao_fourier_integrals"] += perf_counter()-phase_started
        phase = image_context.phase
        pair_value += phase*pair

        projection_started = perf_counter()
        if factorized_projection is None:
            center_i_projected = np.einsum(
                "xgij,gp->xijp", center_i, weighted_conjugate,
                optimize=True,
            )
            center_j_projected = np.einsum(
                "xgij,gp->xijp", center_j, weighted_conjugate,
                optimize=True,
            )
            moment_projected = np.empty(
                (3, 3, nao, nao, weighted_conjugate.shape[1]),
                dtype=np.complex128,
            )
            for x in range(3):
                momentum_weighted = (
                    momentum[:, x, None]*weighted_conjugate
                )
                moment_projected[x] = np.einsum(
                    "ygij,gp->yijp", moment, momentum_weighted,
                    optimize=True,
                )
        else:
            (
                center_i_projected,
                center_j_projected,
                moment_projected,
            ) = factorized_projection(
                center_i, center_j, moment, momentum, weighted_conjugate
            )
        if timings is not None:
            timings["ao_pair_response_matmul"] += (
                perf_counter()-projection_started
            )

        shifted_centers = image_context.shifted_centers
        assembly_started = perf_counter()

        def projected_response(x, y):
            value = -center_i_projected[x]*centers[:, y, None, None]
            value -= center_j_projected[x]*shifted_centers[None, :, y, None]
            value += 1j*moment_projected[x, y]
            return value

        for component, (x, y) in enumerate(indices):
            response = projected_response(x, y)
            if x != y:
                response += projected_response(y, x)
                response *= 0.5
            response_accumulator[component] += phase*response.reshape(
                nao*nao, -1
            )
        if timings is not None:
            elapsed = perf_counter()-assembly_started
            timings["ao_pair_response_assembly"] += elapsed
            timings["ao_pair_response_contraction"] += (
                perf_counter()-projection_started
            )
        del center_i_projected, center_j_projected, moment_projected

    return pair_value.reshape(len(vectors), nao*nao)


def _molecular_ft_overlap_center_derivatives(mol, grid_vectors, shell_slice):
    """Evaluate overlap and i/j derivatives in one shell traversal."""

    vectors = np.asarray(grid_vectors, dtype=float)
    gv_transposed = np.asarray(vectors.T, order="C")
    extension = _load_short_range_strain_extension()
    callback = getattr(extension, mol._add_suffix("DCDT_ft_ovlp_ipij"))
    overlap_name = mol._add_suffix("GTO_ft_ovlp")
    ao_locations = moleintor.make_loc(mol._bas, overlap_name)
    i0, i1, j0, j1 = (int(value) for value in shell_slice)
    ni = int(ao_locations[i1] - ao_locations[i0])
    nj = int(ao_locations[j1] - ao_locations[j0])
    output = np.zeros((7, nj, ni, len(vectors)), dtype=np.complex128)
    overlap_mask = np.ones((i1 - i0, j1 - j0), dtype=np.int8, order="F")
    driver = libcgto.GTO_ft_fill_drv
    evaluator = getattr(libcgto, "GTO_Gv_general")
    sorter = getattr(libcgto, "GTO_ft_zfill_s1")
    driver(
        callback,
        evaluator,
        sorter,
        output.ctypes.data_as(ctypes.c_void_p),
        overlap_mask.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(7),
        (ctypes.c_int * 4)(i0, i1, j0, j1),
        ao_locations.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(0.0),
        gv_transposed.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_double * 1)(0.0),
        lib.c_null_ptr(),
        (ctypes.c_int * 3)(0, 0, 0),
        ctypes.c_int(len(vectors)),
        mol._atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(mol.natm),
        mol._bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(mol.nbas),
        mol._env.ctypes.data_as(ctypes.c_void_p),
    )
    output = np.rollaxis(output, -2, -3)
    output = np.rollaxis(output, -1, -3)
    return output[0], output[1:4], output[4:7]


def _short_range_driver_context(cell, auxcell, omega, precision):
    """Build PySCF-identical ME prescreening data for the strain callback."""

    refmap, unique_atoms, unique_basis, unique_basis_locations = (
        rsdf_helper._get_refuniq_map(cell)
    )
    auxmap, _aux_atoms, unique_auxiliary, unique_aux_locations = (
        rsdf_helper._get_refuniq_map(auxcell)
    )
    distance_step = 1.0 / rsdf_helper.BOHR
    auxiliary_schwartz = rsdf_helper._get_schwartz_data(
        unique_auxiliary, omega, keep1ctr=False, safe=True
    )
    distance_cutoffs = rsdf_helper._get_schwartz_dcut(
        unique_basis, omega, precision / auxiliary_schwartz.max(),
        r0=cell.rcut,
    )
    distance_lists = rsdf_helper._make_dijs_lst(
        distance_cutoffs, distance_step
    )
    distance_locations = np.cumsum(
        [0] + [len(values) for values in distance_lists]
    ).astype(np.int32)
    zero_estimates = [np.zeros_like(values) for values in distance_lists]
    three_center_cutoffs = rsdf_helper._get_3c2e_Rcuts(
        unique_basis, unique_auxiliary, distance_lists, omega, precision,
        "ME", zero_estimates,
    )
    basis_exponents = np.asarray(
        [np.asarray(shell[1:])[:, 0].min() for shell in unique_basis]
    )
    atom_cutoffs = rsdf_helper._get_atom_Rcuts_3c(
        three_center_cutoffs, distance_lists, basis_exponents,
        unique_basis_locations, unique_aux_locations,
    )
    lattice_vectors = np.asarray(
        rsdf_helper._get_Lsmin(cell, atom_cutoffs, unique_atoms),
        dtype=float, order="C",
    )

    intor = cell._add_suffix("int3c2e_ip1")
    atm, bas, env = gto.conc_env(
        cell._atm, cell._bas, cell._env,
        cell._atm, cell._bas, cell._env,
    )
    ao_locations = moleintor.make_loc(bas, intor)
    auxiliary_locations = auxcell.ao_loc_nr(
        auxcell.cart or "ssc" in intor
    )
    ao_locations = np.asarray(
        np.hstack([ao_locations, ao_locations[-1] + auxiliary_locations[1:]]),
        dtype=np.int32,
    )
    atm, bas, env = gto.conc_env(
        atm, bas, env, auxcell._atm, auxcell._bas, auxcell._env
    )
    env[gto.PTR_RANGE_OMEGA] = -omega
    optimizer = _vhf.make_cintopt(atm, bas, env, intor)
    rsdf_helper.libpbc.CINTdel_pairdata_optimizer(optimizer)
    return {
        "reference_unique_map": np.asarray(refmap, dtype=np.int32),
        "auxiliary_unique_map": np.asarray(auxmap, dtype=np.int32),
        "naux_unique": int(len(unique_auxiliary)),
        "unique_exponents": np.asarray(basis_exponents, dtype=float),
        "distance_cutoffs": np.asarray(distance_cutoffs, dtype=float) ** 2,
        "distance_step": float(distance_step),
        "three_center_cutoffs": np.asarray(three_center_cutoffs, dtype=float) ** 2,
        "distance_locations": distance_locations,
        "lattice_vectors": lattice_vectors,
        "shell_pair_mask": np.ones(
            (cell.nbas, cell.nbas), dtype=np.int8, order="C"
        ),
        "ao_locations": ao_locations,
        "optimizer": optimizer,
        "atm": np.asarray(atm, dtype=np.int32),
        "bas": np.asarray(bas, dtype=np.int32),
        "env": np.asarray(env, dtype=float),
    }


def _load_short_range_strain_extension():
    pyscf_lib = Path(pyscf.__file__).resolve().parent / "lib"
    libcint = pyscf_lib / "deps" / "lib" / "libcint.so"
    libcgto_path = pyscf_lib / "libcgto.so"
    for library in (libcint, libcgto_path):
        if not library.is_file():
            raise ImportError(f"required PySCF library was not found at {library}")
    ctypes.CDLL(str(libcint), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(libcgto_path), mode=ctypes.RTLD_GLOBAL)
    module = importlib.import_module("pyscf_pbc_dcdft_stress._sr3c_strain")
    return ctypes.CDLL(module.__file__)


def _ao_centers(cell):
    centers = np.empty((cell.nao_nr(), 3), dtype=float)
    for atom_id, entry in enumerate(np.asarray(cell.aoslice_by_atom(), dtype=int)):
        p0, p1 = int(entry[2]), int(entry[3])
        centers[p0:p1] = np.asarray(cell.atom_coord(atom_id), dtype=float)
    return centers


def _kpoint(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"k-point must have shape (3,), got {point.shape}")
    return point


def _real_if_close(value, tolerance=1e-10):
    array = np.asarray(value)
    if array.size and np.max(np.abs(array.imag)) <= tolerance:
        return array.real
    return array
