"""Analytic reciprocal-kernel and G=0 pieces of the GDF metric strain.

These functions own genuine homogeneous-strain derivatives.  They do not
differentiate displaced cells and can be tested independently before AO
Fourier and short-range image responses are added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import gto
from pyscf.gto import ft_ao as molecular_ft_ao
from pyscf.pbc.df import rsdf
from pyscf.pbc.df import ft_ao as periodic_ft_ao
from pyscf.pbc.df.rsdf_builder import estimate_rs_2c2e_rcut


_DEFAULT_RECIPROCAL_BLOCK_BYTES = 2 * (1 << 30)


@dataclass(frozen=True)
class WeightedCoulombStrain:
    """Weighted reciprocal Coulomb kernel and its raw strain derivative."""

    value: np.ndarray
    raw_derivative: np.ndarray

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class G0MetricStrain:
    """Gamma G=0 auxiliary metric and its raw strain derivative."""

    value: np.ndarray
    raw_derivative: np.ndarray

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class AuxiliaryFourierStrain:
    """Analytic auxiliary Gaussian Fourier value and strain derivative."""

    value: np.ndarray
    raw_derivative: np.ndarray

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class LongRangeMetricStrain:
    """Long-range auxiliary metric and resolved analytic strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    auxiliary_response: np.ndarray
    kernel_response: np.ndarray
    reciprocal_block_size: int
    grid_points: int
    estimated_block_work_bytes: int

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class ReciprocalMetricStrain:
    """Long-range plus G=0 auxiliary metric strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    long_range: LongRangeMetricStrain
    g0: G0MetricStrain

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class ShortRangeMetricStrain:
    """Image-resolved short-range auxiliary metric strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    lattice_vectors: int
    rcut_bohr: float

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class GammaAuxiliaryMetricStrain:
    """Complete Gamma GDF auxiliary metric strain decomposition."""

    value: np.ndarray
    raw_derivative: np.ndarray
    short_range: ShortRangeMetricStrain
    reciprocal: ReciprocalMetricStrain

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class KPointAuxiliaryMetricStrain:
    """Complete complex GDF metric response at one momentum transfer."""

    value: np.ndarray
    raw_derivative: np.ndarray
    short_range: ShortRangeMetricStrain
    reciprocal: ReciprocalMetricStrain
    momentum_transfer: np.ndarray

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


def weighted_coulomb_strain(cell, grid_vectors, *, omega=None):
    """Return ``weighted_coulG`` and ``d(weighted_coulG)/d epsilon``.

    The 3D reciprocal integration weight is ``1 / volume``.  Positive omega
    selects the long-range ``erf(omega*r)/r`` kernel and negative omega the
    short-range complement, following PySCF's convention.  The G=0 point is
    removed and handled by the separate analytic metric correction.
    """

    if int(cell.dimension) != 3:
        raise NotImplementedError("GDF metric strain currently supports 3D cells")
    vectors = np.asarray(grid_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"grid_vectors must have shape (nG,3), got {vectors.shape}")
    range_parameter = float(cell.omega if omega is None else omega)
    norm2 = np.einsum("gx,gx->g", vectors, vectors)
    remove_g0 = bool(len(vectors) and np.linalg.norm(vectors[0]) < 1e-12)
    safe_norm2 = norm2.copy()
    if remove_g0:
        safe_norm2[0] = np.inf

    bare = 4.0 * np.pi / safe_norm2
    dyadic = np.einsum("gx,gy->xyg", vectors, vectors)
    derivative = dyadic * bare[None, None, :] * (2.0 / safe_norm2)[None, None, :]
    if range_parameter < 0:
        damping = np.exp(-0.25 / range_parameter**2 * safe_norm2)
        derivative *= 1.0 - damping
        derivative -= (
            damping[None, None, :]
            * (0.5 / range_parameter**2)
            * dyadic
            * bare[None, None, :]
        )
        bare *= 1.0 - damping
    elif range_parameter > 0:
        damping = np.exp(-0.25 / range_parameter**2 * safe_norm2)
        derivative *= damping
        derivative += (
            damping[None, None, :]
            * (0.5 / range_parameter**2)
            * dyadic
            * bare[None, None, :]
        )
        bare *= damping

    inverse_volume = 1.0 / float(cell.vol)
    weighted_value = inverse_volume * bare
    weighted_derivative = inverse_volume * derivative
    weighted_derivative -= (
        np.eye(3)[:, :, None] * weighted_value[None, None, :]
    )
    if remove_g0:
        weighted_value[0] = 0.0
        weighted_derivative[:, :, 0] = 0.0
    return WeightedCoulombStrain(
        value=np.asarray(weighted_value),
        raw_derivative=np.asarray(weighted_derivative),
    )


def g0_metric_strain(cell, auxcell, omega):
    """Return the Gamma G=0 metric and its exact volume response."""

    range_parameter = abs(float(omega))
    if range_parameter <= 0:
        raise ValueError("G=0 metric strain requires nonzero omega")
    charges = np.asarray(rsdf.get_aux_chg(auxcell), dtype=float)
    coefficient = np.pi / (range_parameter**2 * float(cell.vol))
    value = -coefficient * np.outer(charges, charges)
    derivative = np.zeros((3, 3, len(charges), len(charges)))
    for axis in range(3):
        derivative[axis, axis] = -value
    return G0MetricStrain(value=value, raw_derivative=derivative)


def auxiliary_fourier_strain(auxcell, grid_vectors):
    r"""Differentiate auxiliary AO Fourier factors at fixed fractional centers.

    For one auxiliary Gaussian centered at :math:`R_P`, write

    .. math::

       F_P(G)=\int \phi_P(r-R_P)e^{-iG\cdot r}\,dr.

    The phase :math:`G\cdot R_P` is invariant when the reciprocal vector and
    fractional center transform together.  Only the Gaussian shape response
    remains,

    .. math::

       \frac{\partial F_P}{\partial\epsilon_{xy}}
       = iG_x\int(r_y-R_{Py})\phi_P(r-R_P)e^{-iG\cdot r}\,dr.
    """

    vectors = np.asarray(grid_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"grid_vectors must have shape (nG,3), got {vectors.shape}")
    value = np.asarray(periodic_ft_ao.ft_ao(auxcell, vectors)).T
    global_moment = _fourier_cartesian_moment(auxcell, vectors)
    centers = _ao_centers(auxcell)
    relative_moment = global_moment - np.einsum(
        "py,pg->ypg", centers, value
    )
    derivative = 1j * np.einsum(
        "gx,ypg->xypg", vectors, relative_moment, optimize=True
    )
    return AuxiliaryFourierStrain(
        value=np.asarray(value), raw_derivative=np.asarray(derivative)
    )


def long_range_metric_strain(
    cell, auxcell, omega, mesh, *, reciprocal_block_size=None,
):
    """Return the analytic long-range GDF auxiliary metric strain response."""

    return kpoint_long_range_metric_strain(
        cell, auxcell, omega, mesh, np.zeros(3),
        reciprocal_block_size=reciprocal_block_size,
    )


def kpoint_long_range_metric_strain(
    cell, auxcell, omega, mesh, q, *, reciprocal_block_size=None,
):
    """Return a block-contracted long-range metric at scaled transfer q."""

    mesh_array = np.asarray(mesh, dtype=int)
    if mesh_array.shape != (3,) or np.any(mesh_array <= 0):
        raise ValueError(f"mesh must contain three positive integers: {mesh_array}")
    transfer = _kpoint(q)
    grid_vectors = cell.get_Gv(mesh_array)
    naux = int(auxcell.nao_nr())
    block_size, estimated_block_bytes = _reciprocal_block_size(
        reciprocal_block_size, grid_points=len(grid_vectors), naux=naux,
    )
    value = np.zeros((naux, naux), dtype=np.complex128)
    auxiliary_response = np.empty(
        (3, 3, naux, naux), dtype=np.complex128
    )
    auxiliary_response.fill(0)
    kernel_response = np.empty_like(auxiliary_response)
    kernel_response.fill(0)
    for p0 in range(0, len(grid_vectors), block_size):
        momentum = grid_vectors[p0:p0+block_size]+transfer
        auxiliary = auxiliary_fourier_strain(auxcell, momentum)
        kernel = weighted_coulomb_strain(cell, momentum, omega=float(omega))
        value += (
            auxiliary.value.conj()*kernel.value
        )@auxiliary.value.T
        for x in range(3):
            for y in range(3):
                derivative = auxiliary.raw_derivative[x, y]
                auxiliary_response[x, y] += (
                    (derivative.conj()*kernel.value)@auxiliary.value.T
                    + (auxiliary.value.conj()*kernel.value)@derivative.T
                )
                kernel_response[x, y] += (
                    auxiliary.value.conj()*kernel.raw_derivative[x, y]
                )@auxiliary.value.T
    total = auxiliary_response + kernel_response
    return LongRangeMetricStrain(
        value=_real_if_close(value),
        raw_derivative=_real_if_close(total),
        auxiliary_response=_real_if_close(auxiliary_response),
        kernel_response=_real_if_close(kernel_response),
        reciprocal_block_size=block_size,
        grid_points=int(len(grid_vectors)),
        estimated_block_work_bytes=estimated_block_bytes,
    )


def reciprocal_metric_strain(
    cell, auxcell, omega, mesh, *, reciprocal_block_size=None,
):
    """Return the complete reciprocal and G=0 metric response at Gamma."""

    return kpoint_reciprocal_metric_strain(
        cell, auxcell, omega, mesh, np.zeros(3),
        reciprocal_block_size=reciprocal_block_size,
    )


def kpoint_reciprocal_metric_strain(
    cell, auxcell, omega, mesh, q, *, reciprocal_block_size=None,
):
    """Return reciprocal metric response for one momentum transfer."""

    transfer = _kpoint(q)
    long_range = kpoint_long_range_metric_strain(
        cell, auxcell, omega, mesh, transfer,
        reciprocal_block_size=reciprocal_block_size,
    )
    if np.linalg.norm(transfer) < 1e-10:
        g0 = g0_metric_strain(cell, auxcell, omega)
    else:
        naux = int(auxcell.nao_nr())
        g0 = G0MetricStrain(
            value=np.zeros((naux, naux), dtype=np.complex128),
            raw_derivative=np.zeros(
                (3, 3, naux, naux), dtype=np.complex128
            ),
        )
    return ReciprocalMetricStrain(
        value=_real_if_close(long_range.value + g0.value),
        raw_derivative=_real_if_close(
            long_range.raw_derivative + g0.raw_derivative
        ),
        long_range=long_range,
        g0=g0,
    )


def short_range_metric_strain(auxcell, omega, *, precision=None):
    """Return an analytic image-resolved short-range metric strain.

    The implementation is intentionally transparent for the CPU reference:
    every lattice image is retained so its center velocity under homogeneous
    strain is known.  A later batched C driver may accelerate the same
    contraction without changing the equations.
    """

    return kpoint_short_range_metric_strain(
        auxcell, omega, np.zeros(3), precision=precision
    )


def kpoint_short_range_metric_strain(
    auxcell, omega, q, *, precision=None
):
    """Return the image-resolved short-range metric response at transfer q."""

    range_parameter = abs(float(omega))
    if range_parameter <= 0:
        raise ValueError("short-range metric strain requires nonzero omega")
    target_precision = (
        float(auxcell.precision) ** 1.5
        if precision is None else float(precision)
    )
    rcut = float(
        estimate_rs_2c2e_rcut(auxcell, range_parameter, target_precision)
    )
    lattice_vectors = np.asarray(auxcell.get_lattice_Ls(rcut=rcut), dtype=float)
    transfer = _kpoint(q)
    centers = _ao_centers(auxcell)
    naux = int(auxcell.nao_nr())
    value = np.zeros((naux, naux), dtype=np.complex128)
    derivative = np.zeros((3, 3, naux, naux), dtype=np.complex128)

    with auxcell.with_short_range_coulomb(range_parameter):
        for lattice in lattice_vectors:
            shifted = auxcell.copy(deep=True)
            shifted.set_geom_(auxcell.atom_coords() + lattice, unit="Bohr")
            with shifted.with_short_range_coulomb(range_parameter):
                block = gto.intor_cross("int2c2e", auxcell, shifted)
                ip1 = gto.intor_cross(
                    "int2c2e_ip1", auxcell, shifted, comp=3
                )
                ip2 = gto.intor_cross(
                    "int2c2e_ip2", auxcell, shifted, comp=3
                )
            phase = np.exp(1j * np.dot(transfer, lattice))
            value += phase * np.asarray(block)
            shifted_centers = centers + lattice
            image_derivative = -np.einsum(
                "xij,iy->xyij", ip1, centers, optimize=True
            )
            image_derivative -= np.einsum(
                "xij,jy->xyij", ip2, shifted_centers, optimize=True
            )
            derivative += phase * image_derivative
    return ShortRangeMetricStrain(
        value=_real_if_close(value),
        raw_derivative=_real_if_close(derivative),
        lattice_vectors=int(len(lattice_vectors)),
        rcut_bohr=rcut,
    )


def gamma_auxiliary_metric_strain(
    cell, auxcell, omega, mesh, *, precision=None,
    reciprocal_block_size=None,
):
    """Return short-range, reciprocal, and G=0 metric strain at Gamma."""

    general = kpoint_auxiliary_metric_strain(
        cell, auxcell, omega, mesh, np.zeros(3), precision=precision,
        reciprocal_block_size=reciprocal_block_size,
    )
    return GammaAuxiliaryMetricStrain(
        value=general.value,
        raw_derivative=general.raw_derivative,
        short_range=general.short_range,
        reciprocal=general.reciprocal,
    )


def kpoint_auxiliary_metric_strain(
    cell, auxcell, omega, mesh, q, *, precision=None,
    reciprocal_block_size=None,
):
    """Return the complete complex GDF metric strain at one transfer q."""

    transfer = _kpoint(q)
    short_range = kpoint_short_range_metric_strain(
        auxcell, omega, transfer, precision=precision
    )
    reciprocal = kpoint_reciprocal_metric_strain(
        cell, auxcell, omega, mesh, transfer,
        reciprocal_block_size=reciprocal_block_size,
    )
    return KPointAuxiliaryMetricStrain(
        value=_real_if_close(short_range.value + reciprocal.value),
        raw_derivative=_real_if_close(
            short_range.raw_derivative + reciprocal.raw_derivative
        ),
        short_range=short_range,
        reciprocal=reciprocal,
        momentum_transfer=transfer,
    )


def _reciprocal_block_size(requested, *, grid_points, naux):
    """Bound auxiliary Fourier work while preserving exact accumulation."""

    points = int(grid_points)
    if points < 1:
        raise ValueError("reciprocal grid must contain at least one point")
    bytes_per_point = np.dtype(np.complex128).itemsize*20*int(naux)
    if requested is None:
        block = max(1, _DEFAULT_RECIPROCAL_BLOCK_BYTES//bytes_per_point)
    else:
        block = int(requested)
        if block < 1:
            raise ValueError("reciprocal_block_size must be positive")
    block = min(points, block)
    return block, int(block*bytes_per_point)


def matrix_tensor_to_voigt(value):
    """Project raw ordered strain derivatives onto engineering Voigt strain."""

    tensor = np.asarray(value)
    if tensor.shape[:2] != (3, 3):
        raise ValueError(f"strain derivative must start with (3,3), got {tensor.shape}")
    return np.asarray(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            0.5 * (tensor[1, 2] + tensor[2, 1]),
            0.5 * (tensor[0, 2] + tensor[2, 0]),
            0.5 * (tensor[0, 1] + tensor[1, 0]),
        ]
    )


def _fourier_cartesian_moment(mol, grid_vectors):
    """Return ``integral r_y phi_p exp(-iG.r)`` as ``(3,nao,nG)``."""

    ghost = gto.Mole()
    ghost._atm = np.asarray([[0, 0, 0, 0, 0, 0]], dtype=np.int32)
    ghost._bas = np.asarray([[0, 0, 1, 1, 0, 0, 3, 0]], dtype=np.int32)
    ghost._env = np.zeros(4)
    ghost._env[3] = np.sqrt(4.0 * np.pi)
    ghost._ecpbas = np.zeros((0, gto.BAS_SLOTS), dtype=np.int32)
    ghost.cart = bool(mol.cart)
    ghost._built = True

    combined = gto.conc_mol(mol, ghost)
    shell_slice = (0, mol.nbas, mol.nbas, mol.nbas + 1)
    moment = molecular_ft_ao.ft_aopair(
        combined,
        np.asarray(grid_vectors, dtype=float),
        shls_slice=shell_slice,
        aosym="s1",
        intor="GTO_ft_rc",
        comp=3,
    )
    return np.asarray(moment)[:, :, :, 0].transpose(0, 2, 1)


def _ao_centers(cell):
    centers = np.empty((cell.nao_nr(), 3), dtype=float)
    for atom_id, entry in enumerate(np.asarray(cell.aoslice_by_atom(), dtype=int)):
        p0, p1 = (int(entry[2]), int(entry[3]))
        centers[p0:p1] = np.asarray(cell.atom_coord(atom_id), dtype=float)
    return centers


def _kpoint(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"momentum transfer must have shape (3,), got {point.shape}")
    return point


def _real_if_close(value, tolerance=1e-10):
    array = np.asarray(value)
    if array.size and np.max(np.abs(array.imag)) <= tolerance:
        return array.real
    return array
