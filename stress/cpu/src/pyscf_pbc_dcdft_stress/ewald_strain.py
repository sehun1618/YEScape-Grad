"""Analytic homogeneous-strain derivative of the three-dimensional Ewald sum.

The implementation follows :func:`pyscf.pbc.gto.cell.ewald`, but keeps the
central Ewald splitting parameter, real-space image list, and reciprocal mesh
fixed while differentiating the physical cell.  Holding these numerical
partitions fixed avoids differentiating cutoff-selection heuristics; their
dependence cancels between the converged real- and reciprocal-space sums.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

from .gdf_metric_strain import matrix_tensor_to_voigt


@dataclass(frozen=True)
class EwaldStrain:
    """Ewald energy decomposition and raw Cartesian strain derivative."""

    value: float
    raw_derivative: np.ndarray
    real_value: float
    self_value: float
    reciprocal_value: float
    real_derivative: np.ndarray
    self_derivative: np.ndarray
    reciprocal_derivative: np.ndarray
    eta: float
    cutoff: float
    mesh: tuple[int, int, int]
    lattice_image_count: int

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


def analytic_ewald_strain(cell, *, eta=None, cutoff=None, mesh=None,
                          reciprocal_block_size=262144) -> EwaldStrain:
    """Return the 3D ion-ion Ewald energy and analytic cell derivative.

    Atomic fractional coordinates are fixed.  ``eta``, ``cutoff``, and the
    integer reciprocal ``mesh`` describe the central numerical partition and
    are not physical strain variables.
    """

    if int(getattr(cell, "dimension", 3)) != 3:
        raise NotImplementedError("analytic Ewald strain currently supports 3D cells")
    if bool(getattr(cell, "use_particle_mesh_ewald", False)):
        raise NotImplementedError("particle-mesh Ewald strain is not implemented")
    if cell.a is None:
        raise NotImplementedError("molecular nuclear repulsion has no cell strain")
    if int(cell.natm) == 0:
        zeros = np.zeros((3, 3), dtype=float)
        return EwaldStrain(
            value=0.0, raw_derivative=zeros, real_value=0.0,
            self_value=0.0, reciprocal_value=0.0,
            real_derivative=zeros, self_derivative=zeros,
            reciprocal_derivative=zeros, eta=0.0, cutoff=0.0,
            mesh=(0, 0, 0), lattice_image_count=0,
        )
    if reciprocal_block_size <= 0:
        raise ValueError("reciprocal_block_size must be positive")

    if eta is None or cutoff is None:
        eta, cutoff = cell.get_ewald_params()
    eta = float(eta)
    cutoff = float(cutoff)
    if eta <= 0.0 or cutoff <= 0.0:
        raise ValueError(f"invalid Ewald parameters eta={eta}, cutoff={cutoff}")

    charges = np.asarray(cell.atom_charges(), dtype=float)
    coordinates = np.asarray(cell.atom_coords(), dtype=float)
    if mesh is None:
        log_precision = np.log(
            float(cell.precision) / (float(charges.sum()) * 16.0 * np.pi**2)
        )
        ke_cutoff = -2.0 * eta**2 * log_precision
        mesh = np.asarray(cell.cutoff_to_mesh(ke_cutoff), dtype=int)
    else:
        mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError(f"Ewald mesh must contain three positive integers: {mesh}")

    lattice = np.asarray(cell.get_lattice_Ls(rcut=cutoff), dtype=float)
    real_value, real_derivative = _real_space_response(
        coordinates, charges, lattice, eta
    )
    self_value, self_derivative = _self_response(charges, eta, float(cell.vol))
    reciprocal_value, reciprocal_derivative = _reciprocal_response(
        cell, coordinates, charges, mesh, eta, reciprocal_block_size
    )
    raw_derivative = real_derivative + self_derivative + reciprocal_derivative
    return EwaldStrain(
        value=float(real_value + self_value + reciprocal_value),
        raw_derivative=raw_derivative,
        real_value=float(real_value),
        self_value=float(self_value),
        reciprocal_value=float(reciprocal_value),
        real_derivative=real_derivative,
        self_derivative=self_derivative,
        reciprocal_derivative=reciprocal_derivative,
        eta=eta,
        cutoff=cutoff,
        mesh=tuple(int(value) for value in mesh),
        lattice_image_count=int(len(lattice)),
    )


def _real_space_response(coordinates, charges, lattice, eta):
    charge_products = charges[:, None] * charges[None, :]
    value = 0.0
    derivative = np.zeros((3, 3), dtype=float)
    eta_factor = 2.0 * eta / np.sqrt(np.pi)
    for image in lattice:
        vectors = (
            coordinates[:, None, :] - coordinates[None, :, :]
            + image[None, None, :]
        )
        radii2 = np.einsum("ijx,ijx->ij", vectors, vectors, optimize=True)
        mask = radii2 >= 1.0e-32
        radii = np.ones_like(radii2)
        radii[mask] = np.sqrt(radii2[mask])
        screened = np.zeros_like(radii)
        screened[mask] = erfc(eta * radii[mask]) / radii[mask]
        value += 0.5 * np.einsum(
            "ij,ij->", charge_products, screened, optimize=True
        )

        radial_factor = np.zeros_like(radii)
        radial_factor[mask] = (
            -erfc(eta * radii[mask]) / radii[mask] ** 3
            - eta_factor * np.exp(-(eta * radii[mask]) ** 2)
            / radii[mask] ** 2
        )
        derivative += 0.5 * np.einsum(
            "ij,ij,ijx,ijy->xy", charge_products, radial_factor,
            vectors, vectors, optimize=True,
        )
    return value, derivative


def _self_response(charges, eta, volume):
    point_self = -float(np.dot(charges, charges)) * eta / np.sqrt(np.pi)
    background = -0.5 * float(charges.sum()) ** 2 * np.pi / (eta**2 * volume)
    derivative = -background * np.eye(3)
    return point_self + background, derivative


def _reciprocal_response(cell, coordinates, charges, mesh, eta, block_size):
    reciprocal_vectors, _, weights = cell.get_Gv_weights(mesh)
    if not np.isscalar(weights):
        raise NotImplementedError("analytic Ewald strain requires uniform 3D weights")
    weights = float(weights)
    value = 0.0
    derivative = np.zeros((3, 3), dtype=float)
    identity = np.eye(3)
    for start in range(0, len(reciprocal_vectors), block_size):
        vectors = np.asarray(reciprocal_vectors[start:start + block_size])
        norms2 = np.einsum("gx,gx->g", vectors, vectors, optimize=True)
        mask = norms2 > 1.0e-28
        if not np.any(mask):
            continue
        vectors = vectors[mask]
        norms2 = norms2[mask]
        phases = np.exp(-1j * np.dot(coordinates, vectors.T))
        structure = np.einsum("a,ag->g", charges, phases, optimize=True)
        amplitudes = structure.conj() * structure
        kernel = (
            4.0 * np.pi * weights / norms2
            * np.exp(-norms2 / (4.0 * eta**2))
        )
        weighted_amplitudes = (amplitudes.real * kernel)
        value += 0.5 * float(np.sum(weighted_amplitudes))
        logarithmic_response = (
            -identity[None, :, :]
            + (
                2.0 / norms2 + 1.0 / (2.0 * eta**2)
            )[:, None, None]
            * np.einsum("gx,gy->gxy", vectors, vectors, optimize=True)
        )
        derivative += 0.5 * np.einsum(
            "g,gxy->xy", weighted_amplitudes, logarithmic_response,
            optimize=True,
        )
    return value, derivative
