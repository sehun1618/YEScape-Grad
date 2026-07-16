"""CuPy real-space primitives for full FFTDF nuclear derivatives.

All dense numerical arrays returned by this module remain on the active GPU.
The PySCF ``Cell`` and its integer AO slices stay on the host because they are
small immutable metadata, not a numerical fallback.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.dft.numint import eval_ao_kpts
from gpu4pyscf.pbc import tools


def ao_grid_data(cell, kpts):
    """Return GPU AO values and first spatial derivatives on ``cell.mesh``."""

    kpoint_array = np.asarray(kpts, dtype=float).reshape(-1, 3)
    coordinates_host = cell.gen_uniform_grids(cell.mesh)
    evaluated = eval_ao_kpts(
        cell,
        coordinates_host,
        kpts=kpoint_array,
        deriv=1,
    )
    coordinates = cp.asarray(coordinates_host)
    values = [block[0] for block in evaluated]
    spatial = [block[1:4] for block in evaluated]
    return coordinates, values, spatial


def atom_ao_center_derivatives(cell, values, spatial, atom_id: int, axis: int):
    """Return ``d chi_mu / dR_Ax`` on device for every sampled k-point."""

    p0, p1 = np.asarray(cell.aoslice_by_atom(), dtype=int)[atom_id, 2:]
    output = []
    for ao, derivative in zip(values, spatial, strict=True):
        centered = cp.zeros_like(ao)
        centered[:, p0:p1] = -derivative[axis, :, p0:p1]
        output.append(centered)
    return output


def k_averaged_density(values, density_kpts):
    """Evaluate the spin-summed periodic density on the GPU grid."""

    density = cp.zeros(values[0].shape[0], dtype=cp.complex128)
    for ao, dm in zip(values, density_kpts, strict=True):
        density += cp.einsum("gi,ij,gj->g", ao.conj(), dm, ao)
    return density / len(values)


def k_averaged_density_derivative(values, derivatives, density_kpts):
    """Differentiate the fixed-coefficient AO density on the GPU grid."""

    result = cp.zeros(values[0].shape[0], dtype=cp.complex128)
    for ao, dao, dm in zip(values, derivatives, density_kpts, strict=True):
        result += cp.einsum("gi,ij,gj->g", dao.conj(), dm, ao)
        result += cp.einsum("gi,ij,gj->g", ao.conj(), dm, dao)
    return result / len(values)


def hartree_potential_weighted(cell, density):
    """Apply the reciprocal Coulomb kernel and real-space quadrature weight."""

    coulomb_g = tools.get_coulG(cell, mesh=cell.mesh)
    potential = tools.ifft(tools.fft(density, cell.mesh) * coulomb_g, cell.mesh)
    return potential * (cell.vol / len(coulomb_g))


def phase_factor(coordinates, wavevector):
    """Return ``exp(-i r.q)`` as a CuPy scalar or grid vector."""

    qvector = np.asarray(wavevector, dtype=float)
    if np.max(np.abs(qvector)) < 1.0e-12:
        return cp.asarray(1.0)
    return cp.exp(-1j * coordinates @ cp.asarray(qvector))
