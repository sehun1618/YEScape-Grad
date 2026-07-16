"""Real-space grid primitives shared by FFTDF derivative kernels.

The functions here contain no DC-DFT-specific factors.  They expose AO values,
AO center derivatives, k-averaged densities, and Coulomb convolutions using
the exact ``cell.mesh`` selected by the energy calculation.  Keeping this
layer small makes cutoff/mesh mismatches visible instead of silently building
a second grid inside the gradient code.
"""

from __future__ import annotations

import numpy as np
from pyscf.pbc import tools
from pyscf.pbc.dft.numint import eval_ao_kpts


def ao_grid_data(cell, kpts):
    """Return coordinates, AO values, and first spatial AO derivatives.

    Returns
    -------
    tuple
        ``coords`` has shape ``(ngrids, 3)``.  Each entry of ``ao_values`` has
        shape ``(ngrids, nao)``; each derivative entry has shape
        ``(3, ngrids, nao)``.
    """

    kpoint_array = np.asarray(kpts, dtype=float).reshape(-1, 3)
    coordinates = cell.gen_uniform_grids(cell.mesh)
    evaluated = eval_ao_kpts(
        cell,
        coordinates,
        kpts=kpoint_array,
        deriv=1,
    )
    values = [np.asarray(block[0]) for block in evaluated]
    spatial_derivatives = [np.asarray(block[1:4]) for block in evaluated]
    return coordinates, values, spatial_derivatives


def atom_ao_center_derivatives(
    cell,
    ao_values: list[np.ndarray],
    spatial_derivatives: list[np.ndarray],
    atom_id: int,
    axis: int,
) -> list[np.ndarray]:
    """Return ``d chi_mu / dR_Ax`` on the grid for every k-point."""

    p0, p1 = np.asarray(cell.aoslice_by_atom(), dtype=int)[atom_id, 2:]
    output = []
    for values, derivatives in zip(
        ao_values,
        spatial_derivatives,
        strict=True,
    ):
        centered = np.zeros_like(values)
        # Nuclear translation is the negative electronic-coordinate
        # derivative for AOs centered on the translated atom.
        centered[:, p0:p1] = -derivatives[axis, :, p0:p1]
        output.append(centered)
    return output


def k_averaged_density(
    ao_values: list[np.ndarray],
    density_kpts: np.ndarray,
) -> np.ndarray:
    """Evaluate the spin-summed, equally weighted periodic density."""

    density = np.zeros(ao_values[0].shape[0], dtype=np.complex128)
    for ao, dm in zip(ao_values, np.asarray(density_kpts), strict=True):
        density += np.einsum("gi,ij,gj->g", ao.conj(), dm, ao, optimize=True)
    return density / len(ao_values)


def k_averaged_density_derivative(
    ao_values: list[np.ndarray],
    ao_derivatives: list[np.ndarray],
    density_kpts: np.ndarray,
) -> np.ndarray:
    """Differentiate the grid density at fixed AO density coefficients."""

    derivative = np.zeros(ao_values[0].shape[0], dtype=np.complex128)
    for ao, dao, dm in zip(
        ao_values,
        ao_derivatives,
        np.asarray(density_kpts),
        strict=True,
    ):
        derivative += np.einsum(
            "gi,ij,gj->g", dao.conj(), dm, ao, optimize=True
        )
        derivative += np.einsum(
            "gi,ij,gj->g", ao.conj(), dm, dao, optimize=True
        )
    return derivative / len(ao_values)


def hartree_potential_weighted(cell, density: np.ndarray) -> np.ndarray:
    """Apply the reciprocal Coulomb kernel and real-space quadrature weight."""

    coulomb_g = tools.get_coulG(cell, mesh=cell.mesh)
    potential = tools.ifft(
        tools.fft(np.asarray(density), cell.mesh) * coulomb_g,
        cell.mesh,
    )
    return potential * (cell.vol / len(coulomb_g))


def phase_factor(coordinates: np.ndarray, wavevector: np.ndarray) -> np.ndarray:
    """Return ``exp(-i r.q)`` with a scalar fast path at ``q=0``."""

    if np.max(np.abs(wavevector)) < 1.0e-12:
        return np.array(1.0)
    return np.exp(-1j * coordinates @ wavevector)
