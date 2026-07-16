"""Full fixed-density FFTDF J/K nuclear derivative matrices.

PySCF's CPU FFTDF ``get_j_e1`` and ``get_k_e1`` return AO-row derivatives,
which are sufficient for an ordinary variational gradient after contraction
with the SCF density.  The DC-DFT nuclear response source instead needs the
complete matrix ``dG[D_HF]/dR_Ax`` before its virtual--occupied projection.
This module applies the product rule to every AO occurrence in the finite-grid
FFTDF J and K definitions.

The implementation prioritizes a transparent correctness baseline.  The
exchange kernel is intentionally direct and is suitable for the small systems
used by the Stage-2 completion gate; optimized blocked kernels can replace it
without changing the public tensor contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf.pbc import tools

from .grid import (
    ao_grid_data,
    atom_ao_center_derivatives,
    hartree_potential_weighted,
    k_averaged_density,
    k_averaged_density_derivative,
    phase_factor,
)


@dataclass(frozen=True)
class FFTJKDerivative:
    """Full J and K derivative matrices in Hartree/Bohr."""

    coulomb: np.ndarray
    exchange: np.ndarray

    @property
    def restricted_hf(self) -> np.ndarray:
        """Return ``d(J - K/2)`` for a spin-restricted HF density."""

        return self.coulomb - 0.5 * self.exchange


def coulomb_derivative_matrices(cell, kpts, density_kpts) -> np.ndarray:
    """Return ``dJ_k/dR_Ax`` with shape ``(natm,nkpts,3,nao,nao)``."""

    kpoint_array = np.asarray(kpts, dtype=float).reshape(-1, 3)
    density_array = np.asarray(density_kpts)
    _, ao_values, ao_spatial = ao_grid_data(cell, kpoint_array)
    density_grid = k_averaged_density(ao_values, density_array)
    potential = hartree_potential_weighted(cell, density_grid)
    derivatives = np.zeros(
        (cell.natm, len(kpoint_array), 3, cell.nao_nr(), cell.nao_nr()),
        dtype=np.complex128,
    )

    for atom_id in range(cell.natm):
        for axis in range(3):
            dao_values = atom_ao_center_derivatives(
                cell, ao_values, ao_spatial, atom_id, axis
            )
            density_derivative = k_averaged_density_derivative(
                ao_values, dao_values, density_array
            )
            potential_derivative = hartree_potential_weighted(
                cell, density_derivative
            )
            for k_index, (ao, dao) in enumerate(
                zip(ao_values, dao_values, strict=True)
            ):
                matrix = dao.conj().T @ (potential[:, None] * ao)
                matrix += ao.conj().T @ (potential[:, None] * dao)
                matrix += ao.conj().T @ (potential_derivative[:, None] * ao)
                derivatives[atom_id, k_index, axis] = matrix
    return derivatives


def _exchange_component(
    cell,
    kpts: np.ndarray,
    density_kpts: np.ndarray,
    atom_id: int,
    axis: int,
    *,
    exxdiv,
    with_df,
) -> np.ndarray:
    """Differentiate every AO factor in the FFT exchange matrix."""

    coordinates, ao_values, ao_spatial = ao_grid_data(cell, kpts)
    dao_values = atom_ao_center_derivatives(
        cell, ao_values, ao_spatial, atom_id, axis
    )
    nkpts = len(kpts)
    nao = int(cell.nao_nr())
    ngrids = len(coordinates)
    quadrature_weight = (cell.vol / ngrids) / nkpts
    output = np.zeros((nkpts, nao, nao), dtype=np.complex128)

    for source_k, (ao_source, dao_source, dm_source) in enumerate(
        zip(ao_values, dao_values, density_kpts, strict=True)
    ):
        ao_source_t = np.asarray(ao_source.T, order="C")
        dao_source_t = np.asarray(dao_source.T, order="C")
        density_times_ao = dm_source @ ao_source_t.conj()
        density_times_dao = dm_source @ dao_source_t.conj()

        for target_k, (ao_target, dao_target) in enumerate(
            zip(ao_values, dao_values, strict=True)
        ):
            ao_target_t = np.asarray(ao_target.T, order="C")
            dao_target_t = np.asarray(dao_target.T, order="C")
            qvector = kpts[source_k] - kpts[target_k]
            coulomb_g = tools.get_coulG(
                cell,
                qvector,
                exxdiv,
                with_df,
                cell.mesh,
            )
            phase = phase_factor(coordinates, qvector)
            phase_conj = np.conj(phase)

            for row in range(nao):
                contracted = np.zeros(ngrids, dtype=np.complex128)
                contracted_derivative = np.zeros_like(contracted)
                for source_column in range(nao):
                    pair_density = (
                        ao_target_t[row].conj()
                        * phase
                        * ao_source_t[source_column]
                    )
                    pair_potential = tools.ifft(
                        tools.fft(pair_density, cell.mesh) * coulomb_g,
                        cell.mesh,
                    )
                    pair_density_derivative = (
                        dao_target_t[row].conj()
                        * phase
                        * ao_source_t[source_column]
                    )
                    pair_density_derivative += (
                        ao_target_t[row].conj()
                        * phase
                        * dao_source_t[source_column]
                    )
                    pair_potential_derivative = tools.ifft(
                        tools.fft(pair_density_derivative, cell.mesh)
                        * coulomb_g,
                        cell.mesh,
                    )
                    contracted += (
                        pair_potential * density_times_ao[source_column]
                    )
                    contracted_derivative += (
                        pair_potential_derivative
                        * density_times_ao[source_column]
                    )
                    contracted_derivative += (
                        pair_potential * density_times_dao[source_column]
                    )

                weighted = contracted * phase_conj
                weighted_derivative = contracted_derivative * phase_conj
                output[target_k, row] += quadrature_weight * (
                    weighted_derivative @ ao_target
                    + weighted @ dao_target
                )
    return output


def exchange_derivative_matrices(
    cell,
    kpts,
    density_kpts,
    *,
    exxdiv="ewald",
    with_df,
) -> np.ndarray:
    """Return ``dK_k/dR_Ax`` with shape ``(natm,nkpts,3,nao,nao)``."""

    kpoint_array = np.asarray(kpts, dtype=float).reshape(-1, 3)
    density_array = np.asarray(density_kpts)
    derivatives = np.zeros(
        (cell.natm, len(kpoint_array), 3, cell.nao_nr(), cell.nao_nr()),
        dtype=np.complex128,
    )
    for atom_id in range(cell.natm):
        for axis in range(3):
            derivatives[atom_id, :, axis] = _exchange_component(
                cell,
                kpoint_array,
                density_array,
                atom_id,
                axis,
                exxdiv=exxdiv,
                with_df=with_df,
            )
    return derivatives


def fft_jk_derivative(
    cell,
    kpts,
    density_kpts,
    *,
    exxdiv="ewald",
    with_df,
) -> FFTJKDerivative:
    """Build full fixed-density FFTDF J and K derivative tensors."""

    coulomb = coulomb_derivative_matrices(cell, kpts, density_kpts)
    exchange = exchange_derivative_matrices(
        cell,
        kpts,
        density_kpts,
        exxdiv=exxdiv,
        with_df=with_df,
    )
    return FFTJKDerivative(coulomb=coulomb, exchange=exchange)
