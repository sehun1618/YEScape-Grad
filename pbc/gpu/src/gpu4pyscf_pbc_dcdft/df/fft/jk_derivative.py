"""Full fixed-density FFTDF J/K derivative matrices on the GPU.

GPU4PySCF 1.8.0 provides FFT J/K values and a sampled-k contracted J energy
derivative, but ``get_k_e1_kpts`` is still not implemented.  The upstream J
result is an atom-contracted energy derivative rather than the complete AO
matrix needed by the nonvariational DC-DFT response RHS.  This module therefore
forms ``d(J-K/2)/dR_Ax`` by applying the product rule to every AO occurrence
with CuPy AO collocation, FFTs and contractions.  It intentionally mirrors the
CPU package's transparent correctness kernel so CPU/GPU tensor parity can be
tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from gpu4pyscf.pbc import tools

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
    """Device-resident full J and K derivatives in Hartree/Bohr."""

    coulomb: cp.ndarray
    exchange: cp.ndarray

    @property
    def restricted_hf(self):
        """Return ``d(J-K/2)`` without copying the tensor to the host."""

        return self.coulomb - 0.5 * self.exchange


def coulomb_derivative_matrices(cell, kpts, density_kpts):
    """Return GPU ``dJ_k/dR_Ax`` with ``(A,k,x,p,q)`` ordering."""

    kpoint_array = np.asarray(kpts, dtype=float).reshape(-1, 3)
    density = cp.asarray(density_kpts, order="C")
    _, ao_values, ao_spatial = ao_grid_data(cell, kpoint_array)
    potential = hartree_potential_weighted(
        cell, k_averaged_density(ao_values, density)
    )
    output = cp.zeros(
        (cell.natm, len(kpoint_array), 3, cell.nao_nr(), cell.nao_nr()),
        dtype=cp.complex128,
    )
    for atom_id in range(cell.natm):
        for axis in range(3):
            dao_values = atom_ao_center_derivatives(
                cell, ao_values, ao_spatial, atom_id, axis
            )
            density_derivative = k_averaged_density_derivative(
                ao_values, dao_values, density
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
                output[atom_id, k_index, axis] = matrix
    return output


def _exchange_component(
    cell,
    kpts,
    density_kpts,
    atom_id: int,
    axis: int,
    *,
    exxdiv,
    with_df,
):
    """Differentiate every AO factor in one FFT exchange evaluation."""

    coordinates, ao_values, ao_spatial = ao_grid_data(cell, kpts)
    dao_values = atom_ao_center_derivatives(
        cell, ao_values, ao_spatial, atom_id, axis
    )
    nkpts = len(kpts)
    nao = int(cell.nao_nr())
    ngrids = len(coordinates)
    quadrature_weight = (cell.vol / ngrids) / nkpts
    output = cp.zeros((nkpts, nao, nao), dtype=cp.complex128)

    for source_k, (ao_source, dao_source, dm_source) in enumerate(
        zip(ao_values, dao_values, density_kpts, strict=True)
    ):
        ao_source_t = cp.ascontiguousarray(ao_source.T)
        dao_source_t = cp.ascontiguousarray(dao_source.T)
        density_times_ao = dm_source @ ao_source_t.conj()
        density_times_dao = dm_source @ dao_source_t.conj()

        for target_k, (ao_target, dao_target) in enumerate(
            zip(ao_values, dao_values, strict=True)
        ):
            ao_target_t = cp.ascontiguousarray(ao_target.T)
            dao_target_t = cp.ascontiguousarray(dao_target.T)
            qvector = kpts[source_k] - kpts[target_k]
            coulomb_g = tools.get_coulG(
                cell,
                qvector,
                exxdiv,
                with_df,
                cell.mesh,
                kpts=kpts,
            )
            phase = phase_factor(coordinates, qvector)
            phase_conj = phase.conj()

            for row in range(nao):
                contracted = cp.zeros(ngrids, dtype=cp.complex128)
                derivative = cp.zeros_like(contracted)
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
                    contracted += pair_potential * density_times_ao[source_column]
                    derivative += (
                        pair_potential_derivative
                        * density_times_ao[source_column]
                    )
                    derivative += pair_potential * density_times_dao[source_column]

                weighted = contracted * phase_conj
                weighted_derivative = derivative * phase_conj
                output[target_k, row] += quadrature_weight * (
                    weighted_derivative @ ao_target + weighted @ dao_target
                )
    return output


def exchange_derivative_matrices(
    cell,
    kpts,
    density_kpts,
    *,
    exxdiv="ewald",
    with_df,
):
    """Return GPU ``dK_k/dR_Ax`` with ``(A,k,x,p,q)`` ordering."""

    kpoint_array = np.asarray(kpts, dtype=float).reshape(-1, 3)
    density = cp.asarray(density_kpts, order="C")
    output = cp.zeros(
        (cell.natm, len(kpoint_array), 3, cell.nao_nr(), cell.nao_nr()),
        dtype=cp.complex128,
    )
    for atom_id in range(cell.natm):
        for axis in range(3):
            output[atom_id, :, axis] = _exchange_component(
                cell,
                kpoint_array,
                density,
                atom_id,
                axis,
                exxdiv=exxdiv,
                with_df=with_df,
            )
    return output


def fft_jk_derivative(
    cell,
    kpts,
    density_kpts,
    *,
    exxdiv="ewald",
    with_df,
):
    """Build device-resident full fixed-density J/K derivative tensors."""

    coulomb = coulomb_derivative_matrices(cell, kpts, density_kpts)
    exchange = exchange_derivative_matrices(
        cell,
        kpts,
        density_kpts,
        exxdiv=exxdiv,
        with_df=with_df,
    )
    return FFTJKDerivative(coulomb=coulomb, exchange=exchange)
