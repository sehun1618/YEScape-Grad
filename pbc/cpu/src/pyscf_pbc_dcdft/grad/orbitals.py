"""Occupied/virtual orbital partitions and the DC-DFT response RHS.

The target DFA is evaluated at HF orbitals.  The HF Fock virtual--occupied
block vanishes at convergence, but the target DFA Fock block generally does
not.  That target block is the derivative of the nonvariational energy with
respect to an occupied--virtual orbital rotation and therefore drives the
Z-vector equation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MOPartition:
    """MO blocks for one k-point.

    Attributes
    ----------
    occupied_coeff, virtual_coeff
        AO-to-MO coefficients with shapes ``(nao, nocc)`` and
        ``(nao, nvir)``.
    occupied_mask, virtual_mask
        Masks into the full MO dimension.
    energy_gap
        ``epsilon_a - epsilon_i`` with shape ``(nvir, nocc)`` in Hartree.
    """

    occupied_coeff: np.ndarray
    virtual_coeff: np.ndarray
    occupied_mask: np.ndarray
    virtual_mask: np.ndarray
    energy_gap: np.ndarray

    @property
    def vo_shape(self) -> tuple[int, int]:
        return self.energy_gap.shape


def _kpoint_blocks(value, *, expected_ndim: int, label: str) -> list[np.ndarray]:
    array = np.asarray(value)
    if array.ndim == expected_ndim:
        return [array]
    if array.ndim == expected_ndim + 1:
        return [np.asarray(block) for block in array]
    raise ValueError(
        f"{label} must have ndim {expected_ndim} or {expected_ndim + 1}; "
        f"got shape {array.shape}"
    )


def build_mo_partitions(source_mf) -> list[MOPartition]:
    """Build occupied/virtual partitions from a converged RHF/KRHF object."""

    coeff_blocks = _kpoint_blocks(source_mf.mo_coeff, expected_ndim=2, label="mo_coeff")
    occupation_blocks = _kpoint_blocks(
        source_mf.mo_occ,
        expected_ndim=1,
        label="mo_occ",
    )
    energy_blocks = _kpoint_blocks(
        source_mf.mo_energy,
        expected_ndim=1,
        label="mo_energy",
    )
    if not (len(coeff_blocks) == len(occupation_blocks) == len(energy_blocks)):
        raise ValueError("MO coefficient, occupation, and energy k-point counts differ")

    partitions: list[MOPartition] = []
    for k_index, (coeff, occupation, energy) in enumerate(
        zip(coeff_blocks, occupation_blocks, energy_blocks, strict=True)
    ):
        occupied_mask = np.asarray(occupation) > 0
        virtual_mask = np.asarray(occupation) == 0
        if not np.any(occupied_mask):
            raise ValueError(f"no occupied orbitals at k-point {k_index}")
        if not np.any(virtual_mask):
            raise ValueError(f"no virtual orbitals at k-point {k_index}")
        occupied_coeff = np.asarray(coeff)[:, occupied_mask]
        virtual_coeff = np.asarray(coeff)[:, virtual_mask]
        energy_gap = (
            np.asarray(energy)[virtual_mask, None]
            - np.asarray(energy)[occupied_mask][None, :]
        )
        partitions.append(
            MOPartition(
                occupied_coeff=occupied_coeff,
                virtual_coeff=virtual_coeff,
                occupied_mask=occupied_mask,
                virtual_mask=virtual_mask,
                energy_gap=energy_gap,
            )
        )
    return partitions


def target_fock_ao_blocks(dc_mf) -> list[np.ndarray]:
    """Return target DFA Fock matrices evaluated at the HF density."""

    if dc_mf.source_scf is None or dc_mf.target_ks is None:
        raise RuntimeError("run the DC-DFT energy kernel before building the response RHS")
    density = dc_mf.source_scf.make_rdm1()
    target = dc_mf.target_ks
    fock = np.asarray(target.get_hcore()) + np.asarray(
        target.get_veff(dc_mf.cell, density)
    )
    return _kpoint_blocks(fock, expected_ndim=2, label="target Fock")


def target_fock_vo_blocks(
    fock_blocks: list[np.ndarray],
    partitions: list[MOPartition],
) -> list[np.ndarray]:
    """Project target Fock matrices into ``(virtual, occupied)`` blocks."""

    if len(fock_blocks) != len(partitions):
        raise ValueError("target Fock and MO partition k-point counts differ")
    return [
        partition.virtual_coeff.conj().T
        @ np.asarray(fock)
        @ partition.occupied_coeff
        for fock, partition in zip(fock_blocks, partitions, strict=True)
    ]
