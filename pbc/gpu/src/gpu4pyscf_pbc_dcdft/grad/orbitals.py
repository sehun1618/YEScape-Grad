"""GPU occupied/virtual partitions and target-Fock projections."""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp


@dataclass(frozen=True)
class MOPartition:
    occupied_coeff: cp.ndarray
    virtual_coeff: cp.ndarray
    occupied_mask: cp.ndarray
    virtual_mask: cp.ndarray
    energy_gap: cp.ndarray

    @property
    def vo_shape(self):
        return self.energy_gap.shape


def build_mo_partitions(context):
    """Build one device-resident VO partition per k-point."""

    partitions = []
    for k_index in range(context.nkpts):
        coeff = context.mo_coeff[k_index]
        occupation = context.mo_occ[k_index]
        energy = context.mo_energy[k_index]
        occupied_mask = occupation > 0
        virtual_mask = occupation == 0
        if not bool(cp.any(occupied_mask).item()):
            raise ValueError(f"no occupied orbital at k-point {k_index}")
        if not bool(cp.any(virtual_mask).item()):
            raise ValueError(f"no virtual orbital at k-point {k_index}")
        occupied_coeff = coeff[:, occupied_mask]
        virtual_coeff = coeff[:, virtual_mask]
        energy_gap = (
            energy[virtual_mask, None] - energy[occupied_mask][None, :]
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


def target_fock_vo_blocks(fock_kpts, partitions):
    """Project target-DFA Fock matrices into virtual--occupied blocks."""

    return [
        part.virtual_coeff.conj().T @ fock @ part.occupied_coeff
        for fock, part in zip(fock_kpts, partitions, strict=True)
    ]
