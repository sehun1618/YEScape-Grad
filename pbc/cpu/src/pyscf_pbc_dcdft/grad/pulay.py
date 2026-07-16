"""Overlap/Pulay terms for the frozen-HF target energy."""

from __future__ import annotations

import numpy as np


def contract_target_overlap(
    context,
    overlap_rows: np.ndarray,
    energy_weighted_density: np.ndarray,
) -> np.ndarray:
    """Return the target-Fock overlap term in Hartree/Bohr.

    The AO-row form is

    ``-2/Nk sum_k Tr[S^(A,x)_row D^E_target]``.

    ``D^E_target = 2 C_occ F_target,oo C_occ^H`` differs from the ordinary HF
    energy-weighted density because the differentiated energy is PBE while the
    stationary orbitals come from HF.
    """

    output = np.zeros((context.cell.natm, 3), dtype=float)
    ao_slices = np.asarray(context.cell.aoslice_by_atom(), dtype=int)
    for atom_id in range(context.cell.natm):
        p0, p1 = ao_slices[atom_id, 2:]
        output[atom_id] = (
            -2.0
            * np.einsum(
                "kxij,kji->x",
                overlap_rows[:, :, p0:p1, :],
                energy_weighted_density[:, :, p0:p1],
                optimize=True,
            ).real
            / context.nkpts
        )
    return output
