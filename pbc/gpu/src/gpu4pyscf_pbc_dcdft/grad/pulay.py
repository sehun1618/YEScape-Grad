"""GPU target-Fock overlap/Pulay contraction."""

import cupy as cp
import numpy as np


def contract_target_overlap(context, overlap_rows, weighted_density):
    """Return ``-2/Nk Tr[S_row D_target^E]`` in Hartree/Bohr."""

    output = cp.zeros((context.cell.natm, 3), dtype=cp.float64)
    ao_slices = np.asarray(context.cell.aoslice_by_atom(), dtype=int)
    for atom_id in range(context.cell.natm):
        p0, p1 = ao_slices[atom_id, 2:]
        output[atom_id] = (
            -2.0
            * cp.einsum(
                "kxij,kji->x",
                overlap_rows[:, :, p0:p1, :],
                weighted_density[:, :, p0:p1],
            ).real
            / context.nkpts
        )
    return output
