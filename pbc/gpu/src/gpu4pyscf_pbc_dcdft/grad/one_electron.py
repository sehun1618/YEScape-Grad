"""GPU kinetic, overlap, local-GTH and nonlocal-GTH derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.grad import krhf as gpu_krhf_grad
from gpu4pyscf.pbc.gto import int1e

from ..pseudo import nonlocal_gth_derivative


@dataclass(frozen=True)
class OneElectronDerivatives:
    cell: object
    kpts: np.ndarray
    local_hcore: object
    nonlocal_hcore: cp.ndarray
    overlap_rows: cp.ndarray

    def hcore(self, atom_id: int):
        """Return ``dh_core/dR_A`` with ``(x,k,p,q)`` ordering."""

        local = cp.asarray(self.local_hcore(int(atom_id)))
        nonlocal_part = self.nonlocal_hcore[int(atom_id)]
        # GPU4PySCF stores both providers as (k,x,p,q).
        return (local + nonlocal_part).transpose(1, 0, 2, 3)

    def overlap(self, atom_id: int, axis: int):
        """Return the Hermitian full overlap derivative ``(k,p,q)``."""

        p0, p1 = np.asarray(self.cell.aoslice_by_atom(), dtype=int)[
            int(atom_id), 2:
        ]
        output = cp.zeros(
            (len(self.kpts), self.cell.nao_nr(), self.cell.nao_nr()),
            dtype=self.overlap_rows.dtype,
        )
        output[:, p0:p1, :] = self.overlap_rows[:, int(axis), p0:p1, :]
        return output + output.transpose(0, 2, 1).conj()


def build_one_electron_derivatives(cell, kpts):
    """Construct all one-electron matrix derivatives on the GPU."""

    if not cell._pseudo:
        raise NotImplementedError("GPU Stage 3 supports GTH pseudopotentials")
    kpoint_array = np.asarray(kpts, dtype=float).reshape(-1, 3)
    local_hcore = gpu_krhf_grad.hcore_generator(
        None, cell, kpoint_array
    )
    nonlocal_result = nonlocal_gth_derivative(cell, kpoint_array)
    overlap_rows = -int1e.int1e_ipovlp(cell, kpoint_array)
    return OneElectronDerivatives(
        cell=cell,
        kpts=kpoint_array,
        local_hcore=local_hcore,
        nonlocal_hcore=nonlocal_result.derivative,
        overlap_rows=overlap_rows,
    )
