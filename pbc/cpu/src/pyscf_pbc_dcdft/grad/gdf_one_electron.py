"""GDF-consistent analytic one-electron derivative matrices.

The ordinary PySCF periodic gradient helper differentiates the local GTH
potential represented on ``cell.mesh``.  A GDF mean-field object instead
builds that potential through ``_RSNucBuilder`` and ``_IntPPBuilder``.  Even
though both values converge to the same continuum operator, differentiating
one representation while evaluating the other leaves a finite mesh-dependent
force error.  This module keeps the value and derivative representations
identical.
"""

from __future__ import annotations

import numpy as np
from pyscf.pbc.grad import krhf as pyscf_krhf_grad

from ..df.gdf.local_part1 import LocalPart1Derivative
from ..df.gdf.local_part2 import LocalPart2Derivative
from ..pseudo import nonlocal_gth_derivative
from .one_electron import OneElectronDerivatives


class _GDFLocalHCoreGenerator:
    """Callable returning kinetic plus GDF local-PP derivatives by atom."""

    def __init__(self, cell, kpts):
        self.cell = cell
        self.kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
        self.nkpts = len(self.kpts)
        self.nao = int(cell.nao_nr())
        self._ao_slices = np.asarray(cell.aoslice_by_atom(), dtype=int)
        self._kinetic_rows = np.asarray(
            cell.pbc_intor("int1e_ipkin", kpts=self.kpts)
        )
        self._local_part1 = LocalPart1Derivative(cell, self.kpts)
        self._local_part2 = LocalPart2Derivative(cell, self.kpts)

    def __call__(self, atom_id: int) -> np.ndarray:
        """Return shape ``(3,nkpts,nao,nao)`` for nucleus ``atom_id``."""

        atom_id = int(atom_id)
        if not 0 <= atom_id < self.cell.natm:
            raise IndexError(f"atom_id={atom_id} is outside this cell")
        output = np.empty(
            (3, self.nkpts, self.nao, self.nao), dtype=np.complex128
        )
        for axis in range(3):
            output[axis] = self._kinetic_matrix(atom_id, axis)
            output[axis] += self._local_part1.matrix(atom_id, axis)
            output[axis] += self._local_part2.matrix(atom_id, axis)
        return output

    def _kinetic_matrix(self, atom_id: int, axis: int) -> np.ndarray:
        p0, p1 = self._ao_slices[atom_id, 2:]
        row = self._kinetic_rows[:, axis]
        output = np.zeros(
            (self.nkpts, self.nao, self.nao), dtype=row.dtype
        )
        output[:, p0:p1, :] -= row[:, p0:p1, :]
        output[:, :, p0:p1] -= row[:, p0:p1, :].transpose(0, 2, 1).conj()
        return output


def build_gdf_one_electron_derivatives(cell, kpts) -> OneElectronDerivatives:
    """Build one-electron H1 matrices in the exact PySCF GDF convention."""

    if not cell._pseudo:
        raise NotImplementedError(
            "The GDF route currently supports GTH pseudopotentials only"
        )
    kpoint_array = np.asarray(kpts, dtype=float).reshape(-1, 3)
    local_hcore = _GDFLocalHCoreGenerator(cell, kpoint_array)
    nonlocal_result = nonlocal_gth_derivative(cell, kpoint_array)
    overlap_rows = np.asarray(
        pyscf_krhf_grad.get_ovlp(cell, kpoint_array)
    )
    return OneElectronDerivatives(
        cell=cell,
        kpts=kpoint_array,
        local_hcore=local_hcore,
        nonlocal_hcore=nonlocal_result.derivative,
        overlap_rows=overlap_rows,
    )
