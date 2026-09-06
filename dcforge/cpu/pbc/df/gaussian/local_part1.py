"""Analytic derivative of the Gaussian-DF GTH local-potential part 1.

This module mirrors the ownership boundary of PySCF's private
``_RSNucBuilder`` while exposing a small, package-owned public evaluator.
The value and derivative use the same range-separation parameter, mesh,
diffuse-block filtering, fake nuclear basis, and ``G=0`` convention.
"""

from __future__ import annotations

import numpy as np
from pyscf.pbc.df import rsdf_builder

from .local_part1_long_range import LongRangeLocalPart1
from .local_part1_short_range import ShortRangeLocalPart1


class LocalPart1Derivative:
    """Reusable derivative evaluator for ``_RSNucBuilder.get_pp_loc_part1``."""

    def __init__(self, cell, kpts):
        self.cell = cell
        self.kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
        self.builder = rsdf_builder._RSNucBuilder(cell, self.kpts).build()
        if abs(float(self.builder.omega)) < 1.0e-15:
            raise NotImplementedError(
                "Gaussian-DF local-part1 derivative requires a periodic "
                "range-separated _RSNucBuilder (omega != 0)"
            )
        self.short_range = ShortRangeLocalPart1(self.builder)
        self.long_range = LongRangeLocalPart1(self.builder)

    def matrix(self, atom_id: int, axis: int) -> np.ndarray:
        """Return ``d V_loc,1(k) / dR[A,axis]`` for all k-points."""

        return self.short_range.matrix(atom_id, axis) + self.long_range.matrix(
            atom_id, axis
        )

    def all_matrices(self) -> np.ndarray:
        """Return shape ``(natm,3,nkpts,nao,nao)`` analytic derivatives."""

        output = np.empty(
            (
                self.cell.natm,
                3,
                len(self.kpts),
                self.cell.nao_nr(),
                self.cell.nao_nr(),
            ),
            dtype=np.complex128,
        )
        for atom_id in range(self.cell.natm):
            for axis in range(3):
                output[atom_id, axis] = self.matrix(atom_id, axis)
        return output

    def contract_batch(self, densities, atmlst=None) -> np.ndarray:
        r'''Contract local-part-1 derivatives with several AO densities.

        The expensive short- and long-range setup is shared by every density.
        The long-range owner additionally evaluates all three Cartesian
        components in one reciprocal traversal instead of repeating its
        three-component Fourier kernel once for each requested axis.
        '''
        return (
            self.short_range.contract_batch(densities, atmlst)
            + self.long_range.contract_batch(densities, atmlst)
        )
