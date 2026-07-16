"""One-electron, overlap, and GTH pseudopotential derivatives.

PySCF's sampled-k-point RHF gradient implementation already defines the
kinetic plus local-GTH derivative convention used by its total energy.  This
module reuses that public analytic machinery through a tiny adapter and adds
the project-local nonlocal-GTH matrix derivative required by DC-DFT response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf.pbc.grad import krhf as pyscf_krhf_grad

from ..pseudo import fft_nonlocal_gth_derivative


class _HCoreGradientAdapter:
    """Minimal object satisfying ``pyscf.pbc.grad.krhf.hcore_generator``."""

    @staticmethod
    def get_hcore(cell=None, kpts=None):
        return pyscf_krhf_grad.get_hcore(cell, kpts)


@dataclass(frozen=True)
class OneElectronDerivatives:
    """Analytic one-electron derivative sources.

    ``local_hcore(atom)`` returns shape ``(3,nkpts,nao,nao)`` and includes
    kinetic plus local GTH terms.  ``nonlocal_hcore`` has shape
    ``(natm,nkpts,3,nao,nao)``.  ``overlap_rows`` stores PySCF's AO-row
    derivative with shape ``(nkpts,3,nao,nao)``.
    """

    cell: object
    kpts: np.ndarray
    local_hcore: object
    nonlocal_hcore: np.ndarray
    overlap_rows: np.ndarray

    def hcore(self, atom_id: int) -> np.ndarray:
        """Return full ``d h_core(k) / dR_A`` for all Cartesian directions."""

        local = np.asarray(self.local_hcore(int(atom_id)))
        nonlocal_part = np.asarray(self.nonlocal_hcore[int(atom_id)])
        # Convert nonlocal ordering (k,x,p,q) to local ordering (x,k,p,q).
        return local + nonlocal_part.transpose(1, 0, 2, 3)

    def overlap(self, atom_id: int, axis: int) -> np.ndarray:
        """Return the Hermitian full overlap derivative at all k-points."""

        p0, p1 = np.asarray(self.cell.aoslice_by_atom(), dtype=int)[
            int(atom_id), 2:
        ]
        output = np.zeros(
            (len(self.kpts), self.cell.nao_nr(), self.cell.nao_nr()),
            dtype=self.overlap_rows.dtype,
        )
        output[:, p0:p1, :] = self.overlap_rows[:, int(axis), p0:p1, :]
        return output + output.transpose(0, 2, 1).conj()


def build_one_electron_derivatives(cell, kpts) -> OneElectronDerivatives:
    """Construct analytic one-electron derivative sources for one cell."""

    if not cell._pseudo:
        raise NotImplementedError(
            "Stage 2 currently supports GTH pseudopotentials; all-electron "
            "periodic one-electron derivatives are outside this gate"
        )
    kpoint_array = np.asarray(kpts, dtype=float).reshape(-1, 3)
    local_hcore = pyscf_krhf_grad.hcore_generator(
        _HCoreGradientAdapter(),
        cell,
        kpoint_array,
    )
    # FFTDF.get_pp evaluates the nonlocal GTH projector on ``cell.mesh``.
    # Its derivative must retain that finite reciprocal representation; the
    # exact integral projector used by GDF belongs to gdf_one_electron.py.
    nonlocal_result = fft_nonlocal_gth_derivative(cell, kpoint_array)
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
