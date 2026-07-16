"""Device-resident normalized view of one converged GPU DC-DFT method."""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from pyscf_pbc_dcdft.lib.kpoints import normalize_kpts
from ..dft.workspace import compatible_target_workspace


def _with_kpoint_axis(value, *, matrix_rank: int, label: str):
    array = cp.asarray(value)
    if array.ndim == matrix_rank:
        return array[None]
    if array.ndim == matrix_rank + 1:
        return array
    raise ValueError(f"{label} has unsupported shape {array.shape}")


@dataclass(frozen=True)
class RestrictedGradientContext:
    """GPU arrays and host metadata shared by all gradient providers."""

    dc_mf: object
    source_mf: object
    target_ks: object
    kpts: np.ndarray
    density: cp.ndarray
    mo_coeff: cp.ndarray
    mo_occ: cp.ndarray
    mo_energy: cp.ndarray
    target_evaluation: object | None

    @classmethod
    def from_dc_method(cls, dc_mf):
        source = getattr(dc_mf, "source_scf", None)
        target = getattr(dc_mf, "target_ks", None)
        if source is None or target is None or not dc_mf.converged:
            raise RuntimeError("run the GPU DC-DFT energy kernel first")
        kpts = normalize_kpts(dc_mf.kpts)
        target_evaluation = compatible_target_workspace(dc_mf)
        density = (
            source.make_rdm1()
            if target_evaluation is None
            else target_evaluation.density
        )
        arrays = {
            "density": _with_kpoint_axis(
                density, matrix_rank=2, label="density"
            ),
            "mo_coeff": _with_kpoint_axis(
                source.mo_coeff, matrix_rank=2, label="mo_coeff"
            ),
            "mo_occ": _with_kpoint_axis(
                source.mo_occ, matrix_rank=1, label="mo_occ"
            ),
            "mo_energy": _with_kpoint_axis(
                source.mo_energy, matrix_rank=1, label="mo_energy"
            ),
        }
        for label, array in arrays.items():
            if len(array) != len(kpts):
                raise ValueError(
                    f"{label} has {len(array)} blocks; expected {len(kpts)}"
                )
        return cls(
            dc_mf=dc_mf,
            source_mf=source,
            target_ks=target,
            kpts=np.asarray(kpts),
            target_evaluation=target_evaluation,
            **arrays,
        )

    @property
    def cell(self):
        return self.dc_mf.cell

    @property
    def nkpts(self):
        return len(self.kpts)
