"""Normalized k-point view of a restricted DC-DFT calculation.

The public :class:`~pyscf_pbc_dcdft.dft.DCRKS` driver intentionally stores
Gamma-point quantities in the same two-dimensional form as PySCF ``RHF``.
The analytic-gradient algebra, however, is considerably easier to audit when
Gamma is treated as the special case ``Nk = 1``.  This module performs that
shape conversion once.  Downstream modules can consequently use one explicit
``(nkpts, ...)`` convention without scattered Gamma branches.

All k-points are in reciprocal Bohr, density matrices are spin-summed AO
densities, and energies are in Hartree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..lib.kpoints import normalize_kpts
from ..dft.state import require_current_state


def _with_kpoint_axis(value, *, matrix_rank: int, label: str) -> np.ndarray:
    """Return ``value`` with one leading k-point axis.

    Parameters
    ----------
    value
        Gamma-point or k-resolved array.
    matrix_rank
        Number of dimensions in one k-point block.  AO matrices use ``2``;
        MO energy and occupation vectors use ``1``.
    label
        Name used in shape-error messages.
    """

    array = np.asarray(value)
    if array.ndim == matrix_rank:
        return array[None]
    if array.ndim == matrix_rank + 1:
        return array
    raise ValueError(
        f"{label} must have ndim {matrix_rank} or {matrix_rank + 1}; "
        f"got shape {array.shape}"
    )


@dataclass(frozen=True)
class RestrictedGradientContext:
    """Immutable arrays shared by all restricted gradient components.

    Attributes
    ----------
    dc_mf
        Converged public DC-DFT driver.
    source_mf
        Converged RHF/KRHF object that defines the density.
    target_ks
        Non-SCF RKS/KRKS object used to evaluate the target DFA.
    kpts
        Array with shape ``(nkpts, 3)``.
    density
        Spin-summed HF AO density with shape ``(nkpts, nao, nao)``.
    """

    dc_mf: object
    source_mf: object
    target_ks: object
    kpts: np.ndarray
    density: np.ndarray
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    mo_energy: np.ndarray

    @classmethod
    def from_dc_method(cls, dc_mf) -> "RestrictedGradientContext":
        """Build a checked context after ``dc_mf.kernel()`` has completed."""

        require_current_state(dc_mf)
        source = getattr(dc_mf, "source_scf", None)
        target = getattr(dc_mf, "target_ks", None)
        if source is None or target is None or not bool(getattr(dc_mf, "converged", False)):
            raise RuntimeError("run the DC-DFT energy kernel before its gradient")

        kpts = normalize_kpts(dc_mf.kpts)
        density = _with_kpoint_axis(
            source.make_rdm1(),
            matrix_rank=2,
            label="HF density",
        )
        mo_coeff = _with_kpoint_axis(
            source.mo_coeff,
            matrix_rank=2,
            label="MO coefficients",
        )
        mo_occ = _with_kpoint_axis(
            source.mo_occ,
            matrix_rank=1,
            label="MO occupations",
        )
        mo_energy = _with_kpoint_axis(
            source.mo_energy,
            matrix_rank=1,
            label="MO energies",
        )
        nkpts = len(kpts)
        arrays = {
            "density": density,
            "mo_coeff": mo_coeff,
            "mo_occ": mo_occ,
            "mo_energy": mo_energy,
        }
        for label, array in arrays.items():
            if len(array) != nkpts:
                raise ValueError(
                    f"{label} has {len(array)} k-point blocks; expected {nkpts}"
                )
        return cls(
            dc_mf=dc_mf,
            source_mf=source,
            target_ks=target,
            kpts=np.asarray(kpts),
            density=np.asarray(density),
            mo_coeff=np.asarray(mo_coeff),
            mo_occ=np.asarray(mo_occ),
            mo_energy=np.asarray(mo_energy),
        )

    @property
    def cell(self):
        return self.dc_mf.cell

    @property
    def nkpts(self) -> int:
        return len(self.kpts)

    @property
    def nao(self) -> int:
        return int(self.density.shape[-1])
