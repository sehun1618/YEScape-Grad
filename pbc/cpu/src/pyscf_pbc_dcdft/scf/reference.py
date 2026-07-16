"""Construct the variational HF reference whose density feeds DC-DFT."""

from __future__ import annotations

import numpy as np
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc.scf import hf as pbc_hf

from ..lib.kpoints import normalize_kpts


class DensityFittedRHF(pbc_hf.RHF):
    """Gamma RHF that always evaluates J/K through ``with_df``.

    PySCF's ordinary ``RHF.get_jk`` may switch to an in-core ERI path for a
    small cell even when an FFTDF/GDF object has been assigned.  That is a
    valid SCF optimization, but it would make a backend-specific DC gradient
    differentiate a different approximate Hamiltonian.  This subclass keeps
    the public RHF data layout while making backend ownership explicit.
    """

    def get_jk(
        self,
        cell=None,
        dm=None,
        hermi=1,
        kpt=None,
        kpts_band=None,
        with_j=True,
        with_k=True,
        omega=None,
        **kwargs,
    ):
        if cell is None:
            cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        if kpt is None:
            kpt = self.kpt
        density = np.asarray(dm)
        return self.with_df.get_jk(
            density,
            hermi=hermi,
            kpts=kpt,
            kpts_band=kpts_band,
            with_j=with_j,
            with_k=with_k,
            omega=omega,
            exxdiv=self.exxdiv,
        )


def build_reference(
    cell,
    *,
    kpts=None,
    kpoint_mode: bool,
    with_df,
    exxdiv="ewald",
    conv_tol: float = 1.0e-10,
    conv_tol_grad: float | None = None,
    max_cycle: int = 80,
):
    """Return an unexecuted RHF or KRHF reference object.

    Source and target calculations receive the same explicit DF provider.  A
    backend mismatch would otherwise define two different approximate energy
    models and make analytic-vs-FD comparison ambiguous.
    """

    normalized_kpts = normalize_kpts(kpts)
    if kpoint_mode:
        mf = pbc_scf.KRHF(cell, kpts=normalized_kpts, exxdiv=exxdiv)
    else:
        mf = DensityFittedRHF(cell, exxdiv=exxdiv)
    mf.with_df = with_df
    mf.conv_tol = float(conv_tol)
    if conv_tol_grad is not None:
        mf.conv_tol_grad = float(conv_tol_grad)
    mf.max_cycle = int(max_cycle)
    return mf
