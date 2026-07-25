"""Build restricted GPU RHF/KRHF references with explicit DF ownership."""

from __future__ import annotations

import numpy as np
from gpu4pyscf.pbc.dft import numint as gpu_numint
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.scf.hf import RHF as GPU4PySCFRHF
from gpu4pyscf.pbc.scf.khf import KRHF as GPU4PySCFKRHF
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh


def selected_df_hcore(method, cell=None, points=None, *, sampled_k=None):
    """Return kinetic plus PP/nuclear matrices from ``method.with_df``.

    GPU4PySCF 1.8 routes small-mesh PBC hcore evaluations through
    ``multigrid_v2`` even when a Gaussian DF provider was selected.  DC-DFT
    differentiates the selected provider's one-electron operator, so source
    HF and target energy evaluation must use that same owner explicitly.
    """

    if cell is None:
        cell = method.cell
    if sampled_k is None:
        # StreamObject delegates unknown attributes to attached providers, so
        # ``hasattr(method, 'kpts')`` is true even for Gamma RHF.  The actual
        # single-k classes expose the ``kpt`` property on their class.
        sampled_k = not hasattr(type(method), "kpt")
    if sampled_k:
        if points is None:
            points = method.kpts
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        reference_points = np.asarray(method.kpts).reshape(-1, 3)
        in_bvk_cell = len(points) == len(reference_points)
        bvk_kmesh = (
            kpts_to_kmesh(cell, points, bound_by_supmol=True)
            if in_bvk_cell else None
        )
        hcore = (
            method.with_df.get_pp(points)
            if cell.pseudo else method.with_df.get_nuc(points)
        )
        hcore += int1e.int1e_kin(cell, points, bvk_kmesh)
        return hcore

    if points is None:
        points = method.kpt
    point = np.asarray(points, dtype=float).reshape(3)
    hcore = (
        method.with_df.get_pp(point)
        if cell.pseudo else method.with_df.get_nuc(point)
    )
    hcore += int1e.int1e_kin(cell, point)
    return hcore


class _SelectedDFRHF(GPU4PySCFRHF):
    def get_hcore(self, cell=None, kpt=None):
        return selected_df_hcore(self, cell, kpt, sampled_k=False)


class _SelectedDFKRHF(GPU4PySCFKRHF):
    def get_hcore(self, cell=None, kpts=None):
        return selected_df_hcore(self, cell, kpts, sampled_k=True)


def build_reference(
    cell,
    *,
    kpts,
    kpoint_mode: bool,
    with_df,
    exxdiv,
    conv_tol: float,
    conv_tol_grad: float | None,
    max_cycle: int,
):
    """Return an unrun GPU HF object bound to the requested DF provider.

    ``NumInt/KNumInt`` is assigned deliberately so that GPU4PySCF construction
    cannot replace the selected FFTDF, GDF, or RSDF ownership contract.
    """

    backend = getattr(with_df, "backend_name", None)
    if backend not in {"fftdf", "gdf", "rsdf"}:
        raise NotImplementedError(
            "GPU reference builder supports project-local FFTDF, GDF, or RSDF"
        )
    if kpoint_mode:
        reference = _SelectedDFKRHF(cell, kpts=kpts, exxdiv=exxdiv)
        reference._numint = gpu_numint.KNumInt()
    else:
        reference = _SelectedDFRHF(cell, exxdiv=exxdiv)
        reference._numint = gpu_numint.NumInt()
    reference.with_df = with_df
    reference.conv_tol = float(conv_tol)
    if conv_tol_grad is not None:
        reference.conv_tol_grad = float(conv_tol_grad)
    reference.max_cycle = int(max_cycle)
    return reference
