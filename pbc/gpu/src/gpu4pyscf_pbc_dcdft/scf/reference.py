"""Build restricted GPU RHF/KRHF references with explicit DF ownership."""

from __future__ import annotations

from gpu4pyscf.pbc import scf as gpu_scf
from gpu4pyscf.pbc.dft import numint as gpu_numint


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
        reference = gpu_scf.KRHF(cell, kpts=kpts, exxdiv=exxdiv)
        reference._numint = gpu_numint.KNumInt()
    else:
        reference = gpu_scf.RHF(cell, exxdiv=exxdiv)
        reference._numint = gpu_numint.NumInt()
    reference.with_df = with_df
    reference.conv_tol = float(conv_tol)
    if conv_tol_grad is not None:
        reference.conv_tol_grad = float(conv_tol_grad)
    reference.max_cycle = int(max_cycle)
    return reference
