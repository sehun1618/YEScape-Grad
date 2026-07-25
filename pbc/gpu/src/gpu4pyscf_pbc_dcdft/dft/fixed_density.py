"""GPU evaluation of ``E_DFA[n_HF]`` without target-orbital optimization."""

from __future__ import annotations

from typing import Any

import cupy as cp
from gpu4pyscf.pbc import dft as gpu_dft
from pyscf_pbc_dcdft.lib.kpoints import normalize_kpts
from pyscf_pbc_dcdft.lib.types import DCEnergyComponents, DCEnergyResult

from ..lib import real_scalar
from .target_numint import (
    evaluate_target_hcore,
    evaluate_target_veff,
    target_numint_metadata,
)
from .workspace import TargetEvaluationWorkspace


def build_target_ks(
    cell,
    *,
    kpts,
    kpoint_mode: bool,
    xc: str,
    with_df,
):
    """Build a non-SCF GPU RKS/KRKS target using the same FFTDF provider."""

    if kpoint_mode:
        target = gpu_dft.KRKS(cell, kpts=normalize_kpts(kpts), xc=xc)
    else:
        target = gpu_dft.RKS(cell, xc=xc)
    target.with_df = with_df
    target.build()
    return target


def evaluate_energy(
    reference_mf,
    *,
    xc: str,
    kpts,
    kpoint_mode: bool,
    with_df,
    target_integrator="uniform",
    target_multigrid_mesh=None,
) -> tuple[DCEnergyResult, Any]:
    """Evaluate the target DFA energy using CuPy density and Fock arrays."""

    density = cp.asarray(reference_mf.make_rdm1())
    target = build_target_ks(
        reference_mf.cell,
        kpts=kpts,
        kpoint_mode=kpoint_mode,
        xc=xc,
        with_df=with_df,
    )
    hcore = cp.asarray(
        evaluate_target_hcore(target, kpoint_mode=kpoint_mode)
    )
    target_evaluation = evaluate_target_veff(
        target,
        density,
        with_df=with_df,
        kpts=kpts,
        target_integrator=target_integrator,
        target_multigrid_mesh=target_multigrid_mesh,
    )
    veff = target_evaluation.veff
    target._dcdft_target_numint_metadata = target_evaluation.metadata
    if kpoint_mode:
        nkpts = len(normalize_kpts(kpts))
        one_electron = cp.einsum("kij,kji->", hcore, density) / nkpts
    else:
        nkpts = 1
        one_electron = cp.einsum("ij,ji->", hcore, density)

    ecoul = getattr(veff, "ecoul", None)
    exc = getattr(veff, "exc", None)
    if ecoul is None or exc is None:
        raise RuntimeError("GPU target Veff did not expose ecoul/exc")
    total = target.energy_tot(dm=density, h1e=hcore, vhf=veff)
    components = DCEnergyComponents(
        one_electron=real_scalar(one_electron, label="one-electron energy"),
        coulomb=real_scalar(ecoul, label="Coulomb energy"),
        xc=real_scalar(exc, label="XC energy"),
        nuclear=float(reference_mf.cell.energy_nuc()),
        total=real_scalar(total, label="total DC-DFT energy"),
    )
    result = DCEnergyResult(
        energy=components.total,
        reference_energy=float(reference_mf.e_tot),
        target_xc=str(xc),
        nkpts=nkpts,
        components=components,
        metadata={
            "density_source": reference_mf.__class__.__name__,
            "target_method": target.__class__.__name__,
            "df_backend": getattr(with_df, "backend_name", "fftdf"),
            "device": "gpu",
            "kpoint_mode": bool(kpoint_mode),
            "formula": "E_DFA[n_HF]",
            "target_numint": target_numint_metadata(target),
            "target_integrator": target_evaluation.metadata["mode"],
            "silent_cpu_fallback": False,
        },
    )
    target._dcdft_evaluation_workspace = TargetEvaluationWorkspace.create(
        source_mf=reference_mf,
        target_ks=target,
        with_df=with_df,
        xc=xc,
        kpts=kpts,
        density=density,
        hcore=hcore,
        veff=veff,
        target_integrator=target_evaluation.metadata["mode"],
        target_numint=target_evaluation.numerical_integrator,
        ownership=target_evaluation.metadata,
        # Fingerprint the user setting, not the mesh resolved internally by
        # multigrid.  The workspace still records the resolved cell mesh when
        # this value is None.
        target_multigrid_mesh=target_multigrid_mesh,
    )
    return result, target
