"""Target DFA energy evaluated at an externally supplied HF density."""

from __future__ import annotations

from typing import Any

import numpy as np
from pyscf.pbc import dft as pbc_dft

from ..lib.kpoints import normalize_kpts
from ..lib.types import DCEnergyComponents, DCEnergyResult


def _real_scalar(value, *, label: str, imag_tol: float = 1.0e-9) -> float:
    """Return the real part after guarding the imaginary numerical residual."""

    scalar = complex(np.asarray(value).item())
    if abs(scalar.imag) > imag_tol:
        raise ValueError(f"{label} has an imaginary residual {scalar.imag:.3e}")
    return float(scalar.real)


def build_target_ks(cell, *, kpts, kpoint_mode: bool, xc: str, with_df):
    """Build the non-SCF target KS object with the selected DF provider."""

    if kpoint_mode:
        target = pbc_dft.KRKS(cell, kpts=normalize_kpts(kpts))
    else:
        target = pbc_dft.RKS(cell)
    target.xc = xc
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
) -> tuple[DCEnergyResult, Any]:
    """Evaluate ``E_DFA[n_HF]`` without optimizing target KS orbitals.

    For sampled k-points, PySCF stores one AO density matrix per k-point and
    uses equal weights.  Consequently the one-electron contribution is

    ``(1 / Nk) * sum_k Tr[D_k h_k]``.

    ``veff.ecoul`` and ``veff.exc`` already follow PySCF's k-point weighting and
    must not receive another ``1/Nk`` factor here.
    """

    density = np.asarray(reference_mf.make_rdm1())
    target = build_target_ks(
        reference_mf.cell,
        kpts=kpts,
        kpoint_mode=kpoint_mode,
        xc=xc,
        with_df=with_df,
    )
    hcore = np.asarray(target.get_hcore())
    veff = target.get_veff(reference_mf.cell, density)

    if kpoint_mode:
        nkpts = len(normalize_kpts(kpts))
        one_electron_raw = np.einsum("kij,kji->", hcore, density) / nkpts
    else:
        nkpts = 1
        one_electron_raw = np.einsum("ij,ji->", hcore, density)

    ecoul = getattr(veff, "ecoul", None)
    exc = getattr(veff, "exc", None)
    if ecoul is None or exc is None:
        raise RuntimeError("target KS effective potential did not expose ecoul/exc")

    total_raw = target.energy_tot(dm=density, h1e=hcore, vhf=veff)
    nuclear_raw = reference_mf.cell.energy_nuc()
    components = DCEnergyComponents(
        one_electron=_real_scalar(one_electron_raw, label="one-electron energy"),
        coulomb=_real_scalar(ecoul, label="Coulomb energy"),
        xc=_real_scalar(exc, label="XC energy"),
        nuclear=_real_scalar(nuclear_raw, label="nuclear energy"),
        total=_real_scalar(total_raw, label="total DC-DFT energy"),
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
            "df_backend": getattr(with_df, "backend_name", with_df.__class__.__name__),
            "kpoint_mode": bool(kpoint_mode),
            "formula": "E_DFA[n_HF]",
        },
    )
    return result, target
