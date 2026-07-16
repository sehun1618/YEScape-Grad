"""Shared PySCF-style driver for restricted periodic DC-DFT."""

from __future__ import annotations

from typing import Any

from pyscf import lib

from .. import df
from ..lib.kpoints import is_gamma, normalize_kpts
from ..scf import build_reference
from .fixed_density import evaluate_energy


class DCRestrictedBase(lib.StreamObject):
    """Common state management for Gamma and sampled-k-point drivers."""

    _keys = {
        "cell",
        "mol",
        "kpts",
        "xc",
        "with_df",
        "conv_tol",
        "conv_tol_grad",
        "max_cycle",
        "exxdiv",
        "verbose",
        "stdout",
        "max_memory",
    }
    kpoint_mode = False

    def __init__(self, cell, *, kpts=None, xc: str = "PBE"):
        self.cell = cell
        self.mol = cell
        self.kpts = normalize_kpts(kpts)
        if not self.kpoint_mode and not is_gamma(self.kpts):
            raise ValueError("DCRKS accepts Gamma only; use DCKRKS for sampled k-points")
        self.xc = str(xc)
        self.with_df = df.FFTDF(cell, kpts=self.kpts)
        self.conv_tol = 1.0e-10
        self.conv_tol_grad = None
        self.max_cycle = 80
        self.exxdiv = "ewald"
        self.verbose = getattr(cell, "verbose", 0)
        self.stdout = getattr(cell, "stdout", None)
        self.max_memory = getattr(cell, "max_memory", 4000)

        self.converged = False
        self.e_tot = None
        self.e_hf = None
        self.mo_coeff = None
        self.mo_occ = None
        self.mo_energy = None
        self.source_scf = None
        self.target_ks = None
        self.dc_energy_result = None

    def density_fit(self, auxbasis=None):
        """Select a GDF provider and return ``self`` like PySCF modifiers."""

        self.with_df = df.GDF(self.cell, kpts=self.kpts, auxbasis=auxbasis)
        return self

    def fft_density_fit(self, *, mesh=None):
        """Select FFTDF, optionally with an explicit mesh, and return ``self``."""

        self.with_df = df.FFTDF(self.cell, kpts=self.kpts, mesh=mesh)
        return self

    def rs_density_fit(self, auxbasis=None):
        """Select RSDF and return ``self`` like PySCF modifiers.

        RSDF is an explicit backend choice.  Its compact source mesh and
        tighter two-center metric mesh are resolved by PySCF when the
        provider is built during the HF reference calculation.
        """

        self.with_df = df.RSDF(
            self.cell, kpts=self.kpts, auxbasis=auxbasis
        )
        return self

    def _build_reference(self):
        return build_reference(
            self.cell,
            kpts=self.kpts,
            kpoint_mode=self.kpoint_mode,
            with_df=self.with_df,
            exxdiv=self.exxdiv,
            conv_tol=self.conv_tol,
            conv_tol_grad=self.conv_tol_grad,
            max_cycle=self.max_cycle,
        )

    def kernel(self, dm0=None, **kwargs) -> float:
        """Run HF once, then evaluate the target DFA at the frozen HF density."""

        if kwargs:
            self.set(**kwargs)
        reference = self._build_reference()
        reference.verbose = self.verbose
        reference.kernel(dm0=dm0)
        if not reference.converged:
            raise RuntimeError("HF reference did not converge; DC-DFT energy is undefined")

        result, target = evaluate_energy(
            reference,
            xc=self.xc,
            kpts=self.kpts,
            kpoint_mode=self.kpoint_mode,
            with_df=self.with_df,
        )
        self.source_scf = reference
        self.target_ks = target
        self.dc_energy_result = result
        self.converged = True
        self.e_tot = result.energy
        self.e_hf = result.reference_energy
        self.mo_coeff = reference.mo_coeff
        self.mo_occ = reference.mo_occ
        self.mo_energy = reference.mo_energy
        return result.energy

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        """Return the converged HF density used by the target DFA."""

        if self.source_scf is None:
            raise RuntimeError("run kernel() before requesting the HF density")
        return self.source_scf.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)

    def nuc_grad_method(self):
        """Return the route-specific nonvariational gradient object."""

        from ..grad import DCKGradients, DCGradients

        cls = DCKGradients if self.kpoint_mode else DCGradients
        return cls(self)

    Gradients = nuc_grad_method

    def to_gpu(self):
        """Convert through the separately installed GPU extension package."""

        try:
            from gpu4pyscf_pbc_dcdft import to_gpu
        except ImportError as exc:
            raise ImportError(
                "Install gpu4pyscf-pbc-dcdft to use the GPU DC-DFT route"
            ) from exc
        return to_gpu(self)

    def metadata(self) -> dict[str, Any]:
        capability = (
            self.with_df.capability()
            if hasattr(self.with_df, "capability")
            else {"class": self.with_df.__class__.__name__}
        )
        return {
            "method": self.__class__.__name__,
            "xc": self.xc,
            "nkpts": len(self.kpts),
            "df": capability,
            "mesh": [int(value) for value in self.cell.mesh],
            "ke_cutoff_ha": (
                None
                if self.cell.ke_cutoff is None
                else float(self.cell.ke_cutoff)
            ),
            "energy_complete": True,
            "analytic_gradient_complete": bool(
                capability.get("analytic_gradient", False)
            ),
        }
