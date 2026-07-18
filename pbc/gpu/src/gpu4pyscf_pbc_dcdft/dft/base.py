"""PySCF-style restricted GPU DC-DFT driver."""

from __future__ import annotations

from time import perf_counter

import cupy as cp
from pyscf import lib
from pyscf_pbc_dcdft.lib.kpoints import is_gamma, normalize_kpts
from pyscf_pbc_dcdft.dft.state import (
    invalidate_computed_state,
    mark_current_state,
    require_current_state,
    state_is_current,
    validate_restricted_input,
    validate_restricted_occupations,
)
from pyscf_pbc_dcdft.dft.target_integrator import (
    UNIFORM,
    normalize_target_integrator,
    normalize_target_multigrid_mesh,
)

from .. import df
from ..scf import build_reference
from .fixed_density import evaluate_energy


class DCRestrictedBase(lib.StreamObject):
    """Shared state and execution flow for GPU Gamma/k-point drivers."""

    _keys = {
        "cell", "mol", "kpts", "xc", "with_df", "conv_tol",
        "conv_tol_grad", "max_cycle",
        "exxdiv", "verbose", "stdout", "max_memory",
        "progress_callback",
        "target_integrator", "target_multigrid_mesh",
    }
    kpoint_mode = False

    def __init__(self, cell, *, kpts=None, xc: str = "PBE"):
        self.cell = cell
        self.mol = cell
        self.kpts = normalize_kpts(kpts)
        if not self.kpoint_mode and not is_gamma(self.kpts):
            raise ValueError("DCRKS accepts Gamma only; use DCKRKS")
        self.xc = str(xc)
        self.with_df = df.FFTDF(cell, kpts=self.kpts)
        self.conv_tol = 1.0e-10
        self.conv_tol_grad = None
        self.max_cycle = 80
        self.exxdiv = "ewald"
        self.verbose = getattr(cell, "verbose", 0)
        self.stdout = getattr(cell, "stdout", None)
        self.max_memory = getattr(cell, "max_memory", 4000)
        self.target_integrator = UNIFORM
        self.target_multigrid_mesh = None
        self.converged = False
        self.e_tot = None
        self.e_hf = None
        self.mo_coeff = None
        self.mo_occ = None
        self.mo_energy = None
        self.source_scf = None
        self.target_ks = None
        self.target_evaluation = None
        self.dc_energy_result = None
        self.timings_s = {}
        self.progress_callback = None
        self._dcdft_input_signature = None

    def set_target_integrator(self, mode=UNIFORM, *, mesh=None):
        """Select uniform, XC-only multigrid, or J+XC multigrid target work."""

        self.target_integrator = normalize_target_integrator(mode)
        self.target_multigrid_mesh = normalize_target_multigrid_mesh(mesh)
        if (
            self.target_integrator == UNIFORM
            and self.target_multigrid_mesh is not None
        ):
            raise ValueError(
                "target_multigrid_mesh requires a multigrid target_integrator"
            )
        invalidate_computed_state(self)
        return self

    def _progress(self, stage: str, **details) -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage, **details)

    def fft_density_fit(self, *, mesh=None):
        self.with_df = df.FFTDF(self.cell, kpts=self.kpts, mesh=mesh)
        invalidate_computed_state(self)
        return self

    def density_fit(self, auxbasis=None, *, setup_mode="native"):
        """Select native GPU GDF and return ``self`` like PySCF modifiers."""

        self.with_df = df.GDF(
            self.cell,
            kpts=self.kpts,
            auxbasis=auxbasis,
            setup_mode=setup_mode,
        )
        invalidate_computed_state(self)
        return self

    def rs_density_fit(self, auxbasis=None, *, central_setup="reference"):
        """Select GPU RSDF with explicit central-factor ownership."""

        self.with_df = df.RSDF(
            self.cell,
            kpts=self.kpts,
            auxbasis=auxbasis,
            central_setup=central_setup,
        )
        invalidate_computed_state(self)
        return self

    def kernel(self, dm0=None, **kwargs) -> float:
        """Run GPU HF and evaluate the frozen-density target-DFA energy."""

        if kwargs:
            self.set(**kwargs)
        invalidate_computed_state(self)
        validate_restricted_input(self)
        reference = build_reference(
            self.cell,
            kpts=self.kpts,
            kpoint_mode=self.kpoint_mode,
            with_df=self.with_df,
            exxdiv=self.exxdiv,
            conv_tol=self.conv_tol,
            conv_tol_grad=self.conv_tol_grad,
            max_cycle=self.max_cycle,
        )
        reference.verbose = self.verbose
        self._progress("source_hf_scf", state="started")
        start = perf_counter()
        reference.kernel(dm0=dm0)
        cp.cuda.Stream.null.synchronize()
        self.timings_s["hf_scf"] = perf_counter() - start
        self._progress(
            "source_hf_scf",
            state="complete",
            wall_time_s=self.timings_s["hf_scf"],
            converged=bool(reference.converged),
        )
        if not reference.converged:
            raise RuntimeError("GPU HF reference did not converge")
        validate_restricted_occupations(
            reference.mo_occ,
            nkpts=len(self.kpts),
            nelectron=int(self.cell.nelectron),
        )
        self._progress("target_dfa_energy", state="started", xc=self.xc)
        start = perf_counter()
        result, target = evaluate_energy(
            reference,
            xc=self.xc,
            kpts=self.kpts,
            kpoint_mode=self.kpoint_mode,
            with_df=self.with_df,
            target_integrator=self.target_integrator,
            target_multigrid_mesh=self.target_multigrid_mesh,
        )
        cp.cuda.Stream.null.synchronize()
        self.timings_s["target_pbe_energy"] = perf_counter() - start
        self._progress(
            "target_dfa_energy",
            state="complete",
            xc=self.xc,
            wall_time_s=self.timings_s["target_pbe_energy"],
        )
        self.timings_s["energy_total"] = (
            self.timings_s["hf_scf"] + self.timings_s["target_pbe_energy"]
        )
        self.source_scf = reference
        self.target_ks = target
        self.target_evaluation = target._dcdft_evaluation_workspace
        self.dc_energy_result = result
        self.converged = True
        self.e_tot = result.energy
        self.e_hf = result.reference_energy
        self.mo_coeff = reference.mo_coeff
        self.mo_occ = reference.mo_occ
        self.mo_energy = reference.mo_energy
        mark_current_state(self)
        return result.energy

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        require_current_state(self)
        return self.source_scf.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)

    def nuc_grad_method(self):
        """Return the Gamma or sampled-k-point GPU gradient object."""

        from ..grad import DCKGradients, DCGradients

        require_current_state(self)
        cls = DCKGradients if self.kpoint_mode else DCGradients
        return cls(self)

    Gradients = nuc_grad_method

    def to_cpu(self):
        """Create the corresponding CPU input object without reusing GPU arrays."""

        if self.target_integrator != UNIFORM:
            raise NotImplementedError(
                "CPU conversion cannot preserve a GPU multigrid target integrator"
            )
        from pyscf_pbc_dcdft import df as cpu_df
        from pyscf_pbc_dcdft.dft import DCKRKS as CPUDCKRKS
        from pyscf_pbc_dcdft.dft import DCRKS as CPUDCRKS

        cls = CPUDCKRKS if self.kpoint_mode else CPUDCRKS
        cpu_mf = cls(self.cell, kpts=self.kpts, xc=self.xc)
        backend = getattr(self.with_df, "backend_name", None)
        if backend == "fftdf":
            cpu_mf.with_df = cpu_df.FFTDF(
                self.cell, kpts=self.kpts, mesh=self.with_df.mesh
            )
        elif backend == "gdf":
            cpu_mf.with_df = cpu_df.GDF(
                self.cell,
                kpts=self.kpts,
                auxbasis=self.with_df.auxbasis,
            )
        elif backend == "rsdf":
            cpu_mf.with_df = self.with_df.to_cpu()
        else:
            raise NotImplementedError(
                f"cannot convert unsupported GPU DF backend {backend!r}"
            )
        for name in (
            "conv_tol", "conv_tol_grad", "max_cycle", "exxdiv", "verbose",
            "max_memory",
        ):
            setattr(cpu_mf, name, getattr(self, name))
        return cpu_mf

    def metadata(self):
        return {
            "method": self.__class__.__name__,
            "xc": self.xc,
            "nkpts": len(self.kpts),
            "df": self.with_df.capability(),
            "target_integrator": self.target_integrator,
            "target_numint_backend": (
                "numint"
                if self.target_integrator == UNIFORM
                else "multigrid_v2"
            ),
            "target_multigrid_mesh": (
                None
                if self.target_multigrid_mesh is None
                else [int(value) for value in self.target_multigrid_mesh]
            ),
            "target_evaluation_workspace": (
                self.target_evaluation is not None
            ),
            "mesh": [int(value) for value in self.cell.mesh],
            "ke_cutoff_ha": (
                None
                if self.cell.ke_cutoff is None
                else float(self.cell.ke_cutoff)
            ),
            "energy_complete": state_is_current(self),
            "analytic_gradient_complete": bool(
                self.with_df.capability().get("analytic_gradient", False)
            ),
            "silent_cpu_fallback": False,
            "timings_s": dict(self.timings_s),
        }
