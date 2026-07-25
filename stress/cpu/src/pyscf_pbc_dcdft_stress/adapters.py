"""Read-only energy adapters from PBC DC-DFT into the stress workspace."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .results import EnergyEvaluation


@dataclass(frozen=True)
class GammaGDFEnergyEvaluator:
    """Rebuild one restricted Gamma/GDF DC-DFT state for each strained cell."""

    xc: str = "PBE"
    auxbasis: str | None = None
    conv_tol: float = 1.0e-10
    conv_tol_grad: float | None = None
    max_cycle: int = 80
    exxdiv: str = "ewald"
    verbose: int = 0

    @classmethod
    def from_method(cls, method):
        """Capture numerical settings without mutating the existing method."""

        if bool(getattr(method, "kpoint_mode", False)):
            raise NotImplementedError("S1 evaluator supports Gamma-point DCRKS only")
        provider = getattr(method, "with_df", None)
        if getattr(provider, "backend_name", None) != "gdf":
            raise NotImplementedError("S1 evaluator supports the GDF backend only")
        if getattr(method, "target_integrator", "uniform") != "uniform":
            raise NotImplementedError("S1 evaluator supports uniform target integration only")
        if getattr(method, "dispersion", None) is not None:
            raise NotImplementedError("S1 stress does not yet include dispersion stress")
        return cls(
            xc=str(method.xc), auxbasis=getattr(provider, "auxbasis", None),
            conv_tol=float(method.conv_tol),
            conv_tol_grad=getattr(method, "conv_tol_grad", None),
            max_cycle=int(method.max_cycle), exxdiv=str(method.exxdiv),
            verbose=int(method.verbose),
        )

    def __call__(self, cell, *, dm0=None, kpts=None) -> EnergyEvaluation:
        if kpts is not None and np.max(np.abs(np.asarray(kpts))) > 1.0e-12:
            raise NotImplementedError("S1 evaluator supports Gamma-point energy only")

        return _evaluate_gamma_gaussian_df(
            cell, backend="gdf", xc=self.xc, auxbasis=self.auxbasis,
            conv_tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
            max_cycle=self.max_cycle, exxdiv=self.exxdiv,
            verbose=self.verbose, dm0=dm0,
        )


@dataclass(frozen=True)
class GammaRSDFEnergyEvaluator:
    """Rebuild one restricted Gamma/RSDF DC-DFT state per strained cell."""

    xc: str = "PBE"
    auxbasis: str | None = None
    conv_tol: float = 1.0e-10
    conv_tol_grad: float | None = None
    max_cycle: int = 80
    exxdiv: str = "ewald"
    verbose: int = 0

    @classmethod
    def from_method(cls, method):
        if bool(getattr(method, "kpoint_mode", False)):
            raise NotImplementedError("RSDF stress evaluator supports Gamma only")
        provider = getattr(method, "with_df", None)
        if getattr(provider, "backend_name", None) != "rsdf":
            raise NotImplementedError("RSDF stress evaluator requires RSDF")
        if getattr(method, "target_integrator", "uniform") != "uniform":
            raise NotImplementedError("RSDF stress evaluator requires uniform integration")
        if getattr(method, "dispersion", None) is not None:
            raise NotImplementedError("dispersion stress is outside S5")
        return cls(
            xc=str(method.xc), auxbasis=getattr(provider, "auxbasis", None),
            conv_tol=float(method.conv_tol),
            conv_tol_grad=getattr(method, "conv_tol_grad", None),
            max_cycle=int(method.max_cycle), exxdiv=str(method.exxdiv),
            verbose=int(method.verbose),
        )

    def __call__(self, cell, *, dm0=None, kpts=None) -> EnergyEvaluation:
        if kpts is not None and np.max(np.abs(np.asarray(kpts))) > 1.0e-12:
            raise NotImplementedError("RSDF stress evaluator supports Gamma only")
        return _evaluate_gamma_gaussian_df(
            cell, backend="rsdf", xc=self.xc, auxbasis=self.auxbasis,
            conv_tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
            max_cycle=self.max_cycle, exxdiv=self.exxdiv,
            verbose=self.verbose, dm0=dm0,
        )


@dataclass(frozen=True)
class KPointGaussianDFEnergyEvaluator:
    """Rebuild a restricted sampled-k DC state at fixed fractional k-points."""

    scaled_kpts: np.ndarray
    df_backend: str = "gdf"
    xc: str = "PBE"
    auxbasis: str | None = None
    conv_tol: float = 1.0e-10
    conv_tol_grad: float | None = None
    max_cycle: int = 80
    exxdiv: str = "ewald"
    verbose: int = 0

    def __post_init__(self):
        points = np.asarray(self.scaled_kpts, dtype=float).reshape(-1, 3)
        if len(points) < 2:
            raise ValueError("sampled-k evaluator requires at least two k-points")
        if self.df_backend not in {"gdf", "rsdf"}:
            raise ValueError(f"unsupported DF backend {self.df_backend!r}")
        object.__setattr__(self, "scaled_kpts", points.copy())

    @classmethod
    def from_method(cls, method):
        if not bool(getattr(method, "kpoint_mode", False)):
            raise NotImplementedError("sampled-k evaluator requires DCKRKS")
        provider = getattr(method, "with_df", None)
        backend = getattr(provider, "backend_name", None)
        if backend not in {"gdf", "rsdf"}:
            raise NotImplementedError("sampled-k evaluator supports GDF/RSDF")
        if getattr(method, "target_integrator", "uniform") != "uniform":
            raise NotImplementedError("sampled-k evaluator requires uniform integration")
        if getattr(method, "dispersion", None) is not None:
            raise NotImplementedError("dispersion stress is outside S6")
        return cls(
            scaled_kpts=np.asarray(method.cell.get_scaled_kpts(method.kpts)),
            df_backend=backend, xc=str(method.xc),
            auxbasis=getattr(provider, "auxbasis", None),
            conv_tol=float(method.conv_tol),
            conv_tol_grad=getattr(method, "conv_tol_grad", None),
            max_cycle=int(method.max_cycle), exxdiv=str(method.exxdiv),
            verbose=int(method.verbose),
        )

    def __call__(self, cell, *, dm0=None, kpts=None) -> EnergyEvaluation:
        from pyscf_pbc_dcdft import dft

        points = np.asarray(cell.get_abs_kpts(self.scaled_kpts), dtype=float)
        if kpts is not None and not np.allclose(
            np.asarray(kpts).reshape(-1, 3), points, atol=1e-12, rtol=0
        ):
            raise ValueError("explicit k-points do not match fixed fractional sampling")
        method = dft.DCKRKS(cell, kpts=points, xc=self.xc)
        if self.df_backend == "gdf":
            method = method.density_fit(auxbasis=self.auxbasis)
        else:
            method = method.rs_density_fit(auxbasis=self.auxbasis)
        method.conv_tol = self.conv_tol
        method.conv_tol_grad = self.conv_tol_grad
        method.max_cycle = self.max_cycle
        method.exxdiv = self.exxdiv
        method.verbose = self.verbose
        start = time.perf_counter()
        energy = method.kernel(dm0=dm0)
        wall_seconds = time.perf_counter() - start
        if not method.converged:
            raise RuntimeError("strained sampled-k DC-DFT energy did not converge")
        return EnergyEvaluation(
            energy_ha=float(energy), wall_seconds=wall_seconds,
            converged=True, density=np.asarray(method.source_scf.make_rdm1()),
            metadata={
                "driver": method.__class__.__name__,
                "xc": self.xc,
                "df_backend": self.df_backend,
                "nkpts": len(points),
                "scaled_kpts": self.scaled_kpts.tolist(),
                "absolute_kpts": points.tolist(),
                "mesh": [int(value) for value in cell.mesh],
                "reference_energy_ha": float(method.e_hf),
            },
        )


def _evaluate_gamma_gaussian_df(
    cell, *, backend, xc, auxbasis, conv_tol, conv_tol_grad, max_cycle,
    exxdiv, verbose, dm0,
):
    from pyscf_pbc_dcdft import dft

    method = dft.DCRKS(cell, xc=xc)
    if backend == "gdf":
        method = method.density_fit(auxbasis=auxbasis)
    elif backend == "rsdf":
        method = method.rs_density_fit(auxbasis=auxbasis)
    else:
        raise ValueError(f"unsupported Gaussian DF backend {backend!r}")
    method.conv_tol = float(conv_tol)
    method.conv_tol_grad = conv_tol_grad
    method.max_cycle = int(max_cycle)
    method.exxdiv = exxdiv
    method.verbose = int(verbose)
    start = time.perf_counter()
    energy = method.kernel(dm0=dm0)
    wall_seconds = time.perf_counter() - start
    if not method.converged:
        raise RuntimeError(
            f"strained Gamma/{backend.upper()} DC-DFT energy did not converge"
        )
    density = np.asarray(method.source_scf.make_rdm1())
    return EnergyEvaluation(
        energy_ha=float(energy), wall_seconds=wall_seconds,
        converged=True, density=density,
        metadata={
            "driver": method.__class__.__name__,
            "xc": xc,
            "df_backend": backend,
            "auxbasis": auxbasis,
            "mesh": [int(value) for value in cell.mesh],
            "ke_cutoff_ha": (
                None if cell.ke_cutoff is None else float(cell.ke_cutoff)
            ),
            "reference_energy_ha": float(method.e_hf),
        },
    )
