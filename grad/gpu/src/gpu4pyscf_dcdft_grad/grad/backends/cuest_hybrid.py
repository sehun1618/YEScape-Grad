#!/usr/bin/env python
"""Shared-DF/MO-direct adjoint DC gradient with selective cuEST OE/XC.

This is an experimental successor to
``rdc_grad_gpu_pure_s1opt_adjoint_cuest_scf.py``.  It combines four
optimizations while keeping the density-fitted RHF gradient algebra intact:

1. The KS Fock and HF CPHF response share one GPU4PySCF DF/CDERI object.
2. CPHF and the adjoint response use ``gpu4pyscf.df.hessian.jk.get_jk`` to
   return only the requested MO/occupied block, rather than full AO J/K.
3. cuEST is used selectively for one-electron/XC gradient kernels; DF-J/K
   remains on the shared GPU4PySCF object (the low-memory hybrid route).  The
   default defers creation of the cuEST XC handle until after the high-memory
   direct DF derivative.  An eager mode also uses cuEST OE/XC in the KS Fock.
4. ``4*Z:F1`` is evaluated directly by the DF derivative-energy kernel with
   explicit low-rank density factors, without constructing ``Hessian.make_h1``.

The shared CDERI is released immediately after the adjoint response, its last
consumer.  In the default late mode, the direct DF derivative is evaluated
before the cuEST XC handle is created; cuEST plans are then released after the
KS skeleton.  cuEST does not currently expose the cross DF derivative needed
by the direct term, so that exact term intentionally stays on GPU4PySCF.

cuEST's XC wrapper requires Stratmann partitioning without atomic-radii
adjustment.  The GPU4PySCF comparison mode uses the same grid definition, so
``--gradient-backend gpu4pyscf`` is an apples-to-apples reference for the
selective cuEST gradient path.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import cupy as cp
import numpy as np
from pyscf import gto
from pyscf.dft.gen_grid import stratmann
from pyscf.lib import logger

from gpu4pyscf.df.hessian import jk as df_hessian_jk
from gpu4pyscf.dft import rks as gpu_rks
from gpu4pyscf.lib.cuest_wrapper import apply_cuest_wrapper
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.scf import cphf as gpu_cphf
from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf.scf._response_functions import (
    _gen_rhf_response as _gen_rhf_response_gpu,
)

from ..rhf import _contract, _read_xyz
from .cuest_scf import (
    DCGradientGPUCuESTSCF,
    _MODULUS_KEY,
    _SLICE_KEY,
    _build_parser as _build_cuest_scf_parser,
    _direct_contracted_z_f1_factorized,
    _explicit_auxbasis,
    build_cuest_rhf_handoff,
)


_GRADIENT_BACKENDS = (
    "cuest_hybrid_late",
    "cuest_hybrid",
    "gpu4pyscf",
)
_RESPONSE_BACKENDS = ("mo_direct", "ao")


def _release_device_cache() -> None:
    """Return unused CuPy blocks to CUDA after a large plan is destroyed."""

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()


def _used_device_gib() -> float:
    free, total = cp.cuda.runtime.memGetInfo()
    return (total - free) / 2**30


def _df_hf_response_mo(
    dfobj,
    dm_tag,
    left_coeff,
    occ_coeff,
    *,
    direct_scf_tol: float = 1.0e-14,
):
    """Return ``left.T @ (J-.5K)[dm] @ occ`` without AO J/K output.

    ``dm_tag.mo1`` and ``dm_tag.occ_coeff`` must describe the same symmetric
    RHF response density as ``dm_tag``.  GPU4PySCF's Hessian JK kernel uses
    those factors for exchange and performs the final AO->MO projection in
    the DF loop.
    """

    vj_mo, vk_mo = df_hessian_jk.get_jk(
        dfobj,
        dm_tag,
        left_coeff,
        occ_coeff,
        hermi=1,
        with_j=True,
        with_k=True,
        direct_scf_tol=direct_scf_tol,
    )
    nset = int(dm_tag.reshape(-1, dm_tag.shape[-1], dm_tag.shape[-1]).shape[0])
    out = (vj_mo - 0.5 * vk_mo).reshape(
        nset, left_coeff.shape[1], occ_coeff.shape[1]
    )
    del vj_mo, vk_mo
    return out


def _make_mo_direct_ax(
    dfobj,
    Cv,
    Co,
    *,
    direct_scf_tol: float = 1.0e-14,
):
    """Build the CPHF ``A*x`` callback returning the virtual/occupied block."""

    nvir = Cv.shape[1]
    nocc = Co.shape[1]
    co2 = Co * 2.0

    def ax_vo(x_vo):
        original_shape = x_vo.shape
        x = cp.asarray(x_vo).reshape(-1, nvir, nocc)

        # dm = 2 * [(Cv X) Co.T + h.c.].  The factor tags are consumed by
        # df.hessian.jk for its occupied-factor exchange implementation.
        mo1 = _contract("pa,kai->kpi", Cv, x)
        dm_x = _contract("kpi,qi->kpq", mo1, co2)
        dm_x = dm_x + dm_x.transpose(0, 2, 1)
        dm_x = tag_array(dm_x, mo1=mo1, occ_coeff=Co)

        out = _df_hf_response_mo(
            dfobj,
            dm_x,
            Cv,
            Co,
            direct_scf_tol=direct_scf_tol,
        )
        del mo1, dm_x
        return out.reshape(original_shape)

    return ax_vo


def _adjoint_z_ax_mo_direct(
    dfobj,
    Z,
    Co,
    Cv,
    S1_mo_o,
    so,
    *,
    direct_scf_tol: float = 1.0e-14,
):
    """Return ``Z:Ax(S1)_vo`` using one MO-direct HF response.

    Here the physical density is ``Dz+Dz.T`` rather than the CPHF convention
    ``2*(Dz+Dz.T)``.  Consequently the exchange factor tag contains
    ``0.5*(Cv@Z)`` while the explicit density passed to J remains unscaled.
    """

    z_left = Cv @ Z
    dz_sym = z_left @ Co.T
    dz_sym = dz_sym + dz_sym.T
    # df.hessian.jk consistently expects a leading density-set dimension,
    # including for a single response density.
    dz_sym = tag_array(
        dz_sym[None], mo1=(0.5 * z_left)[None], occ_coeff=Co
    )
    vz_oo = _df_hf_response_mo(
        dfobj,
        dz_sym,
        Co,
        Co,
        direct_scf_tol=direct_scf_tol,
    )[0]
    z_ax = 2.0 * _contract("Aij,ij->A", S1_mo_o[:, so, :], vz_oo)
    del z_left, dz_sym, vz_oo
    return z_ax


class DCGradientGPUCuESTHybrid(DCGradientGPUCuESTSCF):
    """DC gradient with shared DF, MO-direct response, and cuEST OE/XC."""

    def __init__(
        self,
        *args,
        gradient_backend: str = "cuest_hybrid_late",
        response_backend: str = "mo_direct",
        release_shared_df: bool = True,
        release_cuest_plans: bool = True,
        cuest_math_mode: str = "default",
        cuest_workspace_gb: float = 2.0,
        cuest_exchange_slice_count: int = 6,
        cuest_exchange_modulus_count: int = 9,
        **kwargs,
    ):
        # The parent flag is unused because this class controls the exact
        # last-use point in its reordered kernel.
        super().__init__(*args, release_cphf_df=False, **kwargs)
        gradient_backend = gradient_backend.lower().replace("-", "_")
        response_backend = response_backend.lower().replace("-", "_")
        if gradient_backend not in _GRADIENT_BACKENDS:
            raise ValueError(
                f"gradient_backend must be one of {_GRADIENT_BACKENDS}"
            )
        if response_backend not in _RESPONSE_BACKENDS:
            raise ValueError(
                f"response_backend must be one of {_RESPONSE_BACKENDS}"
            )
        if cuest_workspace_gb <= 0:
            raise ValueError("cuest_workspace_gb must be positive")
        if cuest_exchange_slice_count <= 0 or cuest_exchange_modulus_count <= 0:
            raise ValueError("cuEST exchange factor counts must be positive")

        self.gradient_backend = gradient_backend
        self.response_backend = response_backend
        self.release_shared_df = bool(release_shared_df)
        self.release_cuest_plans = bool(release_cuest_plans)
        self.cuest_math_mode = cuest_math_mode
        self.cuest_workspace_bytes = int(cuest_workspace_gb * 1.0e9)
        self.cuest_exchange_slice_count = int(cuest_exchange_slice_count)
        self.cuest_exchange_modulus_count = int(cuest_exchange_modulus_count)
        self.optimization_diagnostics: dict[str, Any] = {}

    def _wrap_cuest_oe_xc(self, ks_base, log):
        """Wrap an existing KS object while leaving its DF-J/K untouched."""

        ks_cuest = apply_cuest_wrapper(ks_base)
        ks_cuest.maximum_workspace_bytes = self.cuest_workspace_bytes

        # This assignment resets/rebuilds the wrapper helpers after the
        # workspace value is set.  J/K then dispatches to ks_base.with_df,
        # which is the same object used by HF CPHF below.
        ks_cuest.turn_on_cuest_jk = False
        ks_cuest.additional_precision_control_parameters.update(
            {
                _SLICE_KEY: self.cuest_exchange_slice_count,
                _MODULUS_KEY: self.cuest_exchange_modulus_count,
            }
        )
        ks_cuest.math_mode = self.cuest_math_mode
        log.info("  cuEST OE/XC enabled; DF-J/K remains on GPU4PySCF")
        return ks_cuest

    def _build_ks_method(self, shared_df, auxbasis, log):
        """Create a same-grid KS method with all DF J/K on ``shared_df``."""

        mol = self.mol
        ks_base = gpu_rks.RKS(mol, xc=self.xc).density_fit(
            auxbasis=auxbasis
        )
        ks_base.with_df = shared_df
        ks_base.with_df.auxbasis = auxbasis

        # cuEST requires this grid partition; use it in both branches so the
        # backend switch does not silently change the mathematical quadrature.
        ks_base.grids.atom_grid = self.atom_grid
        ks_base.grids.becke_scheme = stratmann
        ks_base.grids.radii_adjust = None
        if getattr(ks_base, "nlcgrids", None) is not None:
            ks_base.nlcgrids.atom_grid = self.atom_grid
            ks_base.nlcgrids.becke_scheme = stratmann
            ks_base.nlcgrids.radii_adjust = None

        if self.gradient_backend == "gpu4pyscf":
            log.info("  gradient backend: GPU4PySCF (Stratmann reference)")
            return ks_base
        if self.gradient_backend == "cuest_hybrid_late":
            log.info(
                "  gradient backend: deferred cuEST OE/XC; "
                "GPU4PySCF KS Fock"
            )
            return ks_base

        log.info("  gradient backend: eager cuEST OE/XC")
        return self._wrap_cuest_oe_xc(ks_base, log)

    @staticmethod
    def _release_df(dfobj) -> float:
        cp.cuda.runtime.deviceSynchronize()
        free_before, _ = cp.cuda.runtime.memGetInfo()
        dfobj.reset()
        _release_device_cache()
        free_after, _ = cp.cuda.runtime.memGetInfo()
        return (free_after - free_before) / 2**30

    def kernel(self, verbose=None):
        mol = self.mol
        if verbose is None:
            verbose = mol.verbose
        log = logger.new_logger(mol, verbose)

        t_total = time.perf_counter()
        stage_times: dict[str, float] = {}
        memory: dict[str, float | bool] = {}
        log.info("\n" + "=" * 72)
        log.info("DC gradient: shared DF + MO-direct response + cuEST hybrid")
        log.info("=" * 72)

        if not hasattr(self.mf_hf, "with_df") or self.mf_hf.with_df is None:
            raise ValueError("This optimized path requires density-fitted RHF")

        auxbasis = _explicit_auxbasis(mol, self.mf_hf.with_df.auxbasis)
        shared_df = self.mf_hf.with_df
        shared_df.auxbasis = auxbasis

        C = cp.asarray(self.mf_hf.mo_coeff)
        e = cp.asarray(self.mf_hf.mo_energy)
        occ = cp.asarray(self.mf_hf.mo_occ)
        nao, nmo = C.shape
        nocc = int(cp.sum(occ > 0))
        nvir = nmo - nocc
        so = slice(0, nocc)
        sv = slice(nocc, nmo)
        Co = C[:, so]
        Cv = C[:, sv]
        Co2 = Co * 2.0
        atom_slices = [
            mol.aoslice_by_atom()[ia, 2:4] for ia in range(mol.natm)
        ]

        dm_hf = 2.0 * Co @ Co.T
        dm_hf = tag_array(dm_hf, mo_coeff=C, mo_occ=occ)

        # ---------------------------------------------------------------
        # 1) KS Fock: eager cuEST or GPU4PySCF, always with shared DF-J
        # ---------------------------------------------------------------
        log.info("\n[1/7] KS Fock")
        t0 = time.perf_counter()
        ks_gpu = self._build_ks_method(shared_df, auxbasis, log)
        shared_df_identity = ks_gpu.with_df is shared_df
        F_KS = cp.asarray(ks_gpu.get_fock(dm=dm_hf))
        F_KS_vo = Cv.T @ F_KS @ Co
        F_KS_oo = Co.T @ F_KS @ Co
        stage_times["ks_fock"] = time.perf_counter() - t0
        log.info("  %.2f s", stage_times["ks_fock"])

        # ---------------------------------------------------------------
        # 2) CPHF: the same shared CDERI, preferably direct MO output
        # ---------------------------------------------------------------
        log.info("[2/7] Z-vector (%s)", self.response_backend)
        t0 = time.perf_counter()
        hf_gpu = gpu_hf.RHF(mol).density_fit(auxbasis=auxbasis)
        hf_gpu.with_df = shared_df
        hf_gpu.mo_coeff = C
        hf_gpu.mo_energy = e
        hf_gpu.mo_occ = occ
        hf_gpu.converged = True
        hf_gpu.e_tot = self.mf_hf.e_tot

        hf_resp = None
        if self.response_backend == "mo_direct":
            Ax_vo = _make_mo_direct_ax(
                shared_df,
                Cv,
                Co,
                direct_scf_tol=hf_gpu.direct_scf_tol,
            )
        else:
            hf_resp = _gen_rhf_response_gpu(
                hf_gpu, mo_coeff=C, mo_occ=occ, hermi=1
            )

            def Ax_vo(x_vo):
                original_shape = x_vo.shape
                x = cp.asarray(x_vo).reshape(-1, nvir, nocc)
                mo1 = _contract("pa,kai->kpi", Cv, x)
                dm_x = _contract("kpi,qi->kpq", mo1, Co2)
                dm_x = dm_x + dm_x.transpose(0, 2, 1)
                dm_x = tag_array(
                    dm_x, mo1=mo1, occ_coeff=Co, mo_occ=occ
                )
                v_dm = hf_resp(dm_x)
                tmp = _contract("ap,kpq->kaq", Cv.T, v_dm)
                out = _contract("kaq,qi->kai", tmp, Co)
                del mo1, dm_x, v_dm, tmp
                return out.reshape(original_shape)

        Z, converged = gpu_cphf.solve(
            Ax_vo,
            e,
            occ,
            F_KS_vo,
            max_cycle=self.cphf_max_cycle,
            tol=self.cphf_tol,
        )
        if converged is not None and not converged:
            log.warn("GPU CPHF did not converge")
        stage_times["z_vector"] = time.perf_counter() - t0
        log.info("  %.2f s", stage_times["z_vector"])

        # ---------------------------------------------------------------
        # 3) S1 occupied-column AO->MO transform
        # ---------------------------------------------------------------
        log.info("[3/7] S1 AO->MO (occupied columns)")
        t0 = time.perf_counter()
        ks_gpu.mo_coeff = C
        ks_gpu.mo_occ = occ
        ks_gpu.mo_energy = e
        ks_gpu.converged = True

        h1e = cp.asarray(ks_gpu.get_hcore(mol))
        ks_gpu.e_tot = float(
            cp.einsum("ij,ji->", dm_hf, h1e + F_KS) * 0.5
        ) + ks_gpu.energy_nuc()
        ks_grad_gpu = ks_gpu.Gradients()
        ks_grad_gpu.grid_response = self.grid_response
        S1_ao = cp.asarray(ks_grad_gpu.get_ovlp(mol))

        natm = mol.natm
        npert = 3 * natm
        S1_mo_o = cp.empty((npert, nmo, nocc), dtype=C.dtype)
        for ia, (p0, p1) in enumerate(atom_slices):
            for ic in range(3):
                s1_block = cp.zeros((nao, nao), dtype=C.dtype)
                s1_block[p0:p1] = S1_ao[ic, p0:p1]
                s1_block = s1_block + s1_block.T
                S1_mo_o[3 * ia + ic] = C.T @ s1_block @ Co
        del S1_ao, F_KS
        stage_times["s1_ao_to_mo"] = time.perf_counter() - t0
        log.info("  %.2f s", stage_times["s1_ao_to_mo"])

        # ---------------------------------------------------------------
        # 4) Adjoint response at the shared CDERI's last-use point
        # ---------------------------------------------------------------
        log.info("[4/7] Adjoint Z:Ax(S1) (%s)", self.response_backend)
        t0 = time.perf_counter()
        if self.response_backend == "mo_direct":
            z_ax = _adjoint_z_ax_mo_direct(
                shared_df,
                Z,
                Co,
                Cv,
                S1_mo_o,
                so,
                direct_scf_tol=hf_gpu.direct_scf_tol,
            )
        else:
            from ..rhf import _adjoint_z_ax

            z_ax = _adjoint_z_ax(hf_resp, Z, Co, Cv, S1_mo_o, so)
        stage_times["adjoint_response"] = time.perf_counter() - t0
        log.info("  %.2f s", stage_times["adjoint_response"])

        # No subsequent operation consumes the SCF/CPHF CDERI.  Skeleton and
        # direct Z:F1 use derivative-integral kernels and can rebuild only the
        # smaller metadata they need.
        if self.release_shared_df:
            released = self._release_df(shared_df)
            memory["released_shared_df_gib"] = released
            log.info("  released shared KS/CPHF DF storage: %.2f GiB", released)
        else:
            memory["released_shared_df_gib"] = 0.0
        memory["used_after_shared_df_release_gib"] = _used_device_gib()

        # ---------------------------------------------------------------
        # 5) Direct factorized 4*Z:F1 (no make_h1 array)
        #
        # In late-cuEST mode this deliberately precedes construction of the
        # cuEST XC handle.  cuEST retains a process-level CUDA allocation even
        # after its Python handle is destroyed; deferring it keeps that memory
        # out of the largest DF derivative stage.
        # ---------------------------------------------------------------
        log.info("[5/7] Direct factorized 4*Z:F1")
        t0 = time.perf_counter()
        g_zf1 = _direct_contracted_z_f1_factorized(
            mol, hf_gpu, C, occ, Co, Cv, Z
        )
        stage_times["direct_z_f1"] = time.perf_counter() - t0
        log.info("  %.2f s", stage_times["direct_z_f1"])
        _release_device_cache()

        # ---------------------------------------------------------------
        # 6) KS skeleton; cuEST handles OE/XC analytical gradients only
        # ---------------------------------------------------------------
        if self.gradient_backend == "cuest_hybrid_late":
            # S1 used the ordinary gradient object's overlap derivative.
            # Recreate only the skeleton gradient object after wrapping.
            del ks_grad_gpu
            ks_gpu = self._wrap_cuest_oe_xc(ks_gpu, log)
            ks_grad_gpu = ks_gpu.Gradients()
            ks_grad_gpu.grid_response = self.grid_response

        log.info("[6/7] KS skeleton (%s)", self.gradient_backend)
        t0 = time.perf_counter()
        e_occ = e[so]
        g_skel = self._skeleton_gpu(
            mol,
            ks_gpu,
            ks_grad_gpu,
            F_KS_oo,
            S1_mo_o,
            so,
            log,
            e_occ,
        )
        stage_times["skeleton"] = time.perf_counter() - t0
        log.info("  %.2f s", stage_times["skeleton"])

        handles = getattr(ks_gpu, "handles", None)
        plan_flags = {
            "cuest_oe_plan_built": bool(
                handles is not None and handles.oeintplan_handle is not None
            ),
            "cuest_df_plan_built": bool(
                handles is not None and handles.dfintplan_handle is not None
            ),
            "cuest_xc_plan_built": bool(
                handles is not None and handles.xcintplan_handle is not None
            ),
        }
        if self.release_cuest_plans and handles is not None:
            ks_gpu.reset()
        elif handles is not None:
            # Explicit debug mode: retain references so HandleBundle.__del__
            # does not immediately destroy the plans when locals are removed.
            self._retained_ks_gpu = ks_gpu
            self._retained_ks_grad_gpu = ks_grad_gpu

        # Neither the KS method nor its numerical grids are used after the
        # skeleton.  Destroy both wrappers and return reusable CuPy blocks.
        del ks_grad_gpu, ks_gpu, handles, F_KS_oo, h1e, dm_hf
        _release_device_cache()
        memory["used_after_ks_cleanup_gib"] = _used_device_gib()

        # A gradient implementation is allowed to rebuild the DF object.  It
        # is no longer needed after the skeleton.
        if (
            self.release_shared_df
            and getattr(shared_df, "_cderi", None) is not None
        ):
            memory["released_df_rebuilt_by_skeleton_gib"] = self._release_df(
                shared_df
            )

        # ---------------------------------------------------------------
        # 7) Scalar assembly; no B, Se, AxS, dm_s, or v_resp arrays
        # ---------------------------------------------------------------
        log.info("[7/7] Scalar non-variational assembly")
        t0 = time.perf_counter()
        z_se = _contract(
            "ai,Aai->A", Z * e_occ[None, :], S1_mo_o[:, sv, :]
        )
        g_nv = g_zf1.reshape(-1) - 4.0 * z_se - 2.0 * z_ax
        g_tot = g_skel + g_nv.reshape(natm, 3)
        g_np = cp.asnumpy(g_tot)
        stage_times["scalar_assembly"] = time.perf_counter() - t0

        total_seconds = time.perf_counter() - t_total
        self.memory_diagnostics = memory
        self.optimization_diagnostics = {
            "gradient_backend": self.gradient_backend,
            "response_backend": self.response_backend,
            "shared_df_identity": bool(
                shared_df_identity and hf_gpu.with_df is shared_df
            ),
            "stage_times_seconds": stage_times,
            "total_seconds": total_seconds,
            "direct_z_f1_backend": "gpu4pyscf_df_factorized_cross_derivative",
            **plan_flags,
        }

        del g_zf1, z_se, z_ax, g_nv, g_tot
        log.info("\n" + "=" * 72)
        log.info(
            "Total %.2f s | max|gradient| %.6e",
            total_seconds,
            np.max(np.abs(g_np)),
        )
        log.info("=" * 72 + "\n")
        return g_np


def kernel(mf, xc="PBE", verbose=None, **kwargs):
    """Evaluate the combined shared-DF/selective-cuEST DC gradient."""

    return DCGradientGPUCuESTHybrid(mf, xc=xc, **kwargs).kernel(
        verbose=verbose
    )


def _build_parser():
    parser = _build_cuest_scf_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "keep_cphf_df":
            action.help = "Retain the shared KS/CPHF CDERI (debug only)"
    parser.add_argument(
        "--gradient-backend",
        choices=("cuest-hybrid-late", "cuest-hybrid", "gpu4pyscf"),
        default="cuest-hybrid-late",
        help=(
            "Deferred low-peak cuEST gradient (default), eager cuEST "
            "Fock+gradient, or same-grid GPU4PySCF reference"
        ),
    )
    parser.add_argument(
        "--response-backend",
        choices=("mo-direct", "ao"),
        default="mo-direct",
        help="CPHF/adjoint DF response output representation",
    )
    parser.add_argument(
        "--keep-cuest-gradient-plans",
        action="store_true",
        help="Retain cuEST OE/XC plans after the skeleton (debug only)",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    atom = _read_xyz(args.xyz)
    mol = gto.M(
        atom=atom,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=args.unit,
        verbose=args.verbose,
        max_memory=args.max_memory,
    )
    requested_auxbasis = (
        None if args.auxbasis.lower() == "auto" else args.auxbasis
    )

    if args.scf_backend == "cuest":
        mf, scf_info = build_cuest_rhf_handoff(
            mol,
            auxbasis=requested_auxbasis,
            conv_tol=args.scf_tol,
            max_cycle=args.scf_max_cycle,
            workspace_gb=args.cuest_workspace_gb,
            math_mode=args.cuest_math_mode,
            exchange_slice_count=args.cuest_exchange_slice_count,
            exchange_modulus_count=args.cuest_exchange_modulus_count,
        )
    else:
        resolved_auxbasis = _explicit_auxbasis(mol, requested_auxbasis)
        mf = gpu_hf.RHF(mol).density_fit(auxbasis=resolved_auxbasis)
        mf.conv_tol = args.scf_tol
        mf.max_cycle = args.scf_max_cycle
        t0 = time.perf_counter()
        energy = mf.kernel()
        if not mf.converged:
            raise RuntimeError(
                f"GPU4PySCF DF-RHF did not converge in "
                f"{args.scf_max_cycle} cycles"
            )
        scf_info = {
            "backend": "gpu4pyscf",
            "scf_seconds": time.perf_counter() - t0,
            "energy_eh": float(energy),
        }

    method = DCGradientGPUCuESTHybrid(
        mf,
        xc=args.xc,
        cphf_tol=args.cphf_tol,
        cphf_max_cycle=args.cphf_max_cycle,
        atom_grid=tuple(args.atom_grid),
        grid_response=not args.no_grid_response,
        gradient_backend=args.gradient_backend,
        response_backend=args.response_backend,
        release_shared_df=not args.keep_cphf_df,
        release_cuest_plans=not args.keep_cuest_gradient_plans,
        cuest_math_mode=args.cuest_math_mode,
        cuest_workspace_gb=args.cuest_workspace_gb,
        cuest_exchange_slice_count=args.cuest_exchange_slice_count,
        cuest_exchange_modulus_count=args.cuest_exchange_modulus_count,
    )
    gradient = method.kernel(verbose=args.verbose)

    print("\nSCF diagnostics:")
    for key, value in scf_info.items():
        if key != "auxbasis":
            print(f"  {key}: {value}")
    print("Optimization diagnostics:")
    for key, value in method.optimization_diagnostics.items():
        print(f"  {key}: {value}")
    print("Memory diagnostics:")
    for key, value in method.memory_diagnostics.items():
        print(f"  {key}: {value}")
    print(f"DC-{args.xc} gradient (Eh/Bohr):")
    print(np.array2string(gradient, precision=12, suppress_small=False))

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            output,
            gradient,
            fmt="%.16e",
            header=(
                f"DC-{args.xc} gradient (Eh/Bohr); source={args.xyz}; "
                f"scf_backend={args.scf_backend}; "
                f"gradient_backend={args.gradient_backend}; "
                f"response_backend={args.response_backend}"
            ),
        )
        print(f"Gradient written to: {output}")
    return 0


__all__ = [
    "DCGradientGPUCuESTHybrid",
    "_adjoint_z_ax_mo_direct",
    "_df_hf_response_mo",
    "_make_mo_direct_ax",
    "kernel",
]


if __name__ == "__main__":
    raise SystemExit(main())
