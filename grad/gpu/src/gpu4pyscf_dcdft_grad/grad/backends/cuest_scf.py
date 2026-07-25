#!/usr/bin/env python
"""cuEST-RHF handoff + memory-ordered adjoint DC-DFT gradient.

This module combines the exact/validated optimizations that were measured for
water32/def2-SVPD:

* full cuEST density-fitted RHF SCF;
* immediate destruction of cuEST plans after copying converged MO data;
* GPU4PySCF adjoint DC-gradient evaluation;
* adjoint ``Z:Ax(S1)`` before direct ``Z:F1``;
* release of the CPHF CDERI after its last use;
* explicit low-rank factors for the direct DF derivative contraction.

cuEST is used for the reference RHF SCF only.  The subsequent DC-gradient KS
Fock, CPHF, skeleton, and derivative contractions use GPU4PySCF.  This avoids
retaining the dense cuEST DF plan during the high-memory derivative stage.
"""

from __future__ import annotations

import gc
import time

import cupy as cp
import numpy as np
from pyscf import df, gto

from gpu4pyscf.lib.cuest_wrapper import apply_cuest_wrapper
from gpu4pyscf.scf import hf as gpu_hf

from ..rhf import (
    DCGradientGPU as _AdjointDCGradientGPU,
    _adjoint_z_ax,
    _adjoint_z_ax_mo_direct,
    _build_parser as _build_adjoint_parser,
    _direct_contracted_z_f1,
    _direct_contracted_z_f1_factorized,
    _read_xyz,
)


_SLICE_KEY = (
    "CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_SLICE_COUNT"
)
_MODULUS_KEY = (
    "CUEST_DFSYMMETRICEXCHANGECOMPUTE_PARAMETERS_INT8_MODULUS_COUNT"
)


def _explicit_auxbasis(mol, auxbasis):
    """Resolve PySCF's lazy auxiliary basis for the cuEST wrapper."""

    if auxbasis is not None:
        return auxbasis
    return df.addons.make_auxmol(mol, None).basis


def build_cuest_rhf_handoff(
    mol, *, auxbasis=None, conv_tol=1.0e-10, max_cycle=100,
    workspace_gb=2.0, math_mode="default", exchange_slice_count=6,
    exchange_modulus_count=9,
):
    """Run full-cuEST DF-RHF, release its plans, and return GPU4PySCF orbitals.

    Returns
    -------
    handoff_mf
        A converged GPU4PySCF DFRHF object containing the cuEST orbitals and
        energies, but no built CDERI.
    diagnostics
        SCF timing and device-memory information before/after plan release.
    """

    if workspace_gb <= 0:
        raise ValueError("workspace_gb must be positive")
    if exchange_slice_count <= 0 or exchange_modulus_count <= 0:
        raise ValueError("cuEST exchange factor counts must be positive")

    resolved_auxbasis = _explicit_auxbasis(mol, auxbasis)
    base = gpu_hf.RHF(mol).density_fit(auxbasis=resolved_auxbasis)
    wrapped = apply_cuest_wrapper(base)
    wrapped.maximum_workspace_bytes = int(workspace_gb * 1.0e9)
    wrapped.additional_precision_control_parameters.update(
        {
            _SLICE_KEY: int(exchange_slice_count),
            _MODULUS_KEY: int(exchange_modulus_count),
        }
    )
    wrapped.math_mode = math_mode
    wrapped.conv_tol = conv_tol
    wrapped.max_cycle = max_cycle

    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    energy = wrapped.kernel()
    cp.cuda.runtime.deviceSynchronize()
    scf_seconds = time.perf_counter() - t0
    if not wrapped.converged:
        raise RuntimeError(f"cuEST DF-RHF did not converge in {max_cycle} cycles")

    # The copies must outlive CuESTWrapper.reset()/HandleBundle destruction.
    mo_coeff = cp.array(wrapped.mo_coeff)
    mo_energy = cp.array(wrapped.mo_energy)
    mo_occ = cp.array(wrapped.mo_occ)
    e_tot = float(wrapped.e_tot)

    free_before_reset, total_memory = cp.cuda.runtime.memGetInfo()
    wrapped.reset()
    del wrapped, base
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()
    free_after_reset, _ = cp.cuda.runtime.memGetInfo()

    handoff = gpu_hf.RHF(mol).density_fit(auxbasis=resolved_auxbasis)
    handoff.mo_coeff = mo_coeff
    handoff.mo_energy = mo_energy
    handoff.mo_occ = mo_occ
    handoff.e_tot = e_tot
    handoff.converged = True

    diagnostics = {
        "backend": "full_cuest_handoff",
        "math_mode": math_mode,
        "workspace_gb": float(workspace_gb),
        "exchange_slice_count": int(exchange_slice_count),
        "exchange_modulus_count": int(exchange_modulus_count),
        "scf_seconds": scf_seconds,
        "energy_eh": float(energy),
        "released_device_gib": (free_after_reset - free_before_reset) / 2**30,
        "device_used_after_reset_gib": (total_memory - free_after_reset) / 2**30,
        "auxbasis": resolved_auxbasis,
    }
    return handoff, diagnostics


class DCGradientGPUCuESTSCF(_AdjointDCGradientGPU):
    """Adjoint DC gradient with last-use release of the CPHF DF storage."""

    def __init__(self, *args, release_cphf_df=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.release_cphf_df = bool(release_cphf_df)
        self.memory_diagnostics = {}

    def _nonvariational_gpu(
        self, mol, hf_gpu, hf_resp, C, e, occ, so, sv, Co, Cv, Co2, Z,
        S1_oo, z_se, S1_mo_o, atom_slices, log, **kwargs,
    ):
        if not hasattr(hf_gpu, "with_df") or hf_gpu.with_df is None:
            return super()._nonvariational_gpu(
                mol, hf_gpu, hf_resp, C, e, occ, so, sv, Co, Cv, Co2, Z,
                S1_oo, z_se, S1_mo_o, atom_slices, log, **kwargs,
            )

        natm = mol.natm

        # The response CDERI is needed here for the last time.
        log.debug("  adjoint Z:Ax(S1)_vo before DF release...")
        t_ax = time.time()
        if self.response_backend == "mo_direct":
            z_ax = _adjoint_z_ax_mo_direct(
                hf_gpu.with_df, Z, Co, Cv, S1_oo,
                direct_scf_tol=hf_gpu.direct_scf_tol,
            )
        else:
            z_ax = _adjoint_z_ax(hf_resp, Z, Co, Cv, S1_oo)
        log.debug("  adjoint Z:Ax(S1)_vo: %.2f s", time.time() - t_ax)

        released_gib = 0.0
        if self.release_cphf_df:
            cp.cuda.runtime.deviceSynchronize()
            free_before, _ = cp.cuda.runtime.memGetInfo()
            # reset(None) keeps auxmol/auxbasis, which the derivative-energy
            # kernel below can reuse, while dropping CDERI and metric factors.
            hf_gpu.with_df.reset()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()
            cp.cuda.runtime.deviceSynchronize()
            free_after, _ = cp.cuda.runtime.memGetInfo()
            released_gib = (free_after - free_before) / 2**30
            log.info("  released CPHF DF storage: %.2f GiB", released_gib)

        log.debug("  factorized direct 4*Z:F1_vo...")
        t_f1 = time.time()
        direct_z_f1 = (
            _direct_contracted_z_f1_factorized
            if self.factorized_z_f1 else _direct_contracted_z_f1
        )
        g_zf1 = direct_z_f1(
            mol, hf_gpu, C, occ, Co, Cv, Z,
            memory_gb=self.df_deriv_memory_gb,
        )
        log.debug("  factorized direct 4*Z:F1_vo: %.2f s", time.time() - t_f1)

        g_nv = g_zf1.reshape(-1) - 4.0 * z_se - 2.0 * z_ax
        self.memory_diagnostics = {
            "released_cphf_df_gib": released_gib,
            "release_cphf_df": self.release_cphf_df,
        }
        del g_zf1, z_se, z_ax
        return g_nv.reshape(natm, 3)


def kernel(mf, xc="PBE", verbose=None, **kwargs):
    """Evaluate the memory-ordered adjoint gradient for a handoff mean field."""

    return DCGradientGPUCuESTSCF(mf, xc=xc, **kwargs).kernel(verbose=verbose)


def _build_parser():
    parser = _build_adjoint_parser()
    parser.description = __doc__
    parser.add_argument(
        "--scf-backend",
        choices=("cuest", "gpu4pyscf"),
        default="cuest",
    )
    parser.add_argument("--cuest-workspace-gb", type=float, default=2.0)
    parser.add_argument(
        "--cuest-math-mode",
        choices=("default", "native_fp64"),
        default="default",
    )
    parser.add_argument("--cuest-exchange-slice-count", type=int, default=6)
    parser.add_argument("--cuest-exchange-modulus-count", type=int, default=9)
    parser.add_argument(
        "--keep-cphf-df",
        action="store_true",
        help="Do not release CPHF CDERI before direct Z:F1 (debugging only)",
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
        mf = gpu_hf.RHF(mol).density_fit(auxbasis=requested_auxbasis)
        mf.conv_tol = args.scf_tol
        mf.max_cycle = args.scf_max_cycle
        t0 = time.perf_counter()
        energy = mf.kernel()
        if not mf.converged:
            raise RuntimeError(
                f"GPU4PySCF DF-RHF did not converge in {args.scf_max_cycle} cycles"
            )
        scf_info = {
            "backend": "gpu4pyscf",
            "scf_seconds": time.perf_counter() - t0,
            "energy_eh": float(energy),
        }

    method = DCGradientGPUCuESTSCF(
        mf,
        xc=args.xc,
        cphf_tol=args.cphf_tol,
        cphf_max_cycle=args.cphf_max_cycle,
        atom_grid=tuple(args.atom_grid),
        grid_response=not args.no_grid_response,
        release_cphf_df=not args.keep_cphf_df,
        response_backend=args.response_backend,
        share_df=not args.separate_df,
        factorized_z_f1=not args.dense_z_f1,
        df_deriv_memory_gb=args.df_deriv_memory_gb,
    )
    gradient = method.kernel(verbose=args.verbose)

    print("\nSCF diagnostics:")
    for key, value in scf_info.items():
        if key != "auxbasis":
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
                f"scf_backend={args.scf_backend}"
            ),
        )
        print(f"Gradient written to: {output}")
    return 0


__all__ = [
    "DCGradientGPUCuESTSCF",
    "build_cuest_rhf_handoff",
    "kernel",
]


if __name__ == "__main__":
    raise SystemExit(main())
