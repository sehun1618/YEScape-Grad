"""Frozen-HF target-DFA derivative terms evaluated with CuPy kernels."""

from __future__ import annotations

from time import perf_counter

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.dft import multigrid_v2
from gpu4pyscf.pbc.dft import numint as gpu_numint
from gpu4pyscf.pbc.grad.krks import (
    _d1_dot_,
    _gga_grad_sum_,
    _tau_grad_dot_,
)
from ..dft.target_numint import evaluate_target_hcore, evaluate_target_veff
from pyscf_pbc_dcdft.dft.target_integrator import (
    MULTIGRID_J_XC,
    MULTIGRID_XC,
    UNIFORM,
)


def target_fock_kpts(context, *, return_reuse=False):
    """Return target-DFA Fock matrices and optionally the reuse decision."""

    workspace = context.target_evaluation
    if workspace is not None:
        fock = _with_kpoint_axis(workspace.target_fock)
        return (fock, True) if return_reuse else fock

    hcore = cp.asarray(evaluate_target_hcore(context.target_ks))
    evaluation = evaluate_target_veff(
        context.target_ks,
        context.source_mf.make_rdm1(),
        with_df=context.dc_mf.with_df,
        kpts=context.kpts,
        target_integrator=context.dc_mf.target_integrator,
        target_multigrid_mesh=context.dc_mf.target_multigrid_mesh,
    )
    veff = cp.asarray(evaluation.veff)
    fock = _with_kpoint_axis(hcore) + _with_kpoint_axis(veff)
    return (fock, False) if return_reuse else fock


def _with_kpoint_axis(value):
    array = cp.asarray(value)
    return array[None] if array.ndim == 2 else array


def target_energy_weighted_density(context, target_fock):
    """Build ``2 C_occ F_target,oo C_occ^H`` at each k-point."""

    output = []
    for coeff, occupation, fock in zip(
        context.mo_coeff, context.mo_occ, target_fock, strict=True
    ):
        occupied = occupation > 0
        occupied_coeff = coeff[:, occupied]
        fock_oo = occupied_coeff.conj().T @ fock @ occupied_coeff
        output.append(
            2.0 * occupied_coeff @ fock_oo @ occupied_coeff.conj().T
        )
    return cp.stack(output)


def target_xc_row_derivative(context):
    """Return semilocal XC AO-row derivatives as ``(x,k,p,q)`` on device.

    This is the uncontracted counterpart of
    :func:`gpu4pyscf.pbc.grad.krks.get_vxc`. Keeping the AO matrix is necessary
    for the DC-DFT explicit decomposition and CPU/GPU equation-level parity.
    """

    target = context.target_ks
    ni = gpu_numint.KNumInt()
    if ni.libxc.is_hybrid_xc(target.xc):
        raise NotImplementedError(
            "GPU target derivative supports pure semilocal LDA/GGA/MGGA only"
        )
    grids = target.grids
    if grids.coords is None:
        grids.build()
    xctype = ni._xc_type(target.xc)
    if xctype not in {"LDA", "GGA", "MGGA"}:
        raise NotImplementedError(
            f"GPU target XC derivative does not support {xctype}"
        )
    nao = context.cell.nao
    vmat = cp.zeros(
        (context.nkpts, 3, nao, nao), dtype=context.density.dtype
    )
    ao_deriv = 1 if xctype == "LDA" else 2
    for ao_ks, weight, _ in ni.block_loop(
        context.cell,
        grids,
        ao_deriv,
        context.kpts,
        sort_grids=True,
    ):
        if xctype == "LDA":
            rho = ni.eval_rho(
                context.cell,
                ao_ks[:, 0],
                context.density,
                xctype=xctype,
                hermi=1,
            )
            vxc = ni.eval_xc_eff(
                target.xc, rho, deriv=1, xctype=xctype
            )[1]
            weighted = weight * vxc[0]
            aow = cp.einsum("kpi,p->kpi", ao_ks[:, 0], weighted)
            for k_index in range(context.nkpts):
                vmat[k_index] += _d1_dot_(
                    ao_ks[k_index, 1:4], aow[k_index]
                )
        elif xctype == "GGA":
            rho = ni.eval_rho(
                context.cell,
                ao_ks[:, :4],
                context.density,
                xctype=xctype,
                hermi=1,
            )
            vxc = ni.eval_xc_eff(
                target.xc, rho, deriv=1, xctype=xctype
            )[1]
            weighted = weight * vxc
            weighted[0] *= 0.5
            for k_index in range(context.nkpts):
                vmat[k_index] += _gga_grad_sum_(
                    ao_ks[k_index], weighted
                )
        else:
            rho = ni.eval_rho(
                context.cell,
                ao_ks[:, :4],
                context.density,
                xctype=xctype,
                hermi=1,
            )
            vxc = ni.eval_xc_eff(
                target.xc, rho, deriv=1, xctype=xctype
            )[1]
            weighted = weight * vxc
            weighted[0] *= 0.5
            # PySCF defines tau with a factor 1/2.  This weight is the
            # functional derivative with respect to the unscaled gradient
            # product, so the same factor must be applied once here.
            weighted[4] *= 0.5
            for k_index in range(context.nkpts):
                vmat[k_index] += _gga_grad_sum_(
                    ao_ks[k_index], weighted[:4]
                )
                vmat[k_index] += _tau_grad_dot_(
                    ao_ks[k_index], weighted[4]
                )
    # GPU4PySCF's low-level accumulation uses electronic-coordinate AO
    # derivatives. Nuclear-center derivatives carry the opposite sign.
    return -vmat.transpose(1, 0, 2, 3)


def contract_xc_row_derivative(context, row_derivative):
    """Contract target-XC AO-row matrices into ``(Natom,3)``."""

    output = cp.zeros((context.cell.natm, 3), dtype=cp.float64)
    ao_slices = np.asarray(context.cell.aoslice_by_atom(), dtype=int)
    for atom_id in range(context.cell.natm):
        p0, p1 = ao_slices[atom_id, 2:]
        output[atom_id] = (
            2.0
            * cp.einsum(
                "xkij,kji->x",
                row_derivative[:, :, p0:p1, :],
                context.density[:, :, p0:p1],
            ).real
            / context.nkpts
        )
    return output


def target_xc_atom_gradient(context):
    """Return the target numerical-integration gradient on device.

    ``multigrid_xc`` returns XC only.  ``multigrid_j_xc`` returns the
    combined target Coulomb plus XC derivative; callers must therefore omit
    the selected-DF target-J derivative in the latter mode.
    """

    workspace = context.target_evaluation
    mode = (
        UNIFORM if workspace is None else workspace.target_integrator
    )
    if mode in {MULTIGRID_XC, MULTIGRID_J_XC}:
        if workspace is None or workspace.target_numint is None:
            raise RuntimeError(
                "multigrid target gradient requires the energy workspace"
            )
        cp.cuda.Stream.null.synchronize()
        evaluation_start = perf_counter()
        value = multigrid_v2.get_veff_ip1(
            workspace.target_numint,
            context.target_ks.xc,
            context.density,
            hermi=1,
            kpts=context.kpts,
            with_j=(mode == MULTIGRID_J_XC),
            with_pseudo_vloc_orbital_derivative=False,
            verbose=context.dc_mf.verbose,
        )
        cp.cuda.Stream.null.synchronize()
        workspace.ownership["gradient_evaluation_wall_time_s"] = float(
            perf_counter() - evaluation_start
        )
        workspace.ownership["gradient_reused_energy_multigrid_tasks"] = True
        # Upstream currently returns the per-cell derivative multiplied by Nk.
        return cp.asarray(value) / context.nkpts
    if mode != UNIFORM:
        raise ValueError(f"unsupported target integrator {mode!r}")
    return contract_xc_row_derivative(
        context, target_xc_row_derivative(context)
    )


def release_completed_target_multigrid(context):
    """Release multigrid task caches after the target derivative is complete.

    The fixed-density energy workspace keeps ``MultiGridNumInt`` tasks so the
    analytic gradient can reuse them.  Once ``get_veff_ip1`` has returned its
    small atom-by-Cartesian result, those grid and Gaussian-pair caches are no
    longer needed by the remaining GDF/RSDF response.  Dropping them here
    prevents two individually large workspaces from overlapping in memory.
    """

    workspace = context.target_evaluation
    if (
        workspace is None
        or workspace.target_integrator == UNIFORM
        or workspace.target_numint is None
    ):
        return None

    cp.cuda.Stream.null.synchronize()
    free_before, total_bytes = cp.cuda.Device().mem_info
    pool = cp.get_default_memory_pool()
    pool_used_before = int(pool.used_bytes())
    pool_reserved_before = int(pool.total_bytes())

    target_numint = workspace.target_numint
    task_count = len(getattr(target_numint, "tasks", ()) or ())
    target_numint.reset()

    # CuPy retains unused device blocks by default.  At this ownership
    # boundary retaining them only raises the following RSDF response peak.
    cp.cuda.Stream.null.synchronize()
    pool.free_all_blocks()
    pinned_pool = cp.get_default_pinned_memory_pool()
    if hasattr(pinned_pool, "free_all_blocks"):
        pinned_pool.free_all_blocks()
    try:
        cp.fft.config.get_plan_cache().clear()
    except (AttributeError, RuntimeError):
        # Some CuPy builds do not expose a configurable FFT plan cache.
        pass
    cp.cuda.Stream.null.synchronize()

    free_after, _ = cp.cuda.Device().mem_info
    metadata = {
        "target_multigrid_cache_released_after_gradient": True,
        "released_multigrid_task_count": int(task_count),
        "release_total_device_bytes": int(total_bytes),
        "release_free_device_bytes_before": int(free_before),
        "release_free_device_bytes_after": int(free_after),
        "release_free_device_bytes_delta": int(free_after - free_before),
        "release_pool_used_bytes_before": pool_used_before,
        "release_pool_used_bytes_after": int(pool.used_bytes()),
        "release_pool_reserved_bytes_before": pool_reserved_before,
        "release_pool_reserved_bytes_after": int(pool.total_bytes()),
    }
    workspace.ownership.update(metadata)
    return metadata


def contract_coulomb_full_derivative(context, coulomb_derivative):
    """Contract ``1/2 Tr[D dJ]`` from the full matrix derivative."""

    return 0.5 * cp.einsum(
        "akxij,kji->ax", coulomb_derivative, context.density
    ).real / context.nkpts
