"""Frozen-HF target-DFA derivative terms evaluated with CuPy kernels."""

from __future__ import annotations

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.dft import numint as gpu_numint
from gpu4pyscf.pbc.grad.krks import (
    _d1_dot_,
    _gga_grad_sum_,
    _tau_grad_dot_,
)
from ..dft.target_numint import evaluate_target_hcore, evaluate_target_veff


def target_fock_kpts(context, *, return_reuse=False):
    """Return target-DFA Fock matrices and optionally the reuse decision."""

    workspace = context.target_evaluation
    if workspace is not None:
        fock = _with_kpoint_axis(workspace.target_fock)
        return (fock, True) if return_reuse else fock

    hcore = cp.asarray(evaluate_target_hcore(context.target_ks))
    veff = cp.asarray(
        evaluate_target_veff(
            context.target_ks,
            context.source_mf.make_rdm1(),
            with_df=context.dc_mf.with_df,
            kpts=context.kpts,
        )
    )
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
    """Return the target XC nuclear gradient as ``(Natom,3)`` on device."""

    return contract_xc_row_derivative(
        context, target_xc_row_derivative(context)
    )


def contract_coulomb_full_derivative(context, coulomb_derivative):
    """Contract ``1/2 Tr[D dJ]`` from the full matrix derivative."""

    return 0.5 * cp.einsum(
        "akxij,kji->ax", coulomb_derivative, context.density
    ).real / context.nkpts
