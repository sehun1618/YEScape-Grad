"""Compatibility adapters for GPU4PySCF private DF response kernels."""

from __future__ import annotations

import cupy as cp

from gpu4pyscf.lib.cupy_helper import tag_array


try:
    from gpu4pyscf.df.hessian import jk as _legacy_df_hessian_jk
except ImportError:
    _legacy_df_hessian_jk = None

if _legacy_df_hessian_jk is None:
    from gpu4pyscf.df.hessian import rhf as _df_hessian_rhf


def _pad_spin_pair(alpha, beta):
    """Stack alpha/beta tensors after padding only their final dimension."""

    alpha = cp.asarray(alpha)
    beta = cp.asarray(beta)
    width = max(alpha.shape[-1], beta.shape[-1])
    shape = (2,) + alpha.shape[:-1] + (width,)
    output = cp.zeros(shape, dtype=cp.result_type(alpha.dtype, beta.dtype))
    output[0, ..., :alpha.shape[-1]] = alpha
    output[1, ..., :beta.shape[-1]] = beta
    return output


def df_hessian_get_jk_rhf(
    dfobj, dms, left_coeff, occ_coeff, *, direct_scf_tol=1.0e-14,
):
    """Return RHF DF J/K response in the requested MO block."""

    if _legacy_df_hessian_jk is not None:
        return _legacy_df_hessian_jk.get_jk(
            dfobj, dms, left_coeff, occ_coeff, hermi=1,
            with_j=True, with_k=True, direct_scf_tol=direct_scf_tol,
        )

    # GPU4PySCF 1.8 moved this kernel to df.hessian.rhf._get_jk. The
    # occupation vector is unused by its restricted output formatter; the
    # occupied coefficients remain attached to dms.
    mo_occ = cp.full(left_coeff.shape[1], 2.0)
    return _df_hessian_rhf._get_jk(
        dfobj, dms, left_coeff, mo_occ, hermi=1,
        with_j=True, with_k=True,
    )


def df_hessian_get_jk_uhf(
    dfobj, dms, left_coeff, occ_coeff, *, direct_scf_tol=1.0e-14,
):
    """Return UHF DF J/K response while hiding 1.8 occupied-space padding."""

    if _legacy_df_hessian_jk is not None:
        return _legacy_df_hessian_jk.get_jk(
            dfobj, dms, left_coeff, occ_coeff, hermi=1,
            with_j=True, with_k=True, direct_scf_tol=direct_scf_tol,
        )

    left_a, left_b = map(cp.asarray, left_coeff)
    occ_a, occ_b = map(cp.asarray, occ_coeff)
    mo1_a, mo1_b = dms.mo1
    tagged_dms = tag_array(
        dms, mo1=_pad_spin_pair(mo1_a, mo1_b),
        occ_coeff=_pad_spin_pair(occ_a, occ_b),
        symmetrize=getattr(dms, "symmetrize", 1),
    )
    padded_left = _pad_spin_pair(left_a, left_b)
    nocc_a, nocc_b = occ_a.shape[1], occ_b.shape[1]
    mo_occ = cp.zeros((2, max(nocc_a, nocc_b)))
    mo_occ[0, :nocc_a] = 1.0
    mo_occ[1, :nocc_b] = 1.0

    vj, vk = _df_hessian_rhf._get_jk(
        dfobj, tagged_dms, padded_left, mo_occ, hermi=1,
        with_j=True, with_k=True,
    )
    padded_width = padded_left.shape[-1]
    alpha_stop = padded_width*nocc_a

    def remove_padding(value):
        alpha = value[:, :alpha_stop].reshape(-1, padded_width, nocc_a)
        beta = value[:, alpha_stop:].reshape(-1, padded_width, nocc_b)
        alpha = alpha[:, :left_a.shape[1]].reshape(len(value), -1)
        beta = beta[:, :left_b.shape[1]].reshape(len(value), -1)
        return cp.concatenate((alpha, beta), axis=1)

    return remove_padding(vj), remove_padding(vk)
