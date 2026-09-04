'''MO-direct density-fitted GPU UHF response actions.'''

import cupy as cp
from gpu4pyscf.df.hessian import rhf as df_hessian_rhf
from gpu4pyscf.lib.cupy_helper import contract, tag_array


def _pad_spin_pair(alpha, beta):
    '''Stack unequal alpha/beta MO dimensions for GPU4PySCF's dense API.

    Args:
        alpha, beta : cupy.ndarray
            Spin tensors with identical leading dimensions and possibly
            different final MO dimensions.

    Returns:
        cupy.ndarray
            Spin-first tensor padded to the larger final dimension.

    Notes:
        UHF alpha and beta spaces may have different occupied or requested MO
        dimensions, while the private DF-Hessian kernel accepts one
        rectangular spin stack. Padding is confined to this provider adapter
        and removed from its return values, so the public Z-vector blocks
        retain their exact spin shapes. A separate helper keeps this
        representation workaround out of the spin-response algebra in
        ``gen_vind``.
    '''
    alpha = cp.asarray(alpha)
    beta = cp.asarray(beta)
    width = max(alpha.shape[-1], beta.shape[-1])
    out = cp.zeros(
        (2,) + alpha.shape[:-1] + (width,),
        dtype=cp.result_type(alpha.dtype, beta.dtype))
    out[0, ..., :alpha.shape[-1]] = alpha
    out[1, ..., :beta.shape[-1]] = beta
    return out


def _get_jk_mo(dfobj, dm, left_coeff, occ_coeff):
    '''Return requested alpha/beta MO blocks of the UHF DF response.

    Args:
        dfobj : gpu4pyscf.df.df.DF
            Shared source-UHF density-fitting provider.
        dm : tagged cupy.ndarray
            Concatenated alpha/beta response densities with separate ``mo1``
            and occupied-coefficient metadata.
        left_coeff : tuple of two cupy.ndarray
            Alpha/beta coefficients defining the requested left MO indices.
        occ_coeff : tuple of two cupy.ndarray
            Alpha/beta occupied coefficients defining the right MO indices.

    Returns:
        tuple of two cupy.ndarray
            Exact-shape alpha/beta response blocks after removing padding.

    Notes:
        ``gen_vind`` and ``get_z_response_oo`` need VO and OO blocks,
        respectively. This adapter owns the common padding, private-kernel
        call and unpadding. It is intentionally local to DF because a
        conventional UHF response has no padded three-center MO-output
        contract.
    '''
    left_a, left_b = map(cp.asarray, left_coeff)
    occ_a, occ_b = map(cp.asarray, occ_coeff)
    mo1_a, mo1_b = dm.mo1
    tagged_dm = tag_array(
        dm, mo1=_pad_spin_pair(mo1_a, mo1_b),
        occ_coeff=_pad_spin_pair(occ_a, occ_b),
        symmetrize=getattr(dm, 'symmetrize', 1))
    padded_left = _pad_spin_pair(left_a, left_b)
    nocca, noccb = occ_a.shape[1], occ_b.shape[1]
    mo_occ = cp.zeros((2, max(nocca, noccb)))
    mo_occ[0, :nocca] = 1
    mo_occ[1, :noccb] = 1

    vj, vk = df_hessian_rhf._get_jk(
        dfobj, tagged_dm, padded_left, mo_occ,
        hermi=1, with_j=True, with_k=True)
    value = vj - vk
    width = padded_left.shape[-1]
    alpha_stop = width * nocca

    alpha = value[:, :alpha_stop].reshape(-1, width, nocca)
    beta = value[:, alpha_stop:].reshape(-1, width, noccb)
    alpha = alpha[:, :left_a.shape[1]]
    beta = beta[:, :left_b.shape[1]]
    return alpha, beta


def gen_vind(mf_hf, Co, Cv):
    '''Build a spin-coupled UCPHF action returning only DF VO blocks.

    Args:
        mf_hf : gpu4pyscf.df.df_jk._DFHF
            Density-fitted source-UHF object.
        Co, Cv : tuple of two cupy.ndarray
            Alpha/beta occupied and virtual source coefficients.

    Returns:
        callable
            Packed spin-coupled two-electron Hessian action in VO space.

    Notes:
        The conventional GPU UHF ``gen_vind`` builds two AO response
        potentials and projects them afterward. This DF version passes the
        same coupled spin densities to the three-center Hessian kernel and
        receives only the two VO blocks. The functions coexist so
        ``response_backend='ao'`` remains an independent oracle while
        ``'mo_direct'`` avoids AO response outputs. The common UCPHF solver and
        physical-residual diagnostic are shared.
    '''
    Coa, Cob = Co
    Cva, Cvb = Cv
    nova = Cva.shape[1] * Coa.shape[1]
    novb = Cvb.shape[1] * Cob.shape[1]

    def vind(x):
        shape = cp.asarray(x).shape
        x = cp.asarray(x).reshape(-1, nova + novb)
        nset = x.shape[0]
        xa = x[:, :nova].reshape(nset, Cva.shape[1], Coa.shape[1])
        xb = x[:, nova:].reshape(nset, Cvb.shape[1], Cob.shape[1])
        mo1a = contract('pa,kai->kpi', Cva, xa)
        mo1b = contract('pa,kai->kpi', Cvb, xb)
        dma = contract('kpi,qi->kpq', mo1a, Coa)
        dmb = contract('kpi,qi->kpq', mo1b, Cob)
        dma = dma + dma.transpose(0, 2, 1).conj()
        dmb = dmb + dmb.transpose(0, 2, 1).conj()
        dm = tag_array(
            cp.concatenate((dma, dmb), axis=0),
            mo1=(mo1a, mo1b), occ_coeff=(Coa, Cob))
        va, vb = _get_jk_mo(
            mf_hf.with_df, dm, (Cva, Cvb), (Coa, Cob))
        out = cp.concatenate(
            (va.reshape(nset, nova), vb.reshape(nset, novb)), axis=1)
        return out.reshape(shape)

    return vind


def get_z_response_oo(mf_hf, Co, Cv, zvec):
    '''Return both spin OO adjoint blocks without full AO potentials.

    Args:
        mf_hf : gpu4pyscf.df.df_jk._DFHF
            Density-fitted source-UHF object.
        Co, Cv : tuple of two cupy.ndarray
            Alpha/beta occupied and virtual source coefficients.
        zvec : tuple of two cupy.ndarray
            Positive-RHS alpha/beta Z-vectors.

    Returns:
        tuple of two cupy.ndarray
            Alpha/beta ``C_o^dagger G[D_Z] C_o`` blocks.

    Notes:
        This DF provider is paired with the common provider-independent
        ``grad_ovlp_response`` contraction. The AO implementation is retained
        for conventional sources and parity tests; this implementation exists
        because DF can perform the final OO projection inside its three-center
        loop and avoid materializing two ``(nao,nao)`` potentials.
    '''
    Coa, Cob = Co
    Cva, Cvb = Cv
    za, zb = zvec
    zleft_a = Cva @ za
    zleft_b = Cvb @ zb
    dma = zleft_a @ Coa.conj().T
    dmb = zleft_b @ Cob.conj().T
    dm = tag_array(
        cp.concatenate(
            ((dma + dma.conj().T)[None],
             (dmb + dmb.conj().T)[None]), axis=0),
        mo1=(zleft_a[None], zleft_b[None]),
        occ_coeff=(Coa, Cob))
    va, vb = _get_jk_mo(
        mf_hf.with_df, dm, (Coa, Cob), (Coa, Cob))
    return va[0], vb[0]
