'''MO-direct density-fitted GPU RHF response actions.'''

import cupy as cp
from gpu4pyscf.df.hessian import rhf as df_hessian_rhf
from gpu4pyscf.lib.cupy_helper import contract, tag_array


def _get_jk_mo(dfobj, dm, left_coeff, occ_coeff):
    '''Return a requested RHF MO block of ``J[dm]-K[dm]/2``.

    This private adapter centralizes GPU4PySCF's DF-Hessian return layout for
    both public operations below. ``gen_vind`` requests a VO block, whereas
    ``get_z_response_oo`` requests an OO block. They share this helper because
    only the left projection changes; neither operation needs an AO response
    potential.

    Args:
        dfobj : gpu4pyscf.df.df.DF
            Source-RHF three-center density-fitting provider.
        dm : tagged cupy.ndarray of shape (nset, nao, nao)
            Hermitian response densities carrying GPU4PySCF ``mo1`` metadata.
        left_coeff : cupy.ndarray of shape (nao, nleft)
            MO coefficients for the requested left index.
        occ_coeff : cupy.ndarray of shape (nao, nocc)
            Occupied coefficients defining the right index.

    Returns:
        cupy.ndarray of shape (nset, nleft, nocc)
            Requested MO response block.
    '''
    mo_occ = cp.full(left_coeff.shape[1], 2.0)
    vj, vk = df_hessian_rhf._get_jk(
        dfobj, dm, left_coeff, mo_occ,
        hermi=1, with_j=True, with_k=True)
    nset = dm.reshape(-1, dm.shape[-1], dm.shape[-1]).shape[0]
    return (vj - .5 * vk).reshape(
        nset, left_coeff.shape[1], occ_coeff.shape[1])


def gen_vind(mf_hf, Co, Cv):
    '''Build an RHF CPHF action returning only the DF VO response block.

    Args:
        mf_hf : gpu4pyscf.df.df_jk._DFHF
            Density-fitted source RHF object.
        Co : cupy.ndarray of shape (nao, nocc)
            Occupied source coefficients.
        Cv : cupy.ndarray of shape (nao, nvir)
            Virtual source coefficients.

    Returns:
        callable
            Batched two-electron RHF Hessian action in VO space.

    Notes:
        GPU4PySCF's DF Hessian kernel performs the final AO-to-MO projection
        inside the three-center loop.  No ``(nrhs,nao,nao)`` response
        potential is returned to the CPHF solver.

        The conventional function ``dcforge.gpu.grad.rhf_zvec.gen_vind``
        remains necessary as an AO reference and for non-DF sources. This DF
        function is separate because it depends on a private three-center
        kernel and a tagged-density contract that the conventional provider
        does not have. The common Z solver accepts either callable, so the
        mathematical equation and residual check are not duplicated.
    '''
    nvir, nocc = Cv.shape[1], Co.shape[1]
    Co2 = Co * 2

    def vind(x):
        shape = cp.asarray(x).shape
        x = cp.asarray(x).reshape(-1, nvir, nocc)
        mo1 = contract('pa,kai->kpi', Cv, x)
        dm = contract('kpi,qi->kpq', mo1, Co2)
        dm = dm + dm.transpose(0, 2, 1).conj()
        dm = tag_array(dm, mo1=mo1, occ_coeff=Co)
        return _get_jk_mo(mf_hf.with_df, dm, Cv, Co).reshape(shape)

    return vind


def get_z_response_oo(mf_hf, Co, Cv, zvec):
    '''Return ``C_o^dagger G^HF[D_Z]C_o`` without an AO response output.

    Args:
        mf_hf : gpu4pyscf.df.df_jk._DFHF
            Density-fitted source-RHF object.
        Co : cupy.ndarray of shape (nao, nocc)
            Occupied source coefficients.
        Cv : cupy.ndarray of shape (nao, nvir)
            Virtual source coefficients.
        zvec : cupy.ndarray of shape (nvir, nocc)
            Positive-RHS Z-vector.

    Returns:
        cupy.ndarray of shape (nocc, nocc)
            Source-HF response to ``D_Z`` projected directly to OO space.

    Notes:
        This is the DF counterpart of
        ``dcforge.gpu.grad.rhf_nv.get_z_response_oo``. The conventional
        function forms ``G[D_Z]`` in AO space and then projects it; this
        function asks the DF-Hessian kernel for the OO block directly. Both
        remain available because the AO route is the independent numerical
        reference and the MO-direct route avoids a full ``(nao,nao)`` response
        output in production.
    '''
    zleft = Cv @ zvec
    dm = zleft @ Co.conj().T
    dm = dm + dm.conj().T
    dm = tag_array(
        dm[None], mo1=(.5 * zleft)[None], occ_coeff=Co)
    return _get_jk_mo(mf_hf.with_df, dm, Co, Co)[0]
