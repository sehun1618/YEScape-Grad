# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''UHF source Fock nuclear derivatives; SCF remains upstream.'''

import numpy as np
from pyscf import lib


def grad_fock_make_h1(mf_grad, Co, Cv, zvec, mo_coeff, mo_occ,
                      atmlst):
    '''Return the source-Fock term through PySCF ``Hessian.make_h1``.

    This is the unrestricted numerical reference for the direct contracted
    route. Conventional UHF retains the alpha and beta ``make_h1`` tensors.
    DF-UHF instead consumes PySCF's ``df.hessian.uhf._gen_jk`` atom iterator
    and immediately projects each yielded spin Fock derivative to its VO
    block. This removes the complete atom-indexed Fock tensor from the
    YEScape-Grad layer, while PySCF may still retain internal J/K buffers.

    The returned matrices are the explicit source-operator derivatives at
    the unperturbed spin densities.  They do not include the induced UHF
    density response ``G^UHF[P^lambda]`` or the separately assembled overlap
    response, and therefore are not the total self-consistent ``dF/dlambda``.

    Args:
        mf_grad : yescape_grad.cpu.grad.unsc.Gradients
            Active conventional or density-fitted gradient object.
        Co, Cv : tuple of two ndarray
            Alpha/beta occupied and virtual source MO coefficients.
        zvec : tuple of two ndarray
            Positive-RHS alpha/beta Z-vectors.
        mo_coeff : tuple of two ndarray
            Complete alpha/beta source MO coefficients.
        mo_occ : tuple of two ndarray
            Exact unrestricted source occupations.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            The ``-2 sum_σ Z^σ F^UHF,σ,lambda_vo`` contribution in Ha/Bohr.
    '''
    mf_hess = mf_grad.base.mf_source.Hessian()
    if hasattr(mf_hess, 'auxbasis_response'):
        mf_hess.auxbasis_response = 2 * int(mf_grad.auxbasis_response)
    if hasattr(mf_grad.base.mf_source, 'with_df'):
        from pyscf.df.hessian import uhf as df_hessian_uhf

        de = np.zeros((len(atmlst), 3))
        peak_bytes = 0
        for k, (ia, h1, vj1, vk1) in enumerate(df_hessian_uhf._gen_jk(
            mf_hess, mo_coeff, mo_occ, atmlst=atmlst, with_k=True,
        )):
            for Co_s, Cv_s, z_s, vk1_s in zip(Co, Cv, zvec, vk1):
                h1ao = h1 + vj1 - vk1_s
                f1vo = lib.einsum(
                    'xuv,ua,vi->xai', h1ao, Cv_s.conj(), Co_s)
                de[k] -= 2 * np.real(
                    lib.einsum('ai,xai->x', z_s.conj(), f1vo))
                peak_bytes = max(peak_bytes, h1ao.nbytes + f1vo.nbytes)
        mf_grad.fock_deriv_intermediate_bytes = peak_bytes
        mf_grad.fock_deriv_storage = 'atom_stream'
        return de

    h1ao = mf_hess.make_h1(mo_coeff, mo_occ, atmlst=atmlst)
    mf_grad.fock_deriv_intermediate_bytes = sum(
        h1ao_s[ia].nbytes for h1ao_s in h1ao for ia in atmlst
    )
    mf_grad.fock_deriv_storage = 'materialized'

    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        for Co_s, Cv_s, z_s, h1_s in zip(Co, Cv, zvec, h1ao):
            f1vo = lib.einsum(
                'xuv,ua,vi->xai', h1_s[ia], Cv_s.conj(), Co_s)
            de[k] -= 2 * np.real(
                lib.einsum('ai,xai->x', z_s.conj(), f1vo))
    return de


def grad_fock_direct(mf_grad, Co, Cv, zvec, dm0, atmlst, with_k=True):
    r'''Contract the spin-resolved source-Fock term without ``make_h1``.

    ``with_k=False`` selects the pure-UKS hcore/J term with two total densities
    only. Spin-specific exchange densities and their auxiliary response are
    not passed to the integral provider.

    Define one Hermitian transition density for each spin,

    .. math::

        D_Z^{\sigma} = C_v^{\sigma} Z^{\sigma} (C_o^{\sigma})^\dagger
              + C_o^{\sigma} (Z^{\sigma})^\dagger (C_v^{\sigma})^\dagger.

    With ``P = P^alpha + P^beta`` and
    ``D_Z = D_Z^alpha + D_Z^beta``, the required contraction is

    .. math::

        -\operatorname{Tr}[D_Z h^\lambda]
        -\operatorname{Tr}[D_Z J^\lambda[P]]
        +\sum_{\sigma}\operatorname{Tr}[D_Z^{\sigma} K^\lambda[P^{\sigma}]].

    A single gradient J/K call evaluates the Coulomb cross pair
    ``(P,D_Z)`` and the two same-spin exchange cross pairs.  The primary-AO
    row contributions and, for DF, both auxiliary-center polarization orders
    are retained.  No ``(2,N_atom,3,nao,nao)`` first-order Fock tensor is
    formed.

    Args:
        mf_grad : yescape_grad.cpu.grad.unsc.Gradients
            Active conventional or density-fitted gradient object.
        Co, Cv : tuple of two ndarray
            Alpha/beta occupied and virtual source MO coefficients.
        zvec : tuple of two ndarray
            Positive-RHS alpha/beta Z-vectors.
        dm0 : ndarray of shape (2, nao, nao)
            Spin-resolved source-UHF AO densities.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Directly contracted source-Fock term in Ha/Bohr.
    '''
    mol = mf_grad.mol
    dm0 = np.asarray(dm0)
    dz = []
    for Co_s, Cv_s, z_s in zip(Co, Cv, zvec):
        dz_s = Cv_s @ z_s @ Co_s.conj().T
        dz.append(dz_s + dz_s.conj().T)
    dz = np.asarray(dz)

    dm = dm0.sum(axis=0)
    dmz = dz.sum(axis=0)
    if with_k:
        dms = np.stack((dm, dmz, dm0[0], dz[0], dm0[1], dz[1]))
        vj, vk = mf_grad.get_jk(mol, dms, hermi=1)
    else:
        # Pure UKS needs only total-density Coulomb pairs, not spin exchange.
        dms = np.stack((dm, dmz))
        vj, vk = mf_grad.get_j(mol, dms, hermi=1), None

    de = np.zeros((len(atmlst), 3))
    aoslices = mol.aoslice_by_atom()
    hcore_deriv = mf_grad.hcore_generator(mol)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]

        # Each UHF spin orbital has occupation one, hence the hcore factor is
        # -1 instead of the restricted spatial-orbital factor -2.
        de[k] = -lib.einsum(
            'xuv,uv->x', hcore_deriv(ia), dmz
        ).real

        # Coulomb couples total alpha+beta densities.  The two lines are the
        # two polarization orders (P,D_Z) and (D_Z,P).
        de[k] -= 2 * lib.einsum(
            'xuv,uv->x', vj[0, :, p0:p1], dmz[p0:p1]
        ).real
        de[k] -= 2 * lib.einsum(
            'xuv,uv->x', vj[1, :, p0:p1], dm[p0:p1]
        ).real

        # Exchange couples only equal spins.  Indices (2,3) are alpha and
        # (4,5) beta source/transition-density pairs in dms.
        for i, j in (((2, 3), (4, 5)) if with_k else ()):
            de[k] += 2 * lib.einsum(
                'xuv,uv->x', vk[i, :, p0:p1], dms[j, p0:p1]
            ).real
            de[k] += 2 * lib.einsum(
                'xuv,uv->x', vk[j, :, p0:p1], dms[i, p0:p1]
            ).real

    if hasattr(vj, 'aux'):
        # DF auxiliary-center motion obeys the same cross-pair structure.
        # Both orders are necessary because the derivative acts on the shared
        # auxiliary representation as well as on the primary AO centers.
        vj_aux = np.asarray(vj.aux)
        de -= (vj_aux[0, 1] + vj_aux[1, 0])[atmlst]
        if with_k:
            vk_aux = np.asarray(vk.aux)
            de += (
                vk_aux[2, 3] + vk_aux[3, 2]
                + vk_aux[4, 5] + vk_aux[5, 4]
            )[atmlst]

    mf_grad.fock_deriv_intermediate_bytes = (
        dms.nbytes + np.asarray(vj).nbytes + (vk.nbytes if with_k else 0)
    )
    mf_grad.fock_deriv_storage = 'direct_contracted'
    return de
