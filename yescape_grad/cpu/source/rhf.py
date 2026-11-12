# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''RHF source Fock nuclear derivatives; SCF remains upstream.'''

import numpy as np
from pyscf import lib


def grad_fock_make_h1(mf_grad, Co, Cv, zvec, mo_coeff, mo_occ,
                      atmlst):
    '''Return the source-Fock term through PySCF ``Hessian.make_h1``.

    This is the reference implementation for the direct contracted path.
    Conventional HF uses the public ``make_h1`` result and therefore retains
    one complete AO first-order Fock matrix per requested center.  DF-HF uses
    PySCF's atom iterator ``df.hessian.rhf._gen_jk`` and immediately projects
    each yielded Fock derivative to the VO block before discarding it.  The
    latter avoids retaining the complete atom-indexed Fock tensor in
    YEScape-Grad, although the private PySCF iterator may keep its own J/K buffers.

    ``make_h1`` differentiates the source-HF one- and two-electron operators
    at the unperturbed density.  It is the explicit ``F^HF,lambda`` entering
    the CPHF perturbation, not the total derivative after adding the induced
    density response ``G^HF[P^lambda]``.  Moving-basis overlap terms are also
    assembled separately in :func:`nonvariational_correction`.

    Args:
        mf_grad : yescape_grad.cpu.grad.rnsc.Gradients
            Active conventional or density-fitted gradient object.
        Co : ndarray of shape (nao, nocc)
            Occupied source MO coefficients.
        Cv : ndarray of shape (nao, nvir)
            Virtual source MO coefficients.
        zvec : ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        mo_coeff : ndarray of shape (nao, nmo)
            Complete source MO coefficients.
        mo_occ : ndarray of shape (nmo,)
            Exact restricted source occupations.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            The ``-4 Z F^HF,lambda_vo`` contribution in Ha/Bohr.
    '''
    mf_hess = mf_grad.base.mf_source.Hessian()
    if hasattr(mf_hess, 'auxbasis_response'):
        # PySCF DF-Hessian level 2 includes both primary- and auxiliary-center
        # motion in the first-order source Fock.
        mf_hess.auxbasis_response = 2 * int(mf_grad.auxbasis_response)
    if hasattr(mf_grad.base.mf_source, 'with_df'):
        from pyscf.df.hessian import rhf as df_hessian_rhf

        de = np.zeros((len(atmlst), 3))
        peak_bytes = 0
        for k, (ia, h1, vj1, vk1) in enumerate(df_hessian_rhf._gen_jk(
            mf_hess, mo_coeff, mo_occ, atmlst=atmlst, with_k=True,
        )):
            h1ao = h1 + vj1 - .5 * vk1
            f1vo = lib.einsum(
                'xuv,ua,vi->xai', h1ao, Cv.conj(), Co)
            de[k] = -4 * lib.einsum('ai,xai->x', zvec, f1vo)
            peak_bytes = max(peak_bytes, h1ao.nbytes + f1vo.nbytes)
        mf_grad.fock_deriv_intermediate_bytes = peak_bytes
        mf_grad.fock_deriv_storage = 'atom_stream'
        return de

    h1ao = mf_hess.make_h1(mo_coeff, mo_occ, atmlst=atmlst)
    mf_grad.fock_deriv_intermediate_bytes = sum(
        h1ao[ia].nbytes for ia in atmlst
    )
    mf_grad.fock_deriv_storage = 'materialized'

    de = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        f1vo = lib.einsum('xuv,ua,vi->xai', h1ao[ia], Cv.conj(), Co)
        de[k] = -4 * lib.einsum('ai,xai->x', zvec, f1vo)
    return de


def grad_fock_direct(mf_grad, Co, Cv, zvec, dm0, atmlst, with_k=True):
    r'''Contract ``-4 Z F^HF,lambda_vo`` without building ``make_h1``.

    ``with_k=False`` reuses only the hcore/J polarization for pure KS sources;
    no K integral or K auxiliary-response array is constructed in that case.

    Define the Hermitian Z-transition density

    .. math::

        D_Z = C_v Z C_o^\dagger + C_o Z^\dagger C_v^\dagger.

    Since the real-RHF first-order Fock is Hermitian,

    .. math::

        -4 Z^\dagger F^{\lambda}_{vo}
        = -2\operatorname{Tr}[D_Z F^{\lambda}].

    PySCF's gradient J/K provider is evaluated for ``[P_HF, D_Z]``.  The two
    primary-basis row contractions and, for DF, the two cross auxiliary terms
    are the polarized derivative of the source-HF two-electron energy.  This
    avoids the ``(N_atom,3,nao,nao)`` first-order Fock tensor and reuses the
    active conventional or DF provider through normal method resolution.

    Args:
        mf_grad : yescape_grad.cpu.grad.rnsc.Gradients
            Active conventional or density-fitted gradient object.
        Co : ndarray of shape (nao, nocc)
            Occupied source MO coefficients.
        Cv : ndarray of shape (nao, nvir)
            Virtual source MO coefficients.
        zvec : ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        dm0 : ndarray of shape (nao, nao)
            Closed-shell source-HF AO density ``P_HF``.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            The directly contracted source-Fock term in Ha/Bohr.
    '''
    mol = mf_grad.mol
    dz = Cv @ zvec @ Co.conj().T
    dz = dz + dz.conj().T
    dms = np.stack((dm0, dz))
    if with_k:
        vj, vk = mf_grad.get_jk(mol, dms, hermi=1)
        vhf = vj - .5 * vk
    else:
        # Pure KS uses the same polarization algebra but no exchange integral.
        vj, vk = mf_grad.get_j(mol, dms, hermi=1), None
        vhf = vj
    vhf0, vhfz = vhf

    de = np.zeros((len(atmlst), 3))
    aoslices = mol.aoslice_by_atom()
    hcore_deriv = mf_grad.hcore_generator(mol)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]

        # -2 Tr[D_Z h^lambda].  hcore_generator already returns the complete
        # derivative for displacement of atom ia.
        de[k] = -2 * lib.einsum('xuv,uv->x', hcore_deriv(ia), dz).real

        # Polarization of the two-electron derivative.  Each AO derivative
        # returned by get_jk acts on one primary-basis center, hence the factor
        # two implicit in each -4 row contraction.
        de[k] -= 4 * lib.einsum(
            'xuv,uv->x', vhf0[:, p0:p1], dz[p0:p1]
        ).real
        de[k] -= 4 * lib.einsum(
            'xuv,uv->x', vhfz[:, p0:p1], dm0[p0:p1]
        ).real

    if hasattr(vj, 'aux'):
        # vj.aux[m,n] is the auxiliary-center derivative cross term between
        # density m and density n.  Both polarization orders are required.
        aux = np.asarray(vj.aux)
        if with_k:
            aux = aux - .5 * np.asarray(vk.aux)
        de -= 2 * (aux[0, 1] + aux[1, 0])[atmlst]

    mf_grad.fock_deriv_intermediate_bytes = (
        dms.nbytes + np.asarray(vj).nbytes + (vk.nbytes if with_k else 0)
    )
    mf_grad.fock_deriv_storage = 'direct_contracted'
    return de
