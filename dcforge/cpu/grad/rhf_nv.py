'''Nonvariational ``4 Z^dagger V^lambda`` molecular gradient terms.'''

import numpy as np
from pyscf import lib


def make_s1_oo_vo(mol, Co, Cv, s1, atmlst):
    '''Build the moving-basis overlap blocks for each nuclear perturbation.

    The PDF convention uses ``S^lambda_oo`` and ``S^lambda_vo`` in the
    perturbation-specific CPHF source ``V^lambda``.  PySCF supplies the AO
    overlap derivative with the derivative acting on one AO center.  Selecting
    the AO rows of atom ``I`` and adding the Hermitian partner constructs the
    complete perturbation ``lambda = R_Ialpha`` without forming a full
    ``(3*natm, nmo, nmo)`` MO tensor.

    Args:
        mol : pyscf.gto.Mole
            Molecule defining the atom-local AO slices.
        Co : ndarray of shape (nao, nocc)
            Occupied source-RHF MO coefficients ``C_o``.
        Cv : ndarray of shape (nao, nvir)
            Virtual source-RHF MO coefficients ``C_v``.
        s1 : ndarray of shape (3, nao, nao)
            PySCF AO overlap derivative returned by ``Gradients.get_ovlp``.
        atmlst : sequence of int
            Nuclear centers whose Cartesian perturbations are requested.

    Returns:
        s1oo : ndarray of shape (len(atmlst), 3, nocc, nocc)
            Occupied-occupied overlap blocks ``S^lambda_oo``.
        s1vo : ndarray of shape (len(atmlst), 3, nvir, nocc)
            Virtual-occupied overlap blocks ``S^lambda_vo``.
    '''
    nocc = Co.shape[1]
    nvir = Cv.shape[1]
    s1oo = np.zeros((len(atmlst), 3, nocc, nocc), dtype=Co.dtype)
    s1vo = np.zeros((len(atmlst), 3, nvir, nocc), dtype=Co.dtype)
    aoslices = mol.aoslice_by_atom()

    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]

        # s1co is shared by the OO and VO contractions.  This is the molecular
        # CPU analogue of retaining only the occupied columns of S^lambda C.
        s1co = lib.einsum('xpv,vi->xpi', s1[:, p0:p1], Co)

        # Add the ket-center contribution to obtain the full Hermitian
        # S^lambda_oo block for lambda = R_ia,x.
        s1oo[k] = lib.einsum('xpi,pj->xij', s1co, Co[p0:p1].conj())
        s1oo[k] += s1oo[k].transpose(0, 2, 1).conj()

        # The two contractions are the bra- and ket-center pieces of
        # S^lambda_vo = C_v^dagger S^lambda C_o.
        s1vo[k] = lib.einsum('xpi,pa->xai', s1co, Cv[p0:p1].conj())
        s1cv = lib.einsum('xpv,va->xpa', s1[:, p0:p1], Cv)
        s1vo[k] += lib.einsum('xpa,pi->xai', s1cv, Co[p0:p1].conj())
    return s1oo, s1vo


def grad_fock_make_h1(mf_grad, Co, Cv, zvec, mo_coeff, mo_occ,
                      atmlst):
    '''Return the source-Fock term through PySCF ``Hessian.make_h1``.

    This is the reference implementation for the direct contracted path.
    Conventional HF uses the public ``make_h1`` result and therefore retains
    one complete AO first-order Fock matrix per requested center.  DF-HF uses
    PySCF's atom iterator ``df.hessian.rhf._gen_jk`` and immediately projects
    each yielded Fock derivative to the VO block before discarding it.  The
    latter avoids retaining the complete atom-indexed Fock tensor in
    DC-Forge, although the private PySCF iterator may keep its own J/K buffers.

    ``make_h1`` differentiates the source-HF one- and two-electron operators
    at the unperturbed density.  It is the explicit ``F^HF,lambda`` entering
    the CPHF perturbation, not the total derivative after adding the induced
    density response ``G^HF[P^lambda]``.  Moving-basis overlap terms are also
    assembled separately in :func:`nonvariational_correction`.

    Args:
        mf_grad : dcforge.cpu.grad.rhf_dft.Gradients
            Active conventional or density-fitted gradient object.
        Co : ndarray of shape (nao, nocc)
            Occupied source-RHF MO coefficients.
        Cv : ndarray of shape (nao, nvir)
            Virtual source-RHF MO coefficients.
        zvec : ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        mo_coeff : ndarray of shape (nao, nmo)
            Complete source-RHF MO coefficients.
        mo_occ : ndarray of shape (nmo,)
            Exact restricted source occupations.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            The ``-4 Z F^HF,lambda_vo`` contribution in Ha/Bohr.
    '''
    mf_hess = mf_grad.base.mf_hf.Hessian()
    if hasattr(mf_hess, 'auxbasis_response'):
        # PySCF DF-Hessian level 2 includes both primary- and auxiliary-center
        # motion in the first-order source Fock.
        mf_hess.auxbasis_response = 2 * int(mf_grad.auxbasis_response)
    if hasattr(mf_grad.base.mf_hf, 'with_df'):
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


def grad_fock_direct(mf_grad, Co, Cv, zvec, dm0, atmlst):
    r'''Contract ``-4 Z F^HF,lambda_vo`` without building ``make_h1``.

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
        mf_grad : dcforge.cpu.grad.rhf_dft.Gradients
            Active conventional or density-fitted gradient object.
        Co : ndarray of shape (nao, nocc)
            Occupied source-RHF MO coefficients.
        Cv : ndarray of shape (nao, nvir)
            Virtual source-RHF MO coefficients.
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
    vj, vk = mf_grad.get_jk(mol, dms, hermi=1)
    vhf0 = vj[0] - .5 * vk[0]
    vhfz = vj[1] - .5 * vk[1]

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
        aux = np.asarray(vj.aux) - .5 * np.asarray(vk.aux)
        de -= 2 * (aux[0, 1] + aux[1, 0])[atmlst]

    mf_grad.fock_deriv_intermediate_bytes = (
        dms.nbytes + np.asarray(vj).nbytes + np.asarray(vk).nbytes
    )
    mf_grad.fock_deriv_storage = 'direct_contracted'
    return de


def grad_ovlp_energy(s1vo, e_occ, zvec):
    '''Return the ``+4 Z S^lambda_vo epsilon_o`` source term.

    Args:
        s1vo : ndarray of shape (nset, 3, nvir, nocc)
            Perturbation-specific overlap blocks ``S^lambda_vo``.
        e_occ : ndarray of shape (nocc,)
            Occupied source-RHF orbital energies ``epsilon_i``.
        zvec : ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.

    Returns:
        ndarray of shape (nset, 3)
            Overlap-energy part of ``4 Z V^lambda`` in Ha/Bohr.
    '''
    se_vo = s1vo * e_occ[None, None, None, :]
    return 4 * lib.einsum('ai,kxai->kx', zvec, se_vo)


def grad_ovlp_response_full(Co, Cv, zvec, s1oo, fvind):
    '''Return the overlap response with all perturbation densities.

    In the PDF notation,

    ``V^lambda = S^lambda_vo epsilon_o - F^HF,lambda_vo
                  - G^HF[P^lambda_oo]_vo``.

    The AO density built below is ``-P^lambda_oo``.  Consequently its induced
    source-HF potential is exactly the PDF term
    ``-G^HF[P^lambda_oo]_vo`` after the restricted response factors are
    applied.

    Args:
        Co : ndarray of shape (nao, nocc)
            Occupied source-RHF MO coefficients.
        Cv : ndarray of shape (nao, nvir)
            Virtual source-RHF MO coefficients.
        zvec : ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        s1oo : ndarray of shape (nset, 3, nocc, nocc)
            Occupied-occupied moving-basis overlap response.
        fvind : callable
            AO source-HF response returned by ``mf_hf.gen_response``.

    Returns:
        ndarray of shape (nset, 3)
            Overlap-density response contribution in Ha/Bohr.

    Notes:
        This is the full ``3N`` numerical oracle.  The production gradient
        uses :func:`grad_ovlp_response`, which applies the same self-adjoint
        HF response identity with one AO Z-density.
    '''
    nset, ncart = s1oo.shape[:2]
    nao = Co.shape[0]

    tmp = lib.einsum('kxij,ui->kxuj', s1oo, Co)
    dm1 = lib.einsum('kxuj,vj->kxuv', tmp, Co.conj())
    dm1 = dm1 + dm1.transpose(0, 1, 3, 2).conj()

    # dm1 = -P^lambda_oo because the moving-basis density response is
    # P^lambda_oo = -2 C_o S^lambda_oo C_o^dagger.  The inner restricted factor
    # 2 and the final contraction factor 2 give
    # 4 Z^dagger {-G^HF[P^lambda_oo]_vo}.
    v1ao = fvind(dm1.reshape(-1, nao, nao)) * 2
    v1vo = lib.einsum('Auv,ua,vi->Aai', v1ao, Cv.conj(), Co)
    v1vo = v1vo.reshape(nset, ncart, Cv.shape[1], Co.shape[1])
    return 2 * lib.einsum('ai,kxai->kx', zvec, v1vo)


def grad_ovlp_response(Co, Cv, zvec, s1oo, fvind):
    '''Return the overlap response with one self-adjoint HF response.

    The full perturbation route contracts one source-HF response for every
    nuclear coordinate.  For the real molecular RHF response operator,

    ``4 Z^dagger G^HF[P^lambda_oo]_vo``
    ``= 4 S^lambda_oo : C_o^dagger G^HF[D_Z] C_o``,

    where ``D_Z = C_v Z C_o^dagger + h.c.``.  The right-hand side needs one
    AO response density regardless of ``3N``.

    Args:
        Co : ndarray of shape (nao, nocc)
            Occupied source-RHF MO coefficients.
        Cv : ndarray of shape (nao, nvir)
            Virtual source-RHF MO coefficients.
        zvec : ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        s1oo : ndarray of shape (nset, 3, nocc, nocc)
            Occupied-occupied moving-basis overlap response.
        fvind : callable
            Self-adjoint AO source-HF response action.

    Returns:
        ndarray of shape (nset, 3)
            Overlap-density response contribution in Ha/Bohr.
    '''
    dz = Cv @ zvec @ Co.conj().T
    vz = fvind(dz + dz.conj().T)
    vz_oo = Co.conj().T @ vz @ Co
    return 4 * lib.einsum('kxij,ij->kx', s1oo, vz_oo).real


def nonvariational_correction(mf_grad, Co, Cv, mo_energy, zvec, s1,
                              de_fock, fvind, atmlst=None):
    '''Assemble the perturbation-specific ``4 Z^dagger V^lambda`` correction.

    The source defined in the molecular-gradient PDF is

    ``V^lambda = S^lambda_vo epsilon_o - F^HF,lambda_vo
                  - G^HF[P^lambda_oo]_vo``.

    This function retains its three contributions separately on ``mf_grad``
    before returning their sum.

    Args:
        mf_grad : dcforge.cpu.grad.rhf_dft.Gradients
            Active gradient object that receives the ``de_*`` term ledger.
        Co : ndarray of shape (nao, nocc)
            Occupied source-RHF MO coefficients.
        Cv : ndarray of shape (nao, nvir)
            Virtual source-RHF MO coefficients.
        mo_energy : ndarray of shape (nmo,)
            Canonical source-RHF orbital energies.
        zvec : ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        s1 : ndarray of shape (3, nao, nao)
            PySCF AO overlap derivative.
        de_fock : ndarray of shape (len(atmlst), 3)
            Precontracted ``-4 Z F^HF,lambda_vo`` term from the selected
            source-Fock derivative implementation.
        fvind : callable
            AO source-HF response action.

    Kwargs:
        atmlst : sequence of int or None
            Nuclear centers included in the gradient. All atoms are used when
            omitted.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Nonvariational electronic gradient in Ha/Bohr.
    '''
    mol = mf_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    e_occ = mo_energy[:Co.shape[1]]

    s1oo, s1vo = make_s1_oo_vo(mol, Co, Cv, s1, atmlst)
    de_ovlp_energy = grad_ovlp_energy(s1vo, e_occ, zvec)
    de_ovlp_response = grad_ovlp_response(Co, Cv, zvec, s1oo, fvind)

    mf_grad.s1oo = s1oo
    mf_grad.s1vo = s1vo
    mf_grad.de_fock = de_fock
    mf_grad.de_ovlp_energy = de_ovlp_energy
    mf_grad.de_ovlp_response = de_ovlp_response
    mf_grad.de_nvar = de_fock + de_ovlp_energy + de_ovlp_response
    return mf_grad.de_nvar
