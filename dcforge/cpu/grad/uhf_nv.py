# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Spin-resolved nonvariational molecular UHF-DFT gradient terms.'''

import numpy as np
from pyscf import lib

from dcforge.cpu.grad.rhf_nv import make_s1_oo_vo


def grad_fock_make_h1(mf_grad, Co, Cv, zvec, mo_coeff, mo_occ,
                      atmlst):
    '''Return the source-Fock term through PySCF ``Hessian.make_h1``.

    This is the unrestricted numerical reference for the direct contracted
    route. Conventional UHF retains the alpha and beta ``make_h1`` tensors.
    DF-UHF instead consumes PySCF's ``df.hessian.uhf._gen_jk`` atom iterator
    and immediately projects each yielded spin Fock derivative to its VO
    block. This removes the complete atom-indexed Fock tensor from the
    DC-Forge layer, while PySCF may still retain internal J/K buffers.

    The returned matrices are the explicit source-operator derivatives at
    the unperturbed spin densities.  They do not include the induced UHF
    density response ``G^UHF[P^lambda]`` or the separately assembled overlap
    response, and therefore are not the total self-consistent ``dF/dlambda``.

    Args:
        mf_grad : dcforge.cpu.grad.uhf_dft.Gradients
            Active conventional or density-fitted gradient object.
        Co, Cv : tuple of two ndarray
            Alpha/beta occupied and virtual source-UHF MO coefficients.
        zvec : tuple of two ndarray
            Positive-RHS alpha/beta Z-vectors.
        mo_coeff : tuple of two ndarray
            Complete alpha/beta source-UHF MO coefficients.
        mo_occ : tuple of two ndarray
            Exact unrestricted source occupations.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            The ``-2 sum_s Z^s F^UHF,s,lambda_vo`` contribution in Ha/Bohr.
    '''
    mf_hess = mf_grad.base.mf_hf.Hessian()
    if hasattr(mf_hess, 'auxbasis_response'):
        mf_hess.auxbasis_response = 2 * int(mf_grad.auxbasis_response)
    if hasattr(mf_grad.base.mf_hf, 'with_df'):
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


def grad_fock_direct(mf_grad, Co, Cv, zvec, dm0, atmlst):
    r'''Contract the spin-resolved source-Fock term without ``make_h1``.

    Define one Hermitian transition density for each spin,

    .. math::

        D_Z^s = C_v^s Z^s (C_o^s)^\dagger
              + C_o^s (Z^s)^\dagger (C_v^s)^\dagger.

    With ``P = P^alpha + P^beta`` and
    ``D_Z = D_Z^alpha + D_Z^beta``, the required contraction is

    .. math::

        -\operatorname{Tr}[D_Z h^\lambda]
        -\operatorname{Tr}[D_Z J^\lambda[P]]
        +\sum_s\operatorname{Tr}[D_Z^s K^\lambda[P^s]].

    A single gradient J/K call evaluates the Coulomb cross pair
    ``(P,D_Z)`` and the two same-spin exchange cross pairs.  The primary-AO
    row contributions and, for DF, both auxiliary-center polarization orders
    are retained.  No ``(2,N_atom,3,nao,nao)`` first-order Fock tensor is
    formed.

    Args:
        mf_grad : dcforge.cpu.grad.uhf_dft.Gradients
            Active conventional or density-fitted gradient object.
        Co, Cv : tuple of two ndarray
            Alpha/beta occupied and virtual source-UHF MO coefficients.
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
    dms = np.stack((dm, dmz, dm0[0], dz[0], dm0[1], dz[1]))
    vj, vk = mf_grad.get_jk(mol, dms, hermi=1)

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
        for i, j in ((2, 3), (4, 5)):
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
        vk_aux = np.asarray(vk.aux)
        de -= (vj_aux[0, 1] + vj_aux[1, 0])[atmlst]
        de += (
            vk_aux[2, 3] + vk_aux[3, 2]
            + vk_aux[4, 5] + vk_aux[5, 4]
        )[atmlst]

    mf_grad.fock_deriv_intermediate_bytes = (
        dms.nbytes + np.asarray(vj).nbytes + np.asarray(vk).nbytes
    )
    mf_grad.fock_deriv_storage = 'direct_contracted'
    return de


def grad_ovlp_energy(s1vo, e_occ, zvec):
    '''Return ``+2 sum_s Z^s S^lambda,vo_s epsilon^s_o``.

    Args:
        s1vo : tuple of two ndarray
            Spin-resolved overlap VO blocks, each ``(nset,3,nvir_s,nocc_s)``.
        e_occ : tuple of two ndarray
            Occupied source-UHF orbital energies.
        zvec : tuple of two ndarray
            Positive-RHS alpha/beta Z-vectors.

    Returns:
        ndarray of shape (nset, 3)
            Overlap-energy contribution in Ha/Bohr.
    '''
    de = np.zeros(s1vo[0].shape[:2])
    for s1vo_s, e_s, z_s in zip(s1vo, e_occ, zvec):
        se_vo = s1vo_s * e_s[None, None, None, :]
        de += 2 * np.real(
            lib.einsum('ai,kxai->kx', z_s.conj(), se_vo))
    return de


def grad_ovlp_response(Co, Cv, zvec, s1oo, fvind):
    '''Return the overlap-Hessian term with one adjoint UHF response.

    The direct legacy order builds ``3*natm`` spin-density perturbations from
    ``S^lambda_oo`` and applies ``G^UHF`` to all of them. Self-adjointness gives

    ``sum_s Z^s : G^UHF[P^lambda_oo]^s_vo``
    `` = sum_s S^lambda,oo_s : (C_o^s)^dagger G^UHF[D_Z]^s C_o^s``,

    where ``D_Z^s = C_v^s Z^s (C_o^s)^dagger + h.c.``. The implemented order
    calls the expensive response provider once, independent of atom count.

    Args:
        Co, Cv : tuple of two ndarray
            Alpha/beta occupied and virtual MO coefficients.
        zvec : tuple of two ndarray
            Positive-RHS alpha/beta Z-vectors.
        s1oo : tuple of two ndarray
            Spin-resolved moving-basis OO blocks.
        fvind : callable
            Coupled alpha/beta AO UHF response action.

    Returns:
        ndarray of shape (nset, 3)
            Overlap-Hessian response contribution in Ha/Bohr.

    Notes:
        AO-to-MO projection is deliberately evaluated as two matrix products,
        not one four-index ``einsum``. This follows the faster contraction
        ordering identified in the legacy CPU/GPU comparison.
    '''
    dmz = []
    for Co_s, Cv_s, z_s in zip(Co, Cv, zvec):
        dz = Cv_s @ z_s @ Co_s.conj().T
        dmz.append(dz + dz.conj().T)

    vz = fvind(np.asarray(dmz))
    de = np.zeros(s1oo[0].shape[:2])
    for Co_s, s1oo_s, vz_s in zip(Co, s1oo, vz):
        tmp = Co_s.conj().T @ vz_s
        vzoo = tmp @ Co_s
        de += np.real(lib.einsum('kxij,ij->kx', s1oo_s, vzoo))
    return de


def nonvariational_correction(mf_grad, Co, Cv, mo_energy, zvec, s1,
                              de_fock, fvind, atmlst=None):
    '''Assemble the spin-resolved ``2 sum_s Z^s V^lambda_s`` correction.

    Args:
        mf_grad : dcforge.cpu.grad.uhf_dft.Gradients
            Active gradient object receiving the term ledger.
        Co, Cv : tuple of two ndarray
            Alpha/beta occupied and virtual MO coefficients.
        mo_energy : tuple of two ndarray
            Canonical source-UHF orbital energies.
        zvec : tuple of two ndarray
            Positive-RHS alpha/beta Z-vectors.
        s1 : ndarray of shape (3, nao, nao)
            PySCF one-center AO overlap derivative.
        de_fock : ndarray of shape (len(atmlst), 3)
            Precontracted spin-resolved source-Fock derivative term.
        fvind : callable
            Coupled alpha/beta AO UHF response.

    Kwargs:
        atmlst : sequence of int or None
            Nuclear centers included in the gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Unrestricted nonvariational correction in Ha/Bohr.
    '''
    mol = mf_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)

    s1oo = []
    s1vo = []
    e_occ = []
    for Co_s, Cv_s, e_s in zip(Co, Cv, mo_energy):
        oo_s, vo_s = make_s1_oo_vo(mol, Co_s, Cv_s, s1, atmlst)
        s1oo.append(oo_s)
        s1vo.append(vo_s)
        e_occ.append(e_s[:Co_s.shape[1]])

    de_ovlp_energy = grad_ovlp_energy(s1vo, e_occ, zvec)
    de_ovlp_response = grad_ovlp_response(Co, Cv, zvec, s1oo, fvind)

    mf_grad.de_fock = de_fock
    mf_grad.de_ovlp_energy = de_ovlp_energy
    mf_grad.de_ovlp_response = de_ovlp_response
    mf_grad.de_nvar = de_fock + de_ovlp_energy + de_ovlp_response
    mf_grad.overlap_response_calls = 1
    mf_grad.overlap_response_nrhs = 1
    return mf_grad.de_nvar
