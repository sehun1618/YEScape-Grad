# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Source-neutral overlap and nonvariational gradient assembly.'''

import numpy as np
from pyscf import lib


def make_s1_oo_vo(mol, Co, Cv, s1, atmlst):
    '''Build the moving-basis overlap blocks for each nuclear perturbation.

    The overlap response uses ``S^lambda_oo`` and ``S^lambda_vo`` in the
    perturbation-specific CPHF source ``V^lambda``.  PySCF supplies the AO
    overlap derivative with the derivative acting on one AO center.  Selecting
    the AO rows of atom ``I`` and adding the Hermitian partner constructs the
    complete perturbation ``lambda = R_Ialpha`` without forming a full
    ``(3*natm, nmo, nmo)`` MO tensor.

    Args:
        mol : pyscf.gto.Mole
            Molecule defining the atom-local AO slices.
        Co : ndarray of shape (nao, nocc)
            Occupied source MO coefficients ``C_o``.
        Cv : ndarray of shape (nao, nvir)
            Virtual source MO coefficients ``C_v``.
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


def grad_ovlp_energy(s1vo, e_occ, zvec):
    '''Return the ``+4 Z S^lambda_vo epsilon_o`` source term.

    Args:
        s1vo : ndarray of shape (nset, 3, nvir, nocc)
            Perturbation-specific overlap blocks ``S^lambda_vo``.
        e_occ : ndarray of shape (nocc,)
            Occupied source-SCF orbital energies ``epsilon_i``.
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

    With the positive-RHS Z-vector convention,

    ``V^lambda = S^lambda_vo epsilon_o - F^source,lambda_vo
                  - G^source[P^lambda_oo]_vo``.

    The AO density built below is ``-P^lambda_oo``.  Consequently its induced
    source-SCF potential is exactly the term
    ``-G^source[P^lambda_oo]_vo`` after the restricted response factors are
    applied.

    Args:
        Co : ndarray of shape (nao, nocc)
            Occupied source-SCF MO coefficients.
        Cv : ndarray of shape (nao, nvir)
            Virtual source-SCF MO coefficients.
        zvec : ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        s1oo : ndarray of shape (nset, 3, nocc, nocc)
            Occupied-occupied moving-basis overlap response.
        fvind : callable
            AO source-SCF response returned by ``mf_source.gen_response``.

    Returns:
        ndarray of shape (nset, 3)
            Overlap-density response contribution in Ha/Bohr.

    Notes:
        This is the full ``3N`` numerical reference.  The production gradient
        uses :func:`grad_ovlp_response`, which applies the same self-adjoint
        HF/KS response identity with one AO Z-density.
    '''
    nset, ncart = s1oo.shape[:2]
    nao = Co.shape[0]

    tmp = lib.einsum('kxij,ui->kxuj', s1oo, Co)
    dm1 = lib.einsum('kxuj,vj->kxuv', tmp, Co.conj())
    dm1 = dm1 + dm1.transpose(0, 1, 3, 2).conj()

    # dm1 = -P^lambda_oo because the moving-basis density response is
    # P^lambda_oo = -2 C_o S^lambda_oo C_o^dagger.  The inner restricted factor
    # 2 and the final contraction factor 2 give
    # 4 Z^dagger {-G^source[P^lambda_oo]_vo}.
    v1ao = fvind(dm1.reshape(-1, nao, nao)) * 2
    v1vo = lib.einsum('Auv,ua,vi->Aai', v1ao, Cv.conj(), Co)
    v1vo = v1vo.reshape(nset, ncart, Cv.shape[1], Co.shape[1])
    return 2 * lib.einsum('ai,kxai->kx', zvec, v1vo)


def grad_ovlp_response(Co, Cv, zvec, s1oo, fvind, response=None):
    '''Return the overlap response with one self-adjoint source response.

    The full perturbation route contracts one source-SCF response for every
    nuclear coordinate.  For the real molecular HF/KS response operator,

    ``-4 Z^dagger G^source[P^lambda_oo]_vo``
    ``= 4 S^lambda_oo : C_o^dagger G^source[D_Z] C_o``,

    where ``P^lambda_oo = -2 C_o S^lambda_oo C_o^dagger`` and
    ``D_Z = C_v Z C_o^dagger + h.c.``. The right-hand side needs one
    AO response density regardless of ``3N``.

    Args:
        Co : ndarray of shape (nao, nocc)
            Occupied source-SCF MO coefficients.
        Cv : ndarray of shape (nao, nvir)
            Virtual source-SCF MO coefficients.
        zvec : ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        s1oo : ndarray of shape (nset, 3, nocc, nocc)
            Occupied-occupied moving-basis overlap response.
        fvind : callable
            Self-adjoint AO source-SCF response action.

    Returns:
        ndarray of shape (nset, 3)
            Overlap-density response contribution in Ha/Bohr.
    '''
    if response is None:
        dz = Cv @ zvec @ Co.conj().T
        vz = fvind(dz + dz.conj().T)
        vz_oo = Co.conj().T @ vz @ Co
    else:
        vz_oo = response.oo(zvec)
    return 4 * lib.einsum('kxij,ij->kx', s1oo, vz_oo).real


def nonvariational_correction(mf_grad, Co, Cv, mo_energy, zvec, s1,
                              de_fock, fvind, atmlst=None, response=None):
    '''Assemble the perturbation-specific ``4 Z^dagger V^lambda`` correction.

    The perturbation-specific source is

    ``V^lambda = S^lambda_vo epsilon_o - F^source,lambda_vo
                  - G^source[P^lambda_oo]_vo``.

    This function retains its three contributions separately on ``mf_grad``
    before returning their sum.

    Args:
        mf_grad : yescape_grad.cpu.grad.rnsc.Gradients
            Active gradient object that receives the ``de_*`` term ledger.
        Co : ndarray of shape (nao, nocc)
            Occupied source-SCF MO coefficients.
        Cv : ndarray of shape (nao, nvir)
            Virtual source-SCF MO coefficients.
        mo_energy : ndarray of shape (nmo,)
            Canonical source-SCF orbital energies.
        zvec : ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        s1 : ndarray of shape (3, nao, nao)
            PySCF AO overlap derivative.
        de_fock : ndarray of shape (len(atmlst), 3)
            Precontracted ``-4 Z F^source,lambda_vo`` term from the selected
            source-Fock derivative implementation.
        fvind : callable
            AO source-SCF response action.

    Kwargs:
        response : MOResponse or None
            Per-call DF-HF provider; None uses the existing AO response.
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
    de_ovlp_response = grad_ovlp_response(Co, Cv, zvec, s1oo, fvind, response)

    mf_grad.s1oo = s1oo
    mf_grad.s1vo = s1vo
    mf_grad.de_fock = de_fock
    mf_grad.de_ovlp_energy = de_ovlp_energy
    mf_grad.de_ovlp_response = de_ovlp_response
    mf_grad.de_nvar = de_fock + de_ovlp_energy + de_ovlp_response
    return mf_grad.de_nvar
