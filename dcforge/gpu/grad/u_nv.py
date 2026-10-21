# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Source-neutral overlap and nonvariational gradient assembly.'''

import cupy as cp
from gpu4pyscf.lib.cupy_helper import contract

# S^lambda_oo/vo projection is spin-independent. Unrestricted sources use it
# separately to alpha and beta MO coefficients.
from dcforge.gpu.grad.r_nv import (
    make_s1_oo_and_grad_ovlp_energy as _make_s1_oo_and_grad_ovlp_energy,
)


def make_s1_oo_and_grad_ovlp_energy(
    mol, Co, Cv, e_occ, zvec, s1, atmlst,
):
    '''Build both spin ``S^lambda_oo`` blocks and stream the VO term.

    This is the unrestricted production wrapper around the restricted fused
    helper. The overlap transformation itself is spin-independent, but the
    unrestricted energy term sums two occupation-one spin contributions. The
    wrapper therefore applies the same atom-local transformation separately
    to alpha and beta and changes the restricted factor four to factor two per
    spin.

    It exists separately from the full spin-wise ``make_s1_oo_vo`` reference for
    the same reason as the restricted helper: production retains only both ``s1oo``
    tensors, whereas a reference caller may need the actual ``s1vo`` arrays.
    Making this distinction explicit prevents an accidental
    ``(2,3N,nvir_σ,nocc_σ)`` production allocation.

    Args:
        mol : pyscf.gto.Mole
            Molecule defining atom-local AO slices.
        Co, Cv : tuple of two cupy.ndarray
            Alpha/beta occupied and virtual source coefficients.
        e_occ : tuple of two cupy.ndarray
            Alpha/beta occupied source orbital energies.
        zvec : tuple of two cupy.ndarray
            Positive-RHS alpha/beta Z-vectors.
        s1 : cupy.ndarray of shape (3, nao, nao)
            One-center AO overlap derivative.
        atmlst : sequence of int
            Requested nuclear centers.

    Returns:
        s1oo : tuple of two cupy.ndarray
            Alpha/beta occupied overlap blocks.
        de_ovlp_energy : cupy.ndarray of shape (len(atmlst), 3)
            Immediately contracted spin-summed overlap-energy term.
    '''
    s1oo = []
    de = cp.zeros((len(atmlst), 3))
    for Co_s, Cv_s, e_s, z_s in zip(Co, Cv, e_occ, zvec):
        s1oo_s, de_s = _make_s1_oo_and_grad_ovlp_energy(
            mol, Co_s, Cv_s, e_s, z_s, s1, atmlst)
        s1oo.append(s1oo_s)
        # The restricted helper includes factor four.  Each unrestricted spin
        # channel carries factor two, hence one half of its returned term.
        de += .5 * de_s
    return tuple(s1oo), de


def grad_ovlp_energy(s1vo, e_occ, zvec):
    '''Return ``+2 sum_σ Z^σ S^lambda,vo_σ epsilon^σ_o`` on the GPU.'''
    de = cp.zeros(s1vo[0].shape[:2])
    for s1vo_s, e_s, z_s in zip(s1vo, e_occ, zvec):
        se_vo = s1vo_s * e_s[None, None, None, :]
        de += 2 * contract('ai,kxai->kx', z_s.conj(), se_vo).real
    return de


def grad_ovlp_response_full(Co, Cv, zvec, s1oo, fvind):
    '''Return the unrestricted overlap response with the full perturbation batch.

    This ``3N`` implementation is a numerical reference for the production
    single-adjoint order and is not called by the default gradient driver.
    '''
    nset, ncart = s1oo[0].shape[:2]
    nao = Co[0].shape[0]
    dm1 = []
    for Co_s, s1oo_s in zip(Co, s1oo):
        tmp = contract('kxij,ui->kxuj', s1oo_s, Co_s)
        dm_s = contract('kxuj,vj->kxuv', tmp, Co_s.conj())
        dm1.append(dm_s + dm_s.transpose(0, 1, 3, 2).conj())
    dm1 = cp.stack(dm1).reshape(2, -1, nao, nao)
    v1ao = fvind(dm1)

    de = cp.zeros((nset, ncart))
    for Co_s, Cv_s, z_s, v1_s in zip(Co, Cv, zvec, v1ao):
        tmp = contract('Auv,ua->Aav', v1_s, Cv_s.conj())
        v1vo = contract('Aav,vi->Aai', tmp, Co_s)
        v1vo = v1vo.reshape(nset, ncart, Cv_s.shape[1], Co_s.shape[1])
        de += contract('ai,kxai->kx', z_s.conj(), v1vo).real
    return de


def get_z_response_oo(Co, Cv, zvec, fvind):
    '''Return ``(C_o^σ)^dagger G^source[D_Z]^σ C_o^σ`` for each spin σ.

    Unlike :func:`grad_ovlp_response_full`, this production provider applies
    the coupled source-SCF response to one alpha/beta Z-density pair rather than all
    ``3N`` geometry densities. It is kept distinct from the final
    ``grad_ovlp_response`` contraction so a DF subclass can replace only the
    AO-potential construction with an MO-direct implementation.
    '''
    dmz = []
    for Co_s, Cv_s, z_s in zip(Co, Cv, zvec):
        dz_s = Cv_s @ z_s @ Co_s.conj().T
        dmz.append(dz_s + dz_s.conj().T)

    vz = fvind(cp.stack(dmz))
    vzoo = []
    for Co_s, vz_s in zip(Co, vz):
        tmp = Co_s.conj().T @ vz_s
        vzoo.append(tmp @ Co_s)
    return tuple(vzoo)


def grad_ovlp_response(s1oo, vzoo):
    '''Contract the unrestricted overlap response from provider-owned OO blocks.

    The self-adjoint UHF/UKS response identity replaces ``3N`` pairs of
    alpha/beta perturbation densities by one pair of Z-transition densities.
    AO-to-MO projection is kept as two matrix multiplications per spin.
    '''
    de = cp.zeros(s1oo[0].shape[:2])
    for s1oo_s, vzoo_s in zip(s1oo, vzoo):
        de += contract('kxij,ij->kx', s1oo_s, vzoo_s).real
    return de


def nonvariational_correction(
    mf_grad, s1oo, de_fock, de_ovlp_energy, de_ovlp_response,
):
    '''Assemble the explicit GPU ``2 sum_σ Z^σ V^lambda_σ`` terms.

    Args:
        mf_grad : dcforge.gpu.grad.unsc.Gradients
            Active gradient object receiving the host term ledger.
        s1oo : tuple of two cupy.ndarray
            Alpha/beta occupied overlap blocks retained from the single
            overlap transformation.
        de_fock : cupy.ndarray of shape (len(atmlst), 3)
            Precontracted spin-resolved source-Fock derivative term.
        de_ovlp_energy : cupy.ndarray of shape (len(atmlst), 3)
            Immediately contracted ``S^lambda_vo epsilon_o`` term.
        de_ovlp_response : cupy.ndarray of shape (len(atmlst), 3)
            Provider-independent adjoint overlap-response contraction.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Host unrestricted nonvariational correction in Ha/Bohr.
    '''
    mf_grad.s1oo = s1oo
    mf_grad.s1vo = None
    mf_grad.de_fock = cp.asnumpy(de_fock)
    mf_grad.de_ovlp_energy = cp.asnumpy(de_ovlp_energy)
    mf_grad.de_ovlp_response = cp.asnumpy(de_ovlp_response)
    mf_grad.de_nvar = (
        mf_grad.de_fock
        + mf_grad.de_ovlp_energy
        + mf_grad.de_ovlp_response
    )
    mf_grad.overlap_response_calls = 1
    mf_grad.overlap_response_nrhs = 1
    mf_grad.overlap_transform_calls = 1
    return mf_grad.de_nvar
