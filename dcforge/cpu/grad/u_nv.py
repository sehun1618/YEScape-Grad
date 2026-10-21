# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Source-neutral overlap and nonvariational gradient assembly.'''

import numpy as np
from pyscf import lib

from dcforge.cpu.grad.r_nv import make_s1_oo_vo


def grad_ovlp_energy(s1vo, e_occ, zvec):
    '''Return ``+2 sum_σ Z^σ S^lambda,vo_σ epsilon^σ_o``.

    Args:
        s1vo : tuple of two ndarray
            Spin-resolved overlap VO blocks, each ``(nset,3,nvir_σ,nocc_σ)``.
        e_occ : tuple of two ndarray
            Occupied source-SCF orbital energies.
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
    '''Return the overlap-Hessian term with one adjoint source response.

    The direct legacy order builds ``3*natm`` spin-density perturbations from
    ``S^lambda_oo`` and applies ``G^source`` to all of them. Self-adjointness gives

    ``-2 sum_σ Z^σ : G^source[P^lambda_oo]^σ_vo``
    `` = sum_σ S^lambda,oo_σ : (C_o^σ)^dagger G^source[D_Z]^σ C_o^σ``,

    where ``P^lambda,σ_oo = -C_o^σ S^lambda,σ_oo (C_o^σ)^dagger`` and
    ``D_Z^σ = C_v^σ Z^σ (C_o^σ)^dagger + h.c.``. G acts on both spin
    densities together, including alpha/beta XC-kernel coupling. The implemented order
    calls the expensive response provider once, independent of atom count.

    Args:
        Co, Cv : tuple of two ndarray
            Alpha/beta occupied and virtual MO coefficients.
        zvec : tuple of two ndarray
            Positive-RHS alpha/beta Z-vectors.
        s1oo : tuple of two ndarray
            Spin-resolved moving-basis OO blocks.
        fvind : callable
            Coupled alpha/beta AO source response action.

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
    '''Assemble the spin-resolved ``2 sum_σ Z^σ V^lambda_σ`` correction.

    Args:
        mf_grad : dcforge.cpu.grad.unsc.Gradients
            Active gradient object receiving the term ledger.
        Co, Cv : tuple of two ndarray
            Alpha/beta occupied and virtual MO coefficients.
        mo_energy : tuple of two ndarray
            Canonical source-SCF orbital energies.
        zvec : tuple of two ndarray
            Positive-RHS alpha/beta Z-vectors.
        s1 : ndarray of shape (3, nao, nao)
            PySCF one-center AO overlap derivative.
        de_fock : ndarray of shape (len(atmlst), 3)
            Precontracted spin-resolved source-Fock derivative term.
        fvind : callable
            Coupled alpha/beta AO source response.

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
