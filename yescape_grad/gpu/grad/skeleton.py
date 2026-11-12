# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Skeleton terms of molecular GPU target-on-source gradients.'''

import cupy as cp
import numpy as np
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import uhf as uhf_grad
from gpu4pyscf.lib.cupy_helper import contract


def grad_ovlp_correction(foo, e_occ, s1oo):
    '''Replace the standard diagonal energy-weighted overlap by target F_oo.

    The upstream skeleton uses the orbital energies passed to it, here
    ``diag(epsilon_i^source)``. A target-on-source gradient instead requires
    the full target block ``F^target_oo``. The difference is

    ``-2 Tr[(F^target_oo - diag(epsilon_i^source)) S^lambda_oo]``.

    Args:
        foo : cupy.ndarray of shape (nocc, nocc)
            Target Fock occupied-occupied block.
        e_occ : cupy.ndarray of shape (nocc,)
            Restricted source-SCF occupied orbital energies.
        s1oo : cupy.ndarray of shape (nset, 3, nocc, nocc)
            Full moving-basis occupied overlap blocks.

    Returns:
        cupy.ndarray of shape (nset, 3)
            Noncanonical target-overlap correction in Ha/Bohr.
    '''
    dfock = foo - cp.diag(e_occ)
    return -2 * contract('kxij,ij->kx', s1oo, dfock).real


def skeleton_gradient(mf_grad, foo, e_occ, s1oo, mo_energy, mo_coeff,
                      mo_occ, atmlst):
    '''Return the electronic GPU skeleton before the Z-vector correction.

    GPU4PySCF 1.8.1 owns the current one-electron, XC/grid and provider J/K
    contractions through ``rhf_grad.grad_elec`` and the active gradient
    object's ``energy_ee`` method.  A DF subclass therefore selects DF J/K
    and auxiliary response through normal method resolution; this common
    skeleton does not import a DF provider.

    The upstream variational overlap term is then changed from source
    ``diag(epsilon_o)`` to the full target ``F_oo``.
    '''
    de_standard = rhf_grad.grad_elec(
        mf_grad, mo_energy, mo_coeff, mo_occ, atmlst)
    de_ovlp_correction = cp.asnumpy(
        grad_ovlp_correction(foo, e_occ, s1oo)
    )

    mf_grad.de_standard_elec = np.asarray(de_standard)
    mf_grad.de_ovlp_correction = de_ovlp_correction
    mf_grad.de_skel_elec = mf_grad.de_standard_elec + de_ovlp_correction
    return mf_grad.de_skel_elec


def grad_ovlp_correction_uhf(foo, e_occ, s1oo):
    '''Replace spin-diagonal source energies by the target UKS ``F_oo``.

    Args:
        foo : tuple of two cupy.ndarray
            Target alpha/beta occupied-occupied Fock blocks.
        e_occ : tuple of two cupy.ndarray
            Unrestricted source-SCF occupied orbital energies.
        s1oo : tuple of two cupy.ndarray
            Spin-resolved moving-basis occupied overlap blocks.

    Returns:
        cupy.ndarray of shape (nset, 3)
            Unrestricted noncanonical overlap correction in Ha/Bohr.

    Notes:
        Each occupied unrestricted spin orbital has occupation one, so each spin contraction
        carries factor ``-1`` rather than the restricted factor ``-2``.
    '''
    de = cp.zeros(s1oo[0].shape[:2])
    for foo_s, e_s, s1oo_s in zip(foo, e_occ, s1oo):
        dfock_s = foo_s - cp.diag(e_s)
        de -= contract('kxij,ij->kx', s1oo_s, dfock_s).real
    return de


def skeleton_gradient_uhf(mf_grad, foo, e_occ, s1oo, mo_energy,
                          mo_coeff, mo_occ, atmlst):
    '''Return the electronic GPU unrestricted skeleton before the Z correction.

    GPU4PySCF UHF owns the one-electron and ordinary energy-weighted overlap
    terms. Dynamic dispatch through ``mf_grad.energy_ee`` selects target UKS
    XC/grid and DF J/K/auxiliary response. The final correction replaces the
    source diagonal orbital energies by the full target spin ``F_oo`` blocks.
    '''
    de_standard = uhf_grad.grad_elec(
        mf_grad, mo_energy, mo_coeff, mo_occ, atmlst)
    de_ovlp_correction = cp.asnumpy(
        grad_ovlp_correction_uhf(foo, e_occ, s1oo))

    mf_grad.de_standard_elec = np.asarray(de_standard)
    mf_grad.de_ovlp_correction = de_ovlp_correction
    mf_grad.de_skel_elec = mf_grad.de_standard_elec + de_ovlp_correction
    return mf_grad.de_skel_elec
