'''Gamma-point Gaussian-DF J/K derivatives from analytic CDERI response.'''

import numpy as np
from pyscf import lib
from pyscf.pbc import tools as pbc_tools


def grad_jk(cderi, cderi1, dm):
    r'''Return fixed-density Gamma-point ``J^lambda`` and ``K^lambda``.

    Args:
        cderi : ndarray of shape (naux, nao, nao)
            Central PySCF CDERI factor.
        cderi1 : ndarray of shape (natm, 3, naux, nao, nao)
            Complete source plus metric derivative of ``cderi``.
        dm : ndarray of shape (nao, nao)
            Closed-shell AO density including occupation factor two.

    Returns:
        tuple
            Coulomb and exchange matrix derivatives with shape
            ``(natm, 3, nao, nao)``.
    '''
    cderi = np.asarray(cderi)
    cderi1 = np.asarray(cderi1)
    dm = np.asarray(dm)
    if cderi1.shape[2:] != cderi.shape:
        raise ValueError('CDERI value and derivative shapes are inconsistent')
    if dm.shape != cderi.shape[1:]:
        raise ValueError('density and CDERI AO dimensions are inconsistent')

    rho = lib.einsum('Pij,ji->P', cderi, dm)
    rho1 = lib.einsum('axPij,ji->axP', cderi1, dm)
    vj1 = lib.einsum('P,axPij->axij', rho, cderi1)
    vj1 += lib.einsum('axP,Pij->axij', rho1, cderi)

    vk1 = lib.einsum('axPij,jk,Pkl->axil', cderi1, dm, cderi)
    vk1 += lib.einsum('Pij,jk,axPkl->axil', cderi, dm, cderi1)
    return _real_if_close(vj1), _real_if_close(vk1)


def grad_j_energy(dm, vj1):
    '''Return ``0.5 Tr[P J^lambda]`` in Ha/Bohr.'''
    return .5 * lib.einsum('axij,ji->ax', vj1, dm).real


def grad_ewald_exchange(cell, dm, s1):
    r'''Return the derivative of PySCF's ``exxdiv='ewald'`` K correction.

    PySCF adds ``K_G0 = madelung * S P S`` to the CDERI exchange matrix.
    The lattice and Madelung scalar are fixed for nuclear gradients, but both
    overlap factors move with the atom-centered AO basis.

    Args:
        cell : pyscf.pbc.gto.Cell
            Fixed three-dimensional cell.
        dm : ndarray of shape (nao, nao)
            AO density matrix.
        s1 : ndarray of shape (natm, 3, nao, nao)
            Full Hermitian overlap derivative for each atom.

    Returns:
        ndarray of shape (natm, 3, nao, nao)
            Derivative of the Ewald exchange correction.
    '''
    overlap = np.asarray(cell.pbc_intor('int1e_ovlp', hermi=1))
    madelung = float(pbc_tools.madelung(cell, np.zeros((1, 3))))
    return madelung * (
        lib.einsum('axij,jk,kl->axil', s1, dm, overlap)
        + lib.einsum('ij,jk,axkl->axil', overlap, dm, s1)
    )


def _real_if_close(value, tol=1e-11):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tol:
        return value.real
    return value
