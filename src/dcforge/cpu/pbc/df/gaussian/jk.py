'''Gamma and sampled-k Gaussian-DF J/K analytic derivatives.'''

import numpy as np
from pyscf import lib
from pyscf.pbc import tools as pbc_tools

from dcforge.cpu.pbc.df.gaussian.storage import (
    get_cderi_deriv_pair,
    get_cderi_pair,
)


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


def grad_jk_kpts(cderi, cderi1, dm):
    r'''Return sampled-k fixed-density GDF ``J^lambda`` and ``K^lambda``.

    Args:
        cderi : mapping
            Canonical lower-triangular ``(ki,kj)`` CDERI blocks with shape
            ``(naux,nao,nao)``.
        cderi1 : mapping
            Matching derivative blocks with shape
            ``(natm,3,naux,nao,nao)``.
        dm : ndarray of shape (nkpts,nao,nao)
            Closed-shell k-resolved AO density including occupation two.

    Returns:
        tuple
            ``(vj1,vk1)`` with shape ``(natm,3,nkpts,nao,nao)``.  Each
            matrix contains the complete two-sided CDERI derivative and the
            same internal ``1/nkpts`` normalization as PySCF GDF ``get_jk``.
    '''
    dm = np.asarray(dm)
    if dm.ndim != 3 or dm.shape[1] != dm.shape[2]:
        raise ValueError('sampled-k density must have shape (nkpts,nao,nao)')
    nkpts, nao, _ = dm.shape
    try:
        first = np.asarray(next(iter(cderi1.values())))
    except StopIteration as err:
        raise ValueError('CDERI derivative mapping cannot be empty') from err
    if first.ndim != 5 or first.shape[1] != 3:
        raise ValueError(
            'CDERI derivatives must have shape (natm,3,naux,nao,nao)')
    natm = first.shape[0]
    shape = (natm, 3, nkpts, nao, nao)
    dtype = np.result_type(dm, *cderi.values(), *cderi1.values())
    vj1 = np.zeros(shape, dtype=dtype)
    vk1 = np.zeros(shape, dtype=dtype)

    rho = sum(
        lib.einsum(
            'Pij,ji->P', get_cderi_pair(cderi, k, k), dm[k])
        for k in range(nkpts))
    for ia in range(natm):
        for xyz in range(3):
            rho1 = sum(
                lib.einsum(
                    'Pij,ji->P',
                    get_cderi_deriv_pair(cderi1, k, k, ia, xyz),
                    dm[k])
                for k in range(nkpts))
            for ki in range(nkpts):
                c_ii = get_cderi_pair(cderi, ki, ki)
                c1_ii = get_cderi_deriv_pair(
                    cderi1, ki, ki, ia, xyz)
                vj1[ia, xyz, ki] = (
                    lib.einsum('P,Pij->ij', rho, c1_ii)
                    + lib.einsum('P,Pij->ij', rho1, c_ii)
                ) / nkpts

                exchange = np.zeros((nao, nao), dtype=dtype)
                for kj in range(nkpts):
                    c_ij = get_cderi_pair(cderi, ki, kj)
                    c_ji = get_cderi_pair(cderi, kj, ki)
                    c1_ij = get_cderi_deriv_pair(
                        cderi1, ki, kj, ia, xyz)
                    c1_ji = get_cderi_deriv_pair(
                        cderi1, kj, ki, ia, xyz)
                    exchange += lib.einsum(
                        'Pij,jk,Pkl->il', c1_ij, dm[kj], c_ji)
                    exchange += lib.einsum(
                        'Pij,jk,Pkl->il', c_ij, dm[kj], c1_ji)
                vk1[ia, xyz, ki] = exchange / nkpts
    return _real_if_close(vj1), _real_if_close(vk1)


def grad_j_energy_kpts(dm, vj1):
    r'''Return ``0.5/Nk sum_k Tr[P_k J_k^lambda]`` in Ha/Bohr.'''
    dm = np.asarray(dm)
    vj1 = np.asarray(vj1)
    if vj1.shape[2:] != (len(dm), dm.shape[1], dm.shape[2]):
        raise ValueError('sampled-k J derivative and density shapes differ')
    return (
        .5 * lib.einsum('axkij,kji->ax', vj1, dm).real / len(dm))


def grad_ewald_exchange_kpts(cell, kpts, dm, s1):
    r'''Return sampled-k PySCF ``exxdiv='ewald'`` exchange derivative.

    Args:
        cell, kpts, dm
            Fixed Cell, explicit full-BZ k-points and closed-shell density.
        s1 : ndarray of shape (natm,3,nkpts,nao,nao)
            Complete Hermitian overlap derivative.

    Returns:
        ndarray
            ``K_G0^lambda`` with the same shape as ``s1``.
    '''
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    dm = np.asarray(dm)
    s1 = np.asarray(s1)
    if s1.shape[2:] != (len(kpts), dm.shape[1], dm.shape[2]):
        raise ValueError('sampled-k overlap derivative shape is inconsistent')
    overlap = np.asarray(cell.pbc_intor(
        'int1e_ovlp', hermi=1, kpts=kpts))
    madelung = float(pbc_tools.madelung(cell, kpts))
    return madelung * (
        lib.einsum('axkij,kjl,klm->axkim', s1, dm, overlap)
        + lib.einsum('kij,kjl,axklm->axkim', overlap, dm, s1)
    )


def _real_if_close(value, tol=1e-11):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tol:
        return value.real
    return value
