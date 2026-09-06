'''Finite-mesh analytic derivative of the nonlocal GTH potential.'''

import numpy as np
from pyscf import gto, lib
from pyscf.pbc.df import ft_ao
from pyscf.pbc.gto.pseudo import pp


def get_vppnl_deriv(cell, kpts):
    r'''Return ``dV_nl(k)/dR_A`` on the FFTDF reciprocal mesh.

    FFTDF evaluates each nonlocal GTH channel as

    ``V_nl(k) = vol^-1 B(k)^dagger h B(k)``.

    This function differentiates both the pseudopotential projector structure
    factor and the atom-owned Fourier AO columns.  It uses no displaced cell.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built cell carrying a GTH pseudopotential.
        kpts : ndarray of shape (nkpts, 3)
            Sampling points in reciprocal Bohr.

    Returns:
        ndarray of shape (natm, nkpts, 3, nao, nao)
            Complete nonlocal-pseudopotential derivative in Ha/Bohr.

    Notes:
        The projector-center derivative contains ``+i G`` whereas the AO
        translation contains ``-i(G+k)``.  They coincide only at Gamma after
        all product-rule contractions have been assembled.
    '''
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    Gv = np.asarray(cell.get_Gv(cell.mesh))
    SI = np.asarray(cell.get_SI(mesh=cell.mesh))
    helper, ptr = _projector_mol()
    nao = cell.nao_nr()
    v1 = np.zeros(
        (cell.natm, len(kpts), 3, nao, nao), dtype=np.complex128)

    for k, kpt in enumerate(kpts):
        v1[:, k] = _get_vppnl_deriv_kpt(
            cell, kpt, Gv, SI, helper, ptr)
    return v1 / cell.vol


def _get_vppnl_deriv_kpt(cell, kpt, Gv, SI, helper, ptr):
    Gk = Gv + kpt
    G_rad = np.linalg.norm(Gk, axis=1)
    aoG = ft_ao.ft_ao(cell, Gv, kpt=kpt) / np.sqrt(cell.vol)
    nao = cell.nao_nr()
    v1 = np.zeros((cell.natm, 3, nao, nao), dtype=np.complex128)

    for ia in range(cell.natm):
        for proj, hl in _projector_blocks(
                cell, ia, Gk, G_rad, helper, ptr):
            proj = SI[ia].conj() * proj
            half = lib.einsum('nmg,gp->nmp', proj, aoG)
            half_h = lib.einsum('ij,jmp->imp', hl, half)

            for ja in range(cell.natm):
                for x in range(3):
                    if ja == ia:
                        dproj = lib.einsum(
                            'nmg,gp->nmp', 1j * Gv[:, x] * proj,
                            aoG)
                        v1[ja, x] += _product_rule(
                            half, half_h, hl, dproj)
                    daoG = _ao_center_deriv(cell, aoG, Gk, ja, x)
                    dhalf = lib.einsum('nmg,gp->nmp', proj, daoG)
                    v1[ja, x] += _product_rule(
                        half, half_h, hl, dhalf)
    return v1


def _projector_blocks(cell, ia, Gk, G_rad, helper, ptr):
    symbol = cell.atom_symbol(ia)
    if symbol not in cell._pseudo:
        return
    for l, channel in enumerate(cell._pseudo[symbol][5:]):
        rl, nl, hl = channel
        if nl <= 0:
            continue
        helper._bas[0, gto.ANG_OF] = l
        helper._env[ptr + 3] = .5 * rl ** 2
        helper._env[ptr + 4] = rl ** (l + 1.5) * np.pi ** 1.25
        Ylm = helper.eval_gto('GTOval', Gk)
        proj = np.empty((nl, 2 * l + 1, len(Gk)), dtype=np.complex128)
        for i in range(nl):
            proj[i] = Ylm.T * pp._qli(G_rad * rl, l, i)
        yield proj, np.asarray(hl)


def _ao_center_deriv(cell, aoG, Gk, ia, x):
    p0, p1 = cell.aoslice_by_atom()[ia, 2:]
    daoG = np.zeros_like(aoG)
    daoG[:, p0:p1] = -1j * Gk[:, x, None] * aoG[:, p0:p1]
    return daoG


def _product_rule(half, half_h, hl, dhalf):
    dhalf_h = lib.einsum('ij,jmp->imp', hl, dhalf)
    return (
        lib.einsum('imp,imq->pq', dhalf.conj(), half_h)
        + lib.einsum('imp,imq->pq', half.conj(), dhalf_h)
    )


def _projector_mol():
    mol = gto.Mole()
    mol._atm = np.zeros((1, gto.ATM_SLOTS), dtype=np.int32)
    mol._bas = np.zeros((1, gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    mol._env = np.zeros(ptr + 10)
    mol._bas[0, gto.NPRIM_OF] = 1
    mol._bas[0, gto.NCTR_OF] = 1
    mol._bas[0, gto.PTR_EXP] = ptr + 3
    mol._bas[0, gto.PTR_COEFF] = ptr + 4
    return mol, ptr
