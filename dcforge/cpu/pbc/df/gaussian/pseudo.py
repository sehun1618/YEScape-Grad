'''Exact-integral nonlocal GTH nuclear-gradient primitives.'''

import numpy as np
from pyscf import lib
from pyscf.pbc.gto.pseudo import pp_int


def get_vppnl_deriv(cell, kpts):
    r'''Return the Gaussian-DF nonlocal GTH matrix derivative.

    PySCF GDF and RSGDF evaluate each nonlocal channel from analytic projector
    half-integrals ``I`` as ``V_nl = I^dagger h I``.  This routine applies the
    product rule to the same half-integrals, including projector-center and
    atom-owned AO-center translations.

    Returns:
        ndarray of shape (natm, nkpts, 3, nao, nao)
            Exact-integral nonlocal pseudopotential response in Ha/Bohr.
    '''
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    fakecell, hl_blocks = pp_int.fake_cell_vnl(cell)
    values = pp_int._int_vnl(cell, fakecell, hl_blocks, kpts)
    derivative_intors = (
        'int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2')
    derivatives = pp_int._int_vnl(
        cell, fakecell, hl_blocks, kpts, derivative_intors, comp=3)

    # PySCF's first historical intor is ip1 rather than ip2.  Its own
    # vppnl_nuc_grad performs this sign conversion before contraction.
    if len(derivatives[0]) > 0:
        for k in range(len(kpts)):
            derivatives[0][k] *= -1

    nao = cell.nao_nr()
    v1_projector = np.zeros(
        (cell.natm, len(kpts), 3, nao, nao), dtype=np.complex128)
    v1_ao = np.zeros_like(v1_projector)
    aoslices = cell.aoslice_by_atom()
    for k in range(len(kpts)):
        offsets = [0, 0, 0]
        for shell, hl in enumerate(hl_blocks):
            l = int(fakecell.bas_angular(shell))
            nm = 2 * l + 1
            nl = np.asarray(hl).shape[0]
            proj, proj1 = _projector_shell(
                values, derivatives, k, offsets, nl, nm, nao)
            half1 = lib.einsum(
                'ixmp,ij,jmq->xpq', proj1.conj(), hl, proj)
            ia = int(fakecell._bas[shell, 0])
            v1_projector[ia, k] += half1 + half1.transpose(
                0, 2, 1).conj()
            for ja in range(cell.natm):
                p0, p1 = aoslices[ja, 2:]
                v1_ao[ja, k, :, p0:p1] -= half1[:, p0:p1]
                v1_ao[ja, k, :, :, p0:p1] -= half1[:, p0:p1].transpose(
                    0, 2, 1).conj()
    return v1_projector + v1_ao


def contract_vppnl_deriv(cell, densities, kpts=None, atmlst=None):
    r'''Contract nonlocal GTH derivatives with several densities at once.

    The projector half-integrals and their derivatives are generated once.
    A leading density-set contraction then produces only
    ``(nset,natm,3)`` output, avoiding the full atom-by-AO-square derivative
    retained by :func:`get_vppnl_deriv`.
    '''
    kpts = np.zeros((1, 3)) if kpts is None else np.asarray(
        kpts, dtype=float).reshape(-1, 3)
    nao = cell.nao_nr()
    batch = np.asarray(densities)
    if batch.ndim == 2:
        batch = batch[None, None]
    elif batch.ndim == 3:
        batch = batch[:, None] if len(kpts) == 1 else batch[None]
    if batch.shape[1:] != (len(kpts), nao, nao):
        raise ValueError(
            'nonlocal density batch must have shape '
            f'(nset,{len(kpts)},{nao},{nao}), got {batch.shape}')
    if atmlst is None:
        atmlst = range(cell.natm)
    atmlst = [int(atom_id) for atom_id in atmlst]

    fakecell, hl_blocks = pp_int.fake_cell_vnl(cell)
    values = pp_int._int_vnl(cell, fakecell, hl_blocks, kpts)
    derivative_intors = (
        'int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2')
    derivatives = pp_int._int_vnl(
        cell, fakecell, hl_blocks, kpts, derivative_intors, comp=3)
    if len(derivatives[0]) > 0:
        for k in range(len(kpts)):
            derivatives[0][k] *= -1

    gradient = np.zeros(
        (len(batch), cell.natm, 3), dtype=np.complex128)
    aoslices = cell.aoslice_by_atom()
    for k in range(len(kpts)):
        dm_hermitian = batch[:, k] + batch[:, k].transpose(0, 2, 1).conj()
        derivative_matrix = np.zeros((3, nao, nao), dtype=np.complex128)
        offsets = [0, 0, 0]
        for shell, hl in enumerate(hl_blocks):
            angular = int(fakecell.bas_angular(shell))
            projector_count = 2 * angular + 1
            radial_count = np.asarray(hl).shape[0]
            proj, proj1 = _projector_shell(
                values, derivatives, k, offsets,
                radial_count, projector_count, nao)
            half1 = lib.einsum(
                'ixmp,ij,jmq->xpq', proj1.conj(), hl, proj)
            derivative_matrix += half1
            atom_id = int(fakecell._bas[shell, 0])
            gradient[:, atom_id] += lib.einsum(
                'xpq,sqp->sx', half1, dm_hermitian)
        for atom_id, (_sh0, _sh1, p0, p1) in enumerate(aoslices):
            gradient[:, atom_id] -= lib.einsum(
                'xpq,sqp->sx', derivative_matrix[:, p0:p1],
                dm_hermitian[:, :, p0:p1])
    return gradient[:, atmlst].real


def _projector_shell(values, derivatives, k, offsets, nl, nm, nao):
    proj = np.empty((nl, nm, nao), dtype=np.complex128)
    proj1 = np.empty((nl, 3, nm, nao), dtype=np.complex128)
    for i in range(nl):
        p0 = offsets[i]
        p1 = p0 + nm
        proj[i] = np.asarray(values[i][k])[p0:p1]
        proj1[i] = np.asarray(derivatives[i][k])[:, p0:p1]
        offsets[i] = p1
    return proj, proj1
