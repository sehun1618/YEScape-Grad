'''Sampled-k nonvariational ``4 Re Z^dagger V^lambda`` terms.'''

import numpy as np
from pyscf import lib

from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.df.rsdf import RSDF
from dcforge.cpu.pbc.grad.krhf_skeleton import real_gradient_term
from dcforge.cpu.pbc.grad.krhf_zvec import make_z_dm


def make_s1_oo_vo(cell, Co, Cv, s1, atmlst):
    r'''Build k-resolved moving-basis :math:`S^\lambda_{oo}` and
    :math:`S^\lambda_{vo}` blocks.

    Args:
        cell : pyscf.pbc.gto.Cell
            Active fixed periodic Cell.
        Co, Cv : sequence of ndarray
            Complex occupied and virtual coefficient blocks.
        s1 : ndarray of shape (nkpts,3,nao,nao)
            PySCF AO-row overlap derivative ``-<nabla chi|chi>``.
        atmlst : sequence of int
            Nuclear centers requested by the gradient object.

    Returns:
        tuple
            ``s1oo`` with shape ``(natm_selected,3,nkpts,nocc,nocc)`` and
            ``s1vo`` with shape ``(natm_selected,3,nkpts,nvir,nocc)``.

    Notes:
        Each block includes both the atom-owned AO row and its Hermitian
        conjugate column.  This is required at non-Gamma k-points, where a
        plain transpose would lose the Bloch-orbital conjugation.
    '''
    Co = np.asarray(Co)
    Cv = np.asarray(Cv)
    nkpts = len(Co)
    nocc = Co.shape[2]
    nvir = Cv.shape[2]
    s1oo = np.zeros(
        (len(atmlst), 3, nkpts, nocc, nocc), dtype=Co.dtype)
    s1vo = np.zeros(
        (len(atmlst), 3, nkpts, nvir, nocc), dtype=Co.dtype)
    aoslices = cell.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        s1row = s1[:, :, p0:p1]
        s1co = lib.einsum('kxpv,kvi->kxpi', s1row, Co)
        # Project the atom-owned AO row as C_left^H S_row C_o.  Keeping the
        # conjugated coefficient on the left is essential at non-Gamma
        # k-points; the historical spelling returned the transpose of S_oo.
        oo = lib.einsum(
            'kpa,kxpi->kxai', Co[:, p0:p1].conj(), s1co)
        oo += oo.transpose(0, 1, 3, 2).conj()
        vo = lib.einsum(
            'kpa,kxpi->kxai', Cv[:, p0:p1].conj(), s1co)
        s1cv = lib.einsum('kxpv,kva->kxpa', s1row, Cv)
        # The conjugate-column contribution is
        # C_v^H S_row^H C_o = (S_row C_v)^H C_o.  Conjugating C_o instead
        # would be valid only for real orbitals and silently corrupts the
        # complex sampled-k CPHF right-hand side.
        vo += lib.einsum(
            'kxpa,kpi->kxai', s1cv.conj(), Co[:, p0:p1])
        s1oo[k] = oo.transpose(1, 0, 2, 3)
        s1vo[k] = vo.transpose(1, 0, 2, 3)
    return s1oo, s1vo


def make_s1_ao(cell, s1):
    r'''Expand AO-row overlap derivatives to complete atom derivatives.

    Args:
        cell : pyscf.pbc.gto.Cell
            Active periodic Cell.
        s1 : ndarray of shape (nkpts,3,nao,nao)
            PySCF row derivative ``-<nabla chi_mu|chi_nu>``.

    Returns:
        ndarray of shape (natm,3,nkpts,nao,nao)
            Complete :math:`S^\lambda_k`, including the atom-owned AO row
            and its Hermitian-conjugate column.

    Notes:
        The complete matrix is needed by the sampled-k GDF Ewald exchange
        correction ``K_G0=madelung*S*P*S``.  FFTDF uses row-contracted
        derivatives and does not materialize this tensor.
    '''
    s1 = np.asarray(s1)
    if s1.ndim != 4 or s1.shape[1] != 3:
        raise ValueError(
            'AO-row overlap derivative must have shape (nkpts,3,nao,nao)')
    nkpts, _, nao, nao1 = s1.shape
    if nao != nao1 or nao != cell.nao_nr():
        raise ValueError('overlap derivative AO dimensions are inconsistent')
    full = np.zeros(
        (cell.natm, 3, nkpts, nao, nao), dtype=s1.dtype)
    for ia, aoslice in enumerate(cell.aoslice_by_atom()):
        p0, p1 = aoslice[2:]
        full[ia, :, :, p0:p1] += s1[:, :, p0:p1].transpose(
            1, 0, 2, 3)
        full[ia, :, :, :, p0:p1] += s1[:, :, p0:p1].transpose(
            1, 0, 3, 2).conj()
    return full


def grad_fock_direct(mf_grad, Co, Cv, zvec, dm0, h1ao, atmlst):
    r'''Return sampled-k ``-4 Re sum_k w_k Z_k^dagger F^HF,lambda_vo``.

    The one-electron part contracts the complete Hcore derivative with
    ``D_Z``.  For J/K, PySCF FFTDF AO-row derivatives are evaluated for both
    ``P_HF`` and ``D_Z`` in one call.  The two polarization orders and the
    conjugate AO-row side restore the complete source-Fock derivative without
    storing ``(natm,3,nkpts,nao,nao)`` J/K matrices.
    '''
    mf = mf_grad.base
    nkpts = len(mf.kpts)
    dz = make_z_dm(Co, Cv, zvec)
    de_hcore = real_gradient_term(
        mf_grad,
        -2 * lib.einsum(
            'axkij,kji->ax', h1ao[atmlst], dz) / nkpts,
        'de_fock_hcore')

    if isinstance(mf.with_df, (GDF, RSDF)):
        vhf1_source = mf_grad.vj1_source - .5 * mf_grad.vk1_source
        vhf1_metric = mf_grad.vj1_metric - .5 * mf_grad.vk1_metric
        vhf1_ewald = -.5 * mf_grad.vk1_ewald
        de_source = real_gradient_term(
            mf_grad,
            -2 * lib.einsum(
                'axkij,kji->ax', vhf1_source[atmlst], dz) / nkpts,
            'de_fock_source')
        de_metric = real_gradient_term(
            mf_grad,
            -2 * lib.einsum(
                'axkij,kji->ax', vhf1_metric[atmlst], dz) / nkpts,
            'de_fock_metric')
        de_ewald = real_gradient_term(
            mf_grad,
            -2 * lib.einsum(
                'axkij,kji->ax', vhf1_ewald[atmlst], dz) / nkpts,
            'de_fock_ewald')
        mf_grad.de_fock_hcore = de_hcore
        mf_grad.de_fock_source = de_source
        mf_grad.de_fock_metric = de_metric
        mf_grad.de_fock_ewald = de_ewald
        mf_grad.de_fock_jk = de_source + de_metric + de_ewald
        return de_hcore + mf_grad.de_fock_jk

    dms = np.stack((dm0, dz))
    vj1, vk1 = mf.with_df.get_jk_e1(
        dms, mf.kpts, exxdiv=mf.mf_hf.exxdiv)
    vhf0 = np.asarray(vj1)[:, 0] - .5 * np.asarray(vk1)[:, 0]
    vhfz = np.asarray(vj1)[:, 1] - .5 * np.asarray(vk1)[:, 1]
    de_jk = []
    aoslices = mf.cell.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        value = (
            -4 * lib.einsum(
                'xkij,kji->x', vhf0[:, :, p0:p1],
                dz[:, :, p0:p1]) / nkpts)
        value -= (
            4 * lib.einsum(
                'xkij,kji->x', vhfz[:, :, p0:p1],
                dm0[:, :, p0:p1]) / nkpts)
        de_jk.append(value)
    de_jk = real_gradient_term(
        mf_grad, de_jk, 'de_fock_jk', expect_real=False)
    mf_grad.de_fock_hcore = de_hcore
    mf_grad.de_fock_jk = de_jk
    return de_hcore + de_jk


def grad_ovlp_energy(mf_grad, s1vo, e_occ, zvec, nkpts):
    r'''Return ``+4/Nk Re sum_k Z_k^dagger S^lambda_vo,k epsilon_o,k``.'''
    zvec = np.asarray(zvec)
    e_occ = np.asarray(e_occ)
    se_vo = s1vo * e_occ[None, None, :, None, :]
    value = (
        4 * lib.einsum(
            'kvi,txkvi->tx', zvec.conj(), se_vo) / nkpts)
    return real_gradient_term(
        mf_grad, value, 'de_ovlp_energy', expect_real=False)


def grad_ovlp_response(mf_grad, Co, Cv, zvec, s1oo, fvind, nkpts):
    r'''Return the sampled-k moving-metric HF response term.

    With ``D_Z=C_v Z C_o^H+h.c.`` and ``V_Z=G_HF[D_Z]``, self-adjointness of
    the HF J/K map gives

    ``+4/Nk Re sum_k Tr[S^lambda_oo,k V_Z,oo,k]``.

    The final contraction uses matrix indices ``ij,ji``.  At real Gamma this
    is indistinguishable from the historical symmetric ``ij,ij`` spelling;
    at complex k-points the transpose is mathematically essential.
    '''
    dz = make_z_dm(Co, Cv, zvec)
    vz = np.asarray(fvind(dz))
    vzoo = []
    for Co_k, vz_k in zip(Co, vz, strict=True):
        cov = lib.dot(np.asarray(Co_k).conj().T, vz_k)
        vzoo.append(lib.dot(cov, Co_k))
    vzoo = np.asarray(vzoo)
    value = (
        4 * lib.einsum(
            'axkij,kji->ax', s1oo, vzoo) / nkpts)
    return real_gradient_term(mf_grad, value, 'de_ovlp_response')


def nonvariational_correction(mf_grad, Co, Cv, mo_energy, mo_occ, zvec,
                              s1, de_fock, fvind, atmlst):
    '''Assemble the three sampled-k positive-RHS Z-vector gradient terms.'''
    nkpts = len(Co)
    s1oo, s1vo = make_s1_oo_vo(
        mf_grad.cell, Co, Cv, s1, atmlst)
    e_occ = np.asarray([
        energy[occupation > 0]
        for energy, occupation in zip(
            np.asarray(mo_energy), np.asarray(mo_occ), strict=True)
    ])
    de_ovlp_energy = grad_ovlp_energy(
        mf_grad, s1vo, e_occ, zvec, nkpts)
    de_ovlp_response = grad_ovlp_response(
        mf_grad, Co, Cv, zvec, s1oo, fvind, nkpts)
    mf_grad.s1oo = s1oo
    mf_grad.s1vo = s1vo
    mf_grad.de_fock = de_fock
    mf_grad.de_ovlp_energy = de_ovlp_energy
    mf_grad.de_ovlp_response = de_ovlp_response
    mf_grad.de_nvar = de_fock + de_ovlp_energy + de_ovlp_response
    return mf_grad.de_nvar
