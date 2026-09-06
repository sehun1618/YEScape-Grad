'''Sampled-k nonvariational ``4 Re Z^dagger V^lambda`` terms.'''

import numpy as np
from pyscf import lib

from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.grad.krhf_skeleton import real_gradient_term
from dcforge.cpu.pbc.grad.krhf_zvec import make_z_dm
from dcforge.cpu.pbc.kpts import make_occ_masks


def make_s1_oo_vo(cell, Co, Cv, s1, atmlst):
    r'''Build k-resolved :math:`S^\lambda_{oo}` and
    :math:`S^\lambda_{vo}` blocks.

    ``s1`` is PySCF's atom-owned AO-row overlap derivative with shape
    ``(nkpts,3,nao,nao)``.  Each returned block includes that row and its
    Hermitian-conjugate column.  The conjugation is essential for complex
    Bloch orbitals and cannot be replaced by a plain transpose.
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
    for atom_id, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        s1row = s1[:, :, p0:p1]
        s1co = lib.einsum('kxpv,kvi->kxpi', s1row, Co)
        oo = lib.einsum(
            'kpa,kxpi->kxai', Co[:, p0:p1].conj(), s1co)
        oo += oo.transpose(0, 1, 3, 2).conj()
        vo = lib.einsum(
            'kpa,kxpi->kxai', Cv[:, p0:p1].conj(), s1co)
        s1cv = lib.einsum('kxpv,kva->kxpa', s1row, Cv)
        vo += lib.einsum(
            'kxpa,kpi->kxai', s1cv.conj(), Co[:, p0:p1])
        s1oo[atom_id] = oo.transpose(1, 0, 2, 3)
        s1vo[atom_id] = vo.transpose(1, 0, 2, 3)
    return s1oo, s1vo


def make_s1_ao(cell, s1):
    r'''Expand AO-row overlap derivatives to complete atom derivatives.

    The sampled-k GDF Ewald exchange correction contains two overlap factors,
    so it needs both the atom-owned AO row and its Hermitian-conjugate column.

    Returns:
        ndarray
            ``S^lambda`` with shape ``(natm,3,nkpts,nao,nao)``.
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
    r'''Return :math:`-4\,\mathrm{Re}\sum_kw_kZ_k^\dagger F^{\lambda}_{vo,k}`.

    The one-electron term contracts the complete Hcore derivative with the
    Z-density.  PySCF FFTDF evaluates AO-row J/K derivatives for both the HF
    density and Z-density in one call; the two polarization orders restore
    the complete two-electron source-Fock derivative without storing a full
    atom-resolved Fock-derivative tensor.
    '''
    mf = mf_grad.base
    nkpts = len(mf.kpts)
    dz = make_z_dm(Co, Cv, zvec)
    de_hcore = real_gradient_term(
        mf_grad,
        -2 * lib.einsum(
            'axkij,kji->ax', h1ao[atmlst], dz) / nkpts,
        'de_fock_hcore')

    if isinstance(mf.with_df, GDF):
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
    for ia in atmlst:
        p0, p1 = aoslices[ia, 2:]
        value = -4 * lib.einsum(
            'xkij,kji->x', vhf0[:, :, p0:p1],
            dz[:, :, p0:p1]) / nkpts
        value -= 4 * lib.einsum(
            'xkij,kji->x', vhfz[:, :, p0:p1],
            dm0[:, :, p0:p1]) / nkpts
        de_jk.append(value)
    de_jk = real_gradient_term(
        mf_grad, de_jk, 'de_fock_jk', expect_real=False)
    mf_grad.de_fock_hcore = de_hcore
    mf_grad.de_fock_jk = de_jk
    return de_hcore + de_jk


def grad_ovlp_energy(mf_grad, s1vo, e_occ, zvec, nkpts):
    r'''Return :math:`4N_k^{-1}\mathrm{Re}\sum_kZ_k^\dagger
    S^\lambda_{vo,k}\epsilon_{o,k}`.'''
    zvec = np.asarray(zvec)
    e_occ = np.asarray(e_occ)
    se_vo = s1vo * e_occ[None, None, :, None, :]
    value = 4 * lib.einsum(
        'kvi,txkvi->tx', zvec.conj(), se_vo) / nkpts
    return real_gradient_term(
        mf_grad, value, 'de_ovlp_energy', expect_real=False)


def grad_ovlp_response(mf_grad, Co, Cv, zvec, s1oo, fvind, nkpts):
    r'''Return the moving-metric source-HF response term.

    With ``V_Z=G_HF[D_Z]``, self-adjointness of the HF J/K map gives

    ``4/Nk Re sum_k Tr[S^lambda_oo,k V_Z,oo,k]``.

    The final ``ij,ji`` contraction is required at complex k-points.
    '''
    vz = np.asarray(fvind(make_z_dm(Co, Cv, zvec)))
    vzoo = np.asarray([
        lib.dot(lib.dot(Co_k.conj().T, vz_k), Co_k)
        for Co_k, vz_k in zip(Co, vz, strict=True)
    ])
    value = 4 * lib.einsum(
        'axkij,kji->ax', s1oo, vzoo) / nkpts
    return real_gradient_term(mf_grad, value, 'de_ovlp_response')


def nonvariational_correction(mf_grad, Co, Cv, mo_energy, mo_occ, zvec,
                              s1, de_fock, fvind, atmlst):
    '''Assemble the three positive-RHS sampled-k Z-vector terms.'''
    nkpts = len(Co)
    s1oo, s1vo = make_s1_oo_vo(
        mf_grad.cell, Co, Cv, s1, atmlst)
    occupied_masks, _ = make_occ_masks(mo_occ)
    e_occ = np.asarray([
        energy[is_occ]
        for energy, is_occ in zip(
            np.asarray(mo_energy), occupied_masks, strict=True)
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
