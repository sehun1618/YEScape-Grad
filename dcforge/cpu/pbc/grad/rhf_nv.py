'''Nonvariational Gamma-point ``4 Z^dagger V^lambda`` terms.'''

import numpy as np
from pyscf import lib

from dcforge.cpu.pbc.df.gdf import GDF


def make_s1_oo_vo(cell, Co, Cv, s1, atmlst):
    '''Build full moving-basis ``S^lambda_oo`` and ``S^lambda_vo`` blocks.'''
    nocc = Co.shape[1]
    nvir = Cv.shape[1]
    s1oo = np.zeros((len(atmlst), 3, nocc, nocc), dtype=Co.dtype)
    s1vo = np.zeros((len(atmlst), 3, nvir, nocc), dtype=Co.dtype)
    aoslices = cell.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        s1co = lib.einsum('xpv,vi->xpi', s1[:, p0:p1], Co)
        s1oo[k] = lib.einsum('xpi,pj->xij', s1co, Co[p0:p1].conj())
        s1oo[k] += s1oo[k].transpose(0, 2, 1).conj()
        s1vo[k] = lib.einsum('xpi,pa->xai', s1co, Cv[p0:p1].conj())
        s1cv = lib.einsum('xpv,va->xpa', s1[:, p0:p1], Cv)
        s1vo[k] += lib.einsum('xpa,pi->xai', s1cv, Co[p0:p1].conj())
    return s1oo, s1vo


def grad_fock_direct(mf_grad, Co, Cv, zvec, dm0, h1ao, atmlst,
                     de_hcore=None):
    r'''Return ``-4 Z F^HF,lambda_vo`` without storing ``F^lambda``.

    The transition density ``D_Z=C_v Z C_o^dagger+h.c.`` converts the source
    Fock derivative to a cross-density integral derivative.  PySCF FFTDF
    ``get_jk_e1`` is evaluated for ``[P_HF,D_Z]`` in one call; the two
    polarization orders restore all four AO-center derivatives.
    '''
    mf = mf_grad.base
    dz = Cv @ zvec @ Co.conj().T
    dz = dz + dz.conj().T
    if isinstance(mf.with_df, GDF):
        if de_hcore is None:
            de_hcore = -2 * lib.einsum(
                'axij,ji->ax', h1ao[atmlst, :, 0], dz).real
        if mf_grad.vj1 is None:
            de_source = mf_grad._de_fock_source_contracted[atmlst]
            de_metric = mf_grad._de_fock_metric_contracted[atmlst]
        else:
            vhf1_source = mf_grad.vj1_source - .5 * mf_grad.vk1_source
            vhf1_metric = mf_grad.vj1_metric - .5 * mf_grad.vk1_metric
            de_source = -2 * lib.einsum(
                'axij,ji->ax', vhf1_source[atmlst], dz).real
            de_metric = -2 * lib.einsum(
                'axij,ji->ax', vhf1_metric[atmlst], dz).real
        vhf1_ewald = -.5 * mf_grad.vk1_ewald
        de_ewald = -2 * lib.einsum(
            'axij,ji->ax', vhf1_ewald[atmlst], dz).real
        mf_grad.de_fock_hcore = de_hcore
        mf_grad.de_fock_source = de_source
        mf_grad.de_fock_metric = de_metric
        mf_grad.de_fock_ewald = de_ewald
        return de_hcore + de_source + de_metric + de_ewald

    kpts = np.zeros((1, 3))
    dms = np.stack((dm0, dz))[:, None]
    vj, vk = mf.with_df.get_jk_e1(
        dms, kpts, exxdiv=mf.mf_hf.exxdiv)
    vhf0 = np.asarray(vj)[:, 0, 0] - .5 * np.asarray(vk)[:, 0, 0]
    vhfz = np.asarray(vj)[:, 1, 0] - .5 * np.asarray(vk)[:, 1, 0]

    de = np.zeros((len(atmlst), 3))
    aoslices = mf.cell.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de[k] = -2 * lib.einsum(
            'xij,ji->x', h1ao[ia, :, 0], dz).real
        de[k] -= 4 * lib.einsum(
            'xij,ji->x', vhf0[:, p0:p1], dz[:, p0:p1],
        ).real
        de[k] -= 4 * lib.einsum(
            'xij,ji->x', vhfz[:, p0:p1], dm0[:, p0:p1],
        ).real
    return de


def grad_ovlp_energy(s1vo, e_occ, zvec):
    '''Return ``+4 Z S^lambda_vo epsilon_o`` in Ha/Bohr.'''
    se_vo = s1vo * e_occ[None, None, None, :]
    return 4 * lib.einsum('ai,kxai->kx', zvec.conj(), se_vo).real


def grad_ovlp_response(Co, Cv, zvec, s1oo, fvind):
    '''Return the moving-metric HF response with one adjoint density.'''
    dz = Cv @ zvec @ Co.conj().T
    vz = fvind(dz + dz.conj().T)
    vzoo = Co.conj().T @ vz @ Co
    return 4 * lib.einsum('kxij,ij->kx', s1oo, vzoo).real


def nonvariational_correction(mf_grad, Co, Cv, mo_energy, zvec, s1,
                              de_fock, fvind, atmlst):
    '''Assemble the three periodic ``4 Z^dagger V^lambda`` terms.'''
    s1oo, s1vo = make_s1_oo_vo(
        mf_grad.cell, Co, Cv, s1, atmlst)
    e_occ = mo_energy[:Co.shape[1]]
    de_ovlp_energy = grad_ovlp_energy(s1vo, e_occ, zvec)
    de_ovlp_response = grad_ovlp_response(Co, Cv, zvec, s1oo, fvind)
    mf_grad.s1oo = s1oo
    mf_grad.s1vo = s1vo
    mf_grad.de_fock = de_fock
    mf_grad.de_ovlp_energy = de_ovlp_energy
    mf_grad.de_ovlp_response = de_ovlp_response
    mf_grad.de_nvar = de_fock + de_ovlp_energy + de_ovlp_response
    return mf_grad.de_nvar
