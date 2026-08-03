'''Full fixed-density FFTDF J/K nuclear derivatives on the GPU.

GPU4PySCF 1.8.0 provides a contracted FFT Coulomb derivative but does not
provide the complete exchange derivative matrix required by the DC-DFT
Z-vector term.  This module evaluates the finite-grid product rule directly
with GPU AO collocation and FFTs.  It is the Stage-2 correctness route; later
profiling may replace its contraction order without changing this interface.
'''

import cupy as cp
import numpy as np
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.dft.numint import eval_ao_kpts


def _ao_grid_data(cell, kpts):
    '''Return AO values and first spatial derivatives on ``cell.mesh``.'''
    coords_host = cell.gen_uniform_grids(cell.mesh)
    evaluated = eval_ao_kpts(cell, coords_host, kpts=kpts, deriv=1)
    ao = [value[0] for value in evaluated]
    ao1 = [value[1:4] for value in evaluated]
    return cp.asarray(coords_host), ao, ao1


def _atom_ao_deriv(cell, ao, ao1, ia, xyz):
    '''Return ``d chi_mu / dR_I`` for one atom and Cartesian component.'''
    p0, p1 = np.asarray(cell.aoslice_by_atom(), dtype=int)[ia, 2:]
    output = []
    for value, deriv in zip(ao, ao1, strict=True):
        value1 = cp.zeros_like(value)
        # AO-center and electronic-coordinate derivatives have opposite signs.
        value1[:, p0:p1] = -deriv[xyz, :, p0:p1]
        output.append(value1)
    return output


def _density(ao, dms):
    '''Return the k-point averaged real-space AO density.'''
    rho = cp.zeros(ao[0].shape[0], dtype=cp.complex128)
    for ao_k, dm_k in zip(ao, dms, strict=True):
        rho += cp.einsum('gi,ij,gj->g', ao_k.conj(), dm_k, ao_k)
    return rho / len(ao)


def _density_deriv(ao, ao1, dms):
    '''Return the AO-center derivative of the k-averaged density.'''
    rho1 = cp.zeros(ao[0].shape[0], dtype=cp.complex128)
    for ao_k, ao1_k, dm_k in zip(ao, ao1, dms, strict=True):
        rho1 += cp.einsum('gi,ij,gj->g', ao1_k.conj(), dm_k, ao_k)
        rho1 += cp.einsum('gi,ij,gj->g', ao_k.conj(), dm_k, ao1_k)
    return rho1 / len(ao)


def _hartree_potential(cell, rho):
    '''Apply the finite reciprocal Coulomb kernel including grid weight.'''
    coulG = tools.get_coulG(cell, mesh=cell.mesh)
    vj = tools.ifft(tools.fft(rho, cell.mesh) * coulG, cell.mesh)
    return vj * (cell.vol / len(coulG))


def _coulomb_deriv(cell, kpts, dms, ao, ao1):
    '''Return full ``dJ/dR`` matrices with ``(atom,k,xyz,ao,ao)`` layout.'''
    rho = _density(ao, dms)
    vj = _hartree_potential(cell, rho)
    nkpts = len(kpts)
    nao = cell.nao_nr()
    output = cp.zeros(
        (cell.natm, nkpts, 3, nao, nao), dtype=cp.complex128)

    for ia in range(cell.natm):
        for xyz in range(3):
            dao = _atom_ao_deriv(cell, ao, ao1, ia, xyz)
            rho1 = _density_deriv(ao, dao, dms)
            vj1 = _hartree_potential(cell, rho1)
            for k, (ao_k, dao_k) in enumerate(zip(ao, dao, strict=True)):
                value = dao_k.conj().T @ (vj[:, None] * ao_k)
                value += ao_k.conj().T @ (vj[:, None] * dao_k)
                value += ao_k.conj().T @ (vj1[:, None] * ao_k)
                output[ia, k, xyz] = value
    return output


def _exchange_component(cell, kpts, dms, coords, ao, ao1, ia, xyz,
                        exxdiv, with_df):
    '''Differentiate every AO occurrence in one Gamma/sampled-k K build.'''
    dao = _atom_ao_deriv(cell, ao, ao1, ia, xyz)
    nkpts = len(kpts)
    nao = cell.nao_nr()
    ngrids = len(coords)
    weight = cell.vol / ngrids / nkpts
    output = cp.zeros((nkpts, nao, nao), dtype=cp.complex128)

    for ks, (ao_s, dao_s, dm_s) in enumerate(zip(ao, dao, dms, strict=True)):
        ao_s_t = cp.ascontiguousarray(ao_s.T)
        dao_s_t = cp.ascontiguousarray(dao_s.T)
        dm_ao = dm_s @ ao_s_t.conj()
        dm_dao = dm_s @ dao_s_t.conj()
        for kt, (ao_t, dao_t) in enumerate(zip(ao, dao, strict=True)):
            q = kpts[ks] - kpts[kt]
            coulG = tools.get_coulG(
                cell, q, exxdiv, with_df, cell.mesh, kpts=kpts)
            if np.max(np.abs(q)) < 1e-12:
                phase = cp.asarray(1.0)
            else:
                phase = cp.exp(-1j * coords @ cp.asarray(q))
            phase_conj = phase.conj()
            ao_t_t = cp.ascontiguousarray(ao_t.T)
            dao_t_t = cp.ascontiguousarray(dao_t.T)

            for p in range(nao):
                value = cp.zeros(ngrids, dtype=cp.complex128)
                value1 = cp.zeros_like(value)
                for qidx in range(nao):
                    rho_pq = ao_t_t[p].conj() * phase * ao_s_t[qidx]
                    v_pq = tools.ifft(
                        tools.fft(rho_pq, cell.mesh) * coulG, cell.mesh)
                    rho1_pq = (
                        dao_t_t[p].conj() * phase * ao_s_t[qidx]
                        + ao_t_t[p].conj() * phase * dao_s_t[qidx]
                    )
                    v1_pq = tools.ifft(
                        tools.fft(rho1_pq, cell.mesh) * coulG, cell.mesh)
                    value += v_pq * dm_ao[qidx]
                    value1 += v1_pq * dm_ao[qidx]
                    value1 += v_pq * dm_dao[qidx]

                weighted = value * phase_conj
                weighted1 = value1 * phase_conj
                output[kt, p] += weight * (
                    weighted1 @ ao_t + weighted @ dao_t)
    return output


def _exchange_deriv(cell, kpts, dms, coords, ao, ao1, exxdiv, with_df):
    '''Return full ``dK/dR`` matrices with ``(atom,k,xyz,ao,ao)`` layout.'''
    output = cp.zeros(
        (cell.natm, len(kpts), 3, cell.nao_nr(), cell.nao_nr()),
        dtype=cp.complex128)
    for ia in range(cell.natm):
        for xyz in range(3):
            output[ia, :, xyz] = _exchange_component(
                cell, kpts, dms, coords, ao, ao1, ia, xyz,
                exxdiv, with_df)
    return output


def grad_jk(cell, dm, exxdiv='ewald', with_df=None):
    '''Return full fixed-density FFTDF J/K derivative matrices on the GPU.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built Gamma-point periodic cell.
        dm : cupy.ndarray of shape (nao, nao)
            AO density held fixed while differentiating the finite FFT
            representation.

    Kwargs:
        exxdiv : str or None
            Exchange-divergence convention used by source RHF.
        with_df : dcforge.gpu.pbc.df.fft.FFTDF
            Active source/target FFTDF provider.

    Returns:
        tuple
            ``(vj1, vk1)`` with shape ``(natm, 3, nao, nao)``.  Each is the
            complete AO matrix derivative, not an atom-row derivative.
    '''
    if with_df is None:
        raise ValueError('FFTDF derivative requires the active with_df provider')
    kpts = np.zeros((1, 3))
    dm = cp.asarray(dm, order='C')[None]
    coords, ao, ao1 = _ao_grid_data(cell, kpts)
    vj1 = _coulomb_deriv(cell, kpts, dm, ao, ao1)
    vk1 = _exchange_deriv(
        cell, kpts, dm, coords, ao, ao1, exxdiv, with_df)
    return vj1[:, 0], vk1[:, 0]
