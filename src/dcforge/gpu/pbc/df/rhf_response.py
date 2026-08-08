r'''Real-Gamma Gaussian-DF source-RHF response contractions.

The public PySCF response contract maps an AO transition density to
``J[D] - K[D]/2``.  In the RHF orbital Hessian used by a Z-vector solve,
however, every trial density already has the known cross-factor form

.. math::

   D[X] = 2\{(C_v X) C_o^T + C_o (C_v X)^T\}.

The production action contracts only ``(J[D]-K[D]/2) C_occ`` before the final
virtual projection.  Complete AO Coulomb and exchange matrices are therefore
not materialized for every Krylov vector.  The previous factor-aware AO-matrix
implementation remains available as :func:`gen_vind_ao_oracle`; it is an
explicit numerical oracle, not the production default.
'''

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.df.fft_jk import _ewald_exxdiv_for_G0
from gpu4pyscf.pbc.gto.int1e import int1e_ovlp
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero


def gen_vind(mf_hf, Co, Cv):
    r'''Return the direct-VO Gaussian-DF RHF Hessian action.

    Args:
        mf_hf : gpu4pyscf.pbc.scf.hf.RHF
            Converged real-Gamma source RHF.  Its ``with_df`` object must
            expose GPU4PySCF's compressed Gamma CDERI and block iterator.
        Co, Cv : cupy.ndarray
            Occupied and virtual source-RHF coefficient matrices.

    Returns:
        callable
            Function mapping one or more ``X[a,i]`` trial amplitudes to
            ``C_v^T (J[D[X]] - K[D[X]]/2) C_o`` with the input shape.

    Notes:
        For one symmetric real-Gamma CDERI factor, define

        ``A_P = L_P C_o``, ``B_P = L_P (C_v X)`` and
        ``rho_P = Tr(C_o^T B_P)``.  The action accumulates

        ``(J-K/2) C_o = sum_P s_P [4 rho_P A_P``
        ``                            - B_P(A_P^T C_o)``
        ``                            - A_P(B_P^T C_o)]``.

        Its largest accumulated potential is ``(nao,nocc)`` rather than two
        ``(nao,nao)`` AO matrices.  :func:`gen_vind_ao_oracle` retains those
        matrices for regression tests and numerical diagnosis.
    '''
    provider = mf_hf.with_df
    Co = _real_gamma_matrix(Co, 'Co')
    Cv = _real_gamma_matrix(Cv, 'Cv')
    if Co.shape[0] != Cv.shape[0]:
        raise ValueError('occupied and virtual AO dimensions differ')
    _check_provider(provider, Co.shape[0])
    nvir = int(Cv.shape[1])
    nocc = int(Co.shape[1])
    ewald = _make_ewald_intermediate(provider, Co, mf_hf.exxdiv)

    def vind(x):
        shape = cp.asarray(x).shape
        amplitudes = cp.asarray(x, dtype=cp.float64).reshape(-1, nvir, nocc)
        values = [_contract_one_direct_vo(
            provider, Co, Cv, amplitude, ewald)
                  for amplitude in amplitudes]
        return cp.stack(values).reshape(shape)

    vind.backend = 'gaussian_df_direct_vo'
    return vind


def gen_vind_ao_oracle(mf_hf, Co, Cv):
    r'''Return the former factor-aware full-AO response as an oracle.

    Args:
        mf_hf, Co, Cv
            Same objects accepted by :func:`gen_vind`.

    Returns:
        callable
            Numerically independent CDERI action that constructs complete AO
            ``J[D[X]]`` and ``K[D[X]]`` matrices before their VO projection.

    Notes:
        This route is retained deliberately for parity tests and diagnosis.
        It is not selected by the production gradient driver.
    '''
    provider = mf_hf.with_df
    Co = _real_gamma_matrix(Co, 'Co')
    Cv = _real_gamma_matrix(Cv, 'Cv')
    if Co.shape[0] != Cv.shape[0]:
        raise ValueError('occupied and virtual AO dimensions differ')
    _check_provider(provider, Co.shape[0])
    nvir = int(Cv.shape[1])
    nocc = int(Co.shape[1])

    def vind(x):
        shape = cp.asarray(x).shape
        amplitudes = cp.asarray(x, dtype=cp.float64).reshape(-1, nvir, nocc)
        values = [_contract_one_ao_oracle(
            provider, Co, Cv, amplitude, mf_hf.exxdiv)
                  for amplitude in amplitudes]
        return cp.stack(values).reshape(shape)

    vind.backend = 'gaussian_df_cross_factor_ao_oracle'
    return vind


def _contract_one_direct_vo(provider, Co, Cv, amplitude, ewald):
    '''Contract one trial through occupied columns and then into VO space.'''
    left = contract('pa,ai->pi', Cv, amplitude)
    nao, nocc = left.shape
    response_occ = cp.zeros((nao, nocc), dtype=cp.float64)
    naux = int(provider.get_naoaux())
    # One unpacked CDERI block, A_P, B_P and one reusable occupied-occupied
    # contraction are live together.  Include every per-auxiliary workspace
    # in the inherited provider's device-memory block planner.
    extra = 2 * nao * nocc + nocc * nocc
    blksize = provider.get_blksize(extra=extra)
    aux_iter = lib.prange(0, naux, blksize)
    for cderi, _packed, sign in provider.loop_gamma_point(
            blksize, unpack=True, aux_iter=aux_iter):
        cderi = cp.asarray(cderi)
        b_p = contract('Lpq,qi->Lpi', cderi, left)
        a_p = contract('Lpq,qi->Lpi', cderi, Co)

        rho = contract('Lpi,pi->L', b_p, Co)
        contract('L,Lpi->pi', rho, a_p, alpha=4.0 * sign,
                 beta=1.0, out=response_occ)

        # K[D[X]] = 2 sum_P (B_P A_P^T + A_P B_P^T).  Multiplication by
        # -1/2 in the restricted response cancels that leading factor two.
        occupied_metric = contract('Lpi,pj->Lij', a_p, Co)
        contract('Lpi,Lij->pj', b_p, occupied_metric, alpha=-sign,
                 beta=1.0, out=response_occ)
        del occupied_metric
        occupied_metric = contract('Lpi,pj->Lij', b_p, Co)
        contract('Lpi,Lij->pj', a_p, occupied_metric, alpha=-sign,
                 beta=1.0, out=response_occ)

    if ewald is not None:
        overlap, overlap_occ, madelung = ewald
        overlap_left = contract('pq,qi->pi', overlap, left)
        occ_metric = contract('pi,pj->ij', Co, overlap_occ)
        left_metric = contract('pi,pj->ij', left, overlap_occ)
        contract('pi,ij->pj', overlap_left, occ_metric, alpha=-madelung,
                 beta=1.0, out=response_occ)
        contract('pi,ij->pj', overlap_occ, left_metric, alpha=-madelung,
                 beta=1.0, out=response_occ)

    return contract('ap,pi->ai', Cv.T, response_occ)


def _contract_one_ao_oracle(provider, Co, Cv, amplitude, exxdiv):
    '''Contract one trial through complete AO J/K matrices for parity tests.'''
    left = contract('pa,ai->pi', Cv, amplitude)
    nao, nocc = left.shape
    density = 2.0 * (
        contract('pi,qi->pq', left, Co)
        + contract('pi,qi->pq', Co, left))
    vj = cp.zeros((nao, nao), dtype=cp.float64)
    vk = cp.zeros_like(vj)
    naux = int(provider.get_naoaux())
    # One unpacked CDERI block plus C*left and C*Co are live together.  The
    # inherited planner accounts for exactly those per-auxiliary workspaces.
    blksize = provider.get_blksize(extra=2 * nao * nocc)
    aux_iter = lib.prange(0, naux, blksize)
    for cderi, _packed, sign in provider.loop_gamma_point(
            blksize, unpack=True, aux_iter=aux_iter):
        cderi = cp.asarray(cderi)
        c_left = contract('Lpq,qi->Lpi', cderi, left)
        c_occ = contract('Lpq,qi->Lpi', cderi, Co)

        # Tr[C_L D[X]] = 4 Tr[C_L (C_v X) C_o^T] at real Gamma because
        # C_L, D[X] are symmetric.  Keeping this contraction factorized
        # avoids the generic O(naux*nao**3) exchange path.
        rho = contract('Lpi,pi->L', c_left, Co)
        contract('L,Lpq->pq', rho, cderi, alpha=4.0 * sign,
                 beta=1.0, out=vj)
        contract('Lpi,Lqi->pq', c_left, c_occ, alpha=2.0 * sign,
                 beta=1.0, out=vk)
        contract('Lpi,Lqi->pq', c_occ, c_left, alpha=2.0 * sign,
                 beta=1.0, out=vk)

    if str(exxdiv).lower() == 'ewald':
        _ewald_exxdiv_for_G0(
            provider.cell, np.zeros(3), density[None], vk[None])
    elif exxdiv is not None:
        raise NotImplementedError(
            f'Gaussian-DF RHF response does not support exxdiv={exxdiv!r}')
    response = vj - 0.5 * vk
    response_vo = contract('ap,pq->aq', Cv.T, response)
    return contract('aq,qi->ai', response_vo, Co)


def _make_ewald_intermediate(provider, Co, exxdiv):
    '''Return fixed overlap quantities for direct occupied-column projection.'''
    if str(exxdiv).lower() == 'ewald':
        kpt = np.zeros(3)
        overlap = cp.asarray(int1e_ovlp(provider.cell, kpts=kpt))
        overlap_occ = contract('pq,qi->pi', overlap, Co)
        madelung = float(tools.madelung(provider.cell, kpt))
        return overlap, overlap_occ, madelung
    if exxdiv is not None:
        raise NotImplementedError(
            f'direct-VO GDF response does not support exxdiv={exxdiv!r}')
    return None


def _check_provider(provider, nao):
    '''Validate the inherited real-Gamma CDERI iterator contract.'''
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    if len(kpts) != 1 or not is_zero(kpts[0]):
        raise NotImplementedError('factor-aware RHF response requires Gamma')
    if provider.cell.dimension != 3:
        raise NotImplementedError('factor-aware RHF response requires a 3D cell')
    if provider._cderi is None:
        provider.build(j_only=False)
    if getattr(provider, '_cderip', None) is not None:
        raise NotImplementedError(
            'factor-aware RHF response requires a positive Gamma metric')
    if int(provider.nao) != int(nao):
        raise ValueError('CDERI and orbital AO dimensions differ')


def _real_gamma_matrix(value, name):
    '''Return one real rank-two Gamma array or reject complex storage.'''
    value = cp.asarray(value)
    if value.ndim != 2:
        raise ValueError(f'{name} must be a rank-two array')
    if cp.iscomplexobj(value):
        maximum = float(cp.max(cp.abs(value.imag)).get()) if value.size else 0.0
        if maximum > 1.0e-10:
            raise NotImplementedError(
                f'{name} has non-Gamma imaginary component {maximum:.3e}')
        value = value.real
    return cp.asarray(value, dtype=cp.float64)
