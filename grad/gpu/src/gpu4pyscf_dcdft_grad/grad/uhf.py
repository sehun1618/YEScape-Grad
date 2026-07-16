#!/usr/bin/env python
"""
Unrestricted DC-DFT Gradient - 100% GPU Implementation (S1 Optimized)
=====================================================================
Based on udc_grad_gpu_pure.py with the following S1-related optimizations:

[Opt 3.3] energy_tot 중복 호출 제거
  - 기존: ks_gpu.energy_tot(dm=dm_hf) -> J, K, XC를 처음부터 다시 계산
  - 변경: get_fock에서 이미 계산된 Fock matrix로부터 에너지를 직접 조합
         e_tot = 0.5 * sum_s Tr(D_s * (h1e + F_s)) + E_nuc

[Opt 3.4] _contract 함수의 cutensor import 캐시
  - 기존: 매 호출마다 try/except로 cutensor import 시도
  - 변경: 모듈 로드 시 1회만 import, 이후 캐시된 함수 참조

[Opt 3.5] spin별 S1_mo에서 필요한 occupied-column 블록만 계산
  - 기존: S1_mo_a/b shape = (npert, nmo_s, nmo_s)
  - 변경: S1_mo_o_a/b shape = (npert, nmo_s, nocc_s)
  - 근거: 실제 사용되는 슬라이스:
    (a) skeleton correction: S1_mo[:, oo] -> S1_mo_o[:, occ_rows, :]
    (b) Ax(S1) density:      S1_mo[:, oo] -> S1_mo_o[:, occ_rows, :]
    (c) Se = S1 * e_occ:     S1_mo[:, :, occ] -> S1_mo_o
    virtual-virtual 블록은 사용되지 않음.

원본: udc_grad_gpu_pure.py
"""
import numpy as np
import time
import cupy as cp
from pyscf import gto, dft
from pyscf.lib import logger

from gpu4pyscf.dft import uks as gpu_uks
from gpu4pyscf.scf import uhf as gpu_uhf
from gpu4pyscf.scf import ucphf as gpu_ucphf
from gpu4pyscf.scf._response_functions import _gen_uhf_response as _gen_uhf_response_gpu


# ---------------------------------------------------------------------------
# [Opt 3.4] cutensor import를 모듈 로드 시 1회만 수행
# 기존: 매 _contract 호출마다 try/except
# 변경: 모듈 레벨에서 캐시, _contract 내부에서 분기만 수행
# ---------------------------------------------------------------------------
try:
    from gpu4pyscf.lib.cutensor import contract as _cutensor_contract
except ImportError:
    _cutensor_contract = None


def _contract(pattern, a, b):
    """GPU 2-operand contraction (cutensor cached)."""
    if _cutensor_contract is not None:
        return _cutensor_contract(pattern, a, b)
    return cp.einsum(pattern, a, b)


def _as_spin_tuple(obj):
    """Convert unrestricted MO data to a pair of CuPy arrays."""
    if isinstance(obj, (tuple, list)):
        return cp.asarray(obj[0]), cp.asarray(obj[1])
    arr = cp.asarray(obj)
    if arr.ndim >= 1 and arr.shape[0] == 2:
        return arr[0], arr[1]
    raise ValueError(f'Expected unrestricted spin tuple/stack, got shape {arr.shape}')


class DCGradientGPU:
    """100% GPU unrestricted DC-DFT gradient (S1 optimized)."""

    def __init__(self, mf, xc='PBE', cphf_tol=1e-10, cphf_max_cycle=500,
                 atom_grid=None, grid_level=4, grid_response=True):
        self.mf_hf = mf
        self.mol = mf.mol
        self.xc = xc
        self.cphf_tol = cphf_tol
        self.cphf_max_cycle = cphf_max_cycle
        self.atom_grid = atom_grid
        self.grid_level = grid_level
        self.grid_response = grid_response

    def kernel(self, verbose=None):
        mol = self.mol
        if verbose is None:
            verbose = mol.verbose
        log = logger.new_logger(mol, verbose)

        t0 = time.time()
        log.info('\n' + '='*70)
        log.info('UDC-DFT Gradient - 100%% GPU (S1 Optimized)')
        log.info('='*70)

        has_df = hasattr(self.mf_hf, 'with_df') and self.mf_hf.with_df is not None
        auxbasis = self.mf_hf.with_df.auxbasis if has_df else None

        # All MO data on GPU
        Ca, Cb = _as_spin_tuple(self.mf_hf.mo_coeff)
        ea, eb = _as_spin_tuple(self.mf_hf.mo_energy)
        occa, occb = _as_spin_tuple(self.mf_hf.mo_occ)

        nao = Ca.shape[0]
        nmoa, nmob = Ca.shape[1], Cb.shape[1]

        occidxa = cp.where(occa > 0)[0]
        occidxb = cp.where(occb > 0)[0]
        viridxa = cp.where(occa == 0)[0]
        viridxb = cp.where(occb == 0)[0]

        nocca = int(occidxa.size)
        noccb = int(occidxb.size)
        nvira = int(viridxa.size)
        nvirb = int(viridxb.size)
        nova = nvira * nocca
        novb = nvirb * noccb

        Coa = Ca[:, occidxa]
        Cob = Cb[:, occidxb]
        Cva = Ca[:, viridxa]
        Cvb = Cb[:, viridxb]

        dm_hf = cp.stack((Coa @ Coa.T, Cob @ Cob.T))
        atom_slices = [mol.aoslice_by_atom()[ia, 2:4] for ia in range(mol.natm)]

        # ============================================================
        # [1/5] KS Fock on GPU
        # ============================================================
        log.info('\n[1/5] UKS Fock (GPU)')
        t1 = time.time()

        ks_gpu = gpu_uks.UKS(mol, xc=self.xc)
        if has_df:
            ks_gpu = ks_gpu.density_fit(auxbasis=auxbasis)
        ks_gpu.grids.level = self.grid_level
        if self.atom_grid is not None:
            ks_gpu.grids.atom_grid = self.atom_grid

        F_KS = cp.asarray(ks_gpu.get_fock(dm=dm_hf))
        F_KS_a = F_KS[0]
        F_KS_b = F_KS[1]
        F_KS_vo_a = Cva.T @ F_KS_a @ Coa
        F_KS_vo_b = Cvb.T @ F_KS_b @ Cob
        F_KS_oo_a = Coa.T @ F_KS_a @ Coa
        F_KS_oo_b = Cob.T @ F_KS_b @ Cob

        log.info('  %.2f s', time.time() - t1)

        # ============================================================
        # [2/5] Z-vector on GPU (gpu4pyscf UCPHF)
        # ============================================================
        log.info('[2/5] Z-vector (GPU UCPHF)')
        t2 = time.time()

        hf_gpu = gpu_uhf.UHF(mol)
        if has_df:
            hf_gpu = hf_gpu.density_fit(auxbasis=auxbasis)
        hf_gpu.mo_coeff = (Ca, Cb)
        hf_gpu.mo_energy = (ea, eb)
        hf_gpu.mo_occ = (occa, occb)
        hf_gpu.converged = True
        hf_gpu.e_tot = self.mf_hf.e_tot

        hf_resp = _gen_uhf_response_gpu(
            hf_gpu, mo_coeff=(Ca, Cb), mo_occ=(occa, occb), hermi=1
        )

        def Ax_vo(X_vo):
            X = cp.asarray(X_vo).reshape(-1, nova + novb)
            nset = X.shape[0]
            dm_x = cp.zeros((2, nset, nao, nao), dtype=Ca.dtype)
            out = cp.zeros((nset, nova + novb), dtype=Ca.dtype)

            if nova:
                Xa = X[:, :nova].reshape(nset, nvira, nocca)
                mo1a = _contract('pa,kai->kpi', Cva, Xa)
                dma = _contract('kpi,qi->kpq', mo1a, Coa)
                dm_x[0] = dma + dma.transpose(0, 2, 1)

            if novb:
                Xb = X[:, nova:].reshape(nset, nvirb, noccb)
                mo1b = _contract('pa,kai->kpi', Cvb, Xb)
                dmb = _contract('kpi,qi->kpq', mo1b, Cob)
                dm_x[1] = dmb + dmb.transpose(0, 2, 1)

            v_dm = cp.asarray(hf_resp(dm_x))

            if nova:
                tmpa = _contract('ap,kpq->kaq', Cva.T, v_dm[0])
                out[:, :nova] = _contract('kaq,qi->kai', tmpa, Coa).reshape(nset, nova)
            if novb:
                tmpb = _contract('ap,kpq->kaq', Cvb.T, v_dm[1])
                out[:, nova:] = _contract('kaq,qi->kai', tmpb, Cob).reshape(nset, novb)

            return out.reshape(X_vo.shape)

        (Za, Zb), converged = gpu_ucphf.solve(
            Ax_vo, (ea, eb), (occa, occb), (F_KS_vo_a, F_KS_vo_b),
            max_cycle=self.cphf_max_cycle, tol=self.cphf_tol
        )

        if converged is not None and not converged:
            log.warn('GPU UCPHF not converged')

        log.info('  %.2f s', time.time() - t2)

        # ============================================================
        # [3/5] S1 overlap derivative (AO→MO on GPU)
        # ============================================================
        log.info('[3/5] S1 AO→MO (GPU, occ-only)')
        t3 = time.time()

        ks_gpu.mo_coeff = (Ca, Cb)
        ks_gpu.mo_occ = (occa, occb)
        ks_gpu.mo_energy = (ea, eb)
        ks_gpu.converged = True

        # -----------------------------------------------------------
        # [Opt 3.3] energy_tot 중복 호출 제거
        #
        # 기존 코드:
        #   ks_gpu.e_tot = ks_gpu.energy_tot(dm=dm_hf)
        #   → get_fock에서 이미 계산한 J, K, XC를 처음부터 다시 계산
        #
        # 변경: get_fock 결과(F_KS)로부터 에너지를 직접 조합
        #   E_tot = 0.5 * sum_s Tr(D_s * (h1e + F_KS_s)) + E_nuc
        #
        # 주의: DFT에서는 E_xc가 density에 linear하지 않으므로 이 값은 근사.
        # 단, ks_grad.kernel()이 실제로 필요로 하는 것은 ks_gpu.e_tot이
        # 설정되어 있다는 사실이며, gradient에는 vxc/exc intermediate가 사용됨.
        # -----------------------------------------------------------
        h1e = cp.asarray(ks_gpu.get_hcore(mol))
        e_tot = 0.5 * (
            cp.einsum('sij,ji->', dm_hf, h1e) +
            cp.einsum('sij,sji->', dm_hf, F_KS)
        )
        ks_gpu.e_tot = float(e_tot) + ks_gpu.energy_nuc()

        ks_grad_gpu = ks_gpu.nuc_grad_method()
        ks_grad_gpu.grid_response = self.grid_response

        S1_ao = cp.asarray(ks_grad_gpu.get_ovlp(mol))  # (3, nao, nao)

        # -----------------------------------------------------------
        # [Opt 3.5] spin별 S1_mo occ-only:
        #   (npert, nmo_s, nmo_s) → (npert, nmo_s, nocc_s)
        #
        # 실제 사용되는 블록:
        #   S1_mo_o_s[:, occidx_s, :]  = occ-occ block
        #   S1_mo_o_s[:, :, :]         = all rows, occupied columns
        # -----------------------------------------------------------
        natm = mol.natm
        npert = 3 * natm
        S1_mo_o_a = cp.zeros((npert, nmoa, nocca), dtype=Ca.dtype)
        S1_mo_o_b = cp.zeros((npert, nmob, noccb), dtype=Cb.dtype)

        for ia in range(natm):
            p0, p1 = atom_slices[ia]
            for ic in range(3):
                # AO-basis overlap derivative: 원자 ia의 ic 방향
                # s1_block은 p0:p1 행만 nonzero, symmetrize 필요
                s1_block = cp.zeros((nao, nao), dtype=Ca.dtype)
                s1_block[p0:p1] = S1_ao[ic, p0:p1]
                s1_block = s1_block + s1_block.T

                ip = ia * 3 + ic

                # MO 변환: 전체 MO × occ MO만 계산
                # alpha: Ca.T @ s1_block @ Coa → (nmoa, nocca)
                # beta:  Cb.T @ s1_block @ Cob → (nmob, noccb)
                S1_mo_o_a[ip] = Ca.T @ s1_block @ Coa
                S1_mo_o_b[ip] = Cb.T @ s1_block @ Cob

        del F_KS  # F_KS는 더 이상 불필요 (F_KS_vo/F_KS_oo에 보존)

        log.info('  %.2f s', time.time() - t3)

        # ============================================================
        # [4/5] Skeleton gradient (GPU)
        # ============================================================
        log.info('[4/5] Skeleton (GPU)')
        t4 = time.time()

        e_occ_a = ea[occidxa]
        e_occ_b = eb[occidxb]
        g_skel = self._skeleton_gpu(
            mol, ks_gpu, ks_grad_gpu,
            F_KS_oo_a, F_KS_oo_b,
            S1_mo_o_a, S1_mo_o_b,
            occidxa, occidxb, log, e_occ_a, e_occ_b
        )

        log.info('  %.2f s', time.time() - t4)

        # ============================================================
        # [5/5] Non-variational (GPU)
        # ============================================================
        log.info('[5/5] Non-variational (GPU)')
        t5 = time.time()

        g_nv = self._nonvariational_gpu(
            mol, hf_gpu, hf_resp,
            Ca, Cb, ea, eb, occa, occb,
            occidxa, occidxb, viridxa, viridxb,
            Coa, Cob, Za, Zb,
            S1_mo_o_a, S1_mo_o_b, atom_slices, log
        )

        log.info('  %.2f s', time.time() - t5)

        g_tot = g_skel + g_nv
        g_np = cp.asnumpy(g_tot)

        log.info('\n' + '='*70)
        log.info('Total: %.2f s | Max |grad|: %.6e',
                 time.time() - t0, np.max(np.abs(g_np)))
        log.info('='*70 + '\n')

        return g_np

    def _skeleton_gpu(self, mol, ks_gpu, ks_grad_gpu,
                      F_KS_oo_a, F_KS_oo_b,
                      S1_mo_o_a, S1_mo_o_b,
                      occidxa, occidxb, log, e_occ_a, e_occ_b):
        """Skeleton gradient fully on GPU.

        Strategy: compute standard UKS gradient via GPU kernel(), then apply
        unrestricted DC-DFT overlap correction.  Standard UKS uses
        diag(e_occ_s), DC-DFT needs full F_KS_oo_s.

        [Opt 3.5 반영]
        기존: S1_mo_s[:, occidx_s, occidx_s]
        변경: S1_mo_o_s[:, occidx_s, :]
        """
        natm = mol.natm

        # (A) Full standard UKS gradient on GPU
        log.debug('  ks_grad.kernel (GPU)...')
        g_ks = cp.asarray(ks_grad_gpu.kernel())  # (natm, 3)

        # (B) Overlap correction
        # RHF와 달리 UHF density에는 double-occupancy factor가 없으므로
        # symmetrized S1_MO 기준 prefactor는 spin별 -1.
        log.debug('  overlap correction...')
        dFa = F_KS_oo_a - cp.diag(e_occ_a)
        dFb = F_KS_oo_b - cp.diag(e_occ_b)

        g_corr = -1.0 * (
            _contract('Aij,ij->A', S1_mo_o_a[:, occidxa, :], dFa) +
            _contract('Aij,ij->A', S1_mo_o_b[:, occidxb, :], dFb)
        ).reshape(natm, 3)

        g_skel = g_ks + g_corr

        return g_skel

    def _nonvariational_gpu(self, mol, hf_gpu, hf_resp,
                            Ca, Cb, ea, eb, occa, occb,
                            occidxa, occidxb, viridxa, viridxb,
                            Coa, Cob, Za, Zb,
                            S1_mo_o_a, S1_mo_o_b, atom_slices, log):
        """Non-variational correction fully on GPU.

        [Opt 3.5 반영]
        기존: S1_mo_s shape (npert, nmo_s, nmo_s), 3가지 슬라이스 사용
        변경: S1_mo_o_s shape (npert, nmo_s, nocc_s), 대응관계:
          S1_mo_s[:, occidx_s, occidx_s] → S1_mo_o_s[:, occidx_s, :]
          S1_mo_s[:, :, occidx_s]        → S1_mo_o_s
        """
        natm = mol.natm
        nao = Ca.shape[0]
        npert = 3 * natm
        nocca = Coa.shape[1]
        noccb = Cob.shape[1]

        # (A) GPU make_h1:
        # returns (natm, 3, nmo_s, nocc_s) = C_s.T @ h1ao_s @ Co_s
        log.debug('  make_h1 (GPU)...')
        t_h1 = time.time()
        hf_hess = hf_gpu.Hessian()
        F_HF_1a, F_HF_1b = hf_hess.make_h1((Ca, Cb), (occa, occb))
        F_HF_1a = cp.asarray(F_HF_1a).reshape(npert, Ca.shape[1], nocca)
        F_HF_1b = cp.asarray(F_HF_1b).reshape(npert, Cb.shape[1], noccb)
        log.debug('  make_h1: %.2f s, shapes=%s/%s',
                  time.time() - t_h1, F_HF_1a.shape, F_HF_1b.shape)

        # (B) Build density response from S1 and apply UHF response
        log.debug('  Ax(S1) (GPU)...')
        t_ax = time.time()

        # [Opt 3.5] oo block 추출
        # 기존: S_oo_s = S1_mo_s[:, occidx_s, occidx_s]
        # 변경: S_oo_s = S1_mo_o_s[:, occidx_s, :]
        S_oo_a = S1_mo_o_a[:, occidxa, :]
        S_oo_b = S1_mo_o_b[:, occidxb, :]

        dm_s = cp.empty((2, npert, nao, nao), dtype=Ca.dtype)

        mo1_s_a = cp.einsum('pi,Aij->Apj', Coa, S_oo_a)
        dm1_a = cp.einsum('Apj,qj->Apq', mo1_s_a, Coa)
        dm_s[0] = dm1_a + dm1_a.transpose(0, 2, 1)

        mo1_s_b = cp.einsum('pi,Aij->Apj', Cob, S_oo_b)
        dm1_b = cp.einsum('Apj,qj->Apq', mo1_s_b, Cob)
        dm_s[1] = dm1_b + dm1_b.transpose(0, 2, 1)

        # Apply HF response and project: C_s.T @ resp_s(dm_s) @ Co_s
        v_resp = cp.asarray(hf_resp(dm_s))
        AxS_a = cp.einsum('up,Apq,qi->Aui', Ca.T, v_resp[0], Coa)
        AxS_b = cp.einsum('up,Apq,qi->Aui', Cb.T, v_resp[1], Cob)

        del mo1_s_a, mo1_s_b, dm1_a, dm1_b, dm_s, v_resp
        log.debug('  Ax(S1): %.2f s', time.time() - t_ax)

        # (C) Assemble B = F_HF_1 - S1*e_occ - 0.5*AxS, extract vo, contract with Z
        log.debug('  B assembly + contraction (GPU)...')
        t_b = time.time()

        # [Opt 3.5] Se 계산
        # 기존: Se_s = S1_mo_s[:, :, occidx_s] * e_occ_s
        # 변경: Se_s = S1_mo_o_s * e_occ_s
        Se_a = S1_mo_o_a * ea[occidxa][None, None, :]
        Se_b = S1_mo_o_b * eb[occidxb][None, None, :]

        B_a = F_HF_1a - Se_a - 0.5 * AxS_a
        B_b = F_HF_1b - Se_b - 0.5 * AxS_b
        B_vo_a = B_a[:, viridxa, :]
        B_vo_b = B_b[:, viridxb, :]

        # Factor 2 (not 4): unrestricted spin channels have no double occupancy.
        g_nv = 2.0 * (
            _contract('ai,Aai->A', Za, B_vo_a) +
            _contract('ai,Aai->A', Zb, B_vo_b)
        )

        del Se_a, Se_b, B_a, B_b, B_vo_a, B_vo_b

        log.debug('  B + contraction: %.2f s', time.time() - t_b)

        return g_nv.reshape(natm, 3)


UDCGradientGPU = DCGradientGPU


def kernel(mf, xc='PBE', verbose=None):
    """Compute unrestricted DC-DFT gradient on GPU (S1 optimized)."""
    return DCGradientGPU(mf, xc).kernel(verbose=verbose)


__all__ = ["DCGradientGPU", "UDCGradientGPU", "kernel"]
