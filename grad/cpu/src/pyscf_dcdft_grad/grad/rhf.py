"""
Restricted DC-DFT Analytic Gradients (S1 Optimized)
=====================================================
Based on rdc_grad.py with the following S1-related optimizations:

[Opt S1-CPU] nonvariational_correction에서 s1 관련 중복 계산 제거
  - 기존: 두 개의 atom loop에서 s1_co = s1[:, p0:p1] @ Co를 독립적으로 계산
    * Loop 1 (Ax(S1) density): s1_co → s1_oo → dm_s
    * Loop 2 (V^ξ assembly):   s1_co → s1_vo_bra (+ s1_cv → s1_vo_ket)
  - 변경: Loop 1에서 계산한 s1_co를 캐시하여 Loop 2에서 재사용

  참고: CPU 코드는 GPU와 달리 S1_mo 행렬을 구성하지 않음.
  AO 기반 atom-local 연산을 사용하여 이미 메모리 효율적.
  (GPU에서의 S1_mo (nmo×nmo) → S1_mo_o (nmo×nocc) 최적화와는 다른 접근.)

  CPU에서 "S1_mo occ-only"에 해당하는 개념은:
    s1_block_sym @ C (nao×nmo) 대신 s1_block_sym @ Co (nao×nocc)를 계산하는 것이며,
    이는 atom-local 접근에서 s1_co = s1[:, p0:p1] @ Co 계산과 동치.
    CPU 코드는 이미 이 방식을 사용하고 있으므로 추가적인 S1 메모리 최적화는 불필요.

Key Equations (from DC-DFT_Gradient_v4.pdf):
    - Eq. 20:  DC-DFT Energy
    - Eq. 75:  Z-Vector: (A+B)^HF Z = F^KS_vo
    - Eq. 80:  Total Gradient: dE/dξ = g_skel + g_nv
    - Eq. 87:  Non-variational: g_nv = 4 Σ Z_ai V^ξ_ai

원본: rdc_grad.py
"""

import sys
import numpy as np
from typing import Optional, Callable
from pyscf import lib, scf, dft, grad
from pyscf.lib import logger
from pyscf.scf import cphf

from .api import (
    maxabs, split_occ_vir, has_df, get_auxbasis,
    normalize_verbose, print_gradient
)


# ============================================================================
# DC-DFT Energy (PDF Eq. 20) — 변경 없음
# ============================================================================

def energy(mol, xc: str, mo_coeff: np.ndarray, mo_occ: np.ndarray,
           grids=None, density_fit: bool = False, auxbasis: Optional[str] = None) -> float:
    """
    Compute DC-DFT energy for restricted (closed-shell) system.

    PDF Eq. 20:
        E_DC = E_NN + Σ P^HF h + 1/2 Σ P^HF P^HF [(μν|λσ) - c_x/2 (μλ|σν)] + E_xc[ρ^HF]
    """
    mf_hf = scf.RHF(mol)
    if density_fit:
        mf_hf = mf_hf.density_fit(auxbasis=auxbasis)
    mf_hf.mo_coeff = mo_coeff
    mf_hf.mo_occ = mo_occ
    mf_hf.converged = True

    dm_hf = mf_hf.make_rdm1(mo_coeff, mo_occ)

    ks = dft.RKS(mol)
    ks.xc = xc
    if density_fit:
        ks = ks.density_fit(auxbasis=auxbasis)

    if grids is not None:
        ks.grids = grids
    else:
        ks.grids.atom_grid = (99, 590)
    if ks.grids.coords is None:
        ks.grids.build()

    return ks.energy_tot(dm=dm_hf)


# ============================================================================
# Z-Vector Solver (PDF Eq. 75, Section 10) — 변경 없음
# ============================================================================

def solve_z_vector(mo_energy: np.ndarray, mo_occ: np.ndarray,
                   Co: np.ndarray, Cv: np.ndarray,
                   fock_vo: np.ndarray, hf_response: Callable,
                   cphf_tol: float = 1e-10, max_cycle: int = 200,
                   log=None) -> np.ndarray:
    """
    Solve Z-vector equation via CPHF.

    PDF Eq. 75:
        (A + B)^HF Z = F^KS_vo
    """
    if log is None:
        log = logger.Logger(sys.stdout, logger.NOTE)

    nocc = Co.shape[1]
    nvir = Cv.shape[1]

    log.debug(f'Z-vector: nocc={nocc}, nvir={nvir}, nnz(F_vo)={np.count_nonzero(fock_vo)}')
    log.debug(f'Max |F^KS_vo| = {maxabs(fock_vo):.6e}')

    def orbital_hessian_matvec(x_vo):
        x = np.asarray(x_vo).reshape(-1, nvir, nocc)
        out = np.empty_like(x)
        for k in range(x.shape[0]):
            dm1 = Cv @ x[k] @ Co.conj().T
            dm1 = dm1 + dm1.conj().T
            hf_resp_dm = hf_response(dm1) * 2.0
            out[k] = Cv.conj().T @ hf_resp_dm @ Co
        return out.reshape(x_vo.shape)

    log.info(f'Solving CPHF: tol={cphf_tol:.2e}, max_cycle={max_cycle}')
    zvec, _mo_e1 = cphf.solve(
        orbital_hessian_matvec, mo_energy, mo_occ, fock_vo,
        max_cycle=max_cycle, tol=cphf_tol,
    )

    # cphf.solve returns (mo1, mo_e1), not a convergence flag.  Assess the
    # physical orbital-Hessian equation directly instead.
    orbital_energy_diff = mo_energy[nocc:, None] - mo_energy[None, :nocc]
    residual = zvec * orbital_energy_diff + orbital_hessian_matvec(zvec) + fock_vo
    residual_max = maxabs(residual)
    log.debug(f'Z-vector residual = {residual_max:.2e}')
    if residual_max > max(10.0 * cphf_tol, 1e-9):
        log.warn(
            'Large CPHF residual %.2e; consider increasing max_cycle or '
            'loosening tol.',
            residual_max,
        )

    return zvec


# ============================================================================
# Skeleton Gradient (PDF Eq. 81-86) — 변경 없음
# (CPU skeleton은 AO 기반으로 동작하며, S1_mo를 사용하지 않음)
# ============================================================================

def skeleton_gradient(mol, ks_grad, dm_hf: np.ndarray, fock_oo: np.ndarray,
                      Co: np.ndarray, s1: np.ndarray, log=None, optimize='optimal') -> np.ndarray:
    """
    Compute skeleton gradient (variational part).

    PDF Eq. 81:
        g_skel = g^(1e) + g^(2e) + g^(xc) + g^(ovlp) + g^(NN)

    Note: 이 함수는 AO 기반으로 동작하며 S1_mo 행렬을 사용하지 않음.
    overlap correction은 AO 기반 energy-weighted density dme0와
    AO overlap derivative s1을 직접 contract하여 계산.
    """
    if log is None:
        log = logger.Logger(sys.stdout, logger.NOTE)

    natm = mol.natm
    aoslices = mol.aoslice_by_atom()
    g_skel = np.zeros((natm, 3))

    vhf = ks_grad.get_veff(dm=dm_hf)
    hcore_deriv = ks_grad.hcore_generator()

    # Energy-weighted density: dme0 = 2 * Co @ F_KS_oo @ Co.T
    # GPU에서는 S1_mo[:, so, so]와 F_KS_oo를 contract하지만,
    # CPU에서는 AO 기반 dme0와 s1을 직접 contract (동치)
    dme0 = 2.0 * Co @ fock_oo @ Co.conj().T

    log.debug(f'Skeleton: vhf.shape={vhf.shape}, dme0.shape={dme0.shape}')

    for ia in range(natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        g_1e = lib.einsum('xij,ij->x', hcore_deriv(ia), dm_hf)
        g_2e_xc = lib.einsum('xij,ij->x', vhf[:, p0:p1], dm_hf[p0:p1]) * 2.0
        g_ovlp = -lib.einsum('xij,ij->x', s1[:, p0:p1], dme0[p0:p1]) * 2.0
        g_extra = ks_grad.extra_force(ia, locals())
        g_skel[ia] = g_1e + g_2e_xc + g_ovlp + g_extra

    g_nn = grad.rhf.grad_nuc(mol)
    return g_skel + g_nn


# ============================================================================
# Non-Variational Correction (PDF Eq. 87-89) — S1 최적화 적용
# ============================================================================

def nonvariational_correction(mol, Co: np.ndarray, Cv: np.ndarray,
                               mo_energy: np.ndarray, zvec: np.ndarray,
                               s1: np.ndarray, h1ao_list, hf_response: Callable,
                               log=None) -> np.ndarray:
    """
    Compute non-variational gradient correction (S1 optimized).

    PDF Eq. 87:
        g_nv = 4 Σ_{ai} Z_ai V^ξ_ai

    where V^ξ (PDF Eq. 89):
        V^ξ_ai = (F^HF)^ξ_ai - S^ξ_ai ε_i - 1/2 (A+B)^HF S^ξ

    [Opt S1-CPU] s1 관련 중복 계산 제거
    ─────────────────────────────────────
    기존 코드에서는 두 개의 atom loop가 존재:

    Loop 1 (Ax(S1) density, line 268-277 of original):
      for ia in range(natm):
          s1_co = s1[:, p0:p1] @ Co          # ← 계산 (1)
          s1_oo = s1_co.T @ Co[p0:p1]
          ...

    Loop 2 (V^ξ assembly, line 286-307 of original):
      for ia in range(natm):
          s1_co = s1[:, p0:p1] @ Co          # ← 동일 계산 반복 (2)
          s1_vo_bra = Cv[p0:p1].T @ s1_co
          ...

    변경: Loop 1에서 s1_co를 atom별로 캐시하여 Loop 2에서 재사용.
    비용: O(natm × Δp × nao × nocc) 의 GEMM 1회 절약.

    참고: GPU 코드와의 대응 관계
    ─────────────────────────────
    GPU에서의 S1_mo_o (npert, nmo, nocc)의 per-atom 조각은:
      S1_mo_o[ia*3+ic] = C.T @ s1_block_sym @ Co   (nmo × nocc)

    CPU에서 동일한 정보는 다음과 같이 분산:
      s1_co[ic]  = s1[ic, p0:p1, :] @ Co   (Δp × nocc)    ← s1_block의 p0:p1 행만
      s1_oo[ic]  = Co[p0:p1].T @ s1_co + s1_co.T @ Co[p0:p1]   (nocc × nocc)
      s1_vo[ic]  = Cv[p0:p1].T @ s1_co + Co[p0:p1].T @ s1_cv   (nvir × nocc)

    CPU는 atom-local Δp 크기의 행렬만 다루므로 full nao×nao 행렬 구성 불필요.
    이는 GPU의 S1_mo_o 접근법보다 메모리 효율적이나, atom loop 오버헤드가 있음.
    """
    if log is None:
        log = logger.Logger(sys.stdout, logger.NOTE)

    natm = mol.natm
    nao = mol.nao
    nocc = Co.shape[1]
    aoslices = mol.aoslice_by_atom()

    e_occ = mo_energy[:nocc]
    g_nv = np.zeros((natm, 3))

    # ──────────────────────────────────────────────────────────
    # Loop 1: Ax(S1) density 구성
    #
    # 각 원자 ia에 대해:
    #   s1_oo^ξ = S^ξ의 occ-occ block (symmetrized)
    #   dm_s = Co @ s1_oo @ Co.T + h.c.  (overlap-driven density)
    #
    # [Opt S1-CPU] s1_co를 atom별로 캐시
    # ──────────────────────────────────────────────────────────
    dm_s = np.zeros((natm, 3, nao, nao))
    s1_co_cache = {}  # {ia: s1_co}  — atom별 s1_co 캐시

    for ia in range(natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]

        # s1_co = s1[:, p0:p1, :] @ Co
        # shape: (3, Δp, nocc) where Δp = p1 - p0
        # 의미: AO overlap derivative의 atom-ia 행 × occupied MO
        s1_co = lib.einsum('xpv,vi->xpi', s1[:, p0:p1], Co)

        # [Opt S1-CPU] 캐시 저장 — Loop 2에서 재사용
        s1_co_cache[ia] = s1_co

        # s1_oo = Co[p0:p1].T @ s1 @ Co (symmetrized)
        # 이것은 GPU의 S1_mo_o[:, so, :] 에 해당하는 per-atom occ-occ block
        s1_oo = lib.einsum('xpi,pj->xij', s1_co, Co[p0:p1].conj())
        s1_oo = s1_oo + s1_oo.transpose(0, 2, 1)

        # Overlap-driven density response: dm_s = Co @ s1_oo @ Co.T + h.c.
        tmp = lib.einsum('xij,ui->xuj', s1_oo, Co)
        dm1 = lib.einsum('xuj,vj->xuv', tmp, Co.conj())
        dm_s[ia] = dm1 + dm1.transpose(0, 2, 1).conj()

    # Apply (A+B)^HF response to all perturbation densities at once
    ax_ao = hf_response(dm_s.reshape(-1, nao, nao)) * 2.0
    ax_vo = lib.einsum('Auv,ua,vi->Aai', ax_ao, Cv.conj(), Co)

    log.debug(f'ax_vo: max={maxabs(ax_vo):.6e}')

    # ──────────────────────────────────────────────────────────
    # Loop 2: V^ξ assembly and contraction with Z
    #
    # V^ξ_ai = (F^HF)^ξ_ai - S^ξ_ai ε_i - 1/2 (A+B)^HF S^ξ
    #
    # S^ξ_ai의 계산:
    #   s1_vo = s1_vo_bra + s1_vo_ket  (symmetrized vo block)
    #   이것은 GPU의 S1_mo_o[:, sv, :]에 해당
    #
    # [Opt S1-CPU] s1_co를 캐시에서 가져옴 (중복 계산 제거)
    # ──────────────────────────────────────────────────────────
    for ia in range(natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]

        # (1) Fock derivative: F^HF_1 in vo block
        h1ao = h1ao_list[ia]
        f1_vo = lib.einsum('xuv,ua,vi->xai', h1ao, Cv.conj(), Co)

        # (2) Overlap-energy term: S^ξ_ai * ε_i
        # s1_vo = Cv.T @ s1_sym_block @ Co (per atom, symmetrized)
        #
        # [Opt S1-CPU] s1_co를 캐시에서 재사용
        # 기존: s1_co = lib.einsum('xpv,vi->xpi', s1[:, p0:p1], Co)
        # 변경: s1_co = s1_co_cache[ia]
        s1_co = s1_co_cache[ia]

        # s1_vo_bra = Cv[p0:p1].T @ (s1[:, p0:p1] @ Co)
        # = Cv[p0:p1].T @ s1_co
        s1_vo_bra = lib.einsum('xpi,pa->xai', s1_co, Cv[p0:p1].conj())

        # s1_vo_ket = Co[p0:p1].T @ (s1[:, p0:p1] @ Cv)
        # 이 부분은 s1_co로부터 유도 불가 (Cv와의 곱이 필요)
        s1_cv = lib.einsum('xpv,va->xpa', s1[:, p0:p1], Cv)
        s1_vo_ket = lib.einsum('xpa,pi->xai', s1_cv, Co[p0:p1].conj())

        s1_vo = s1_vo_bra + s1_vo_ket
        se_vo = s1_vo * e_occ[None, None, :]

        # (3) Assemble V^ξ = F^HF_1 - S*ε - 0.5 * Ax(S1)
        v1 = f1_vo - se_vo - 0.5 * ax_vo[ia * 3:(ia + 1) * 3]

        # (4) Contract: g_nv = 4 Σ Z V^ξ
        g_nv[ia] = 4.0 * lib.einsum('ai,xai->x', zvec, v1)

    # 캐시 해제
    del s1_co_cache

    log.debug(f'g_nv: max={maxabs(g_nv):.6e}')

    return g_nv


# ============================================================================
# Main Gradient Class
# ============================================================================

class Gradients(lib.StreamObject):
    """
    Restricted DC-DFT analytic gradient calculator (S1 optimized).

    Attributes:
        base: HF mean-field object
        xc: DFT functional
        grids: Integration grids
        cphf_tol: CPHF tolerance
        grid_response: Include grid response
        auxbasis_response: Include auxbasis response
    """

    def __init__(self, mf_hf, xc: str, grids=None, cphf_tol: float = 1e-10,
                 grid_response: bool = True, auxbasis_response: bool = True):
        self.base = mf_hf
        self.mol = mf_hf.mol
        self.xc = xc
        self.grids = grids
        self.cphf_tol = cphf_tol
        self.grid_response = grid_response
        self.auxbasis_response = auxbasis_response

        self.verbose = mf_hf.verbose
        self.stdout = mf_hf.stdout
        self.max_memory = mf_hf.max_memory

        self._keys = {'xc', 'grids', 'cphf_tol', 'grid_response', 'auxbasis_response'}

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s ********', self.__class__, self.base.__class__)
        log.info('Functional = %s', self.xc)
        log.info('CPHF tolerance = %g', self.cphf_tol)
        log.info('Grid response = %s', self.grid_response)

        if has_df(self.base):
            log.info('Density fitting = ON (auxbasis=%s)', get_auxbasis(self.base))
            log.info('Auxbasis response = %s', self.auxbasis_response)
        else:
            log.info('Density fitting = OFF')

        return self

    def kernel(self, mo_coeff=None, mo_occ=None, atmlst=None, verbose=None):
        """Compute DC-DFT gradient."""
        if verbose is not None:
            if isinstance(verbose, bool):
                self.verbose = logger.NOTE if verbose else logger.QUIET
            else:
                self.verbose = verbose
        log = logger.new_logger(self, self.verbose)
        t0 = (logger.process_clock(), logger.perf_counter())

        if mo_coeff is None:
            mo_coeff = self.base.mo_coeff
        if mo_occ is None:
            mo_occ = self.base.mo_occ

        if log.verbose >= logger.NOTE:
            self.dump_flags(log)

        mol = self.mol
        mo_energy = self.base.mo_energy
        Co, Cv, so, sv = split_occ_vir(mo_coeff, mo_occ)
        dm_hf = self.base.make_rdm1(mo_coeff, mo_occ)

        # Build KS object
        ks = dft.RKS(mol)
        ks.xc = self.xc
        if has_df(self.base):
            ks = ks.density_fit(auxbasis=get_auxbasis(self.base))

        if self.grids is not None:
            ks.grids = self.grids
        else:
            ks.grids.atom_grid = (99, 590)
        if ks.grids.coords is None:
            ks.grids.build()

        ni = ks._numint
        cx = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)[2]
        log.note('Exact exchange fraction c_x = %.4f', cx)

        # Compute F^KS[ρ^HF]
        fock_ks_ao = ks.get_fock(dm=dm_hf)
        fock_ks_mo = mo_coeff.conj().T @ fock_ks_ao @ mo_coeff
        fock_vo = fock_ks_mo[sv, so]

        log.note('Max |F^KS_ai| = %.6e', maxabs(fock_vo))

        # Solve Z-vector
        log.info('Solving Z-vector equation...')
        hf_resp = self.base.gen_response(mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=1)
        zvec = solve_z_vector(
            mo_energy, mo_occ, Co, Cv, fock_vo, hf_resp,
            cphf_tol=self.cphf_tol, log=log,
        )

        # Setup gradient objects
        ks_grad = ks.nuc_grad_method()
        ks_grad.grid_response = self.grid_response

        if has_df(ks) and self.auxbasis_response:
            ks_grad.auxbasis_response = True
            log.debug('Auxbasis response enabled')

        s1 = ks_grad.get_ovlp(mol)
        h1ao_list = self.base.Hessian().make_h1(mo_coeff, mo_occ)

        # Skeleton gradient
        log.info('Computing skeleton gradient...')
        fock_oo = fock_ks_mo[so, so]
        g_skel = skeleton_gradient(mol, ks_grad, dm_hf, fock_oo, Co, s1, log=log)

        # Non-variational correction (S1 optimized)
        log.info('Computing non-variational correction (S1 optimized)...')
        g_nv = nonvariational_correction(
            mol, Co, Cv, mo_energy, zvec, s1, h1ao_list, hf_resp, log=log,
        )

        # Total gradient
        de = g_skel + g_nv

        if log.verbose >= logger.NOTE:
            log.note('\n' + '='*70)
            log.note('DC-DFT GRADIENT SUMMARY (S1 Optimized)')
            log.note('='*70)
            print_gradient(mol, g_skel, 'Skeleton gradient (g_skel)')
            print_gradient(mol, g_nv, 'Non-variational correction (g_nv)')
            print_gradient(mol, de, 'Total DC-DFT gradient')
            log.note('Wall time: %.2f s', logger.perf_counter() - t0[1])
            log.note('='*70)

        if atmlst is not None:
            de = de[np.asarray(atmlst)]

        log.timer('DC-DFT gradients (S1 opt)', *t0)
        self.de = de
        return de

    def as_scanner(self):
        raise NotImplementedError('Gradient scanner not yet implemented')


RDC_Gradients = Gradients


__all__ = ["Gradients", "RDC_Gradients", "energy"]
