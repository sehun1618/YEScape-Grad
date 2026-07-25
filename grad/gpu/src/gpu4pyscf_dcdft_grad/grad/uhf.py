#!/usr/bin/env python
"""Unrestricted GPU DC-DFT gradient with spin-coupled adjoint response.

The density-fitted optimized path avoids the perturbation-indexed AO response
and derivative-Fock tensors used by the original implementation.  Its
non-variational correction is evaluated as

    g_nv = g_zf1 - 2*z_se - z_ax,

where each term is already contracted over the alpha and beta orbital spaces.
The UHF response operator is applied once to the Z transition densities, and
self-adjointness moves the 3*Natm overlap responses onto that single result.

The direct ``g_zf1`` contraction uses GPU4PySCF's DF derivative-energy kernel.
For unrestricted densities its exact coefficients are one total-density
Coulomb pair with factor 2 and one exchange pair per spin with factor 4.
These coefficients reproduce ``UHF.Hessian().make_h1`` without materialising
the full derivative Fock matrices.

Earlier S1 optimizations retained below include:

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
import cupy as cp
import numpy as np
import time
from pyscf import df
from pyscf.lib import logger

from gpu4pyscf.df.df_jk import _make_factorized_dm
from gpu4pyscf.df.grad import tdrhf as df_tdrhf_grad
from gpu4pyscf.dft import uks as gpu_uks
from gpu4pyscf.grad import rhf as gpu_rhf_grad
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.scf import uhf as gpu_uhf
from gpu4pyscf.scf import ucphf as gpu_ucphf
from gpu4pyscf.scf._response_functions import _gen_uhf_response as _gen_uhf_response_gpu

from ._gpu4pyscf_compat import df_hessian_get_jk_uhf
from .rhf import _df_derivative_memory_limit, _release_df_storage


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


def _df_uhf_response_mo(dfobj, dm_tag, left_coeff, occ_coeff,
                        *, direct_scf_tol=1.0e-14):
    """Return alpha/beta MO blocks of the UHF response to ``dm_tag``."""

    vj_mo, vk_mo = df_hessian_get_jk_uhf(
        dfobj, dm_tag, left_coeff, occ_coeff,
        direct_scf_tol=direct_scf_tol,
    )
    out = vj_mo - vk_mo
    nset = out.shape[0]
    na = left_coeff[0].shape[1] * occ_coeff[0].shape[1]
    nb = left_coeff[1].shape[1] * occ_coeff[1].shape[1]
    out_a = out[:, :na].reshape(
        nset, left_coeff[0].shape[1], occ_coeff[0].shape[1]
    )
    out_b = out[:, na:na + nb].reshape(
        nset, left_coeff[1].shape[1], occ_coeff[1].shape[1]
    )
    del vj_mo, vk_mo, out
    return out_a, out_b


def _make_mo_direct_uax(dfobj, Cva, Cvb, Coa, Cob,
                        *, direct_scf_tol=1.0e-14):
    """Build a UCPHF callback returning only alpha/beta VO response blocks."""

    nova = Cva.shape[1] * Coa.shape[1]
    novb = Cvb.shape[1] * Cob.shape[1]

    def ax_vo(x_vo):
        original_shape = x_vo.shape
        x = cp.asarray(x_vo).reshape(-1, nova + novb)
        nset = x.shape[0]
        xa = x[:, :nova].reshape(nset, Cva.shape[1], Coa.shape[1])
        xb = x[:, nova:].reshape(nset, Cvb.shape[1], Cob.shape[1])
        mo1a = _contract('pa,kai->kpi', Cva, xa)
        mo1b = _contract('pa,kai->kpi', Cvb, xb)
        dma = _contract('kpi,qi->kpq', mo1a, Coa)
        dmb = _contract('kpi,qi->kpq', mo1b, Cob)
        dma = dma + dma.transpose(0, 2, 1)
        dmb = dmb + dmb.transpose(0, 2, 1)
        dm = tag_array(
            cp.concatenate((dma, dmb), axis=0),
            mo1=(mo1a, mo1b), occ_coeff=(Coa, Cob),
        )
        out_a, out_b = _df_uhf_response_mo(
            dfobj, dm, (Cva, Cvb), (Coa, Cob),
            direct_scf_tol=direct_scf_tol,
        )
        out = cp.concatenate(
            (out_a.reshape(nset, nova), out_b.reshape(nset, novb)), axis=1
        )
        del xa, xb, mo1a, mo1b, dma, dmb, dm, out_a, out_b
        return out.reshape(original_shape)

    return ax_vo


def _build_uhf_s1_oo_and_z_se(
    S1_ao, Coa, Cob, Cva, Cvb, Za, Zb, e_occ_a, e_occ_b, atom_slices,
):
    """Stream overlap derivatives into OO blocks and the scalar Z:Se term."""

    npert = 3 * len(atom_slices)
    s1_oo_a = cp.empty((npert, Coa.shape[1], Coa.shape[1]), dtype=Coa.dtype)
    s1_oo_b = cp.empty((npert, Cob.shape[1], Cob.shape[1]), dtype=Cob.dtype)
    z_se = cp.empty(npert, dtype=Coa.dtype)
    z_e_a = Za * e_occ_a[None, :]
    z_e_b = Zb * e_occ_b[None, :]

    for ia, (p0, p1) in enumerate(atom_slices):
        for ic in range(3):
            ip = 3 * ia + ic
            s_rows = S1_ao[ic, p0:p1]
            value = 0.0
            for Co, Cv, z_e, target in (
                (Coa, Cva, z_e_a, s1_oo_a),
                (Cob, Cvb, z_e_b, s1_oo_b),
            ):
                co_atom = Co[p0:p1]
                cv_atom = Cv[p0:p1]
                s_co = s_rows @ Co
                s_oo = co_atom.T @ s_co + s_co.T @ co_atom
                s_cv = s_rows @ Cv
                s_vo = cv_atom.T @ s_co + s_cv.T @ co_atom
                target[ip] = s_oo
                value = value + _contract('ai,ai->', z_e, s_vo)
            z_se[ip] = value

    del z_e_a, z_e_b
    return s1_oo_a, s1_oo_b, z_se


def _adjoint_uhf_z_ax(hf_resp, Za, Zb, Coa, Cob, Cva, Cvb,
                      s1_oo_a, s1_oo_b):
    """Contract every overlap response through one spin-coupled AO response."""

    dz_a = Cva @ Za @ Coa.T
    dz_b = Cvb @ Zb @ Cob.T
    vz_a, vz_b = hf_resp(cp.stack((dz_a + dz_a.T, dz_b + dz_b.T)))
    vz_oo_a = Coa.T @ vz_a @ Coa
    vz_oo_b = Cob.T @ vz_b @ Cob
    z_ax = (
        _contract('Aij,ij->A', s1_oo_a, vz_oo_a)
        + _contract('Aij,ij->A', s1_oo_b, vz_oo_b)
    )
    del dz_a, dz_b, vz_a, vz_b, vz_oo_a, vz_oo_b
    return z_ax


def _adjoint_uhf_z_ax_mo_direct(
    dfobj, Za, Zb, Coa, Cob, Cva, Cvb, s1_oo_a, s1_oo_b,
    *, direct_scf_tol=1.0e-14,
):
    """MO-direct form of the spin-coupled adjoint overlap contraction."""

    z_left_a = Cva @ Za
    z_left_b = Cvb @ Zb
    dz_a = z_left_a @ Coa.T
    dz_b = z_left_b @ Cob.T
    dm = tag_array(
        cp.concatenate(
            ((dz_a + dz_a.T)[None], (dz_b + dz_b.T)[None]), axis=0
        ),
        mo1=(z_left_a[None], z_left_b[None]), occ_coeff=(Coa, Cob),
    )
    vz_oo_a, vz_oo_b = _df_uhf_response_mo(
        dfobj, dm, (Coa, Cob), (Coa, Cob),
        direct_scf_tol=direct_scf_tol,
    )
    z_ax = (
        _contract('Aij,ij->A', s1_oo_a, vz_oo_a[0])
        + _contract('Aij,ij->A', s1_oo_b, vz_oo_b[0])
    )
    del z_left_a, z_left_b, dz_a, dz_b, dm, vz_oo_a, vz_oo_b
    return z_ax


def _direct_contracted_uhf_z_f1(
    mol, hf_gpu, Ca, Cb, occa, occb, viridxa, viridxb, Coa, Cob, Cva,
    Cvb, Za, Zb, *, memory_gb=None,
):
    """Return spin-summed ``2 * Z_sigma:F1_sigma`` by direct DF contraction."""

    npert = 3 * mol.natm
    hcore_a = gpu_rhf_grad.get_grad_hcore(
        hf_gpu.nuc_grad_method(), mo_coeff=Ca, mo_occ=occa
    ).reshape(npert, Ca.shape[1], Coa.shape[1])
    hcore_b = gpu_rhf_grad.get_grad_hcore(
        hf_gpu.nuc_grad_method(), mo_coeff=Cb, mo_occ=occb
    ).reshape(npert, Cb.shape[1], Cob.shape[1])
    g_hcore = 2.0 * (
        _contract('ai,Aai->A', Za, hcore_a[:, viridxa, :])
        + _contract('ai,Aai->A', Zb, hcore_b[:, viridxb, :])
    )
    del hcore_a, hcore_b

    z_left_a = Cva @ Za
    z_left_b = Cvb @ Zb
    trans_total = _make_factorized_dm(
        cp.concatenate((z_left_a, z_left_b), axis=1),
        cp.concatenate((Coa, Cob), axis=1), symmetrize=1,
    )
    dm_total = _make_factorized_dm(
        cp.concatenate((Coa, Cob), axis=1),
        cp.concatenate((Coa, Cob), axis=1), symmetrize=0,
    )
    trans_a = _make_factorized_dm(z_left_a, Coa, symmetrize=1)
    trans_b = _make_factorized_dm(z_left_b, Cob, symmetrize=1)
    dm_a = _make_factorized_dm(Coa, Coa, symmetrize=0)
    dm_b = _make_factorized_dm(Cob, Cob, symmetrize=0)

    auxmol = hf_gpu.with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(
            mol, auxbasis=hf_gpu.with_df.auxbasis
        )
    int3c2e_opt = df_tdrhf_grad.Int3c2eOpt(mol, auxmol).build()
    with _df_derivative_memory_limit(memory_gb):
        g_jk = df_tdrhf_grad._jk_energies_per_atom(
            int3c2e_opt,
            [[trans_total, dm_total], [trans_a, dm_a], [trans_b, dm_b]],
            j_factor=[2.0, 0.0, 0.0], k_factor=[0.0, 4.0, 4.0],
            sum_results=True,
        )
    out = g_hcore.reshape(mol.natm, 3) + cp.asarray(g_jk)
    del z_left_a, z_left_b, trans_total, dm_total, trans_a, trans_b
    del dm_a, dm_b, int3c2e_opt, g_jk
    return out


class DCGradientGPU:
    """GPU unrestricted DC-DFT gradient with a bounded-memory DF route."""

    def __init__(self, mf, xc='PBE', cphf_tol=1e-10, cphf_max_cycle=500,
                 atom_grid=None, grid_level=4, grid_response=True,
                 response_backend='mo_direct', share_df=True,
                 direct_z_f1=True, df_deriv_memory_gb=None):
        self.mf_hf = mf
        self.mol = mf.mol
        self.xc = xc
        self.cphf_tol = cphf_tol
        self.cphf_max_cycle = cphf_max_cycle
        self.atom_grid = atom_grid
        self.grid_level = grid_level
        self.grid_response = grid_response
        response_backend = response_backend.lower().replace('-', '_')
        if response_backend not in ('mo_direct', 'ao'):
            raise ValueError("response_backend must be 'mo_direct' or 'ao'")
        self.response_backend = response_backend
        self.share_df = bool(share_df)
        self.direct_z_f1 = bool(direct_z_f1)
        self.df_deriv_memory_gb = (
            None if df_deriv_memory_gb is None else float(df_deriv_memory_gb)
        )
        if self.df_deriv_memory_gb is not None and self.df_deriv_memory_gb <= 0:
            raise ValueError('df_deriv_memory_gb must be positive')
        self.memory_diagnostics = {}
        self.optimization_diagnostics = {}

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
        shared_df = self.mf_hf.with_df if has_df and self.share_df else None
        if not self.mf_hf.converged:
            raise RuntimeError('The reference UHF calculation is not converged')

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
            if shared_df is not None:
                ks_gpu.with_df = shared_df
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
            if shared_df is not None:
                hf_gpu.with_df = shared_df
        hf_gpu.mo_coeff = (Ca, Cb)
        hf_gpu.mo_energy = (ea, eb)
        hf_gpu.mo_occ = (occa, occb)
        hf_gpu.converged = True
        hf_gpu.e_tot = self.mf_hf.e_tot

        hf_resp = None
        if has_df and self.response_backend == 'mo_direct':
            Ax_vo = _make_mo_direct_uax(
                hf_gpu.with_df, Cva, Cvb, Coa, Cob,
                direct_scf_tol=hf_gpu.direct_scf_tol,
            )
        else:
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
                    out[:, :nova] = _contract(
                        'kaq,qi->kai', tmpa, Coa
                    ).reshape(nset, nova)
                if novb:
                    tmpb = _contract('ap,kpq->kaq', Cvb.T, v_dm[1])
                    out[:, nova:] = _contract(
                        'kaq,qi->kai', tmpb, Cob
                    ).reshape(nset, novb)

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

        natm = mol.natm
        npert = 3 * natm
        e_occ_a = ea[occidxa]
        e_occ_b = eb[occidxb]
        S1_oo_a, S1_oo_b, z_se = _build_uhf_s1_oo_and_z_se(
            S1_ao, Coa, Cob, Cva, Cvb, Za, Zb, e_occ_a, e_occ_b,
            atom_slices,
        )

        # The full implementation is retained as an explicit reference and
        # for non-DF inputs.  Only that path needs all MO rows of S1.
        S1_mo_o_a = S1_mo_o_b = None
        if not (has_df and self.direct_z_f1):
            S1_mo_o_a = cp.empty((npert, nmoa, nocca), dtype=Ca.dtype)
            S1_mo_o_b = cp.empty((npert, nmob, noccb), dtype=Cb.dtype)
            for ia, (p0, p1) in enumerate(atom_slices):
                for ic in range(3):
                    s1_block = cp.zeros((nao, nao), dtype=Ca.dtype)
                    s1_block[p0:p1] = S1_ao[ic, p0:p1]
                    s1_block = s1_block + s1_block.T
                    ip = 3 * ia + ic
                    S1_mo_o_a[ip] = Ca.T @ s1_block @ Coa
                    S1_mo_o_b[ip] = Cb.T @ s1_block @ Cob

        del F_KS  # F_KS는 더 이상 불필요 (F_KS_vo/F_KS_oo에 보존)

        log.info('  %.2f s', time.time() - t3)

        # ============================================================
        # [4/5] Skeleton gradient (GPU)
        # ============================================================
        log.info('[4/5] Skeleton (GPU)')
        t4 = time.time()

        g_skel = self._skeleton_gpu(
            mol, ks_grad_gpu, F_KS_oo_a, F_KS_oo_b, S1_oo_a, S1_oo_b,
            log, e_occ_a, e_occ_b,
        )

        log.info('  %.2f s', time.time() - t4)

        # ============================================================
        # [5/5] Non-variational (GPU)
        # ============================================================
        log.info('[5/5] Non-variational (GPU)')
        t5 = time.time()

        g_nv = self._nonvariational_gpu(
            mol, hf_gpu, hf_resp, Ca, Cb, ea, eb, occa, occb, occidxa,
            occidxb, viridxa, viridxb, Coa, Cob, Cva, Cvb, Za, Zb,
            S1_oo_a, S1_oo_b, z_se, S1_mo_o_a, S1_mo_o_b, atom_slices,
            log,
        )

        log.info('  %.2f s', time.time() - t5)

        g_tot = g_skel + g_nv
        g_np = cp.asnumpy(g_tot)

        self.optimization_diagnostics.update({
            'response_backend': self.response_backend,
            'shared_df_identity': bool(
                shared_df is not None and hf_gpu.with_df is shared_df
            ),
            'direct_z_f1': bool(has_df and self.direct_z_f1),
            'df_deriv_memory_gb': self.df_deriv_memory_gb,
            'total_seconds': time.time() - t0,
        })

        log.info('\n' + '='*70)
        log.info('Total: %.2f s | Max |grad|: %.6e',
                 time.time() - t0, np.max(np.abs(g_np)))
        log.info('='*70 + '\n')

        return g_np

    def _skeleton_gpu(
        self, mol, ks_grad_gpu, F_KS_oo_a, F_KS_oo_b, S1_oo_a, S1_oo_b,
        log, e_occ_a, e_occ_b,
    ):
        """Skeleton gradient fully on GPU.

        Strategy: compute standard UKS gradient via GPU kernel(), then apply
        unrestricted DC-DFT overlap correction.  Standard UKS uses
        diag(e_occ_s), DC-DFT needs full F_KS_oo_s.

        Only the occupied-occupied overlap blocks are retained.
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
            _contract('Aij,ij->A', S1_oo_a, dFa)
            + _contract('Aij,ij->A', S1_oo_b, dFb)
        ).reshape(natm, 3)

        g_skel = g_ks + g_corr

        return g_skel

    def _nonvariational_gpu(
        self, mol, hf_gpu, hf_resp, Ca, Cb, ea, eb, occa, occb, occidxa,
        occidxb, viridxa, viridxb, Coa, Cob, Cva, Cvb, Za, Zb, S1_oo_a,
        S1_oo_b, z_se, S1_mo_o_a, S1_mo_o_b, atom_slices, log,
    ):
        """Evaluate the bounded-memory spin-coupled correction."""

        has_df = hasattr(hf_gpu, 'with_df') and hf_gpu.with_df is not None
        if not has_df or not self.direct_z_f1:
            log.warn(
                'Using the full UHF non-variational reference path; the '
                'direct contracted route requires density fitting and '
                'direct_z_f1=True'
            )
            if S1_mo_o_a is None or S1_mo_o_b is None:
                raise RuntimeError('The full UHF path requires occupied S1 columns')
            if hf_resp is None:
                hf_resp = _gen_uhf_response_gpu(
                    hf_gpu, mo_coeff=(Ca, Cb), mo_occ=(occa, occb), hermi=1
                )
            return self._nonvariational_full_gpu(
                mol, hf_gpu, hf_resp, Ca, Cb, ea, eb, occa, occb,
                occidxa, occidxb, viridxa, viridxb, Coa, Cob, Za, Zb,
                S1_mo_o_a, S1_mo_o_b, atom_slices, log,
            )

        log.debug('  adjoint spin-coupled Z:Ax(S1) (one response)...')
        t_ax = time.time()
        if self.response_backend == 'mo_direct':
            z_ax = _adjoint_uhf_z_ax_mo_direct(
                hf_gpu.with_df, Za, Zb, Coa, Cob, Cva, Cvb,
                S1_oo_a, S1_oo_b, direct_scf_tol=hf_gpu.direct_scf_tol,
            )
        else:
            z_ax = _adjoint_uhf_z_ax(
                hf_resp, Za, Zb, Coa, Cob, Cva, Cvb, S1_oo_a, S1_oo_b
            )
        adjoint_seconds = time.time() - t_ax
        log.debug('  adjoint Z:Ax(S1): %.2f s', adjoint_seconds)

        # The response CDERI reaches its last use above.  Releasing it before
        # direct F1 prevents two high-water-mark DF representations from
        # coexisting on the device.
        self.memory_diagnostics['released_cphf_df_gib'] = _release_df_storage(
            hf_gpu.with_df, 'UHF CPHF', log
        )

        log.debug('  direct spin-summed 2*Z:F1 (GPU DF derivative)...')
        t_f1 = time.time()
        g_zf1 = _direct_contracted_uhf_z_f1(
            mol, hf_gpu, Ca, Cb, occa, occb, viridxa, viridxb, Coa, Cob,
            Cva, Cvb, Za, Zb, memory_gb=self.df_deriv_memory_gb,
        )
        direct_f1_seconds = time.time() - t_f1
        log.debug('  direct 2*Z:F1: %.2f s', direct_f1_seconds)

        self.optimization_diagnostics.update({
            'adjoint_response_seconds': adjoint_seconds,
            'direct_z_f1_seconds': direct_f1_seconds,
        })
        g_nv = g_zf1.reshape(-1) - 2.0 * z_se - z_ax
        del g_zf1, z_se, z_ax
        return g_nv.reshape(mol.natm, 3)

    def _nonvariational_full_gpu(
        self, mol, hf_gpu, hf_resp, Ca, Cb, ea, eb, occa, occb, occidxa,
        occidxb, viridxa, viridxb, Coa, Cob, Za, Zb, S1_mo_o_a,
        S1_mo_o_b, atom_slices, log,
    ):
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
        F_HF_1a, F_HF_1b = hf_hess.make_h1(
            cp.stack((Ca, Cb)), cp.stack((occa, occb))
        )
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
