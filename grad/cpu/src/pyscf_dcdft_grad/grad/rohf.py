"""
Restricted Open-Shell DC-DFT Analytic Gradients (S1 Optimized)
==============================================================

Standalone CPU implementation for ROHF references.

Implementation note
-------------------
PySCF does not expose a molecular ROHF Hessian `make_h1`; internally, PySCF
also routes ROHF response through the UHF response machinery.  This module
therefore preserves the restricted open-shell orbitals and occupations, then
uses an equivalent UHF representation:

    C_alpha = C_beta = C_ROHF
    occ_alpha = (occ_ROHF > 0)
    occ_beta  = (occ_ROHF == 2)

The resulting density is exactly the ROHF density.  Response, Hessian H1, and
spin-resolved non-variational terms are evaluated in the alpha/beta
representation while the reference orbitals remain restricted.

Key differences from closed-shell RDC:
    - Spin-resolved alpha/beta occupied and virtual spaces
    - UCPHF Z-vector
    - Factor 2 in g_nv instead of closed-shell factor 4
    - Spin-summed skeleton gradient with UKS response at ROHF density
"""

import sys
import numpy as np
from typing import Optional, Callable, Tuple
from pyscf import lib, gto, scf, dft, grad
from pyscf.lib import logger
from pyscf.scf import ucphf, cphf


# ============================================================================
# Utilities
# ============================================================================

def maxabs(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    return 0.0 if arr.size == 0 else float(np.max(np.abs(arr)))


def has_df(mf) -> bool:
    return getattr(mf, 'with_df', None) is not None


def get_auxbasis(mf) -> Optional[str]:
    if has_df(mf):
        return getattr(mf.with_df, 'auxbasis', None)
    return None


def print_gradient(mol, g: np.ndarray, title: str = 'Gradient'):
    print(f'\n{title}:')
    for ia in range(mol.natm):
        symbol = mol.atom_symbol(ia)
        print(f'    {symbol:>2s}  {g[ia,0]:16.10f}  {g[ia,1]:16.10f}  {g[ia,2]:16.10f}')


def rohf_to_uhf_arrays(mf_rohf, mo_coeff=None, mo_occ=None, mo_energy=None):
    """Convert ROHF MO data to alpha/beta arrays without changing orbitals."""
    if mo_coeff is None:
        mo_coeff = mf_rohf.mo_coeff
    if mo_occ is None:
        mo_occ = mf_rohf.mo_occ
    if mo_energy is None:
        mo_energy = mf_rohf.mo_energy

    C = np.asarray(mo_coeff)
    occ = np.asarray(mo_occ)

    occa = np.asarray(occ > 0, dtype=float)
    occb = np.asarray(occ == 2, dtype=float)

    # PySCF ROHF mo_energy may carry spin-dependent canonical energies.
    ea = np.asarray(getattr(mo_energy, 'mo_ea', mo_energy))
    eb = np.asarray(getattr(mo_energy, 'mo_eb', mo_energy))

    return (C, C), (ea, eb), (occa, occb)


def make_uhf_reference(mf_rohf, mo_coeff=None, mo_occ=None, mo_energy=None):
    """Build a UHF-like object carrying the ROHF density and common orbitals."""
    if mo_coeff is None and mo_occ is None and mo_energy is None:
        try:
            mf_uhf = mf_rohf.to_uhf()
            mf_uhf.converged = True
            return mf_uhf
        except Exception:
            pass

    (Ca, Cb), (ea, eb), (occa, occb) = rohf_to_uhf_arrays(
        mf_rohf, mo_coeff=mo_coeff, mo_occ=mo_occ, mo_energy=mo_energy
    )

    mf_uhf = scf.UHF(mf_rohf.mol)
    if has_df(mf_rohf):
        mf_uhf = mf_uhf.density_fit(auxbasis=get_auxbasis(mf_rohf))
    mf_uhf.mo_coeff = (Ca, Cb)
    mf_uhf.mo_energy = (ea, eb)
    mf_uhf.mo_occ = (occa, occb)
    mf_uhf.e_tot = getattr(mf_rohf, 'e_tot', None)
    mf_uhf.converged = True
    mf_uhf.max_memory = mf_rohf.max_memory
    mf_uhf.stdout = mf_rohf.stdout
    mf_uhf.verbose = mf_rohf.verbose
    return mf_uhf


# ============================================================================
# DC-DFT Energy
# ============================================================================

def energy(mol, xc: str, mo_coeff, mo_occ, grids=None,
           density_fit: bool = False, auxbasis: Optional[str] = None) -> float:
    """
    Compute DC-DFT energy for a restricted open-shell density.

    `mo_coeff` is the common ROHF MO coefficient matrix and `mo_occ` has
    PySCF ROHF occupations (2, 1, 0).
    """
    C = np.asarray(mo_coeff)
    occ = np.asarray(mo_occ)
    occa = np.asarray(occ > 0, dtype=float)
    occb = np.asarray(occ == 2, dtype=float)

    mf_hf = scf.UHF(mol)
    if density_fit:
        mf_hf = mf_hf.density_fit(auxbasis=auxbasis)
    mf_hf.mo_coeff = (C, C)
    mf_hf.mo_occ = (occa, occb)
    mf_hf.converged = True
    dm_hf = mf_hf.make_rdm1((C, C), (occa, occb))

    ks = dft.UKS(mol)
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
# Z-Vector Solver
# ============================================================================

def solve_z_vector(Coa: np.ndarray, Cob: np.ndarray,
                   Cva: np.ndarray, Cvb: np.ndarray,
                   ea: np.ndarray, eb: np.ndarray,
                   occa: np.ndarray, occb: np.ndarray,
                   fvo_a: np.ndarray, fvo_b: np.ndarray,
                   uhf_response: Callable,
                   cphf_tol: float = 1e-10, max_cycle: int = 200,
                   log=None) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the spin-resolved Z-vector equation via UCPHF."""
    if log is None:
        log = logger.Logger(sys.stdout, logger.NOTE)

    nocca, noccb = Coa.shape[1], Cob.shape[1]
    nvira, nvirb = Cva.shape[1], Cvb.shape[1]
    nova, novb = nvira * nocca, nvirb * noccb
    nao = Coa.shape[0]

    occa_idx = np.where(occa > 0)[0]
    occb_idx = np.where(occb > 0)[0]
    vira_idx = np.where(occa == 0)[0]
    virb_idx = np.where(occb == 0)[0]

    log.debug(f'Z-vector (ROHF/UHF form): nocca={nocca}, noccb={noccb}, '
              f'nvira={nvira}, nvirb={nvirb}')
    log.debug(f'Max |F^KS_ai(alpha)| = {maxabs(fvo_a):.6e}')
    log.debug(f'Max |F^KS_ai(beta)| = {maxabs(fvo_b):.6e}')

    def vind(x_vo):
        x = np.asarray(x_vo).reshape(-1, nova + novb)
        dm1 = np.zeros((2, x.shape[0], nao, nao))
        out = np.zeros_like(x)

        for k in range(x.shape[0]):
            if nova:
                xa = x[k, :nova].reshape(nvira, nocca)
                dmaa = Cva @ xa @ Coa.conj().T
                dm1[0, k] = dmaa + dmaa.conj().T
            if novb:
                xb = x[k, nova:].reshape(nvirb, noccb)
                dmbb = Cvb @ xb @ Cob.conj().T
                dm1[1, k] = dmbb + dmbb.conj().T

        v1 = uhf_response(dm1)

        for k in range(x.shape[0]):
            if nova:
                out[k, :nova] = (Cva.conj().T @ v1[0, k] @ Coa).reshape(-1)
            if novb:
                out[k, nova:] = (Cvb.conj().T @ v1[1, k] @ Cob).reshape(-1)

        return out.reshape(x_vo.shape)

    def vind_a(xa_vo):
        x = np.asarray(xa_vo).reshape(-1, nova)
        dm1 = np.zeros((2, x.shape[0], nao, nao))
        out = np.zeros_like(x)
        for k in range(x.shape[0]):
            xa = x[k].reshape(nvira, nocca)
            dmaa = Cva @ xa @ Coa.conj().T
            dm1[0, k] = dmaa + dmaa.conj().T
        v1 = uhf_response(dm1)
        for k in range(x.shape[0]):
            out[k] = (Cva.conj().T @ v1[0, k] @ Coa).reshape(-1)
        return out.reshape(xa_vo.shape)

    def vind_b(xb_vo):
        x = np.asarray(xb_vo).reshape(-1, novb)
        dm1 = np.zeros((2, x.shape[0], nao, nao))
        out = np.zeros_like(x)
        for k in range(x.shape[0]):
            xb = x[k].reshape(nvirb, noccb)
            dmbb = Cvb @ xb @ Cob.conj().T
            dm1[1, k] = dmbb + dmbb.conj().T
        v1 = uhf_response(dm1)
        for k in range(x.shape[0]):
            out[k] = (Cvb.conj().T @ v1[1, k] @ Cob).reshape(-1)
        return out.reshape(xb_vo.shape)

    log.info(f'Solving UCPHF: tol={cphf_tol:.2e}, max_cycle={max_cycle}')

    if nova and novb:
        za, zb = ucphf.solve(
            vind, (ea, eb), (occa, occb), (fvo_a, fvo_b),
            max_cycle=max_cycle, tol=cphf_tol,
        )[0]
    elif nova:
        za = cphf.solve(vind_a, ea, occa, fvo_a,
                        max_cycle=max_cycle, tol=cphf_tol)[0]
        zb = np.zeros((nvirb, noccb))
    elif novb:
        zb = cphf.solve(vind_b, eb, occb, fvo_b,
                        max_cycle=max_cycle, tol=cphf_tol)[0]
        za = np.zeros((nvira, nocca))
    else:
        za = np.zeros((nvira, nocca))
        zb = np.zeros((nvirb, noccb))

    if log.verbose >= logger.DEBUG and (nova or novb):
        res_a = za * (ea[vira_idx, None] - ea[None, occa_idx])
        res_b = zb * (eb[virb_idx, None] - eb[None, occb_idx])
        if nova and novb:
            zpack = np.hstack((za.reshape(-1), zb.reshape(-1)))
            resp = vind(zpack.reshape(1, -1))[0]
            res_a += resp[:nova].reshape(nvira, nocca) + fvo_a
            res_b += resp[nova:].reshape(nvirb, noccb) + fvo_b
        elif nova:
            res_a += vind_a(za.reshape(1, -1))[0].reshape(nvira, nocca) + fvo_a
        elif novb:
            res_b += vind_b(zb.reshape(1, -1))[0].reshape(nvirb, noccb) + fvo_b
        log.debug(f'Z-vector residual(alpha) = {maxabs(res_a):.2e}')
        log.debug(f'Z-vector residual(beta) = {maxabs(res_b):.2e}')

    return za, zb


# ============================================================================
# Skeleton Gradient
# ============================================================================

def skeleton_gradient(mol, ks_grad, dm_hf, fock_oo_a, fock_oo_b,
                      Coa, Cob, s1, log=None) -> np.ndarray:
    """Compute the spin-summed skeleton gradient."""
    if log is None:
        log = logger.Logger(sys.stdout, logger.NOTE)

    natm = mol.natm
    aoslices = mol.aoslice_by_atom()
    g_skel = np.zeros((natm, 3))

    vhf = ks_grad.get_veff(dm=dm_hf)
    hcore_deriv = ks_grad.hcore_generator()

    dm_sf = dm_hf[0] + dm_hf[1]
    dme0_sf = Coa @ fock_oo_a @ Coa.conj().T + Cob @ fock_oo_b @ Cob.conj().T

    log.debug(f'Skeleton: vhf.shape={vhf.shape}, dme0_sf.shape={dme0_sf.shape}')

    for ia in range(natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        g_1e = lib.einsum('xij,ij->x', hcore_deriv(ia), dm_sf)
        g_2e_xc = lib.einsum('sxij,sij->x', vhf[:, :, p0:p1], dm_hf[:, p0:p1]) * 2.0
        g_ovlp = -lib.einsum('xij,ij->x', s1[:, p0:p1], dme0_sf[p0:p1]) * 2.0
        g_extra = ks_grad.extra_force(ia, locals())
        g_skel[ia] = g_1e + g_2e_xc + g_ovlp + g_extra

    g_nn = grad.rhf.grad_nuc(mol)
    return g_skel + g_nn


# ============================================================================
# Non-Variational Correction (S1 optimized)
# ============================================================================

def nonvariational_correction(mol, Coa, Cob, Cva, Cvb,
                               e_occ_a, e_occ_b, s1, aoslices,
                               zvec, uhf_response, h1ao_a, h1ao_b,
                               log=None) -> np.ndarray:
    """
    Compute spin-resolved non-variational correction.

    g_nv = 2 * sum_ai (Z_alpha_ai V_alpha_ai + Z_beta_ai V_beta_ai)

    The factor is 2 rather than closed-shell 4 because each spin channel is
    singly occupied in the alpha/beta representation.
    """
    if log is None:
        log = logger.Logger(sys.stdout, logger.NOTE)

    natm = mol.natm
    nao = mol.nao
    za, zb = zvec

    dm_s = np.zeros((2, natm, 3, nao, nao))
    s1_co_cache_a = {}
    s1_co_cache_b = {}

    for ia in range(natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]

        s1_co_a = lib.einsum('xpv,vi->xpi', s1[:, p0:p1], Coa)
        s1_oo_a = lib.einsum('xpi,pj->xij', s1_co_a, Coa[p0:p1].conj())
        s1_oo_a = s1_oo_a + s1_oo_a.transpose(0, 2, 1)
        tmp_a = lib.einsum('xij,ui->xuj', s1_oo_a, Coa)
        dm1_a = lib.einsum('xuj,vj->xuv', tmp_a, Coa.conj())
        dm_s[0, ia] = dm1_a + dm1_a.transpose(0, 2, 1).conj()
        s1_co_cache_a[ia] = s1_co_a

        s1_co_b = lib.einsum('xpv,vi->xpi', s1[:, p0:p1], Cob)
        s1_oo_b = lib.einsum('xpi,pj->xij', s1_co_b, Cob[p0:p1].conj())
        s1_oo_b = s1_oo_b + s1_oo_b.transpose(0, 2, 1)
        tmp_b = lib.einsum('xij,ui->xuj', s1_oo_b, Cob)
        dm1_b = lib.einsum('xuj,vj->xuv', tmp_b, Cob.conj())
        dm_s[1, ia] = dm1_b + dm1_b.transpose(0, 2, 1).conj()
        s1_co_cache_b[ia] = s1_co_b

    ax_ao = uhf_response(dm_s.reshape(2, -1, nao, nao))
    ax_vo_a = lib.einsum('Auv,ua,vi->Aai', ax_ao[0], Cva.conj(), Coa)
    ax_vo_b = lib.einsum('Auv,ua,vi->Aai', ax_ao[1], Cvb.conj(), Cob)

    log.debug(f'ax_vo(alpha): max={maxabs(ax_vo_a):.6e}')
    log.debug(f'ax_vo(beta): max={maxabs(ax_vo_b):.6e}')

    g_nv = np.zeros((natm, 3))

    for ia in range(natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]

        f1_vo_a = lib.einsum('xuv,ua,vi->xai', h1ao_a[ia], Cva.conj(), Coa)
        s1_co_a = s1_co_cache_a[ia]
        s1_vo_bra_a = lib.einsum('xpi,pa->xai', s1_co_a, Cva[p0:p1].conj())
        s1_cv_a = lib.einsum('xpv,va->xpa', s1[:, p0:p1], Cva)
        s1_vo_ket_a = lib.einsum('xpa,pi->xai', s1_cv_a, Coa[p0:p1].conj())
        s1_vo_a = s1_vo_bra_a + s1_vo_ket_a
        v1_a = f1_vo_a - s1_vo_a * e_occ_a[None, None, :] - 0.5 * ax_vo_a[ia * 3:(ia + 1) * 3]

        f1_vo_b = lib.einsum('xuv,ua,vi->xai', h1ao_b[ia], Cvb.conj(), Cob)
        s1_co_b = s1_co_cache_b[ia]
        s1_vo_bra_b = lib.einsum('xpi,pa->xai', s1_co_b, Cvb[p0:p1].conj())
        s1_cv_b = lib.einsum('xpv,va->xpa', s1[:, p0:p1], Cvb)
        s1_vo_ket_b = lib.einsum('xpa,pi->xai', s1_cv_b, Cob[p0:p1].conj())
        s1_vo_b = s1_vo_bra_b + s1_vo_ket_b
        v1_b = f1_vo_b - s1_vo_b * e_occ_b[None, None, :] - 0.5 * ax_vo_b[ia * 3:(ia + 1) * 3]

        g_nv[ia] = 2.0 * (
            lib.einsum('ai,xai->x', za, v1_a) +
            lib.einsum('ai,xai->x', zb, v1_b)
        )

    del s1_co_cache_a, s1_co_cache_b

    log.debug(f'g_nv: max={maxabs(g_nv):.6e}')

    return g_nv


# ============================================================================
# Main Gradient Class
# ============================================================================

class Gradients(lib.StreamObject):
    """Restricted open-shell DC-DFT analytic gradient calculator."""

    def __init__(self, mf_rohf, xc: str, grids=None, cphf_tol: float = 1e-10,
                 grid_response: bool = True, auxbasis_response: bool = True):
        self.base = mf_rohf
        self.mol = mf_rohf.mol
        self.xc = xc
        self.grids = grids
        self.cphf_tol = cphf_tol
        self.grid_response = grid_response
        self.auxbasis_response = auxbasis_response

        self.verbose = mf_rohf.verbose
        self.stdout = mf_rohf.stdout
        self.max_memory = mf_rohf.max_memory

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
        """Compute ROHF-density DC-DFT gradient."""
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
        mf_uhf = make_uhf_reference(self.base, mo_coeff=mo_coeff, mo_occ=mo_occ)

        Ca, Cb = mf_uhf.mo_coeff
        ea, eb = mf_uhf.mo_energy
        occa, occb = mf_uhf.mo_occ

        occa_idx = np.where(occa > 0)[0]
        occb_idx = np.where(occb > 0)[0]
        vira_idx = np.where(occa == 0)[0]
        virb_idx = np.where(occb == 0)[0]

        Coa = Ca[:, occa_idx]
        Cob = Cb[:, occb_idx]
        Cva = Ca[:, vira_idx]
        Cvb = Cb[:, virb_idx]

        dm_hf = mf_uhf.make_rdm1((Ca, Cb), (occa, occb))

        ks = dft.UKS(mol)
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

        fock_ks_ao = ks.get_fock(dm=dm_hf)
        fock_ks_a = Ca.conj().T @ fock_ks_ao[0] @ Ca
        fock_ks_b = Cb.conj().T @ fock_ks_ao[1] @ Cb
        fvo_a = fock_ks_a[np.ix_(vira_idx, occa_idx)]
        fvo_b = fock_ks_b[np.ix_(virb_idx, occb_idx)]

        log.note('Max |F^KS_ai(alpha)| = %.6e', maxabs(fvo_a))
        log.note('Max |F^KS_ai(beta)| = %.6e', maxabs(fvo_b))

        log.info('Solving UCPHF Z-vector equation...')
        uhf_resp = mf_uhf.gen_response(mo_coeff=(Ca, Cb), mo_occ=(occa, occb), hermi=1)
        za, zb = solve_z_vector(
            Coa, Cob, Cva, Cvb, ea, eb, occa, occb, fvo_a, fvo_b,
            uhf_resp, cphf_tol=self.cphf_tol, log=log,
        )

        ks_grad = ks.nuc_grad_method()
        ks_grad.grid_response = self.grid_response
        if has_df(ks) and self.auxbasis_response:
            ks_grad.auxbasis_response = True
            log.debug('Auxbasis response enabled')

        s1 = ks_grad.get_ovlp(mol)
        h1ao_a, h1ao_b = mf_uhf.Hessian().make_h1((Ca, Cb), (occa, occb))
        aoslices = mol.aoslice_by_atom()

        log.info('Computing skeleton gradient...')
        fock_oo_a = fock_ks_a[np.ix_(occa_idx, occa_idx)]
        fock_oo_b = fock_ks_b[np.ix_(occb_idx, occb_idx)]
        g_skel = skeleton_gradient(
            mol, ks_grad, dm_hf, fock_oo_a, fock_oo_b, Coa, Cob, s1, log=log,
        )

        log.info('Computing non-variational correction (S1 optimized)...')
        g_nv = nonvariational_correction(
            mol, Coa, Cob, Cva, Cvb, ea[occa_idx], eb[occb_idx],
            s1, aoslices, (za, zb), uhf_resp, h1ao_a, h1ao_b, log=log,
        )

        de = g_skel + g_nv

        if log.verbose >= logger.NOTE:
            log.note('\n' + '='*70)
            log.note('RODC-DFT GRADIENT SUMMARY (S1 Optimized)')
            log.note('='*70)
            print_gradient(mol, g_skel, 'Skeleton gradient (g_skel)')
            print_gradient(mol, g_nv, 'Non-variational correction (g_nv)')
            print_gradient(mol, de, 'Total RODC-DFT gradient')
            log.note('Wall time: %.2f s', logger.perf_counter() - t0[1])
            log.note('='*70)

        if atmlst is not None:
            de = de[np.asarray(atmlst)]

        log.timer('RODC-DFT gradients (S1 opt)', *t0)
        self.de = de
        return de

    def as_scanner(self):
        raise NotImplementedError('Gradient scanner not yet implemented')


RODC_Gradients = Gradients


def kernel(mf_rohf, xc: str, mo_coeff=None, mo_occ=None,
           grids=None, cphf_tol=1e-10, grid_response=True,
           auxbasis_response=True, verbose=None):
    grad_calc = Gradients(
        mf_rohf, xc,
        grids=grids,
        cphf_tol=cphf_tol,
        grid_response=grid_response,
        auxbasis_response=auxbasis_response,
    )
    return grad_calc.kernel(mo_coeff=mo_coeff, mo_occ=mo_occ, verbose=verbose)


__all__ = ["Gradients", "RODC_Gradients", "kernel"]
