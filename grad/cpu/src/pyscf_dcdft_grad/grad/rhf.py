"""Restricted molecular DC-DFT gradient with adjoint response contraction.

The non-variational term applies the self-adjoint HF response once to the Z
transition density instead of once for every nuclear perturbation.  Overlap
blocks are formed and contracted atom by atom.  For density-fitted references,
PySCF's DF Hessian generator is consumed directly so the returned per-atom
``make_h1`` list is never retained in memory.

Key Equations (from DC-DFT_Gradient_v4.pdf):
    - Eq. 20:  DC-DFT Energy
    - Eq. 75:  Z-Vector: (A+B)^HF Z = F^KS_vo
    - Eq. 80:  Total Gradient: dE/dξ = g_skel + g_nv
    - Eq. 87:  Non-variational: g_nv = 4 Σ Z_ai V^ξ_ai

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


def _iter_h1ao(mf, mo_coeff, mo_occ):
    """Yield one atom's derivative Fock and discard it after consumption."""

    hess = mf.Hessian()
    if has_df(mf):
        from pyscf.df.hessian import rhf as df_hessian_rhf

        for ia, h1, vj1, vk1 in df_hessian_rhf._gen_jk(
            hess, mo_coeff, mo_occ, with_k=True
        ):
            yield ia, h1 + vj1 - 0.5 * vk1
        return

    for ia, h1ao in enumerate(hess.make_h1(mo_coeff, mo_occ)):
        yield ia, h1ao


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

def nonvariational_correction(
    mol, Co, Cv, mo_energy, zvec, s1, h1ao_iter, hf_response, log=None,
):
    """Compute the RHF non-variational correction with one adjoint response.

    Self-adjointness moves the HF response from every nuclear perturbation to
    the single Z transition density. Atom-local overlap blocks are then
    contracted immediately; no ``dm_s``, ``ax_ao``, ``ax_vo``, ``Se`` or ``B``
    tensor is materialized.
    """
    if log is None:
        log = logger.Logger(sys.stdout, logger.NOTE)

    natm = mol.natm
    nocc = Co.shape[1]
    aoslices = mol.aoslice_by_atom()
    e_occ = mo_energy[:nocc]
    g_nv = np.zeros((natm, 3))

    dz = Cv @ zvec @ Co.conj().T
    vz = hf_response(dz + dz.conj().T)
    vz_oo = Co.conj().T @ vz @ Co
    del dz, vz

    if isinstance(h1ao_iter, (list, tuple)):
        h1ao_iter = enumerate(h1ao_iter)
    for ia, h1ao in h1ao_iter:
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        f1_vo = lib.einsum('xuv,ua,vi->xai', h1ao, Cv.conj(), Co)
        s1_co = lib.einsum('xpv,vi->xpi', s1[:, p0:p1], Co)
        s1_oo = lib.einsum('xpi,pj->xij', s1_co, Co[p0:p1].conj())
        s1_oo = s1_oo + s1_oo.transpose(0, 2, 1).conj()
        s1_vo_bra = lib.einsum('xpi,pa->xai', s1_co, Cv[p0:p1].conj())
        s1_cv = lib.einsum('xpv,va->xpa', s1[:, p0:p1], Cv)
        s1_vo_ket = lib.einsum('xpa,pi->xai', s1_cv, Co[p0:p1].conj())
        s1_vo = s1_vo_bra + s1_vo_ket
        z_f1 = lib.einsum('ai,xai->x', zvec, f1_vo)
        z_se = lib.einsum(
            'ai,xai->x', zvec, s1_vo * e_occ[None, None, :]
        )
        z_ax = 2.0 * lib.einsum('xij,ij->x', s1_oo, vz_oo)
        g_nv[ia] = 4.0 * z_f1 - 4.0 * z_se - 2.0 * z_ax

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
        h1ao_iter = _iter_h1ao(self.base, mo_coeff, mo_occ)

        # Skeleton gradient
        log.info('Computing skeleton gradient...')
        fock_oo = fock_ks_mo[so, so]
        g_skel = skeleton_gradient(mol, ks_grad, dm_hf, fock_oo, Co, s1, log=log)

        # Non-variational correction (S1 optimized)
        log.info('Computing non-variational correction (S1 optimized)...')
        g_nv = nonvariational_correction(
            mol, Co, Cv, mo_energy, zvec, s1, h1ao_iter, hf_resp, log=log,
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
