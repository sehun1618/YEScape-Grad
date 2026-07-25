"""Unrestricted molecular DC-DFT gradient with spin-coupled adjoint response.

The alpha/beta Z transition densities are submitted together to one UHF
response call.  Atom-local overlap contractions replace the former
``(2, 3*Natm, NAO, NAO)`` response-density tensor.  Density-fitted derivative
Fock matrices are consumed from PySCF's generator one atom at a time.

Key differences from restricted:
    - Separate alpha/beta spin channels
    - Z-vector becomes (Z_alpha, Z_beta)
    - UCPHF solver instead of CPHF
    - Factor 2 instead of 4 in g_nv (no double occupancy)
"""

import sys
import numpy as np
from typing import Optional, Callable, Tuple
from pyscf import lib, scf, dft, grad
from pyscf.lib import logger
from pyscf.scf import ucphf, cphf

from .api import (
    maxabs, as_spin_tuple, has_df, get_auxbasis,
    normalize_verbose, print_gradient
)


def _iter_h1ao(mf, mo_coeff, mo_occ):
    """Yield alpha/beta derivative Fock matrices one atom at a time."""

    hess = mf.Hessian()
    if has_df(mf):
        from pyscf.df.hessian import uhf as df_hessian_uhf

        for ia, h1, vj1, vk1 in df_hessian_uhf._gen_jk(
            hess, mo_coeff, mo_occ, with_k=True
        ):
            vk1a, vk1b = vk1
            yield ia, h1 + vj1 - vk1a, h1 + vj1 - vk1b
        return

    h1ao_a, h1ao_b = hess.make_h1(mo_coeff, mo_occ)
    for ia, (h1a, h1b) in enumerate(zip(h1ao_a, h1ao_b)):
        yield ia, h1a, h1b


# ============================================================================
# DC-DFT Energy (PDF Eq. 20, Unrestricted)
# ============================================================================

def energy(mol, xc: str, mo_coeff, mo_occ, grids=None,
           density_fit: bool = False, auxbasis: Optional[str] = None) -> float:
    """
    Compute DC-DFT energy for unrestricted (open-shell) system.

    Args:
        mol: PySCF Mole object
        xc: Exchange-correlation functional
        mo_coeff: UHF MO coefficients (alpha, beta)
        mo_occ: UHF MO occupations (alpha, beta)
        grids: DFT integration grids
        density_fit: Use density fitting
        auxbasis: Auxiliary basis

    Returns:
        DC-DFT total energy
    """
    # Build UHF mean-field
    mf_hf = scf.UHF(mol)
    if density_fit:
        mf_hf = mf_hf.density_fit(auxbasis=auxbasis)

    Ca, Cb = as_spin_tuple(mo_coeff)
    occa, occb = as_spin_tuple(mo_occ)

    mf_hf.mo_coeff = (Ca, Cb)
    mf_hf.mo_occ = (occa, occb)
    mf_hf.converged = True

    # UHF density
    dm_hf = mf_hf.make_rdm1((Ca, Cb), (occa, occb))

    # Build UKS object
    ks = dft.UKS(mol)
    ks.xc = xc
    if density_fit:
        ks = ks.density_fit(auxbasis=auxbasis)

    # Setup grids
    if grids is not None:
        ks.grids = grids
    else:
        ks.grids.atom_grid = (99, 590)
    if ks.grids.coords is None:
        ks.grids.build()

    return ks.energy_tot(dm=dm_hf)


# ============================================================================
# Z-Vector Solver (Unrestricted)
# ============================================================================

def solve_z_vector(Coa: np.ndarray, Cob: np.ndarray,
                   Cva: np.ndarray, Cvb: np.ndarray,
                   ea: np.ndarray, eb: np.ndarray,
                   occa: np.ndarray, occb: np.ndarray,
                   fvo_a: np.ndarray, fvo_b: np.ndarray,
                   uhf_response: Callable,
                   cphf_tol: float = 1e-10, max_cycle: int = 200,
                   log=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Z-vector equation for unrestricted case.

    Returns:
        (za, zb): Z-vectors for alpha and beta spins
    """
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

    log.debug(f'Z-vector (UHF): nocca={nocca}, noccb={noccb}, nvira={nvira}, nvirb={nvirb}')
    log.debug(f'Max |F^KS_ai(α)| = {maxabs(fvo_a):.6e}')
    log.debug(f'Max |F^KS_ai(β)| = {maxabs(fvo_b):.6e}')

    # Define combined vind operator
    def vind(x_vo):
        """Apply (A+B)^UHF to x."""
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

    # Separate alpha/beta vind (for edge cases)
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

    # Solve UCPHF
    log.info(f'Solving UCPHF: tol={cphf_tol:.2e}, max_cycle={max_cycle}')

    if nova and novb:
        # Both spins have virtual-occupied pairs
        za, zb = ucphf.solve(
            vind, (ea, eb), (occa, occb), (fvo_a, fvo_b),
            max_cycle=max_cycle, tol=cphf_tol,
        )[0]
    elif nova:
        # Only alpha
        za = cphf.solve(vind_a, ea, occa, fvo_a,
                        max_cycle=max_cycle, tol=cphf_tol)[0]
        zb = np.zeros((nvirb, noccb))
    elif novb:
        # Only beta
        zb = cphf.solve(vind_b, eb, occb, fvo_b,
                        max_cycle=max_cycle, tol=cphf_tol)[0]
        za = np.zeros((nvira, nocca))
    else:
        # No virtual-occupied pairs (shouldn't happen)
        za = np.zeros((nvira, nocca))
        zb = np.zeros((nvirb, noccb))

    # Verify solution
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

        log.debug(f'Z-vector residual(α) = {maxabs(res_a):.2e}')
        log.debug(f'Z-vector residual(β) = {maxabs(res_b):.2e}')

    return za, zb


# ============================================================================
# Skeleton Gradient (Unrestricted)
# ============================================================================

def skeleton_gradient(mol, ks_grad, dm_hf, fock_oo_a, fock_oo_b,
                      Coa, Cob, s1, log=None) -> np.ndarray:
    """
    Compute skeleton gradient for unrestricted case.

    Args:
        mol: Mole object
        ks_grad: UKS gradient object
        dm_hf: UHF density (spin-free total)
        fock_oo_a/b: KS Fock in occ-occ block (alpha/beta)
        Coa/Cob: Occupied MO coefficients (alpha/beta)
        s1: Overlap derivatives
        log: Logger

    Returns:
        g_skel: Skeleton gradient (natm, 3)
    """
    if log is None:
        log = logger.Logger(sys.stdout, logger.NOTE)

    natm = mol.natm
    aoslices = mol.aoslice_by_atom()
    g_skel = np.zeros((natm, 3))

    vhf = ks_grad.get_veff(dm=dm_hf)
    hcore_deriv = ks_grad.hcore_generator()

    # Spin-free density and energy-weighted density
    dm_sf = dm_hf[0] + dm_hf[1]
    dme0_sf = Coa @ fock_oo_a @ Coa.conj().T + Cob @ fock_oo_b @ Cob.conj().T

    for ia in range(natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]

        # One-electron
        g_1e = lib.einsum('xij,ij->x', hcore_deriv(ia), dm_sf)

        # Two-electron + XC (summed over spins)
        g_2e_xc = lib.einsum('sxij,sij->x', vhf[:, :, p0:p1], dm_hf[:, p0:p1]) * 2.0

        # Overlap correction
        g_ovlp = -lib.einsum('xij,ij->x', s1[:, p0:p1], dme0_sf[p0:p1]) * 2.0

        # Extra forces
        g_extra = ks_grad.extra_force(ia, locals())

        g_skel[ia] = g_1e + g_2e_xc + g_ovlp + g_extra

    g_nn = grad.rhf.grad_nuc(mol)
    return g_skel + g_nn


# ============================================================================
# Non-Variational Correction (Unrestricted)
# ============================================================================

def nonvariational_correction(
    mol, Coa, Cob, Cva, Cvb, e_occ_a, e_occ_b, s1, aoslices, zvec,
    uhf_response, h1ao_a, h1ao_b=None, log=None,
):
    """Compute the UHF correction with one spin-coupled adjoint response."""
    if log is None:
        log = logger.Logger(sys.stdout, logger.NOTE)

    natm = mol.natm
    za, zb = zvec
    dz_a = Cva @ za @ Coa.conj().T
    dz_b = Cvb @ zb @ Cob.conj().T
    dz = np.asarray((dz_a + dz_a.conj().T, dz_b + dz_b.conj().T))
    vz_a, vz_b = uhf_response(dz)
    vz_oo_a = Coa.conj().T @ vz_a @ Coa
    vz_oo_b = Cob.conj().T @ vz_b @ Cob
    del dz_a, dz_b, dz, vz_a, vz_b

    de = np.zeros((natm, 3))

    h1ao_iter = (
        h1ao_a if h1ao_b is None
        else ((ia, h1a, h1b) for ia, (h1a, h1b) in enumerate(zip(h1ao_a, h1ao_b)))
    )
    for ia, h1a, h1b in h1ao_iter:
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        f1_vo_a = lib.einsum('xuv,ua,vi->xai', h1a, Cva.conj(), Coa)
        s1_co_a = lib.einsum('xpv,vi->xpi', s1[:, p0:p1], Coa)
        s1_oo_a = lib.einsum('xpi,pj->xij', s1_co_a, Coa[p0:p1].conj())
        s1_oo_a = s1_oo_a + s1_oo_a.transpose(0, 2, 1).conj()
        s1_vo_bra_a = lib.einsum('xpi,pa->xai', s1_co_a, Cva[p0:p1].conj())
        s1_cv_a = lib.einsum('xpv,va->xpa', s1[:, p0:p1], Cva)
        s1_vo_ket_a = lib.einsum('xpa,pi->xai', s1_cv_a, Coa[p0:p1].conj())
        s1_vo_a = s1_vo_bra_a + s1_vo_ket_a
        f1_vo_b = lib.einsum('xuv,ua,vi->xai', h1b, Cvb.conj(), Cob)
        s1_co_b = lib.einsum('xpv,vi->xpi', s1[:, p0:p1], Cob)
        s1_oo_b = lib.einsum('xpi,pj->xij', s1_co_b, Cob[p0:p1].conj())
        s1_oo_b = s1_oo_b + s1_oo_b.transpose(0, 2, 1).conj()
        s1_vo_bra_b = lib.einsum('xpi,pa->xai', s1_co_b, Cvb[p0:p1].conj())
        s1_cv_b = lib.einsum('xpv,va->xpa', s1[:, p0:p1], Cvb)
        s1_vo_ket_b = lib.einsum('xpa,pi->xai', s1_cv_b, Cob[p0:p1].conj())
        s1_vo_b = s1_vo_bra_b + s1_vo_ket_b

        z_f1 = (
            lib.einsum('ai,xai->x', za, f1_vo_a)
            + lib.einsum('ai,xai->x', zb, f1_vo_b)
        )
        z_se = (
            lib.einsum(
                'ai,xai->x', za, s1_vo_a * e_occ_a[None, None, :]
            )
            + lib.einsum(
                'ai,xai->x', zb, s1_vo_b * e_occ_b[None, None, :]
            )
        )
        z_ax = (
            lib.einsum('xij,ij->x', s1_oo_a, vz_oo_a)
            + lib.einsum('xij,ij->x', s1_oo_b, vz_oo_b)
        )
        de[ia] = 2.0 * z_f1 - 2.0 * z_se - z_ax

    log.debug(f'g_nv: max={maxabs(de):.6e}')

    return de


# ============================================================================
# Main Gradient Class
# ============================================================================

class Gradients(lib.StreamObject):
    """
    Unrestricted DC-DFT analytic gradient calculator.
    """

    def __init__(self, mf_hf, xc: str, grids=None, cphf_tol: float = 1e-10,
                 grid_response: bool = True, auxbasis_response: bool = True):
        """
        Args:
            mf_hf: Converged unrestricted HF mean-field
            xc: DFT functional
            grids: Custom grids
            cphf_tol: CPHF tolerance
            grid_response: Include grid response
            auxbasis_response: Include auxbasis response
        """
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
        """Print calculation settings."""
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
        """
        Compute DC-DFT gradient for unrestricted case.

        Returns:
            de: Nuclear gradients (natm, 3)
        """
        if verbose is not None:
            # Convert boolean to logger level (PySCF convention)
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
        Ca, Cb = as_spin_tuple(mo_coeff)
        ea, eb = as_spin_tuple(self.base.mo_energy)
        occa, occb = as_spin_tuple(mo_occ)

        # Split occ/vir
        occa_idx = np.where(occa > 0)[0]
        occb_idx = np.where(occb > 0)[0]
        vira_idx = np.where(occa == 0)[0]
        virb_idx = np.where(occb == 0)[0]

        Coa = Ca[:, occa_idx]
        Cob = Cb[:, occb_idx]
        Cva = Ca[:, vira_idx]
        Cvb = Cb[:, virb_idx]

        dm_hf = self.base.make_rdm1((Ca, Cb), (occa, occb))

        # Build UKS
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

        # Compute F^KS[ρ^HF]
        fock_ks_ao = ks.get_fock(dm=dm_hf)
        fock_ks_a = Ca.conj().T @ fock_ks_ao[0] @ Ca
        fock_ks_b = Cb.conj().T @ fock_ks_ao[1] @ Cb
        fvo_a = fock_ks_a[np.ix_(vira_idx, occa_idx)]
        fvo_b = fock_ks_b[np.ix_(virb_idx, occb_idx)]

        log.note('Max |F^KS_ai(α)| = %.6e', maxabs(fvo_a))
        log.note('Max |F^KS_ai(β)| = %.6e', maxabs(fvo_b))

        # Solve Z-vector
        log.info('Solving UCPHF...')
        uhf_resp = self.base.gen_response(mo_coeff=(Ca, Cb), mo_occ=(occa, occb), hermi=1)
        za, zb = solve_z_vector(
            Coa, Cob, Cva, Cvb, ea, eb, occa, occb, fvo_a, fvo_b,
            uhf_resp, cphf_tol=self.cphf_tol, log=log,
        )

        # Setup gradient
        ks_grad = ks.nuc_grad_method()
        ks_grad.grid_response = self.grid_response

        if has_df(ks) and self.auxbasis_response:
            ks_grad.auxbasis_response = True
            log.debug('Auxbasis response enabled')

        s1 = ks_grad.get_ovlp(mol)
        h1ao_iter = _iter_h1ao(self.base, (Ca, Cb), (occa, occb))
        aoslices = mol.aoslice_by_atom()

        # Skeleton
        log.info('Computing skeleton gradient...')
        fock_oo_a = fock_ks_a[np.ix_(occa_idx, occa_idx)]
        fock_oo_b = fock_ks_b[np.ix_(occb_idx, occb_idx)]
        g_skel = skeleton_gradient(
            mol, ks_grad, dm_hf, fock_oo_a, fock_oo_b, Coa, Cob, s1, log=log,
        )

        # Non-variational
        log.info('Computing non-variational correction...')
        g_nv = nonvariational_correction(
            mol, Coa, Cob, Cva, Cvb, ea[occa_idx], eb[occb_idx],
            s1, aoslices, (za, zb), uhf_resp, h1ao_iter, log=log,
        )

        de = g_skel + g_nv

        if log.verbose >= logger.NOTE:
            log.note('\n' + '='*70)
            log.note('DC-DFT GRADIENT SUMMARY (Unrestricted)')
            log.note('='*70)
            print_gradient(mol, g_skel, 'Skeleton gradient (g_skel)')
            print_gradient(mol, g_nv, 'Non-variational correction (g_nv)')
            print_gradient(mol, de, 'Total DC-DFT gradient')
            log.note('Wall time: %.2f s', logger.perf_counter() - t0[1])
            log.note('='*70)

        if atmlst is not None:
            de = de[np.asarray(atmlst)]

        log.timer('UDC-DFT gradients', *t0)
        self.de = de
        return de


UDC_Gradients = Gradients


__all__ = ["Gradients", "UDC_Gradients", "energy"]
