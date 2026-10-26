# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Unrestricted source-SCF Z-vector solver for molecular non-SCF gradients.'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import cphf, ucphf
from dcforge.tools.cphf import solve as solve_cphf


def gen_vind(Co, Cv, fvind):
    '''Build the spin-coupled unrestricted source-SCF orbital-Hessian response action.

    Args:
        Co : tuple of two ndarray
            Alpha/beta occupied MO coefficients, each ``(nao, nocc_σ)``.
        Cv : tuple of two ndarray
            Alpha/beta virtual MO coefficients, each ``(nao, nvir_σ)``.
        fvind : callable
            AO unrestricted source-SCF response returned by ``mf_source.gen_response``.

    Returns:
        callable
            Vectorized action accepting PySCF's packed alpha/beta VO layout.

    Notes:
        The contraction order follows the measured fast GPU path, but uses
        NumPy/PySCF arrays here. It forms ``C_v X`` first and projects the
        response one MO index at a time. This avoids a Python loop over Krylov
        vectors and avoids a single four-index AO-to-MO ``einsum``.
    '''
    Coa, Cob = Co
    Cva, Cvb = Cv
    nocca, noccb = Coa.shape[1], Cob.shape[1]
    nvira, nvirb = Cva.shape[1], Cvb.shape[1]
    nova, novb = nvira * nocca, nvirb * noccb
    nao = Coa.shape[0]

    def vind(x):
        shape = np.asarray(x).shape
        x = np.asarray(x).reshape(-1, nova + novb)
        nset = x.shape[0]
        dm1 = np.zeros((2, nset, nao, nao), dtype=x.dtype)
        out = np.zeros_like(x)

        if nova:
            xa = x[:, :nova].reshape(nset, nvira, nocca)
            mo1a = lib.einsum('pa,kai->kpi', Cva, xa)
            dmaa = lib.einsum('kpi,qi->kpq', mo1a, Coa.conj())
            dm1[0] = dmaa + dmaa.transpose(0, 2, 1).conj()
        if novb:
            xb = x[:, nova:].reshape(nset, nvirb, noccb)
            mo1b = lib.einsum('pa,kai->kpi', Cvb, xb)
            dmbb = lib.einsum('kpi,qi->kpq', mo1b, Cob.conj())
            dm1[1] = dmbb + dmbb.transpose(0, 2, 1).conj()

        v1ao = fvind(dm1)
        if nova:
            tmpa = lib.einsum('ap,kpq->kaq', Cva.conj().T, v1ao[0])
            v1a = lib.einsum('kaq,qi->kai', tmpa, Coa)
            out[:, :nova] = v1a.reshape(nset, nova)
        if novb:
            tmpb = lib.einsum('ap,kpq->kaq', Cvb.conj().T, v1ao[1])
            v1b = lib.einsum('kaq,qi->kai', tmpb, Cob)
            out[:, nova:] = v1b.reshape(nset, novb)
        return out.reshape(shape)

    return vind


def get_z_residual(zvec, mo_energy, mo_occ, fvo, vind):
    '''Evaluate both spin blocks of the positive-RHS unrestricted source-SCF Z equation.

    Args:
        zvec : tuple of two ndarray
            Positive-RHS alpha/beta Z-vectors ``Z^σ_ai``.
        mo_energy : tuple of two ndarray
            Canonical unrestricted source-SCF orbital energies.
        mo_occ : tuple of two ndarray
            Unrestricted source-SCF occupations.
        fvo : tuple of two ndarray
            Target-UKS ``F^target,alpha_vo`` and ``F^target,beta_vo`` blocks.
        vind : callable
            Packed spin-coupled density-response (J/K and XC kernel) source-SCF Hessian action.

    Returns:
        tuple of two ndarray
            Alpha/beta residual blocks.
    '''
    za, zb = zvec
    occa, occb = mo_occ
    ea, eb = mo_energy
    fvoa, fvob = fvo
    nova = za.size
    packed = np.hstack((za.ravel(), zb.ravel()))
    if packed.size:
        v = vind(packed)
    else:
        v = packed
    ra = (ea[occa == 0, None] - ea[occa > 0]) * za - fvoa
    rb = (eb[occb == 0, None] - eb[occb > 0]) * zb - fvob
    ra += v[:nova].reshape(za.shape)
    rb += v[nova:].reshape(zb.shape)
    return ra, rb


def solve_z_vector(mo_energy, mo_occ, Co, Cv, fvo, fvind,
                   conv_tol=1e-9, max_cycle=200, log=None, vind=None):
    '''Solve the spin-coupled molecular unrestricted source-SCF Z-vector equation.

    The canonical equation is

    ``[(A+B)^source]^T (Z^alpha,Z^beta) = (F^target,alpha_vo,F^target,beta_vo)``.

    PySCF UCPHF uses the negative-RHS convention. The returned amplitudes are
    sign-flipped once so that DC-Forge exposes the positive-RHS convention.

    Args:
        mo_energy : tuple of two ndarray
            Canonical unrestricted source-SCF orbital energies.
        mo_occ : tuple of two ndarray
            Exact ``1/0`` unrestricted source-SCF occupations.
        Co : tuple of two ndarray
            Alpha/beta occupied MO coefficients.
        Cv : tuple of two ndarray
            Alpha/beta virtual MO coefficients.
        fvo : tuple of two ndarray
            Target-UKS virtual-occupied Fock blocks.
        fvind : callable
            AO unrestricted source-SCF response action.

    Kwargs:
        vind : callable or None
            Optional MO-direct solver action (packed spin layout for UHF).
            Final physical residuals always use the independent AO action.
        conv_tol : float
            Tolerance passed directly to ``pyscf.scf.ucphf.solve``.
        max_cycle : int
            Maximum UCPHF Krylov cycles.
        log : pyscf.lib.logger.Logger or None
            Logger passed to PySCF UCPHF.

    Returns:
        zvec : tuple of two ndarray
            Positive-RHS alpha/beta Z-vectors.
        info : dict
            Solver and independently reconstructed residual diagnostics.

    Notes:
        ``s1=None`` is intentional because this is the geometry-independent
        adjoint solve. Nuclear overlap derivatives enter ``V^lambda`` later.
    '''
    if log is None:
        log = logger.new_logger(verbose=logger.WARN)

    ao_vind = gen_vind(Co, Cv, fvind)
    if vind is None:
        vind = ao_vind
    occa, occb = mo_occ
    ea, eb = mo_energy
    fvoa, fvob = fvo
    nova, novb = fvoa.size, fvob.size

    if nova and novb:
        zvec, _ = solve_cphf(ucphf,
            vind, mo_energy, mo_occ, fvo, s1=None,
            max_cycle=max_cycle, tol=conv_tol, verbose=log,
        )
        zvec = (-np.asarray(zvec[0]), -np.asarray(zvec[1]))
        solver = 'pyscf_ucphf'
    elif nova:
        def vind_a(x):
            shape = np.asarray(x).shape
            x = np.asarray(x).reshape(-1, nova)
            packed = np.zeros((x.shape[0], nova + novb), dtype=x.dtype)
            packed[:, :nova] = x
            return vind(packed).reshape(-1, nova + novb)[:, :nova].reshape(shape)

        za, _ = solve_cphf(cphf,
            vind_a, ea, occa, fvoa, s1=None,
            max_cycle=max_cycle, tol=conv_tol, verbose=log,
        )
        zvec = (-np.asarray(za), np.zeros_like(fvob))
        solver = 'pyscf_cphf_alpha'
    elif novb:
        def vind_b(x):
            shape = np.asarray(x).shape
            x = np.asarray(x).reshape(-1, novb)
            packed = np.zeros((x.shape[0], nova + novb), dtype=x.dtype)
            packed[:, nova:] = x
            return vind(packed).reshape(-1, nova + novb)[:, nova:].reshape(shape)

        zb, _ = solve_cphf(cphf,
            vind_b, eb, occb, fvob, s1=None,
            max_cycle=max_cycle, tol=conv_tol, verbose=log,
        )
        zvec = (np.zeros_like(fvoa), -np.asarray(zb))
        solver = 'pyscf_cphf_beta'
    else:
        zvec = (np.zeros_like(fvoa), np.zeros_like(fvob))
        solver = 'zero_dimensional'

    residual = get_z_residual(zvec, mo_energy, mo_occ, fvo, ao_vind)
    rhs_norm_spin = tuple(
        float(np.max(np.abs(x))) if x.size else 0.0 for x in fvo
    )
    residual_spin = tuple(
        float(np.max(np.abs(x))) if x.size else 0.0 for x in residual
    )
    rhs_norm = max(rhs_norm_spin)
    abs_residual = max(residual_spin)
    if rhs_norm > 0:
        relative = abs_residual / rhs_norm
    elif abs_residual == 0:
        relative = 0.0
    else:
        relative = float('inf')

    info = {
        'rhs_max_abs': rhs_norm,
        'rhs_max_abs_spin': rhs_norm_spin,
        'residual_max_abs': abs_residual,
        'residual_max_abs_spin': residual_spin,
        'residual_relative': relative,
        'solver': solver,
    }
    return zvec, info
