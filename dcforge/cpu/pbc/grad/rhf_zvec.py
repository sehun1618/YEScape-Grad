'''Gamma-point periodic RHF Z-vector solver.'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import cphf


def get_z_residual(zvec, mo_energy, mo_occ, fvo, vind):
    '''Return ``[(A+B)^HF Z - F^PBE_vo]`` in the Gamma VO space.'''
    e_a = mo_energy[mo_occ == 0]
    e_i = mo_energy[mo_occ > 0]
    return (e_a[:, None] - e_i) * zvec + vind(zvec) - fvo


def _make_vind(Co, Cv, fvind):
    nocc = Co.shape[1]
    nvir = Cv.shape[1]

    def vind(x):
        shape = np.asarray(x).shape
        x = np.asarray(x).reshape(-1, nvir, nocc)
        v = np.empty_like(x)
        for k in range(len(x)):
            dm1 = Cv @ x[k] @ Co.conj().T
            dm1 = dm1 + dm1.conj().T
            v1ao = fvind(dm1) * 2
            v[k] = Cv.conj().T @ v1ao @ Co
        return v.reshape(shape)
    return vind


def _krylov_solve(mo_energy, mo_occ, fvo, vind, tol, max_cycle,
                  lindep):
    e_a = mo_energy[mo_occ == 0]
    e_i = mo_energy[mo_occ > 0]
    e_ai = 1 / (e_a[:, None] - e_i)
    x0 = -fvo * e_ai
    nvir, nocc = fvo.shape

    def aop(x):
        shape = np.asarray(x).shape
        x = np.asarray(x).reshape(-1, nvir, nocc)
        return (vind(x) * e_ai).reshape(shape)

    x = lib.krylov(
        aop, x0.ravel(), tol=tol, max_cycle=max_cycle,
        lindep=lindep, verbose=logger.WARN,
    )
    return -np.asarray(x).reshape(fvo.shape)


def _dense_solve(mo_energy, mo_occ, fvo, vind):
    nvir, nocc = fvo.shape
    nov = fvo.size
    eye = np.eye(nov).reshape(nov, nvir, nocc)
    e_a = mo_energy[mo_occ == 0]
    e_i = mo_energy[mo_occ > 0]
    gap = (e_a[:, None] - e_i).ravel()
    actions = vind(eye).reshape(nov, nov) + eye.reshape(nov, nov) * gap
    hess = actions.T
    zvec = np.linalg.solve(hess.T, fvo.ravel()).reshape(fvo.shape)
    return zvec, float(np.max(np.abs(hess - hess.T)))


def solve_z_vector(mo_energy, mo_occ, Co, Cv, fvo, fvind,
                   conv_tol=1e-9, residual_tol=1e-8, max_cycle=200,
                   dense_reference_size=256, log=None):
    '''Solve ``[(A+B)^HF]^T Z = F^PBE_vo`` with a residual gate.

    PySCF CPHF remains the primary solver.  Its internal Krylov method can
    terminate through a linear-dependence threshold before the reconstructed
    physical equation reaches the requested tolerance.  DC-Forge therefore
    recomputes the residual, retries with a smaller ``lindep``, and uses a
    dense adjoint reference only for bounded small Gamma problems.

    Args:
        mo_energy, mo_occ : ndarray
            Canonical source-RHF energies and exact 2/0 occupations.
        Co, Cv : ndarray
            Occupied and virtual Gamma-point MO coefficients.
        fvo : ndarray of shape (nvir, nocc)
            Target PBE Fock virtual-occupied block.
        fvind : callable
            Source periodic RHF response action.

    Kwargs:
        conv_tol : float
            Initial PySCF CPHF Krylov tolerance.
        residual_tol : float
            Absolute acceptance threshold for the independently reconstructed
            physical Z-equation residual.
        max_cycle : int
            Maximum Krylov subspace cycles.
        dense_reference_size : int
            Maximum VO dimension eligible for the dense fallback.
        log : pyscf.lib.logger.Logger or None
            PySCF logger for the primary CPHF solve.

    Returns:
        tuple
            Positive-RHS ``zvec`` and a solver diagnostic dictionary.
    '''
    if log is None:
        log = logger.new_logger(verbose=logger.WARN)
    if np.max(np.abs(np.asarray(Co).imag)) > 1e-10:
        raise NotImplementedError(
            'the first Gamma solver requires real time-reversal orbitals')

    mo_energy = np.asarray(mo_energy).real
    mo_occ = np.asarray(mo_occ)
    Co = np.asarray(Co).real
    Cv = np.asarray(Cv).real
    fvo = np.asarray(fvo).real
    vind = _make_vind(Co, Cv, fvind)

    x, _ = cphf.solve(
        vind, mo_energy, mo_occ, fvo, s1=None,
        max_cycle=max_cycle, tol=conv_tol, verbose=log,
    )
    zvec = -np.asarray(x)
    solver = 'pyscf_cphf'
    matrix_asymmetry = None

    residual = get_z_residual(zvec, mo_energy, mo_occ, fvo, vind)
    abs_residual = float(np.max(np.abs(residual)))
    if abs_residual > residual_tol:
        # tol**2 is PySCF's natural Krylov-vector stopping scale.  Reducing
        # lindep prevents the default 1e-13 floor from masking that request.
        krylov_tol = min(conv_tol, residual_tol * .1)
        lindep = min(1e-18, krylov_tol ** 2)
        zvec = _krylov_solve(
            mo_energy, mo_occ, fvo, vind,
            krylov_tol, max_cycle, lindep)
        solver = 'pyscf_krylov_residual_fallback'
        residual = get_z_residual(zvec, mo_energy, mo_occ, fvo, vind)
        abs_residual = float(np.max(np.abs(residual)))

    if abs_residual > residual_tol and fvo.size <= dense_reference_size:
        zvec, matrix_asymmetry = _dense_solve(
            mo_energy, mo_occ, fvo, vind)
        solver = 'dense_adjoint_fallback'
        residual = get_z_residual(zvec, mo_energy, mo_occ, fvo, vind)
        abs_residual = float(np.max(np.abs(residual)))

    if abs_residual > residual_tol:
        raise RuntimeError(
            'periodic Z-vector physical residual did not converge: '
            f'{abs_residual:.3e} > {residual_tol:.3e}'
        )

    rhs_norm = float(np.max(np.abs(fvo))) if fvo.size else 0.0
    info = {
        'rhs_max_abs': rhs_norm,
        'residual_max_abs': abs_residual,
        'residual_relative': (
            abs_residual / rhs_norm if rhs_norm else abs_residual),
        'solver': solver,
        'dimension': int(fvo.size),
        'matrix_asymmetry_max_abs': matrix_asymmetry,
    }
    return zvec, info
