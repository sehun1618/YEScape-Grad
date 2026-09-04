'''Restricted HF Z-vector solver for molecular HF-DFT gradients.'''

import numpy as np
from pyscf.lib import logger
from pyscf.scf import cphf


def get_z_residual(zvec, mo_energy, mo_occ, fvo, vind):
    '''Evaluate the residual of the canonical Z-vector equation.

    The notation follows ``DC_DFT_Analytic_Gradient_Textbook_KR_AB.pdf``:

    ``[(A+B)^HF]^T Z = F^KS_vo``.

    For the real closed-shell RHF reference used here, the orbital Hessian is
    symmetric.  The implemented residual is therefore
    ``(A+B)^HF Z - F^KS_vo``.

    Args:
        zvec : ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector ``Z_ai``.
        mo_energy : ndarray of shape (nmo,)
            Canonical source-RHF orbital energies.
        mo_occ : ndarray of shape (nmo,)
            Restricted occupations. Occupied and virtual orbitals are selected
            by ``mo_occ > 0`` and ``mo_occ == 0``, respectively.
        fvo : ndarray of shape (nvir, nocc)
            Target-KS Fock block ``F^KS_vo`` evaluated on the HF density.
        vind : callable
            Induced HF potential action in the virtual-occupied space.  It
            returns the two-electron part of ``(A+B)^HF zvec``.

    Returns:
        ndarray of shape (nvir, nocc)
            Elementwise residual ``[(A+B)^HF Z - F^KS_vo]_ai``.
    '''
    e_a = mo_energy[mo_occ == 0]
    e_i = mo_energy[mo_occ > 0]
    return (e_a[:, None] - e_i) * zvec + vind(zvec) - fvo


def _relative_residual(abs_residual, rhs_norm):
    '''Return the max-absolute residual relative to ``max|F^KS_vo|``.'''
    if rhs_norm > 0:
        return abs_residual / rhs_norm
    if abs_residual == 0:
        return 0.0
    return float('inf')


def solve_z_vector(mo_energy, mo_occ, Co, Cv, fvo, fvind,
                   conv_tol=1e-9, max_cycle=200, log=None):
    '''Solve the molecular restricted-HF Z-vector equation with PySCF CPHF.

    The canonical equation is

    ``[(A+B)^HF]^T Z = F^KS_vo``.

    The molecular real-RHF Hessian is symmetric, so the transpose is implicit
    in the matrix-vector action below.  PySCF's ``cphf.solve`` uses the
    convention ``(A+B)^HF x = -h1``.  Passing ``h1=F^KS_vo`` therefore returns
    ``x=-Z``; the sign is changed once before exposing the DC-Forge Z-vector.

    Args:
        mo_energy : ndarray of shape (nmo,)
            Canonical source-RHF orbital energies.
        mo_occ : ndarray of shape (nmo,)
            Restricted source-RHF occupations.
        Co : ndarray of shape (nao, nocc)
            Occupied source-RHF MO coefficients ``C_o``.
        Cv : ndarray of shape (nao, nvir)
            Virtual source-RHF MO coefficients ``C_v``.
        fvo : ndarray of shape (nvir, nocc)
            Target-KS Fock block ``F^KS_vo`` evaluated on the HF density.
        fvind : callable
            AO source-HF response returned by ``mf_hf.gen_response``.

    Kwargs:
        conv_tol : float
            Krylov tolerance passed directly to ``pyscf.scf.cphf.solve``.
            It is not interpreted as a bound on the reconstructed physical
            residual.
        max_cycle : int
            Maximum number of PySCF CPHF Krylov cycles.
        log : pyscf.lib.logger.Logger or None
            Logger forwarded to the PySCF CPHF solver.

    Returns:
        zvec : ndarray of shape (nvir, nocc)
            Positive-RHS Z-vector satisfying the canonical equation to the
            accuracy reached by PySCF CPHF.
        info : dict
            Solver name, RHS norm, and independently reconstructed absolute
            and RHS-relative residual diagnostics.

    Notes:
        ``s1=None`` in ``cphf.solve`` is intentional.  This solve is the
        perturbation-independent adjoint equation for ``Z`` at the current
        geometry; it is not the direct first-order orbital response
        ``U^lambda`` for a particular nuclear displacement.  The moving-basis
        quantities ``S^lambda`` enter the perturbation-specific source
        ``V^lambda`` and are contracted with the already solved Z-vector in
        ``nonvariational_correction``.
    '''
    if log is None:
        log = logger.new_logger(verbose=logger.WARN)

    nocc = Co.shape[1]
    nvir = Cv.shape[1]

    def vind(x):
        '''Apply the two-electron part of ``(A+B)^HF`` in the VO space.'''
        shape = np.asarray(x).shape
        x = np.asarray(x).reshape(-1, nvir, nocc)
        v = np.empty_like(x)

        # A VO rotation x_ai produces the Hermitian first-order AO density
        # C_v x C_o^dagger + h.c.  The restricted factor 2 converts the
        # source-HF response into the closed-shell (A+B)^HF action used in the
        # textbook convention.  Multiple RHSs are accepted in PySCF layout.
        for k in range(x.shape[0]):
            dm1 = Cv @ x[k] @ Co.conj().T
            dm1 = dm1 + dm1.conj().T
            v1ao = fvind(dm1) * 2
            v[k] = Cv.conj().T @ v1ao @ Co
        return v.reshape(shape)

    # No geometric overlap perturbation belongs to this adjoint solve.
    # Omitting s1 selects PySCF's fixed-metric VO CPHF equation.  Supplying
    # s1 is required instead for direct U^lambda response to a nuclear
    # displacement, as in pyscf.hessian.rhf.solve_mo1.
    zvec, _ = cphf.solve(vind, mo_energy, mo_occ, fvo, s1=None,
                         max_cycle=max_cycle, tol=conv_tol, verbose=log)

    # PySCF solves (A+B)^HF x = -F^KS_vo.  DC-Forge exposes the textbook
    # positive-RHS vector Z satisfying [(A+B)^HF]^T Z = F^KS_vo.
    zvec = -np.asarray(zvec)

    # PySCF's second return value is not a convergence flag in the s1=None
    # route.  Reconstruct the physical equation residual for diagnostics.
    rhs_norm = float(np.max(np.abs(fvo))) if fvo.size else 0.0
    residual = get_z_residual(zvec, mo_energy, mo_occ, fvo, vind)
    abs_residual = float(np.max(np.abs(residual))) if residual.size else 0.0
    rel_residual = _relative_residual(abs_residual, rhs_norm)

    info = {
        'rhs_max_abs': rhs_norm,
        'residual_max_abs': abs_residual,
        'residual_relative': rel_residual,
        'solver': 'pyscf_cphf',
    }
    return zvec, info
