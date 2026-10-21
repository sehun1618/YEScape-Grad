# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Restricted source-SCF Z-vector solver using GPU4PySCF CPHF.'''

import cupy as cp
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import cphf
from dcforge.tools.cphf import solve as solve_cphf


def get_z_residual(zvec, mo_energy, mo_occ, fvo, vind):
    '''Evaluate ``[(A+B)^source Z - F^target_vo]`` on the GPU.

    Args:
        zvec : cupy.ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        mo_energy : cupy.ndarray of shape (nmo,)
            Source-SCF orbital energies.
        mo_occ : cupy.ndarray of shape (nmo,)
            Exact restricted occupations.
        fvo : cupy.ndarray of shape (nvir, nocc)
            Target-KS Fock virtual-occupied block.
        vind : callable
            GPU virtual-occupied source-SCF response action.

    Returns:
        cupy.ndarray of shape (nvir, nocc)
            Physical Z-equation residual.
    '''
    e_a = mo_energy[mo_occ == 0]
    e_i = mo_energy[mo_occ > 0]
    return (e_a[:, None] - e_i) * zvec + vind(zvec) - fvo


def gen_vind(Co, Cv, fvind):
    '''Build the conventional GPU restricted source-SCF orbital-Hessian action.

    Args:
        Co : cupy.ndarray of shape (nao, nocc)
            Occupied restricted source-SCF coefficients.
        Cv : cupy.ndarray of shape (nao, nvir)
            Virtual restricted source-SCF coefficients.
        fvind : callable
            AO source-SCF response returned by ``mf_source.gen_response``.

    Returns:
        callable
            Batched VO-space action of the density-response part (J/K and source fxc) of
            ``(A+B)^source``.

    Notes:
        Density fitting overrides this provider with an MO-direct action.
        Keeping the solver independent of the AO/DF representation leaves
        the Z-vector equation and residual diagnostics unchanged.
    '''
    Co = cp.asarray(Co)
    Cv = cp.asarray(Cv)
    nocc = Co.shape[1]
    nvir = Cv.shape[1]
    Co2 = Co * 2

    def vind(x):
        shape = cp.asarray(x).shape
        x = cp.asarray(x).reshape(-1, nvir, nocc)
        mo1 = contract('pa,kai->kpi', Cv, x)
        dm1 = contract('kpi,qi->kpq', mo1, Co2)
        dm1 = dm1 + dm1.transpose(0, 2, 1).conj()
        v1ao = fvind(dm1)
        tmp = contract('ap,kpq->kaq', Cv.conj().T, v1ao)
        v1vo = contract('kaq,qi->kai', tmp, Co)
        return v1vo.reshape(shape)

    return vind


def solve_z_vector(mo_energy, mo_occ, fvo, vind,
                   conv_tol=1e-9, max_cycle=200, log=None):
    '''Solve the positive-RHS molecular restricted source-SCF Z-vector equation on the GPU.

    The canonical equation is

    ``[(A+B)^source]^T Z = F^target_vo``.

    GPU4PySCF CPHF follows the PySCF convention
    ``(A+B)^source x = -h1``.  With ``h1=F^target_vo`` it returns ``x=-Z``;
    this function changes the sign exactly once before returning ``zvec``.

    Args:
        mo_energy : cupy.ndarray of shape (nmo,)
            Canonical source-SCF orbital energies.
        mo_occ : cupy.ndarray of shape (nmo,)
            Restricted source occupations.
        fvo : cupy.ndarray of shape (nvir, nocc)
            Target-KS Fock ``F^target_vo`` at the fixed source density.
        vind : callable
            Conventional AO or density-fitted MO-direct source-SCF
            orbital-Hessian action in the VO space.

    Kwargs:
        conv_tol : float
            Tolerance passed to ``gpu4pyscf.scf.cphf.solve``.
        max_cycle : int
            Maximum GPU CPHF Krylov cycles.
        log : gpu4pyscf.lib.logger.Logger or None
            Logger forwarded to GPU4PySCF CPHF.

    Returns:
        zvec : cupy.ndarray of shape (nvir, nocc)
            Positive-RHS Z-vector.
        info : dict
            Solver identity and independently reconstructed residual metrics.

    Notes:
        ``s1=None`` is intentional because this is the perturbation-independent
        adjoint solve.  Nuclear overlap derivatives enter later in
        ``4 Z^dagger V^lambda``.
    '''
    if log is None:
        log = logger.new_logger(verbose=logger.WARN)

    if fvo.size == 0:
        # No virtual-occupied rotations: the adjoint equation is empty, not
        # a Krylov failure. In particular He/STO-3G has this exact limit.
        return cp.zeros_like(fvo), {
            'rhs_max_abs': 0., 'residual_max_abs': 0.,
            'residual_relative': 0., 'solver': 'zero_dimensional',
        }

    mo_energy = cp.asarray(mo_energy)
    mo_occ = cp.asarray(mo_occ)
    fvo = cp.asarray(fvo)

    x, _ = solve_cphf(cphf,
        vind, mo_energy, mo_occ, fvo, s1=None,
        max_cycle=max_cycle,
        tol=conv_tol,
        verbose=log,
    )
    zvec = -cp.asarray(x)

    rhs_norm = float(cp.max(cp.abs(fvo)).get()) if fvo.size else 0.0
    residual = get_z_residual(zvec, mo_energy, mo_occ, fvo, vind)
    abs_residual = (
        float(cp.max(cp.abs(residual)).get()) if residual.size else 0.0
    )
    if rhs_norm > 0:
        rel_residual = abs_residual / rhs_norm
    elif abs_residual == 0:
        rel_residual = 0.0
    else:
        rel_residual = float('inf')

    info = {
        'rhs_max_abs': rhs_norm,
        'residual_max_abs': abs_residual,
        'residual_relative': rel_residual,
        'solver': 'gpu4pyscf_cphf',
    }
    return zvec, info
