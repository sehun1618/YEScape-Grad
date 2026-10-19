'''Unrestricted HF Z-vector solver using GPU4PySCF UCPHF.'''

import cupy as cp
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import cphf, ucphf
from dcforge.tools.cphf import solve as solve_cphf


def gen_vind(Co, Cv, fvind):
    '''Build the spin-coupled GPU UHF orbital-Hessian action.

    Args:
        Co, Cv : tuple of two cupy.ndarray
            Alpha/beta occupied and virtual MO coefficients.
        fvind : callable
            AO UHF response returned by ``mf_hf.gen_response``.

    Returns:
        callable
            Vectorized action accepting GPU4PySCF's packed alpha/beta VO
            amplitudes.
    '''
    Coa, Cob = Co
    Cva, Cvb = Cv
    nocca, noccb = Coa.shape[1], Cob.shape[1]
    nvira, nvirb = Cva.shape[1], Cvb.shape[1]
    nova, novb = nvira * nocca, nvirb * noccb
    nao = Coa.shape[0]

    def vind(x):
        '''Apply the two-electron part of ``(A+B)^UHF`` in packed VO space.'''
        shape = cp.asarray(x).shape
        x = cp.asarray(x).reshape(-1, nova + novb)
        nset = x.shape[0]
        dm1 = cp.zeros((2, nset, nao, nao), dtype=x.dtype)
        out = cp.zeros_like(x)

        if nova:
            xa = x[:, :nova].reshape(nset, nvira, nocca)
            mo1a = contract('pa,kai->kpi', Cva, xa)
            dmaa = contract('kpi,qi->kpq', mo1a, Coa.conj())
            dm1[0] = dmaa + dmaa.transpose(0, 2, 1).conj()
        if novb:
            xb = x[:, nova:].reshape(nset, nvirb, noccb)
            mo1b = contract('pa,kai->kpi', Cvb, xb)
            dmbb = contract('kpi,qi->kpq', mo1b, Cob.conj())
            dm1[1] = dmbb + dmbb.transpose(0, 2, 1).conj()

        v1ao = fvind(dm1)
        if nova:
            tmpa = contract('ap,kpq->kaq', Cva.conj().T, v1ao[0])
            v1a = contract('kaq,qi->kai', tmpa, Coa)
            out[:, :nova] = v1a.reshape(nset, nova)
        if novb:
            tmpb = contract('ap,kpq->kaq', Cvb.conj().T, v1ao[1])
            v1b = contract('kaq,qi->kai', tmpb, Cob)
            out[:, nova:] = v1b.reshape(nset, novb)
        return out.reshape(shape)

    return vind


def get_z_residual(zvec, mo_energy, mo_occ, fvo, vind):
    '''Evaluate both spin blocks of the positive-RHS GPU UHF Z equation.'''
    za, zb = zvec
    occa, occb = mo_occ
    ea, eb = mo_energy
    fvoa, fvob = fvo
    nova = za.size
    packed = cp.hstack((za.ravel(), zb.ravel()))
    if packed.size:
        v = vind(packed)
    else:
        v = packed
    ra = (ea[occa == 0, None] - ea[occa > 0]) * za - fvoa
    rb = (eb[occb == 0, None] - eb[occb > 0]) * zb - fvob
    ra += v[:nova].reshape(za.shape)
    rb += v[nova:].reshape(zb.shape)
    return ra, rb


def solve_z_vector(mo_energy, mo_occ, fvo, vind,
                   conv_tol=1e-9, max_cycle=200, log=None):
    '''Solve the spin-coupled molecular UHF Z-vector equation on the GPU.

    The canonical equation is

    ``[(A+B)^UHF]^T (Z^alpha,Z^beta) = (F^KS,alpha_vo,F^KS,beta_vo)``.

    GPU4PySCF UCPHF uses the negative-RHS convention. The returned amplitudes
    are sign-flipped once so DC-Forge exposes the positive-RHS PDF notation.

    Args:
        mo_energy : tuple or cupy.ndarray of shape (2, nmo)
            Canonical source-UHF orbital energies.
        mo_occ : tuple or cupy.ndarray of shape (2, nmo)
            Exact ``1/0`` source-UHF occupations.
        fvo : tuple of two cupy.ndarray
            Target-UKS virtual-occupied Fock blocks.
        vind : callable
            Conventional AO or DF MO-direct packed spin-coupled Hessian
            action.

    Kwargs:
        conv_tol : float
            Tolerance passed directly to ``gpu4pyscf.scf.ucphf.solve``.
        max_cycle : int
            Maximum GPU UCPHF Krylov cycles.
        log : gpu4pyscf.lib.logger.Logger or None
            Logger passed to GPU4PySCF UCPHF.

    Returns:
        zvec : tuple of two cupy.ndarray
            Positive-RHS alpha/beta Z-vectors.
        info : dict
            Solver and independently reconstructed residual diagnostics.

    Notes:
        ``s1=None`` is intentional. This is the perturbation-independent
        adjoint solve; nuclear overlap derivatives enter ``V^lambda`` later.
    '''
    if log is None:
        log = logger.new_logger(verbose=logger.WARN)

    mo_energy = cp.asarray(mo_energy)
    mo_occ = cp.asarray(mo_occ)
    fvo = tuple(cp.asarray(x) for x in fvo)
    occa, occb = mo_occ
    ea, eb = mo_energy
    fvoa, fvob = fvo
    nova, novb = fvoa.size, fvob.size

    if nova and novb:
        zvec, _ = solve_cphf(ucphf,
            vind, mo_energy, mo_occ, fvo, s1=None,
            max_cycle=max_cycle, tol=conv_tol, verbose=log,
        )
        zvec = (-cp.asarray(zvec[0]), -cp.asarray(zvec[1]))
        solver = 'gpu4pyscf_ucphf'
    elif nova:
        def vind_a(x):
            shape = cp.asarray(x).shape
            x = cp.asarray(x).reshape(-1, nova)
            packed = cp.zeros((x.shape[0], nova + novb), dtype=x.dtype)
            packed[:, :nova] = x
            return vind(packed).reshape(-1, nova + novb)[:, :nova].reshape(shape)

        za, _ = solve_cphf(cphf,
            vind_a, ea, occa, fvoa, s1=None,
            max_cycle=max_cycle, tol=conv_tol, verbose=log,
        )
        zvec = (-cp.asarray(za), cp.zeros_like(fvob))
        solver = 'gpu4pyscf_cphf_alpha'
    elif novb:
        def vind_b(x):
            shape = cp.asarray(x).shape
            x = cp.asarray(x).reshape(-1, novb)
            packed = cp.zeros((x.shape[0], nova + novb), dtype=x.dtype)
            packed[:, nova:] = x
            return vind(packed).reshape(-1, nova + novb)[:, nova:].reshape(shape)

        zb, _ = solve_cphf(cphf,
            vind_b, eb, occb, fvob, s1=None,
            max_cycle=max_cycle, tol=conv_tol, verbose=log,
        )
        zvec = (cp.zeros_like(fvoa), -cp.asarray(zb))
        solver = 'gpu4pyscf_cphf_beta'
    else:
        zvec = (cp.zeros_like(fvoa), cp.zeros_like(fvob))
        solver = 'zero_dimensional'

    residual = get_z_residual(zvec, mo_energy, mo_occ, fvo, vind)
    rhs_norm_spin = tuple(
        float(cp.max(cp.abs(x)).get()) if x.size else 0.0 for x in fvo
    )
    residual_spin = tuple(
        float(cp.max(cp.abs(x)).get()) if x.size else 0.0 for x in residual
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
