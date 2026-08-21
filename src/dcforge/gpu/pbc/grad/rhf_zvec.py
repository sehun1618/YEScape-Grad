'''Gamma-point GPU RHF adjoint Z-vector solver.'''

import time

import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator, gmres, minres
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import cphf


def get_z_residual(zvec, mo_energy, mo_occ, fvo, vind):
    '''Return ``[(A+B)^HF Z - F^target_vo]`` in the Gamma VO space.

    Args:
        zvec : cupy.ndarray of shape (nvir, nocc)
            Candidate positive-right-hand-side adjoint amplitude.
        mo_energy, mo_occ : cupy.ndarray
            Source-RHF canonical orbital energies and 2/0 occupations.
        fvo : cupy.ndarray of shape (nvir, nocc)
            Fixed-density target-DFT Fock virtual-occupied block.
        vind : callable
            Two-electron ``(A+B)^HF`` action in the VO space.

    Returns:
        cupy.ndarray of shape (nvir, nocc)
            Independently reconstructed physical residual.  This is not the
            preconditioned Krylov residual returned by a solver.
    '''
    e_a = mo_energy[mo_occ == 0]
    e_i = mo_energy[mo_occ > 0]
    return (e_a[:, None] - e_i) * zvec + vind(zvec) - fvo


def gen_vind(Co, Cv, fvind):
    '''Build the GPU source-RHF orbital-Hessian action in the VO space.

    Args:
        Co, Cv : cupy.ndarray
            Occupied and virtual source-RHF coefficient matrices.
        fvind : callable
            AO source-HF response returned by GPU4PySCF.

    Returns:
        callable
            Real Gamma-point ``(A+B)^HF`` two-electron VO action.
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
        # GPU4PySCF's current Gamma PBC ``gen_response`` accepts one AO
        # density at a time, unlike its molecular counterpart.  Keep the
        # Krylov/dense solver batch interface and apply that provider action
        # one trial density at a time; the persistent response tensors remain
        # device-resident.
        v1ao = cp.stack([fvind(dm1_k) for dm1_k in dm1])
        tmp = contract('ap,kpq->kaq', Cv.conj().T, v1ao)
        v1vo = contract('kaq,qi->kai', tmp, Co)
        return v1vo.reshape(shape)

    return vind


def _dense_solve(mo_energy, mo_occ, fvo, vind):
    '''Return a small real-Gamma dense adjoint reference solution.

    This bounded oracle explicitly applies the source-HF Hessian to every
    VO basis direction.  It is selected only for small systems when CPHF's
    independently reconstructed physical residual exceeds its acceptance
    threshold.

    Args:
        mo_energy, mo_occ, fvo, vind
            Same quantities accepted by :func:`solve_z_vector`.

    Returns:
        tuple
            ``(zvec, matrix_asymmetry_max_abs)``.  The second item audits the
            row-built adjoint matrix before the dense solve.
    '''
    nvir, nocc = fvo.shape
    nov = fvo.size
    eye = cp.eye(nov, dtype=fvo.dtype).reshape(nov, nvir, nocc)
    e_a = mo_energy[mo_occ == 0]
    e_i = mo_energy[mo_occ > 0]
    action = vind(eye).reshape(nov, nov)
    action += eye.reshape(nov, nov) * (e_a[:, None] - e_i).ravel()
    # ``action[i]`` is A applied to basis vector i; its transpose is the
    # conventional column-major matrix.  The Gamma RHF action is real-linear
    # and symmetric, but retaining this orientation makes the residual check
    # independent of that symmetry assumption.
    matrix = action.T
    zvec = cp.linalg.solve(matrix, fvo.ravel()).reshape(nvir, nocc)
    asymmetry = float(cp.max(cp.abs(matrix - matrix.T)).get())
    return zvec, asymmetry


def _gmres_solve(mo_energy, mo_occ, fvo, vind, krylov_tol, max_cycle,
                 initial_guess=None):
    '''Solve the device-resident VO adjoint equation after CPHF stagnation.

    The operator is evaluated as ``D x + vind(x)``, where ``D`` is the
    canonical occupied--virtual energy-gap diagonal.  ``D^{-1}`` is used as
    a right-independent left preconditioner; it affects Krylov convergence
    only, not the exact adjoint equation.  All vectors remain CuPy arrays.

    Args:
        mo_energy, mo_occ : cupy.ndarray
            Source-RHF canonical spectrum and exact 2/0 occupations.
        fvo : cupy.ndarray of shape (nvir, nocc)
            Positive right-hand side of the adjoint equation.
        vind : callable
            Source-HF two-electron VO Hessian action.
        krylov_tol : float
            Effective relative Krylov stopping criterion resolved by
            :func:`solve_z_vector`.  The solver does not silently tighten it.
        max_cycle : int
            Maximum restarted GMRES cycle count.

    Returns:
        tuple
            ``(zvec, info)`` with ``zvec`` in VO shape and CuPy GMRES's
            nonnegative integer completion code.
    '''
    nvir, nocc = fvo.shape
    nov = fvo.size
    gap = (mo_energy[mo_occ == 0, None] - mo_energy[mo_occ > 0]).ravel()

    def matvec(vector):
        vector = cp.asarray(vector).reshape(nvir, nocc)
        return (gap.reshape(nvir, nocc) * vector + vind(vector)).ravel()

    def precondition(vector):
        return cp.asarray(vector) / gap

    operator = LinearOperator((nov, nov), matvec=matvec, dtype=fvo.dtype)
    inverse_gap = LinearOperator(
        (nov, nov), matvec=precondition, dtype=fvo.dtype)
    iteration_count = 0

    def count_iteration(_residual):
        nonlocal iteration_count
        iteration_count += 1

    zvec, info = gmres(
        operator, fvo.ravel(), M=inverse_gap,
        rtol=float(krylov_tol), atol=0.,
        restart=min(nov, 80), maxiter=int(max_cycle),
        # The diagonal-gap solution is both a useful initial guess and avoids
        # passing GMRES's otherwise exact-zero initial vector into the
        # GPU4PySCF GDF low-rank density factorizer.  The latter has no AO
        # rank to reshape for an exactly zero trial density.  No approximation
        # is introduced: GMRES still solves the full ``gap + vind`` operator.
        x0=(fvo.ravel() / gap if initial_guess is None
            else cp.asarray(initial_guess).ravel()),
        callback=count_iteration,
        callback_type='pr_norm')
    if info < 0:
        raise RuntimeError(f'GPU GMRES failed with info={info}')
    return cp.asarray(zvec).reshape(nvir, nocc), int(info), iteration_count


def _minres_solve(mo_energy, mo_occ, fvo, vind, krylov_tol, max_cycle,
                  initial_guess=None):
    '''Solve the real-symmetric Gamma adjoint with preconditioned MINRES.

    At real Gamma, the restricted orbital Hessian is self-adjoint in the real
    VO rotation basis.  MINRES can therefore avoid GMRES's restarted Arnoldi
    storage.  It is an optional refinement route; the same unpreconditioned
    residual gate in :func:`solve_z_vector` decides acceptance.

    Args:
        mo_energy, mo_occ, fvo, vind, conv_tol, max_cycle
            Same quantities accepted by :func:`_gmres_solve`.

    Returns:
        tuple
            ``(zvec, info, iteration_count)`` in the same convention as
            :func:`_gmres_solve`.
    '''
    nvir, nocc = fvo.shape
    nov = fvo.size
    gap = (mo_energy[mo_occ == 0, None] - mo_energy[mo_occ > 0]).ravel()

    def matvec(vector):
        vector = cp.asarray(vector).reshape(nvir, nocc)
        return (gap.reshape(nvir, nocc) * vector + vind(vector)).ravel()

    def precondition(vector):
        return cp.asarray(vector) / gap

    operator = LinearOperator((nov, nov), matvec=matvec, dtype=fvo.dtype)
    inverse_gap = LinearOperator(
        (nov, nov), matvec=precondition, dtype=fvo.dtype)
    iteration_count = 0

    def count_iteration(_solution):
        nonlocal iteration_count
        iteration_count += 1

    zvec, info = minres(
        operator, fvo.ravel(), M=inverse_gap,
        tol=float(krylov_tol), maxiter=int(max_cycle),
        # A nonzero diagonal-gap initial guess preserves the GDF provider
        # contract just as in GMRES: its factorizer cannot reshape an exact
        # zero transition density, although the mathematical action is zero.
        x0=(fvo.ravel() / gap if initial_guess is None
            else cp.asarray(initial_guess).ravel()),
        callback=count_iteration)
    if info < 0:
        raise RuntimeError(f'GPU MINRES failed with info={info}')
    return cp.asarray(zvec).reshape(nvir, nocc), int(info), iteration_count


def solve_z_vector(mo_energy, mo_occ, fvo, vind, conv_tol=1e-9,
                   residual_tol=1e-8, max_cycle=200,
                   dense_oracle_size=256, refinement_solver='minres',
                   initial_guess=None, krylov_tol=None, log=None):
    '''Solve ``[(A+B)^HF]^T Z = F^target_vo`` with a residual gate.

    ``s1=None`` is deliberate: this is the geometry-independent adjoint
    equation.  Nuclear overlap derivatives enter the explicit response terms
    after this solve.  For a small VO space, GPU4PySCF CPHF is retained as a
    reference primary followed by the dense oracle if required.  For a large
    real-Gamma VO space, the selected matrix-free MINRES/GMRES solver is
    entered directly: first running CPHF and then the same Krylov equation
    would duplicate every expensive GDF response action.

    Args:
        mo_energy, mo_occ : cupy.ndarray
            Canonical source-RHF orbital energies and exact 2/0 occupations.
        fvo : cupy.ndarray of shape (nvir, nocc)
            Target PBE Fock virtual-occupied block.
        vind : callable
            Source periodic RHF two-electron response action in VO space.

    Kwargs:
        conv_tol : float
            Requested GPU4PySCF CPHF Krylov tolerance.
        krylov_tol : float or None
            Explicit matrix-free MINRES/GMRES stopping tolerance. ``None``
            preserves the production safety policy
            ``min(conv_tol, 1e-10)``. A positive explicit value is an
            experiment-only override; it never replaces ``residual_tol`` as
            the physical acceptance gate. Small systems that use the
            GPU4PySCF CPHF/dense route do not consume this option.
        residual_tol : float
            Absolute threshold on the reconstructed unpreconditioned physical
            residual.  This is the acceptance criterion for every solver.
        max_cycle : int
            Maximum CPHF or selected matrix-free Krylov iterations.
        dense_oracle_size : int
            Maximum VO dimension eligible for explicit dense fallback.
        refinement_solver : {'minres', 'gmres'}
            Device-resident Krylov method used only after primary CPHF fails
            the physical residual gate.  ``minres`` is the default for this
            real-Gamma restricted Hessian.  If MINRES completes but does not
            pass the independently reconstructed residual gate, DC-Forge
            retries the same equation with GMRES before rejecting it.
        initial_guess : cupy.ndarray or None
            Optional geometry-scanner guess already projected into the
            current virtual--occupied basis.  It is accepted only for the
            large matrix-free route and only when its independently evaluated
            physical residual is no worse than the diagonal-gap guess.
        log : gpu4pyscf.lib.logger.Logger or None
            Logger passed to the primary GPU4PySCF CPHF solve.

    Returns:
        tuple
            ``(zvec, info)``.  ``info`` records the actual solver route and
            absolute/relative physical residual used for acceptance.
    '''
    if log is None:
        log = logger.new_logger(verbose=logger.WARN)
    mo_energy = cp.asarray(mo_energy)
    mo_occ = cp.asarray(mo_occ)
    fvo = cp.asarray(fvo)
    refinement_solver = str(refinement_solver).lower()
    if refinement_solver not in ('gmres', 'minres'):
        raise ValueError("refinement_solver must be 'gmres' or 'minres'")
    conv_tol = float(conv_tol)
    residual_tol = float(residual_tol)
    if conv_tol <= 0. or residual_tol <= 0.:
        raise ValueError('conv_tol and residual_tol must be positive')
    requested_krylov_tol = (
        None if krylov_tol is None else float(krylov_tol))
    if requested_krylov_tol is not None and requested_krylov_tol <= 0.:
        raise ValueError('krylov_tol must be positive or None')

    total_start = time.perf_counter()
    gap = (mo_energy[mo_occ == 0, None] - mo_energy[mo_occ > 0])
    diagonal_guess = fvo / gap
    initial_guess_requested = initial_guess is not None
    initial_guess_used = False
    initial_guess_fallback_reason = None
    initial_residual_max_abs = None
    diagonal_initial_residual_max_abs = None
    # The dense-oracle threshold is also the safe cost boundary for the
    # upstream CPHF primary.  Above it, one CPHF attempt followed by MINRES
    # evaluates the same expensive Gamma GDF response operator twice.  The
    # direct matrix-free route remains exact and is accepted only through the
    # same independently reconstructed residual below.
    use_cphf_primary = fvo.size <= dense_oracle_size
    production_krylov_tol = min(conv_tol, 1e-10)
    if use_cphf_primary:
        effective_krylov_tol = conv_tol
        krylov_tolerance_policy = 'gpu4pyscf_cphf_requested'
    elif requested_krylov_tol is None:
        effective_krylov_tol = production_krylov_tol
        krylov_tolerance_policy = 'production_safety_cap'
    else:
        effective_krylov_tol = requested_krylov_tol
        krylov_tolerance_policy = 'explicit_matrix_free_experiment'
    selected_initial_guess = None
    if initial_guess_requested and use_cphf_primary:
        initial_guess_fallback_reason = (
            'small VO space uses the GPU4PySCF CPHF/dense oracle route')
    elif initial_guess_requested:
        candidate = cp.asarray(initial_guess, dtype=fvo.dtype)
        if candidate.shape != fvo.shape:
            initial_guess_fallback_reason = (
                f'projected shape {candidate.shape} differs from {fvo.shape}')
        elif not bool(cp.all(cp.isfinite(candidate))):
            initial_guess_fallback_reason = 'projected guess is not finite'
        else:
            candidate_residual = get_z_residual(
                candidate, mo_energy, mo_occ, fvo, vind)
            diagonal_residual = get_z_residual(
                diagonal_guess, mo_energy, mo_occ, fvo, vind)
            initial_residual_max_abs = float(
                cp.max(cp.abs(candidate_residual)).get())
            diagonal_initial_residual_max_abs = float(
                cp.max(cp.abs(diagonal_residual)).get())
            if initial_residual_max_abs <= diagonal_initial_residual_max_abs:
                selected_initial_guess = candidate
                initial_guess_used = True
            else:
                initial_guess_fallback_reason = (
                    'projected physical residual exceeds diagonal-gap guess')
    primary_wall_seconds = 0.
    if use_cphf_primary:
        primary_start = total_start
        x, _ = cphf.solve(
            vind, mo_energy, mo_occ, fvo, s1=None,
            max_cycle=max_cycle, tol=conv_tol, verbose=log)
        primary_wall_seconds = time.perf_counter() - primary_start
        zvec = -cp.asarray(x)
        solver = 'gpu4pyscf_cphf'
        residual = get_z_residual(zvec, mo_energy, mo_occ, fvo, vind)
        residual_max_abs = (
            float(cp.max(cp.abs(residual)).get()) if residual.size else 0.0)
    else:
        zvec = None
        solver = 'matrix_free_direct'
        residual_max_abs = float('inf')
    matrix_asymmetry = None
    refinement_wall_seconds = 0.
    refinement_iterations = None
    cleanup_wall_seconds = 0.
    cleanup_iterations = None
    cleanup_krylov_tol = None
    if residual_max_abs > residual_tol and fvo.size <= dense_oracle_size:
        refinement_start = time.perf_counter()
        zvec, matrix_asymmetry = _dense_solve(
            mo_energy, mo_occ, fvo, vind)
        refinement_wall_seconds = time.perf_counter() - refinement_start
        solver = 'dense_adjoint_fallback'
        residual = get_z_residual(zvec, mo_energy, mo_occ, fvo, vind)
        residual_max_abs = (
            float(cp.max(cp.abs(residual)).get()) if residual.size else 0.0)
    if residual_max_abs > residual_tol:
        refinement_start = time.perf_counter()
        iterative_solve = (
            _minres_solve if refinement_solver == 'minres' else _gmres_solve)
        zvec, iterative_info, refinement_iterations = iterative_solve(
            mo_energy, mo_occ, fvo, vind, effective_krylov_tol, max_cycle,
            initial_guess=selected_initial_guess)
        refinement_wall_seconds = time.perf_counter() - refinement_start
        solver = f'cupyx_{refinement_solver}(info={iterative_info})'
        residual = get_z_residual(zvec, mo_energy, mo_occ, fvo, vind)
        residual_max_abs = (
            float(cp.max(cp.abs(residual)).get()) if residual.size else 0.0)
        # A loose experiment may stop before the unchanged physical residual
        # gate.  Measure the real cost of recovering that gate by restarting
        # the same fixed-precision solver from the approximate Z vector at the
        # established production tolerance.  This cleanup is deliberately
        # absent when the requested residual gate is already satisfied.
        if (residual_max_abs > residual_tol
                and requested_krylov_tol is not None
                and effective_krylov_tol > production_krylov_tol):
            cleanup_start = time.perf_counter()
            cleanup_krylov_tol = production_krylov_tol
            zvec, cleanup_info, cleanup_iterations = iterative_solve(
                mo_energy, mo_occ, fvo, vind, cleanup_krylov_tol, max_cycle,
                initial_guess=zvec)
            cleanup_wall_seconds = time.perf_counter() - cleanup_start
            refinement_wall_seconds += cleanup_wall_seconds
            refinement_iterations += cleanup_iterations
            solver += (
                f'->fp64_cleanup_{refinement_solver}'
                f'(info={cleanup_info})')
            residual = get_z_residual(
                zvec, mo_energy, mo_occ, fvo, vind)
            residual_max_abs = (
                float(cp.max(cp.abs(residual)).get())
                if residual.size else 0.0)
        # MINRES uses the real-Gamma Hessian symmetry.  Preserve a nonsymmetric
        # Krylov oracle for the rare case in which its own stopping criterion
        # is satisfied but the unpreconditioned physical residual is not.
        # This retry changes neither the CPHF equation nor its acceptance
        # threshold; it only chooses a more general exact iterative method.
        if residual_max_abs > residual_tol and refinement_solver == 'minres':
            refinement_start = time.perf_counter()
            zvec, gmres_info, gmres_iterations = _gmres_solve(
                mo_energy, mo_occ, fvo, vind, production_krylov_tol,
                max_cycle, initial_guess=zvec)
            refinement_wall_seconds += time.perf_counter() - refinement_start
            refinement_iterations += gmres_iterations
            solver += f'->cupyx_gmres(info={gmres_info})'
            residual = get_z_residual(zvec, mo_energy, mo_occ, fvo, vind)
            residual_max_abs = (
                float(cp.max(cp.abs(residual)).get()) if residual.size else 0.0)
    rhs_max_abs = float(cp.max(cp.abs(fvo)).get()) if fvo.size else 0.0
    if residual_max_abs > residual_tol:
        raise RuntimeError(
            'periodic GPU Z-vector physical residual did not converge: '
            f'{residual_max_abs:.3e} > {residual_tol:.3e}')
    return zvec, {
        'rhs_max_abs': rhs_max_abs,
        'residual_max_abs': residual_max_abs,
        'residual_relative': (
            residual_max_abs / rhs_max_abs if rhs_max_abs else residual_max_abs),
        'solver': solver,
        'dimension': int(fvo.size),
        'matrix_asymmetry_max_abs': matrix_asymmetry,
        'primary_wall_seconds': primary_wall_seconds,
        'refinement_wall_seconds': refinement_wall_seconds,
        'refinement_iterations': refinement_iterations,
        'requested_conv_tol': conv_tol,
        'requested_krylov_tol': requested_krylov_tol,
        'effective_krylov_tol': effective_krylov_tol,
        'krylov_tolerance_policy': krylov_tolerance_policy,
        'physical_residual_tol': residual_tol,
        'cleanup_krylov_tol': cleanup_krylov_tol,
        'cleanup_wall_seconds': cleanup_wall_seconds,
        'cleanup_iterations': cleanup_iterations,
        'response_backend': getattr(vind, 'backend', 'ao_density_gen_response'),
        'initial_guess_requested': initial_guess_requested,
        'initial_guess_used': initial_guess_used,
        'initial_guess_fallback_reason': initial_guess_fallback_reason,
        'initial_residual_max_abs': initial_residual_max_abs,
        'diagonal_initial_residual_max_abs': (
            diagonal_initial_residual_max_abs),
        'wall_seconds': time.perf_counter() - total_start,
    }
