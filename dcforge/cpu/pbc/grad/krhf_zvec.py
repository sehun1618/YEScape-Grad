'''Dense and matrix-free real-linear sampled-k KRHF Z-vector solvers.'''

import time

import numpy as np
from pyscf import lib
from scipy.sparse.linalg import LinearOperator, gmres

from dcforge.cpu.pbc.df.fft import FFTDF
from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.energy.krhf_dft import KRHFDFT, _target_fock_signature
from dcforge.cpu.pbc.kpts import (
    DENSE_RESPONSE_RESIDUAL_TOL,
    PRODUCTION_RESPONSE_RESIDUAL_TOL,
    make_occ_masks,
    validate_insulating_occupations,
    validate_kresolved_hermitian,
)


def make_mo_partitions(mo_energy, mo_coeff, mo_occ):
    '''Return occupied, virtual, and energy-gap blocks for every k-point.

    Args:
        mo_energy : array_like
            Canonical source-KRHF energies with shape ``(nkpts,nmo)``.
        mo_coeff : array_like
            Complex AO-to-MO coefficients with shape ``(nkpts,nao,nmo)``.
        mo_occ : array_like
            Exact restricted 2/0 occupations with shape ``(nkpts,nmo)``.

    Returns:
        tuple
            ``(Co, Cv, e_ai)`` tuples over k-points.  Coefficient blocks have
            shapes ``(nao,nocc)`` and ``(nao,nvir)``; gaps have shape
            ``(nvir,nocc)`` in Hartree.
    '''
    mo_energy = np.asarray(mo_energy)
    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)
    if mo_energy.ndim != 2 or mo_occ.shape != mo_energy.shape:
        raise ValueError('mo_energy and mo_occ must have shape (nkpts, nmo)')
    if (mo_coeff.ndim != 3 or mo_coeff.shape[0] != mo_energy.shape[0]
            or mo_coeff.shape[2] != mo_energy.shape[1]):
        raise ValueError('mo_coeff must have shape (nkpts, nao, nmo)')
    if not np.all(np.isfinite(mo_coeff)):
        raise ValueError('mo_coeff contains nonfinite values')
    validate_insulating_occupations(mo_energy, mo_occ)
    occupied_masks, virtual_masks = make_occ_masks(mo_occ)

    Co = []
    Cv = []
    e_ai = []
    for energy, coeff, so, sv in zip(
            mo_energy, mo_coeff, occupied_masks, virtual_masks, strict=True):
        Co.append(np.asarray(coeff[:, so]))
        Cv.append(np.asarray(coeff[:, sv]))
        e_ai.append(
            np.asarray(energy[sv, None] - energy[so][None, :]).real)
    return tuple(Co), tuple(Cv), tuple(e_ai)


def make_fvo(mf, Co, Cv):
    r'''Project the cached target Fock to :math:`F^{\rm DFA}_{vo,k}`.

    At converged KRHF orbitals :math:`F^{\rm HF}_{vo,k}` is stationary,
    whereas the target-DFA Fock evaluated on that density generally is not.  This
    target block is the positive right-hand side of
    :math:`A_R^T z_R=f_R`.
    '''
    if not isinstance(mf, KRHFDFT):
        raise TypeError('make_fvo requires a sampled-k KRHFDFT object')
    if mf.e_dft is None or mf.target_fock is None or mf.dm_hf is None:
        raise RuntimeError('run the sampled-k KRHFDFT energy kernel first')
    dm0 = np.asarray(mf.mf_hf.make_rdm1())
    if (mf.target_fock_signature != _target_fock_signature(mf, dm0)
            or not np.array_equal(np.asarray(mf.dm_hf), dm0)):
        raise RuntimeError(
            'source density or target representation changed; rerun KRHFDFT')

    Co, Cv = _validate_coeff_blocks(Co, Cv, len(mf.kpts))
    fock = validate_kresolved_hermitian(
        mf.target_fock, len(mf.kpts), mf.cell.nao_nr(),
        name='target DFA Fock')
    return tuple(
        lib.dot(lib.dot(Cv_k.conj().T, fock_k), Co_k)
        for fock_k, Co_k, Cv_k in zip(fock, Co, Cv, strict=True)
    )


def make_z_dm(Co, Cv, zvec):
    r'''Return :math:`D_{Z,k}=C_{v,k}Z_kC_{o,k}^H+\mathrm{h.c.}`.

    This Hermitian density contains both ``Z`` and ``Z*``.  Consequently the
    sampled-k orbital Hessian is real-linear, not complex-linear.
    '''
    Co, Cv = _validate_coeff_blocks(Co, Cv)
    zvec = _validate_vo_blocks(zvec, Co, Cv, name='zvec')
    dm1 = []
    for Co_k, Cv_k, z_k in zip(Co, Cv, zvec, strict=True):
        one_sided = lib.dot(lib.dot(Cv_k, z_k), Co_k.conj().T)
        dm1.append(one_sided + one_sided.conj().T)
    return np.asarray(dm1)


def gen_hf_response(mf_hf):
    '''Return PySCF's coupled full-k ``J[D]-K[D]/2`` response action.'''
    if not mf_hf.converged:
        raise RuntimeError('source KRHF must be converged')
    if not isinstance(mf_hf.with_df, (FFTDF, GDF)):
        raise NotImplementedError(
            'sampled-k response currently supports CPU FFTDF and GDF only')
    return mf_hf.gen_response(
        mo_coeff=mf_hf.mo_coeff, mo_occ=mf_hf.mo_occ, hermi=1)


def make_hessian_operator(mo_energy, mo_occ, Co, Cv, fvind):
    r'''Return the sampled-k restricted-HF real-linear VO Hessian action.

    For trial blocks :math:`X_k`,

    .. math::

        A[X]_k=(\epsilon_{a,k}-\epsilon_{i,k})X_k
        +2C_{v,k}^\dagger G_{\rm HF}[D_X]C_{o,k},

    where :math:`D_{X,k}=C_{v,k}X_kC_{o,k}^\dagger+\mathrm{h.c.}`.
    One PySCF ``fvind`` call receives every k block, preserving the coupled
    full-Brillouin-zone Hartree and exchange response.
    '''
    mo_energy = np.asarray(mo_energy)
    mo_occ = np.asarray(mo_occ)
    Co, Cv = _validate_coeff_blocks(Co, Cv, len(mo_energy))
    validate_insulating_occupations(mo_energy, mo_occ)
    occupied_masks, virtual_masks = make_occ_masks(mo_occ)
    e_ai = tuple(
        np.asarray(energy[sv, None] - energy[so][None, :]).real
        for energy, so, sv in zip(
            mo_energy, occupied_masks, virtual_masks, strict=True)
    )

    def vind(x):
        x = _validate_vo_blocks(x, Co, Cv, name='trial')
        v1ao = np.asarray(fvind(make_z_dm(Co, Cv, x))) * 2
        nao = Co[0].shape[0]
        if v1ao.shape != (len(Co), nao, nao):
            raise ValueError('KRHF response must return shape (nkpts, nao, nao)')
        if not np.all(np.isfinite(v1ao)):
            raise ValueError('KRHF response contains nonfinite values')
        return tuple(
            gap * x_k + lib.dot(lib.dot(Cv_k.conj().T, v1_k), Co_k)
            for gap, x_k, v1_k, Co_k, Cv_k in zip(
                e_ai, x, v1ao, Co, Cv, strict=True)
        )
    return vind


def flatten_vo(blocks):
    '''Flatten ordered complex VO blocks without object-array storage.'''
    blocks = tuple(np.asarray(block) for block in blocks)
    if not blocks:
        raise ValueError('at least one VO block is required')
    return np.concatenate([block.reshape(-1) for block in blocks]).astype(
        np.complex128, copy=False)


def unflatten_vo(vector, shapes):
    '''Restore one flat complex vector to ordered k-resolved VO blocks.'''
    vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
    shapes = tuple(tuple(int(value) for value in shape) for shape in shapes)
    offsets = np.cumsum([0] + [int(np.prod(shape)) for shape in shapes])
    if vector.size != offsets[-1]:
        raise ValueError('flat complex vector size differs from VO shapes')
    return tuple(
        vector[offsets[k]:offsets[k + 1]].reshape(shape)
        for k, shape in enumerate(shapes)
    )


def complex_to_real(vector):
    '''Map ``z`` to the real-expanded vector ``[Re(z), Im(z)]``.'''
    vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
    return np.concatenate((vector.real, vector.imag))


def real_to_complex(vector):
    '''Invert the real-expanded ``[Re(z), Im(z)]`` representation.'''
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.size % 2:
        raise ValueError('real-linear vector size must be even')
    half = vector.size // 2
    return vector[:half] + 1j * vector[half:]


def build_real_linear_matrix(vind, template):
    '''Materialize the small-system real-linear Hessian ``A_R``.

    Column ``j`` is a fresh physical action on real basis vector ``e_j``.
    The construction retains both ``X`` and ``X*`` channels and is intended
    as a correctness reference, not a production-size solver.
    '''
    template = tuple(np.asarray(block) for block in template)
    shapes = tuple(block.shape for block in template)
    dimension = 2 * flatten_vo(template).size
    matrix = np.empty((dimension, dimension), dtype=np.float64)
    for j in range(dimension):
        x = np.zeros(dimension)
        x[j] = 1
        blocks = unflatten_vo(real_to_complex(x), shapes)
        matrix[:, j] = complex_to_real(flatten_vo(vind(blocks)))
    return matrix, shapes


def apply_real_linear(vind, vector, shapes):
    '''Apply ``A_R`` without trusting a stored matrix.'''
    blocks = unflatten_vo(real_to_complex(vector), shapes)
    return complex_to_real(flatten_vo(vind(blocks)))


def apply_real_adjoint(vind, vector, shapes):
    '''Apply ``A_R.T`` through fresh basis actions.

    This deliberately expensive function makes the accepted adjoint residual
    independent of the matrix passed to ``numpy.linalg.solve``.
    '''
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    result = np.empty_like(vector)
    for j in range(vector.size):
        x = np.zeros_like(vector)
        x[j] = 1
        result[j] = np.dot(vector, apply_real_linear(vind, x, shapes))
    return result


def weighted_vo_inner(x, y, weights, imaginary_tol=None):
    r'''Return :math:`\sum_k w_k\,\mathrm{Re}\,\mathrm{Tr}[X_k^H Y_k]`.'''
    x = tuple(np.asarray(block) for block in x)
    y = tuple(np.asarray(block) for block in y)
    if len(x) != len(y) or not x:
        raise ValueError('weighted VO operands must have equal nonzero blocks')
    weights = _equal_weights(len(x), weights)
    value = 0j
    for k, (x_k, y_k) in enumerate(zip(x, y, strict=True)):
        if x_k.shape != y_k.shape:
            raise ValueError('weighted VO operand block shapes differ')
        value += weights[k] * lib.einsum('ai,ai->', x_k.conj(), y_k)
    if imaginary_tol is not None and abs(value.imag) > imaginary_tol:
        raise ValueError(
            f'weighted VO inner-product imaginary residual '
            f'{abs(value.imag):.3e} exceeds {imaginary_tol:.3e}')
    return float(value.real)


def get_complex_linearity_error(vind, x):
    '''Return ``max|A(iX)-iA(X)|`` for the conjugate-channel diagnostic.'''
    x = tuple(np.asarray(block) for block in x)
    ax = vind(x)
    aix = vind(tuple(1j * block for block in x))
    return max(
        float(np.max(np.abs(aix_k - 1j * ax_k)))
        for aix_k, ax_k in zip(aix, ax, strict=True)
    )


def solve_z_vector(mo_energy, mo_occ, Co, Cv, fvo, fvind,
                   residual_tol=DENSE_RESPONSE_RESIDUAL_TOL,
                   dense_reference_size=512, equation='adjoint',
                   kpts_weights=None):
    r'''Solve the dense sampled-k positive-RHS Z-vector equation.

    ``equation='adjoint'`` solves :math:`A_R^Tz_R=f_R`, the canonical
    DC-Forge equation.  ``'direct'`` is retained only as a Hessian-symmetry
    diagnostic.  Both solutions receive fresh physical residual checks.

    Returns:
        tuple
            Complex k-resolved ``zvec`` blocks and a diagnostic dictionary.
    '''
    if equation not in ('adjoint', 'direct'):
        raise ValueError("equation must be 'adjoint' or 'direct'")
    if residual_tol <= 0:
        raise ValueError('residual_tol must be positive')
    if dense_reference_size < 1:
        raise ValueError('dense_reference_size must be positive')

    Co, Cv = _validate_coeff_blocks(Co, Cv)
    fvo = _validate_vo_blocks(fvo, Co, Cv, name='fvo')
    weights = _equal_weights(len(Co), kpts_weights)
    shapes = tuple(block.shape for block in fvo)
    complex_size = flatten_vo(fvo).size
    dimension = 2 * complex_size
    if dimension > dense_reference_size:
        raise RuntimeError(
            f'dense sampled-k Z-vector dimension {dimension} exceeds '
            f'dense_reference_size={dense_reference_size}; a matrix-free solver '
            'is required')

    vind = make_hessian_operator(mo_energy, mo_occ, Co, Cv, fvind)
    matrix, matrix_shapes = build_real_linear_matrix(vind, fvo)
    if matrix_shapes != shapes:
        raise RuntimeError('internal VO layout changed while building Hessian')
    rhs = complex_to_real(flatten_vo(fvo))
    try:
        z_adjoint_real = np.linalg.solve(matrix.T, rhs)
        z_direct_real = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as err:
        raise RuntimeError('dense sampled-k KRHF Hessian is singular') from err

    adjoint_residual = apply_real_adjoint(vind, z_adjoint_real, shapes) - rhs
    direct_residual = apply_real_linear(vind, z_direct_real, shapes) - rhs
    z_real = z_adjoint_real if equation == 'adjoint' else z_direct_real
    residual = adjoint_residual if equation == 'adjoint' else direct_residual
    matrix_operator = matrix.T if equation == 'adjoint' else matrix
    matrix_residual = matrix_operator @ z_real - rhs

    abs_residual = float(np.max(np.abs(residual))) if residual.size else 0.0
    rhs_norm = float(np.max(np.abs(rhs))) if rhs.size else 0.0
    if abs_residual > residual_tol:
        raise RuntimeError(
            'sampled-k dense Z-vector physical residual failed: '
            f'{abs_residual:.3e} > {residual_tol:.3e}')

    zvec = unflatten_vo(real_to_complex(z_real), shapes)
    info = {
        'solver': 'dense_real_linear',
        'equation': equation,
        'rhs_sign': 'positive',
        'dimension': int(dimension),
        'complex_dimension': int(complex_size),
        'nkpts': len(Co),
        'kpts_weights': weights.copy(),
        'rhs_max_abs': rhs_norm,
        'residual_max_abs': abs_residual,
        'residual_relative': (
            abs_residual / rhs_norm if rhs_norm else abs_residual),
        'matrix_residual_max_abs': float(np.max(np.abs(matrix_residual))),
        'adjoint_residual_max_abs': float(np.max(np.abs(adjoint_residual))),
        'direct_residual_max_abs': float(np.max(np.abs(direct_residual))),
        'condition_number': float(np.linalg.cond(matrix_operator)),
        'matrix_asymmetry_max_abs': float(
            np.max(np.abs(matrix - matrix.T))),
        'direct_adjoint_max_abs': float(
            np.max(np.abs(z_direct_real - z_adjoint_real))),
    }
    return zvec, info


def solve_z_vector_matrix_free(
        mo_energy, mo_occ, Co, Cv, fvo, fvind, conv_tol=1e-9,
        residual_tol=PRODUCTION_RESPONSE_RESIDUAL_TOL, max_cycle=200,
        kpts_weights=None, self_adjoint_tol=1e-10, restart=80):
    r'''Solve the sampled-k positive-RHS adjoint without storing ``A_R``.

    For the supported equal-weight full-BZ representation, the real-expanded
    restricted-HF Hessian is self-adjoint: the orbital-gap contribution is
    real diagonal, and PySCF's ``J[D]-K[D]/2`` response is self-adjoint under
    the same full-k trace.  A deterministic bilinear probe audits this
    identity before the forward action is used as :math:`A_R^T`.

    GMRES uses the occupied-virtual energy gaps as a preconditioner.  Its
    reported convergence is not the acceptance criterion: after GMRES exits,
    a fresh, unpreconditioned operator call reconstructs the physical
    residual of :math:`A_R^Tz_R-f_R=0`.

    Kwargs:
        conv_tol : float
            Relative GMRES convergence request.
        residual_tol : float
            Absolute physical-residual acceptance threshold.
        max_cycle : int
            Maximum GMRES inner iterations.
        kpts_weights : array_like or None
            Equal full-BZ k-point weights.
        self_adjoint_tol : float
            Maximum normalized bilinear-adjoint defect.
        restart : int
            Maximum restarted Arnoldi basis dimension.

    Returns:
        tuple
            Complex k-resolved ``zvec`` blocks and solver diagnostics.  The
            reported workspace is an estimate; dense-Hessian storage is zero.
    '''
    if conv_tol <= 0:
        raise ValueError('conv_tol must be positive')
    if residual_tol <= 0:
        raise ValueError('residual_tol must be positive')
    if max_cycle < 1:
        raise ValueError('max_cycle must be positive')
    if self_adjoint_tol <= 0:
        raise ValueError('self_adjoint_tol must be positive')
    if restart < 1:
        raise ValueError('restart must be positive')

    start = time.perf_counter()
    Co, Cv = _validate_coeff_blocks(Co, Cv)
    fvo = _validate_vo_blocks(fvo, Co, Cv, name='fvo')
    weights = _equal_weights(len(Co), kpts_weights)
    shapes = tuple(block.shape for block in fvo)
    rhs = complex_to_real(flatten_vo(fvo))
    dimension = rhs.size
    if dimension == 0:
        raise ValueError('matrix-free Z-vector requires a nonempty VO space')

    vind = make_hessian_operator(mo_energy, mo_occ, Co, Cv, fvind)
    occupied_masks, virtual_masks = make_occ_masks(mo_occ)
    gaps = []
    for energy, so, sv in zip(
            np.asarray(mo_energy), occupied_masks, virtual_masks, strict=True):
        gaps.append(np.asarray(
            energy[sv, None] - energy[so][None, :]).real.ravel())
    gap = np.concatenate(gaps)
    gap = np.concatenate((gap, gap))
    if gap.shape != rhs.shape or np.any(gap <= 0):
        raise RuntimeError('invalid occupied-virtual gap preconditioner')

    operator_calls = 0
    preconditioner_calls = 0

    def matvec(vector):
        nonlocal operator_calls
        operator_calls += 1
        return apply_real_linear(vind, vector, shapes)

    def precondition(vector):
        nonlocal preconditioner_calls
        preconditioner_calls += 1
        return np.asarray(vector) / gap

    operator = LinearOperator(
        (dimension, dimension), matvec=matvec, dtype=np.float64)
    inverse_gap = LinearOperator(
        (dimension, dimension), matvec=precondition, dtype=np.float64)

    # Equal weights differ from the Euclidean inner product only by a common
    # scalar.  Two reproducible probes catch a provider or representation for
    # which the forward action cannot safely stand in for its transpose.
    rng = np.random.default_rng(17)
    self_adjoint_error = 0.0
    for _ in range(2):
        x = rng.normal(size=dimension)
        y = rng.normal(size=dimension)
        ax = matvec(x)
        ay = matvec(y)
        lhs = np.dot(x, ay)
        rhs_probe = np.dot(ax, y)
        scale = max(1.0, abs(lhs), abs(rhs_probe))
        self_adjoint_error = max(
            self_adjoint_error, abs(lhs - rhs_probe) / scale)
    if self_adjoint_error > self_adjoint_tol:
        raise RuntimeError(
            'sampled-k matrix-free Hessian failed the self-adjoint gate: '
            f'{self_adjoint_error:.3e} > {self_adjoint_tol:.3e}')

    iteration_count = 0

    def count_iteration(_residual):
        nonlocal iteration_count
        iteration_count += 1

    restart = min(int(restart), dimension)
    z_real, solver_info = gmres(
        operator, rhs, M=inverse_gap, x0=rhs / gap,
        rtol=float(conv_tol), atol=0.0, restart=restart,
        maxiter=int(max_cycle), callback=count_iteration,
        callback_type='legacy')
    if solver_info != 0:
        raise RuntimeError(
            'sampled-k matrix-free GMRES did not converge: '
            f'info={solver_info}, iterations={iteration_count}')

    residual = matvec(z_real) - rhs
    abs_residual = float(np.max(np.abs(residual)))
    rhs_norm = float(np.max(np.abs(rhs)))
    if abs_residual > residual_tol:
        raise RuntimeError(
            'sampled-k matrix-free Z-vector physical residual failed: '
            f'{abs_residual:.3e} > {residual_tol:.3e}')

    zvec = unflatten_vo(real_to_complex(z_real), shapes)
    workspace_estimate = int((restart + 5) * dimension * 8)
    info = {
        'solver': 'scipy_gmres_matrix_free',
        'equation': 'adjoint_self_adjoint_action',
        'rhs_sign': 'positive',
        'dimension': int(dimension),
        'complex_dimension': int(dimension // 2),
        'nkpts': len(Co),
        'kpts_weights': weights.copy(),
        'rhs_max_abs': rhs_norm,
        'residual_max_abs': abs_residual,
        'residual_relative': (
            abs_residual / rhs_norm if rhs_norm else abs_residual),
        'matrix_residual_max_abs': None,
        'adjoint_residual_max_abs': abs_residual,
        'direct_residual_max_abs': abs_residual,
        'condition_number': None,
        'matrix_asymmetry_max_abs': None,
        'direct_adjoint_max_abs': None,
        'self_adjoint_probe_error': float(self_adjoint_error),
        'iterations': int(iteration_count),
        'operator_calls': int(operator_calls),
        'preconditioner_calls': int(preconditioner_calls),
        'restart': int(restart),
        'dense_matrix_bytes': 0,
        'workspace_estimate_bytes': workspace_estimate,
        'wall_seconds': time.perf_counter() - start,
    }
    return zvec, info


def _validate_coeff_blocks(Co, Cv, nkpts=None):
    '''Return validated occupied and virtual coefficient tuples.'''
    Co = tuple(np.asarray(block) for block in Co)
    Cv = tuple(np.asarray(block) for block in Cv)
    if not Co or len(Co) != len(Cv):
        raise ValueError('Co and Cv must contain equal nonzero k-point blocks')
    if nkpts is not None and len(Co) != nkpts:
        raise ValueError('coefficient block count differs from nkpts')
    nao = Co[0].shape[0]
    vo_shape = (Cv[0].shape[1], Co[0].shape[1])
    for k, (Co_k, Cv_k) in enumerate(zip(Co, Cv, strict=True)):
        if (Co_k.ndim != 2 or Cv_k.ndim != 2
                or Co_k.shape[0] != nao or Cv_k.shape[0] != nao):
            raise ValueError(f'invalid coefficient shape at k-point {k}')
        if (Cv_k.shape[1], Co_k.shape[1]) != vo_shape:
            raise NotImplementedError(
                'sampled-k response requires the same MO topology at every k-point')
        if not np.all(np.isfinite(Co_k)) or not np.all(np.isfinite(Cv_k)):
            raise ValueError('MO coefficient blocks contain nonfinite values')
    return Co, Cv


def _validate_vo_blocks(blocks, Co, Cv, name):
    '''Return complex VO blocks matching the coefficient partitions.'''
    blocks = tuple(np.asarray(block, dtype=np.complex128) for block in blocks)
    if len(blocks) != len(Co):
        raise ValueError(f'{name} block count differs from k-point count')
    for k, (block, Co_k, Cv_k) in enumerate(
            zip(blocks, Co, Cv, strict=True)):
        shape = (Cv_k.shape[1], Co_k.shape[1])
        if block.shape != shape:
            raise ValueError(
                f'{name} block {k} has shape {block.shape}; expected {shape}')
        if not np.all(np.isfinite(block)):
            raise ValueError(f'{name} block {k} contains nonfinite values')
    return blocks


def _equal_weights(nkpts, weights):
    '''Return the equal full-BZ weights supported by this adjoint.'''
    expected = np.full(nkpts, 1.0 / nkpts)
    if weights is None:
        return expected
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (nkpts,) or not np.all(np.isfinite(weights)):
        raise ValueError('kpts_weights must have shape (nkpts,) and be finite')
    if not np.allclose(weights, expected, atol=2e-15, rtol=2e-15):
        raise NotImplementedError(
            'sampled-k response supports equal full-BZ weights only')
    return expected
