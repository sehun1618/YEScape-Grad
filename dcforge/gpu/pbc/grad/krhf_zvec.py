'''Dense and matrix-free device sampled-k KRHF Z-vector solvers.'''

import time

import cupy as cp
import numpy as np
from cupyx.scipy.sparse.linalg import LinearOperator, gmres

from dcforge.cpu.pbc.kpts import (
    DENSE_RESPONSE_RESIDUAL_TOL,
    PRODUCTION_RESPONSE_RESIDUAL_TOL,
    make_occ_masks,
    validate_insulating_occupations,
)
from dcforge.gpu.pbc._compat import contiguous_int1e_hermitian
from dcforge.gpu.pbc.df.fft import FFTDF
from dcforge.gpu.pbc.df.gdf import GDF
from dcforge.gpu.pbc.energy.krhf_dft import (
    KRHFDFT,
    _target_fock_signature,
)


def make_mo_partitions(mo_energy, mo_coeff, mo_occ):
    '''Return occupied, virtual and energy-gap device blocks at every k.'''
    mo_energy = cp.asarray(mo_energy)
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    if mo_energy.ndim != 2 or mo_occ.shape != mo_energy.shape:
        raise ValueError(
            'mo_energy and mo_occ must have shape (nkpts,nmo)')
    if (mo_coeff.ndim != 3 or mo_coeff.shape[0] != mo_energy.shape[0]
            or mo_coeff.shape[2] != mo_energy.shape[1]):
        raise ValueError('mo_coeff must have shape (nkpts,nao,nmo)')
    if not bool(cp.all(cp.isfinite(mo_coeff)).get()):
        raise ValueError('mo_coeff contains nonfinite values')
    mo_energy_host = cp.asnumpy(mo_energy)
    mo_occ_host = cp.asnumpy(mo_occ)
    validate_insulating_occupations(mo_energy_host, mo_occ_host)
    occupied_masks, virtual_masks = make_occ_masks(mo_occ_host)

    Co = []
    Cv = []
    e_ai = []
    for k in range(len(mo_energy)):
        so = cp.asarray(occupied_masks[k])
        sv = cp.asarray(virtual_masks[k])
        Co.append(cp.asarray(mo_coeff[k][:, so]))
        Cv.append(cp.asarray(mo_coeff[k][:, sv]))
        e_ai.append(
            cp.asarray(
                mo_energy[k][sv, None] - mo_energy[k][so][None, :]).real)
    return tuple(Co), tuple(Cv), tuple(e_ai)


def make_fvo(mf, Co, Cv):
    r'''Project the cached device target Fock to :math:`F^{PBE}_{vo,k}`.'''
    if not isinstance(mf, KRHFDFT):
        raise TypeError('make_fvo requires a GPU sampled-k KRHFDFT object')
    if mf.e_dft is None or mf.target_fock is None or mf.dm_hf is None:
        raise RuntimeError('run the GPU sampled-k energy kernel first')
    dm0 = cp.asarray(mf.mf_hf.make_rdm1())
    signature = _target_fock_signature(mf, dm0)
    if (mf.target_fock_signature != signature
            or not bool(cp.array_equal(mf.dm_hf, dm0).get())):
        raise RuntimeError(
            'source density or target representation changed; rerun KRHFDFT')

    Co, Cv = _validate_coeff_blocks(Co, Cv, len(mf.kpts))
    fock = cp.asarray(mf.target_fock)
    fvo = []
    for k in range(len(Co)):
        fvo.append(Cv[k].conj().T @ fock[k] @ Co[k])
    return tuple(fvo)


def make_z_dm(Co, Cv, zvec):
    r'''Return :math:`D_{Z,k}=C_{v,k}Z_kC_{o,k}^H+\mathrm{h.c.}`.'''
    Co, Cv = _validate_coeff_blocks(Co, Cv)
    zvec = _validate_vo_blocks(zvec, Co, Cv, name='zvec')
    density = []
    for Co_k, Cv_k, z_k in zip(Co, Cv, zvec, strict=True):
        one_sided = (Cv_k @ z_k) @ Co_k.conj().T
        density.append(one_sided + one_sided.conj().T)
    return cp.asarray(density)


def gen_hf_response(mf_hf):
    '''Return native GPU coupled ``J[D]-K[D]/2`` full-k response.'''
    if not mf_hf.converged:
        raise RuntimeError('source GPU KRHF must be converged')
    if not isinstance(mf_hf.with_df, (FFTDF, GDF)):
        raise NotImplementedError(
            'GPU sampled-k response supports FFTDF and GDF only')
    native = mf_hf.gen_response(
        mo_coeff=mf_hf.mo_coeff, mo_occ=mf_hf.mo_occ, hermi=1)

    def response(density):
        '''Apply the native coupled response under the layout workaround.'''
        with contiguous_int1e_hermitian(mf_hf):
            return native(density)
    return response


def make_hessian_operator(mo_energy, mo_occ, Co, Cv, fvind):
    r'''Return the device restricted-HF real-linear VO Hessian action.

    ``A[X]_k = (epsilon_a-epsilon_i) X_k``
    ``         + 2 C_v,k^H G_HF[D_X]_k C_o,k``.
    '''
    mo_energy = cp.asarray(mo_energy)
    mo_occ = cp.asarray(mo_occ)
    Co, Cv = _validate_coeff_blocks(Co, Cv, len(mo_energy))
    mo_energy_host = cp.asnumpy(mo_energy)
    mo_occ_host = cp.asnumpy(mo_occ)
    validate_insulating_occupations(mo_energy_host, mo_occ_host)
    occupied_masks, virtual_masks = make_occ_masks(mo_occ_host)
    gaps = []
    for k in range(len(mo_energy)):
        so = cp.asarray(occupied_masks[k])
        sv = cp.asarray(virtual_masks[k])
        gaps.append(
            cp.asarray(
                mo_energy[k][sv, None]
                - mo_energy[k][so][None, :]).real)

    def vind(x):
        '''Apply gaps plus the coupled GPU KRHF response.'''
        x = _validate_vo_blocks(x, Co, Cv, name='trial')
        dm1 = make_z_dm(Co, Cv, x)
        v1ao = cp.asarray(fvind(dm1)) * 2
        nao = Co[0].shape[0]
        if v1ao.shape != (len(Co), nao, nao):
            raise ValueError(
                'GPU KRHF response must return shape (nkpts,nao,nao)')
        output = []
        for gap, x_k, v1_k, Co_k, Cv_k in zip(
                gaps, x, v1ao, Co, Cv, strict=True):
            output.append(gap * x_k + Cv_k.conj().T @ v1_k @ Co_k)
        return tuple(output)
    return vind


def flatten_vo(blocks):
    '''Flatten complex device VO blocks without object-array storage.'''
    blocks = tuple(cp.asarray(block) for block in blocks)
    if not blocks:
        raise ValueError('at least one VO block is required')
    return cp.concatenate([block.reshape(-1) for block in blocks])


def unflatten_vo(vector, shapes):
    '''Restore one flat device vector to ordered VO blocks.'''
    vector = cp.asarray(vector, dtype=cp.complex128).reshape(-1)
    shapes = tuple(tuple(int(value) for value in shape) for shape in shapes)
    offsets = np.cumsum([0] + [int(np.prod(shape)) for shape in shapes])
    if vector.size != offsets[-1]:
        raise ValueError('flat complex vector size differs from VO shapes')
    return tuple(
        vector[offsets[k]:offsets[k + 1]].reshape(shape)
        for k, shape in enumerate(shapes))


def complex_to_real(vector):
    '''Map device ``z`` to ``[Re(z),Im(z)]``.'''
    vector = cp.asarray(vector, dtype=cp.complex128).reshape(-1)
    return cp.concatenate((vector.real, vector.imag))


def real_to_complex(vector):
    '''Invert the device real-expanded representation.'''
    vector = cp.asarray(vector, dtype=cp.float64).reshape(-1)
    if vector.size % 2:
        raise ValueError('real-linear vector size must be even')
    half = vector.size // 2
    return vector[:half] + 1j * vector[half:]


def build_real_linear_matrix(vind, template):
    '''Materialize the small-system real-linear Hessian on the GPU.'''
    template = tuple(cp.asarray(block) for block in template)
    shapes = tuple(block.shape for block in template)
    dimension = 2 * flatten_vo(template).size
    matrix = cp.empty((dimension, dimension), dtype=cp.float64)
    for j in range(dimension):
        x = cp.zeros(dimension, dtype=cp.float64)
        x[j] = 1
        blocks = unflatten_vo(real_to_complex(x), shapes)
        matrix[:, j] = complex_to_real(flatten_vo(vind(blocks)))
    return matrix, shapes


def apply_real_linear(vind, vector, shapes):
    '''Apply the physical real-linear operator without a stored matrix.'''
    blocks = unflatten_vo(real_to_complex(vector), shapes)
    return complex_to_real(flatten_vo(vind(blocks)))


def apply_real_adjoint(vind, vector, shapes):
    '''Apply ``A_R^T`` through fresh GPU basis actions.'''
    vector = cp.asarray(vector, dtype=cp.float64).reshape(-1)
    result = cp.empty_like(vector)
    for j in range(vector.size):
        x = cp.zeros_like(vector)
        x[j] = 1
        result[j] = cp.dot(vector, apply_real_linear(vind, x, shapes))
    return result


def solve_z_vector(
        mo_energy, mo_occ, Co, Cv, fvo, fvind,
        residual_tol=DENSE_RESPONSE_RESIDUAL_TOL,
        dense_reference_size=512, kpts_weights=None):
    r'''Solve the positive-RHS device equation ``A_R^T z_R=f_R``.

    The Hessian, transpose solve, direct diagnostic and independently
    reconstructed residual remain CuPy arrays.  Only scalar diagnostics are
    copied to the host.
    '''
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
            f'dense GPU Z-vector dimension {dimension} exceeds '
            f'dense_reference_size={dense_reference_size}; use '
            'solve_z_vector_matrix_free for the production response')

    vind = make_hessian_operator(mo_energy, mo_occ, Co, Cv, fvind)
    matrix, matrix_shapes = build_real_linear_matrix(vind, fvo)
    if matrix_shapes != shapes:
        raise RuntimeError('VO layout changed while building GPU Hessian')
    rhs = complex_to_real(flatten_vo(fvo))
    try:
        z_adjoint_real = cp.linalg.solve(matrix.T, rhs)
        z_direct_real = cp.linalg.solve(matrix, rhs)
    except cp.linalg.LinAlgError as err:
        raise RuntimeError('dense GPU sampled-k Hessian is singular') from err

    adjoint_residual = apply_real_adjoint(
        vind, z_adjoint_real, shapes) - rhs
    direct_residual = apply_real_linear(
        vind, z_direct_real, shapes) - rhs
    matrix_residual = matrix.T @ z_adjoint_real - rhs
    abs_residual = _max_abs(adjoint_residual)
    rhs_norm = _max_abs(rhs)
    if abs_residual > residual_tol:
        raise RuntimeError(
            'GPU sampled-k dense Z-vector physical residual failed: '
            f'{abs_residual:.3e} > {residual_tol:.3e}')

    zvec = unflatten_vo(real_to_complex(z_adjoint_real), shapes)
    info = {
        'solver': 'cupy_dense_real_linear',
        'equation': 'adjoint',
        'rhs_sign': 'positive',
        'dimension': int(dimension),
        'complex_dimension': int(complex_size),
        'nkpts': len(Co),
        'kpts_weights': weights.copy(),
        'rhs_max_abs': rhs_norm,
        'residual_max_abs': abs_residual,
        'residual_relative': (
            abs_residual / rhs_norm if rhs_norm else abs_residual),
        'matrix_residual_max_abs': _max_abs(matrix_residual),
        'adjoint_residual_max_abs': abs_residual,
        'direct_residual_max_abs': _max_abs(direct_residual),
        'condition_number': float(cp.linalg.cond(matrix.T).get()),
        'matrix_asymmetry_max_abs': _max_abs(matrix - matrix.T),
        'direct_adjoint_max_abs': _max_abs(
            z_direct_real - z_adjoint_real),
        'dense_matrix_bytes': int(matrix.nbytes),
        'scientific_host_fallback': False,
    }
    return zvec, info


def solve_z_vector_matrix_free(
        mo_energy, mo_occ, Co, Cv, fvo, fvind, conv_tol=1e-9,
        residual_tol=PRODUCTION_RESPONSE_RESIDUAL_TOL, max_cycle=200,
        kpts_weights=None, self_adjoint_tol=1e-10, restart=80):
    r'''Solve ``A_R^T z_R=f_R`` on the GPU without storing ``A_R``.

    Under the first-release equal full-BZ weights, the restricted-HF orbital
    Hessian is self-adjoint in the real-expanded representation.  A pair of
    deterministic device bilinear probes verifies this property before the
    forward action is used as the adjoint action.  CuPy GMRES uses the
    occupied-virtual gaps as a device preconditioner.  Solver completion is
    accepted only after a fresh, unpreconditioned physical residual.

    CuPy's restarted GMRES currently solves its small Arnoldi least-squares
    problem on the host.  This is algorithmic host control rather than a
    CPU quantum-chemistry fallback.  The estimated transferred Hessenberg
    bytes are reported explicitly in ``info``.

    Args:
        mo_energy, mo_occ : cupy.ndarray
            Source-KRHF orbital energies and exact 2/0 occupations.
        Co, Cv : sequence of cupy.ndarray
            Complex occupied and virtual MO coefficient blocks.
        fvo : sequence of cupy.ndarray
            Positive target-Fock VO right-hand-side blocks.
        fvind : callable
            Coupled native GPU KRHF ``J[D]-K[D]/2`` response.

    Kwargs:
        conv_tol : float
            Relative CuPy GMRES convergence request.
        residual_tol : float
            Absolute fresh physical-residual acceptance threshold.
        max_cycle : int
            Maximum CuPy GMRES Arnoldi iterations.
        kpts_weights : array_like or None
            Equal full-BZ weights.
        self_adjoint_tol : float
            Maximum normalized bilinear-probe defect.
        restart : int
            Restarted Arnoldi basis dimension.

    Returns:
        tuple
            Complex device ``zvec`` blocks and a diagnostic dictionary.
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

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    Co, Cv = _validate_coeff_blocks(Co, Cv)
    fvo = _validate_vo_blocks(fvo, Co, Cv, name='fvo')
    weights = _equal_weights(len(Co), kpts_weights)
    shapes = tuple(block.shape for block in fvo)
    rhs = complex_to_real(flatten_vo(fvo))
    dimension = int(rhs.size)
    if dimension == 0:
        raise ValueError('matrix-free Z-vector requires a nonempty VO space')

    vind = make_hessian_operator(mo_energy, mo_occ, Co, Cv, fvind)
    mo_energy = cp.asarray(mo_energy)
    mo_occ = cp.asarray(mo_occ)
    occupied_masks, virtual_masks = make_occ_masks(cp.asnumpy(mo_occ))
    gap = []
    for energy, so_host, sv_host in zip(
            mo_energy, occupied_masks, virtual_masks, strict=True):
        so = cp.asarray(so_host)
        sv = cp.asarray(sv_host)
        gap.append(
            (energy[sv, None] - energy[so][None, :]).real.reshape(-1))
    gap = cp.concatenate(gap)
    gap = cp.concatenate((gap, gap))
    if (gap.shape != rhs.shape
            or not bool(cp.all(cp.isfinite(gap)).get())
            or not bool(cp.all(gap > 0).get())):
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
        return cp.asarray(vector) / gap

    operator = LinearOperator(
        (dimension, dimension), matvec=matvec, dtype=cp.float64)
    inverse_gap = LinearOperator(
        (dimension, dimension), matvec=precondition, dtype=cp.float64)

    rng = cp.random.RandomState(17)
    self_adjoint_error = 0.0
    for _ in range(2):
        x = rng.normal(size=dimension, dtype=cp.float64)
        y = rng.normal(size=dimension, dtype=cp.float64)
        ax = matvec(x)
        ay = matvec(y)
        lhs = cp.dot(x, ay)
        rhs_probe = cp.dot(ax, y)
        scale = cp.maximum(
            1.0, cp.maximum(cp.abs(lhs), cp.abs(rhs_probe)))
        defect = float((cp.abs(lhs - rhs_probe) / scale).get())
        self_adjoint_error = max(self_adjoint_error, defect)
    if self_adjoint_error > self_adjoint_tol:
        raise RuntimeError(
            'GPU sampled-k Hessian failed the self-adjoint gate: '
            f'{self_adjoint_error:.3e} > {self_adjoint_tol:.3e}')

    restart = min(int(restart), dimension)
    restart_cycles = 0

    def count_restart(_residual):
        nonlocal restart_cycles
        restart_cycles += 1

    z_real, solver_info = gmres(
        operator, rhs, M=inverse_gap, x0=rhs / gap,
        rtol=float(conv_tol), atol=0.0, restart=restart,
        maxiter=int(max_cycle), callback=count_restart,
        callback_type='pr_norm')
    if solver_info != 0:
        raise RuntimeError(
            'GPU sampled-k matrix-free GMRES did not converge: '
            f'info={solver_info}, restart_cycles={restart_cycles}')

    residual = matvec(z_real) - rhs
    abs_residual = _max_abs(residual)
    rhs_norm = _max_abs(rhs)
    if abs_residual > residual_tol:
        raise RuntimeError(
            'GPU sampled-k matrix-free Z-vector physical residual failed: '
            f'{abs_residual:.3e} > {residual_tol:.3e}')

    zvec = unflatten_vo(real_to_complex(z_real), shapes)
    workspace_estimate = int((restart + 5) * dimension * 8)
    hessenberg_host_bytes = int(
        restart_cycles * (restart + 1) * restart * 8)
    cp.cuda.Stream.null.synchronize()
    info = {
        'solver': 'cupyx_gmres_matrix_free',
        'equation': 'adjoint_self_adjoint_action',
        'rhs_sign': 'positive',
        'dimension': dimension,
        'complex_dimension': dimension // 2,
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
        'self_adjoint_probe_error': self_adjoint_error,
        'iterations': None,
        'restart_cycles': int(restart_cycles),
        'operator_calls': int(operator_calls),
        'preconditioner_calls': int(preconditioner_calls),
        'restart': int(restart),
        'dense_matrix_bytes': 0,
        'workspace_estimate_bytes': workspace_estimate,
        'krylov_host_control': 'cupyx_restart_least_squares',
        'krylov_host_control_bytes_estimate': hessenberg_host_bytes,
        'scientific_host_fallback': False,
        'wall_seconds': time.perf_counter() - start,
    }
    return zvec, info


def _validate_coeff_blocks(Co, Cv, nkpts=None):
    '''Return validated occupied/virtual device tuples.'''
    Co = tuple(cp.asarray(block) for block in Co)
    Cv = tuple(cp.asarray(block) for block in Cv)
    if not Co or len(Co) != len(Cv):
        raise ValueError('Co and Cv must contain equal nonzero k blocks')
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
                'GPU dense response requires equal VO topology at every k')
        if (not bool(cp.all(cp.isfinite(Co_k)).get())
                or not bool(cp.all(cp.isfinite(Cv_k)).get())):
            raise ValueError('MO coefficient blocks contain nonfinite values')
    return Co, Cv


def _validate_vo_blocks(blocks, Co, Cv, name):
    '''Return complex device VO blocks matching the partitions.'''
    blocks = tuple(cp.asarray(block, dtype=cp.complex128) for block in blocks)
    if len(blocks) != len(Co):
        raise ValueError(f'{name} block count differs from k-point count')
    for k, (block, Co_k, Cv_k) in enumerate(
            zip(blocks, Co, Cv, strict=True)):
        shape = (Cv_k.shape[1], Co_k.shape[1])
        if block.shape != shape:
            raise ValueError(
                f'{name} block {k} has shape {block.shape}; expected {shape}')
        if not bool(cp.all(cp.isfinite(block)).get()):
            raise ValueError(f'{name} block {k} contains nonfinite values')
    return blocks


def _equal_weights(nkpts, weights):
    '''Return the canonical equal full-BZ weights.'''
    expected = np.full(nkpts, 1.0 / nkpts)
    if weights is None:
        return expected
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (nkpts,) or not np.all(np.isfinite(weights)):
        raise ValueError('kpts_weights must have shape (nkpts,) and be finite')
    if not np.allclose(weights, expected, atol=2e-15, rtol=2e-15):
        raise NotImplementedError('GPU dense response requires equal weights')
    return expected


def _max_abs(value):
    '''Return one maximum device modulus as a Python diagnostic scalar.'''
    value = cp.asarray(value)
    return float(cp.max(cp.abs(value)).get()) if value.size else 0.0
