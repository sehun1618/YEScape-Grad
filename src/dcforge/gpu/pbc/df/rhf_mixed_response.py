'''Guarded mixed-precision direct-VO response for real-Gamma GDF.

Only the repeated three-center products in the inner orbital-Hessian action
use FP32.  Energy gaps, Krylov vectors, auxiliary-block accumulation, the
independently reconstructed physical residual, accepted Z-vector and every
nuclear-response contraction remain FP64.  A low-route failure is never
silent: the same FP64 operator is solved again, first from the retained finite
low-precision vector and then from the diagonal-gap guess when that is safer.
'''

import time

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import asarray, contract
from pyscf import lib

from dcforge.gpu.pbc.df.cderi import H5CDERI, copy_gamma_cderi_fp32
from dcforge.gpu.pbc.df.rhf_response import _DirectVOResponse
from dcforge.gpu.pbc.grad.rhf_zvec import (
    _gmres_solve,
    _minres_solve,
    get_z_residual,
    solve_z_vector,
)


def _maximum(value):
    value = cp.asarray(value)
    return float(cp.max(cp.abs(value)).get()) if value.size else 0.


class _ActionCounter:
    '''Count response actions without adding synchronization to the solve.'''

    def __init__(self, action):
        self.action = action
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        return self.action(value)

    def close(self):
        close = getattr(self.action, 'close', None)
        return close() if close is not None else None


class _FP64StorageResponse:
    '''Direct-VO FP64 oracle reading one explicit compressed factor.'''

    backend = 'gdf_direct_vo_fp64_storage_oracle'

    def __init__(self, mf_hf, Co, Cv, storage):
        self._base = _DirectVOResponse(
            mf_hf, Co, Cv, cache_mode='recompute', profile_detail=False)
        self._storage = storage
        self._closed = False
        self.profile = self._base.profile
        self.profile.update({
            'backend': self.backend,
            'provider_cderi_storage': type(storage).__name__,
            'storage_dtype': np.dtype(storage.dtype).name,
            'contraction_dtype': 'float64',
        })
        self._install_iterator()

    def _install_iterator(self):
        base = self._base
        storage = self._storage
        blocking = isinstance(storage, H5CDERI)

        def iterator():
            block_size = base.profile['block_size']
            mapping, _diagonal = base.provider._cderi_idx
            mapping = cp.asarray(mapping)
            rows, columns = divmod(mapping, base.nao)
            dense = cp.zeros(
                (block_size, base.nao, base.nao), dtype=cp.float64)
            for p0, p1 in lib.prange(0, base.naux, block_size):
                # H5CDERI slicing creates a temporary NumPy owner.  A
                # blocking copy is required because GPU4PySCF's optimized
                # ``asarray`` deliberately omits CuPy's pinned staging copy.
                packed = asarray(storage[p0:p1], blocking=blocking)
                block = dense[:p1-p0]
                block[...] = 0.
                block[:, columns, rows] = block[:, rows, columns] = packed
                base.profile['host_to_device_cderi_bytes'] += int(
                    packed.nbytes)
                yield block, packed, 1.

        base._iter_provider_blocks = iterator

    def __call__(self, value):
        if self._closed:
            raise RuntimeError('FP64 storage response is closed')
        return self._base(value)

    def close(self):
        if not self._closed:
            self.profile = self._base.close()
            self._base = None
            self._storage = None
            self._closed = True
        return self.profile


class _FP32StorageFP64Response:
    r'''Evaluate the physical response from FP32 storage in FP64 arithmetic.

    The persistent factor is transferred to the device in FP32 and promoted
    only after the current auxiliary block arrives.  This distinction matters
    for a Water64 factor: asking :class:`FP32CDERIView` for an FP64 host slice
    doubles both the transfer volume and the temporary host traffic on every
    residual or fallback action.  Device-side promotion changes neither the
    stored Hamiltonian nor the direct-VO algebra; it only removes that repeated
    host expansion.

    Notes:
        FP64 arithmetic cannot recover bits already discarded by a ``mixed``
        or ``fp32_build`` GDF profile.  A true native-FP64 reference requires
        rebuilding the provider with ``gdf_precision_profile='fp64'``.
    '''

    backend = 'gdf_direct_vo_fp32_storage_fp64_compute'

    def __init__(self, mf_hf, Co, Cv, storage):
        if np.dtype(storage.dtype) != np.dtype(np.float32):
            raise TypeError('device promotion requires FP32 CDERI storage')
        self._base = _DirectVOResponse(
            mf_hf, Co, Cv, cache_mode='recompute', profile_detail=False)
        self._storage = storage
        self._closed = False
        self.profile = self._base.profile
        self.profile.update({
            'backend': self.backend,
            'provider_cderi_storage': type(storage).__name__,
            'storage_dtype': 'float32',
            'transfer_dtype': 'float32',
            'device_promotion_dtype': 'float64',
            'contraction_dtype': 'float64',
            'host_fp64_slice_materialized': False,
        })
        self._install_iterator()

    def _install_iterator(self):
        base = self._base
        storage = self._storage

        def iterator():
            block_size = base.profile['block_size']
            mapping, _diagonal = base.provider._cderi_idx
            mapping = cp.asarray(mapping)
            rows, columns = divmod(mapping, base.nao)
            dense = cp.zeros(
                (block_size, base.nao, base.nao), dtype=cp.float64)
            for p0, p1 in lib.prange(0, base.naux, block_size):
                packed = asarray(storage[p0:p1])
                if packed.dtype != cp.float32:
                    raise TypeError(
                        'device-promotion transfer changed FP32 dtype')
                block = dense[:p1-p0]
                block[...] = 0.
                # Assignment performs the promotion on the GPU after the
                # half-sized FP32 host-to-device transfer has completed.
                block[:, columns, rows] = block[:, rows, columns] = packed
                base.profile['host_to_device_cderi_bytes'] += int(
                    packed.nbytes)
                yield block, packed, 1.

        base._iter_provider_blocks = iterator

    def __call__(self, value):
        if self._closed:
            raise RuntimeError('FP32-storage FP64 response is closed')
        return self._base(value)

    def close(self):
        if not self._closed:
            self.profile = self._base.close()
            self._base = None
            self._storage = None
            self._closed = True
        return self.profile


class _FP32DirectVOResponse:
    r'''Fixed FP32 three-center action with FP64 block accumulation.'''

    backend = 'gdf_direct_vo_fp32_gemm_fp64_block_sum'

    def __init__(self, mf_hf, Co, Cv, storage):
        self._base = _DirectVOResponse(
            mf_hf, Co, Cv, cache_mode='recompute', profile_detail=False)
        self._storage = storage
        self.Co = self._base.Co
        self.Cv = self._base.Cv
        self.Co32 = self.Co.astype(cp.float32)
        self.Cv32 = self.Cv.astype(cp.float32)
        self.nocc = self._base.nocc
        self.nvir = self._base.nvir
        self.nao = self._base.nao
        self.naux = self._base.naux
        self.ewald = self._base.ewald
        self._dense = None
        self._closed = False
        self.profile = {
            'backend': self.backend,
            'storage_dtype': np.dtype(storage.dtype).name,
            'transfer_dtype': 'float32',
            'multiplication_dtype': 'float32',
            'between_block_accumulation_dtype': 'float64',
            'action_count': 0,
            'host_to_device_cderi_bytes': 0,
            'closed': False,
        }

    def __call__(self, value):
        if self._closed:
            raise RuntimeError('FP32 direct-VO response is closed')
        shape = cp.asarray(value).shape
        amplitudes = cp.asarray(
            value, dtype=cp.float64).reshape(-1, self.nvir, self.nocc)
        result = cp.stack([
            self._contract_one(amplitude) for amplitude in amplitudes])
        return result.reshape(shape)

    def _contract_one(self, amplitude):
        left32 = contract(
            'pa,ai->pi', self.Cv32, amplitude.astype(cp.float32))
        response_occ = cp.zeros((self.nao, self.nocc), dtype=cp.float64)
        for cderi in self._iter_blocks():
            b_p = contract('Lpq,qi->Lpi', cderi, left32)
            a_p = contract('Lpq,qi->Lpi', cderi, self.Co32)
            h_p = contract('Lpi,pj->Lij', a_p, self.Co32)
            rho = contract('Lpi,pi->L', b_p, self.Co32)
            block = cp.zeros((self.nao, self.nocc), dtype=cp.float32)
            contract('L,Lpi->pi', rho, a_p, alpha=4., beta=1., out=block)
            contract(
                'Lpi,Lij->pj', b_p, h_p, alpha=-1., beta=1., out=block)
            b_metric = contract('Lpi,pj->Lij', b_p, self.Co32)
            contract(
                'Lpi,Lij->pj', a_p, b_metric,
                alpha=-1., beta=1., out=block)
            response_occ += block.astype(cp.float64)
        if self.ewald is not None:
            overlap, overlap_occ, madelung = self.ewald
            left = self.Cv @ amplitude
            overlap_left = contract('pq,qi->pi', overlap, left)
            occ_metric = contract('pi,pj->ij', self.Co, overlap_occ)
            left_metric = contract('pi,pj->ij', left, overlap_occ)
            contract(
                'pi,ij->pj', overlap_left, occ_metric,
                alpha=-madelung, beta=1., out=response_occ)
            contract(
                'pi,ij->pj', overlap_occ, left_metric,
                alpha=-madelung, beta=1., out=response_occ)
        self.profile['action_count'] += 1
        return contract('ap,pi->ai', self.Cv.T, response_occ)

    def _iter_blocks(self):
        block_size = self._base.profile['block_size']
        mapping, _diagonal = self._base.provider._cderi_idx
        mapping = cp.asarray(mapping)
        rows, columns = divmod(mapping, self.nao)
        if self._dense is None:
            self._dense = cp.zeros(
                (block_size, self.nao, self.nao), dtype=cp.float32)
        for p0, p1 in lib.prange(0, self.naux, block_size):
            packed = asarray(self._storage[p0:p1])
            if packed.dtype != cp.float32:
                raise TypeError('mixed response transfer changed FP32 dtype')
            block = self._dense[:p1-p0]
            block[...] = 0.
            block[:, columns, rows] = block[:, rows, columns] = packed
            self.profile['host_to_device_cderi_bytes'] += int(packed.nbytes)
            yield block

    def close(self):
        if not self._closed:
            self._dense = None
            self._storage = None
            self.Co32 = self.Cv32 = self.Co = self.Cv = self.ewald = None
            self.profile['fp64_base_profile'] = self._base.close()
            self._base = None
            self._closed = True
            self.profile['closed'] = True
        return self.profile


def _copy_fp32_storage(provider, block_size=64):
    '''Round one completed real-Gamma positive CDERI into mapped FP32.'''
    return copy_gamma_cderi_fp32(provider, block_size=block_size)


def _iterative_correction(
        mo_energy, mo_occ, rhs, action, tolerance, max_cycle,
        solver, initial_guess=None):
    function = _minres_solve if solver == 'minres' else _gmres_solve
    return function(
        mo_energy, mo_occ, rhs, action, tolerance, max_cycle,
        initial_guess=initial_guess)


def solve_mixed_gdf_z_vector(
        mf_hf, mo_energy, mo_occ, Co, Cv, fvo, *,
        conv_tol=1.e-9, residual_tol=1.e-8, max_cycle=200,
        refinement_solver='minres', initial_guess=None,
        require_hdf5=False, inner_tol=1.e-6, correction_tol=1.e-5,
        quantized_residual_tol=1.e-7,
        quantized_hard_reject_tol=1.e-5, max_corrections=3,
        log=None):
    r'''Solve the GDF Z equation with the guarded Z3 mixed algorithm.

    Z3 deliberately separates two residuals which answer different questions.
    The *quantized* residual measures convergence under the inexpensive FP32
    inner operator and is only a cleanup/scheduling diagnostic.  The
    independently reconstructed *physical* residual is evaluated with FP64
    arithmetic and is the sole scientific acceptance gate.  Consequently,
    FP32 roundoff cannot force a costly exact solve merely because its own
    residual floor lies slightly above ``quantized_residual_tol``.

    If physical defect correction still fails, the last finite low-precision
    solution is retained as an exact-solver initial guess.  It is passed only
    when its physical residual improves upon the zero vector, and
    :func:`solve_z_vector` independently compares it with the diagonal-gap
    guess before actually using it.  No failed low route is silently accepted.

    Args:
        mf_hf, mo_energy, mo_occ, Co, Cv, fvo
            Source-RHF state and target occupied-virtual Fock block used by the
            ordinary periodic RHF adjoint equation.

    Kwargs:
        conv_tol, residual_tol, max_cycle, refinement_solver, initial_guess
            Same meanings as in :func:`solve_z_vector`.  ``residual_tol`` is
            the final physical acceptance threshold.
        require_hdf5 : bool
            Require the FP64 source factor to be HDF5-backed.  This is the
            capacity contract of ``precision_policy='mixed_lowmem'``.
        inner_tol : float
            Krylov tolerance for the first fixed-FP32 solve.
        correction_tol : float
            Krylov tolerance for inexpensive quantized/physical defect
            corrections.  Every candidate is still audited with the physical
            operator before acceptance.
        quantized_residual_tol : float
            Advisory FP32-operator cleanup target.  It is intentionally looser
            than the physical gate because FP32 arithmetic has its own
            numerical floor.
        quantized_hard_reject_tol : float
            Above this diagnostic value Z3 skips further low-operator defect
            correction and enters the guarded exact fallback.
        max_corrections : int
            Maximum cleanup steps for each low-operator residual stage.

    Returns:
        tuple
            ``(zvec, z_info, profile)``.  The profile records both residuals,
            every correction, whether a fallback seed was passed and actually
            used, and the CDERI transfer/promotion boundary.

    Notes:
        With a native FP64 provider, the physical residual is evaluated
        against the original FP64 CDERI.  With ``gdf_precision_profile`` set
        to ``mixed`` or ``fp32_build``, the provider owns only an FP32 factor;
        Z3 then uses FP64 arithmetic on that quantized Hamiltonian.  Selecting
        ``precision_policy='fp64'`` changes the Z arithmetic, but recovering a
        native-FP64 Hamiltonian requires rebuilding the complete GDF profile
        as ``fp64``.
    '''
    provider = mf_hf.with_df
    raw = provider._cderi[0]
    if require_hdf5 and not isinstance(raw, H5CDERI):
        raise ValueError(
            "precision_policy='mixed_lowmem' requires "
            "mf.with_df.cderi_storage='hdf5' before mf.kernel()")
    started = time.perf_counter()
    storage = low = exact = None
    storage_profile = None
    fallback = None
    ledger = []
    low_failure = None
    solver = str(refinement_solver).lower()
    if solver not in ('minres', 'gmres'):
        raise ValueError('refinement_solver must be minres or gmres')
    inner_tol = float(inner_tol)
    correction_tol = float(correction_tol)
    physical_tol = float(residual_tol)
    quantized_tol = float(quantized_residual_tol)
    hard_reject_tol = float(quantized_hard_reject_tol)
    max_corrections = int(max_corrections)
    if min(inner_tol, correction_tol, physical_tol, quantized_tol) <= 0.:
        raise ValueError('mixed Z tolerances must be positive')
    if hard_reject_tol < quantized_tol:
        raise ValueError(
            'quantized_hard_reject_tol must not be smaller than '
            'quantized_residual_tol')
    if max_corrections < 0:
        raise ValueError('max_corrections must be nonnegative')
    quantized_residual_max = float('inf')
    quantized_status = 'not_evaluated'
    physical_max = float('inf')
    physical = None
    zvec = None
    initial_guess_requested = initial_guess is not None
    initial_guess_used = False
    initial_guess_fallback_reason = None
    initial_residual_max_abs = None
    diagonal_initial_residual_max_abs = None
    try:
        storage, storage_profile = _copy_fp32_storage(provider)
        low = _ActionCounter(_FP32DirectVOResponse(
            mf_hf, Co, Cv, storage))
        if storage_profile['reused_provider_storage']:
            # ``mixed`` and ``fp32_build`` providers no longer own the
            # discarded FP64 bits.  Transfer their resident factor in FP32
            # and promote only the active device block for physical audits.
            exact = _ActionCounter(_FP32StorageFP64Response(
                mf_hf, Co, Cv, storage))
            physical_operator = 'fp32_cderi_fp64_arithmetic'
        else:
            # Response-only mixed precision retains the native FP64 provider
            # and must audit against that unmodified Hamiltonian.
            exact = _ActionCounter(_FP64StorageResponse(
                mf_hf, Co, Cv, raw))
            physical_operator = 'native_fp64_cderi'
        selected_initial_guess = None
        if initial_guess_requested:
            candidate = cp.asarray(initial_guess, dtype=fvo.dtype)
            if candidate.shape != fvo.shape:
                initial_guess_fallback_reason = (
                    f'projected shape {candidate.shape} differs from '
                    f'{fvo.shape}')
            elif not bool(cp.all(cp.isfinite(candidate))):
                initial_guess_fallback_reason = (
                    'projected guess is not finite')
            else:
                gap = (
                    mo_energy[mo_occ == 0, None]
                    - mo_energy[mo_occ > 0])
                diagonal_guess = fvo / gap
                candidate_residual = get_z_residual(
                    candidate, mo_energy, mo_occ, fvo, exact)
                diagonal_residual = get_z_residual(
                    diagonal_guess, mo_energy, mo_occ, fvo, exact)
                initial_residual_max_abs = _maximum(candidate_residual)
                diagonal_initial_residual_max_abs = _maximum(
                    diagonal_residual)
                if (initial_residual_max_abs
                        <= diagonal_initial_residual_max_abs):
                    selected_initial_guess = candidate
                    initial_guess_used = True
                else:
                    initial_guess_fallback_reason = (
                        'projected FP64 physical residual exceeds '
                        'diagonal-gap guess')
        zvec, info, iterations = _iterative_correction(
            mo_energy, mo_occ, fvo, low, inner_tol, max_cycle,
            solver, initial_guess=selected_initial_guess)
        if not bool(cp.all(cp.isfinite(zvec))):
            raise FloatingPointError('initial mixed Z-vector is nonfinite')
        quantized_residual = get_z_residual(
            zvec, mo_energy, mo_occ, fvo, low)
        quantized_residual_max = _maximum(quantized_residual)
        ledger.append({
            'kind': 'initial_low_precision_solve',
            'iterations': int(iterations),
            'solver_info': int(info),
            'quantized_operator_residual_max_abs': quantized_residual_max,
        })
        for index in range(1, max_corrections + 1):
            if quantized_residual_max <= quantized_tol:
                break
            correction, info, iterations = _iterative_correction(
                mo_energy, mo_occ, -quantized_residual, low,
                correction_tol, max_cycle, solver)
            if not bool(cp.all(cp.isfinite(correction))):
                low_failure = (
                    f'quantized correction {index} is nonfinite')
                break
            candidate = zvec + correction
            candidate_residual = get_z_residual(
                candidate, mo_energy, mo_occ, fvo, low)
            candidate_max = _maximum(candidate_residual)
            reduction = candidate_max / quantized_residual_max
            ledger.append({
                'kind': 'quantized_operator_residual_cleanup',
                'correction': index,
                'iterations': int(iterations),
                'solver_info': int(info),
                'residual_before_max_abs': quantized_residual_max,
                'quantized_operator_residual_max_abs': candidate_max,
                'reduction_ratio': reduction,
            })
            if not np.isfinite(candidate_max):
                low_failure = (
                    f'quantized correction {index} is nonfinite')
                break
            zvec = candidate
            quantized_residual = candidate_residual
            quantized_residual_max = candidate_max
            if reduction >= .95:
                low_failure = (
                    f'quantized correction {index} stagnated')
                break
        quantized_status = (
            'advisory_pass'
            if quantized_residual_max <= quantized_tol else
            'hard_reject'
            if quantized_residual_max > hard_reject_tol else
            'advisory_miss')
        physical = get_z_residual(
            zvec, mo_energy, mo_occ, fvo, exact)
        physical_max = _maximum(physical)
        ledger.append({
            'kind': 'initial_physical_residual',
            'physical_residual_max_abs': physical_max,
            'quantized_gate_status': quantized_status,
        })
        # A hard quantized miss says the low action is too noisy to be a
        # trustworthy correction operator.  It does not by itself reject an
        # already physically converged vector; only ``physical_tol`` does.
        allow_physical_correction = quantized_status != 'hard_reject'
        for index in range(1, max_corrections + 1):
            if physical_max <= physical_tol:
                break
            if not allow_physical_correction:
                break
            correction, info, iterations = _iterative_correction(
                mo_energy, mo_occ, -physical, low,
                correction_tol, max_cycle, solver)
            if not bool(cp.all(cp.isfinite(correction))):
                low_failure = f'physical correction {index} is nonfinite'
                break
            candidate = zvec + correction
            candidate_residual = get_z_residual(
                candidate, mo_energy, mo_occ, fvo, exact)
            candidate_max = _maximum(candidate_residual)
            reduction = candidate_max / physical_max
            ledger.append({
                'kind': 'physical_defect_correction',
                'correction': index,
                'iterations': int(iterations),
                'solver_info': int(info),
                'residual_before_max_abs': physical_max,
                'physical_residual_max_abs': candidate_max,
                'reduction_ratio': reduction,
            })
            if not np.isfinite(candidate_max):
                low_failure = f'physical correction {index} is nonfinite'
                break
            zvec = candidate
            physical = candidate_residual
            physical_max = candidate_max
            if reduction >= .95:
                low_failure = f'physical correction {index} stagnated'
                break
        if physical_max > physical_tol:
            raise RuntimeError(
                low_failure or
                f'physical residual {physical_max:.3e} exceeds '
                f'{physical_tol:.3e}')
    except (MemoryError, cp.cuda.memory.OutOfMemoryError,
            RuntimeError, FloatingPointError) as error:
        if exact is None:
            exact = _ActionCounter(_FP64StorageResponse(
                mf_hf, Co, Cv, raw))
            physical_operator = 'native_fp64_cderi'
        rhs_max = _maximum(fvo)
        seed_preserved = bool(
            zvec is not None and bool(cp.all(cp.isfinite(zvec))))
        if seed_preserved and not np.isfinite(physical_max):
            physical = get_z_residual(
                zvec, mo_energy, mo_occ, fvo, exact)
            physical_max = _maximum(physical)
        seed_passed = bool(
            seed_preserved and np.isfinite(physical_max)
            and physical_max < rhs_max)
        seed_physical_max = (
            float(physical_max) if seed_preserved else None)
        zvec, exact_info = solve_z_vector(
            mo_energy, mo_occ, fvo, exact,
            conv_tol=conv_tol, residual_tol=residual_tol,
            max_cycle=max_cycle, refinement_solver=solver,
            initial_guess=zvec if seed_passed else None, log=log)
        physical_max = float(exact_info['residual_max_abs'])
        fallback = {
            'reason': f'{type(error).__name__}: {error}',
            'low_vector_preserved': seed_preserved,
            'initial_guess_passed_to_exact_solver': seed_passed,
            'initial_guess_used': bool(exact_info.get(
                'initial_guess_used', False)),
            'initial_guess_physical_residual_max_abs': (
                seed_physical_max),
            'zero_guess_residual_max_abs': rhs_max,
            'exact_solver_info': exact_info,
        }
    finally:
        low_profile = None if low is None else low.close()
        exact_profile = None if exact is None else exact.close()
        # Mapped FP32 storage belongs only to this solve.  The response state
        # retains Z, not CDERI, so scanner/reset cannot reuse stale factors.
        storage = None

    rhs_max = _maximum(fvo)
    wall = time.perf_counter() - started
    if fallback is not None:
        z_info = dict(fallback['exact_solver_info'])
        z_info['solver'] = 'mixed_fallback->' + z_info['solver']
        z_info['wall_seconds'] = wall
        z_info['response_backend'] = getattr(
            exact.action, 'backend', _FP64StorageResponse.backend)
    else:
        iteration_count = sum(
            int(item.get('iterations', 0)) for item in ledger)
        z_info = {
            'rhs_max_abs': rhs_max,
            'residual_max_abs': physical_max,
            'residual_relative': (
                physical_max / rhs_max if rhs_max else physical_max),
            'solver': f'mixed_z3_{solver}_physical_gate',
            'dimension': int(cp.asarray(fvo).size),
            'matrix_asymmetry_max_abs': None,
            'primary_wall_seconds': 0.,
            'refinement_wall_seconds': wall,
            'refinement_iterations': iteration_count,
            'requested_conv_tol': float(conv_tol),
            'requested_krylov_tol': None,
            'effective_krylov_tol': inner_tol,
            'krylov_tolerance_policy': (
                'z3_fixed_fp32_inner_physical_residual_gate'),
            'physical_residual_tol': float(residual_tol),
            'cleanup_krylov_tol': correction_tol,
            'cleanup_wall_seconds': 0.,
            'cleanup_iterations': None,
            'response_backend': _FP32DirectVOResponse.backend,
            'initial_guess_requested': initial_guess_requested,
            'initial_guess_used': initial_guess_used,
            'initial_guess_fallback_reason': (
                initial_guess_fallback_reason),
            'initial_residual_max_abs': initial_residual_max_abs,
            'diagonal_initial_residual_max_abs': (
                diagonal_initial_residual_max_abs),
            'wall_seconds': wall,
        }
    profile = {
        'policy': 'mixed_lowmem' if require_hdf5 else 'mixed',
        'algorithm': 'z3_separate_quantized_and_physical_gates',
        'inner_tolerance': inner_tol,
        'correction_tolerance': correction_tol,
        'quantized_residual_tolerance': quantized_tol,
        'quantized_hard_reject_tolerance': hard_reject_tol,
        'quantized_gate_status': quantized_status,
        'max_corrections': max_corrections,
        'physical_operator': physical_operator,
        'storage': storage_profile,
        'low_action': low_profile,
        'physical_residual_action': exact_profile,
        # Compatibility key retained for existing queue/report consumers.
        'fp64_residual_action': exact_profile,
        'quantized_operator_residual_max_abs': quantized_residual_max,
        'physical_residual_max_abs': physical_max,
        'refinement_ledger': ledger,
        'low_route_failure': low_failure,
        'fallback': fallback,
        'wall_seconds': wall,
    }
    return zvec, z_info, profile


__all__ = ['solve_mixed_gdf_z_vector']
