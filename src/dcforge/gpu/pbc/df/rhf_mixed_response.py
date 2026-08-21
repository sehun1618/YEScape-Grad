'''Guarded mixed-precision direct-VO response for real-Gamma GDF.

Only the repeated three-center products in the inner orbital-Hessian action
use FP32.  Energy gaps, Krylov vectors, auxiliary-block accumulation, the
independently reconstructed physical residual, accepted Z-vector and every
nuclear-response contraction remain FP64.  A low-route failure is never
silent: the same FP64 operator is solved from a canonical initial guess.
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
        require_hdf5=False, log=None):
    '''Solve the GDF Z equation with a guarded fixed-FP32 inner operator.

    Returns ``(zvec, z_info, profile)``.  ``profile['fallback']`` is either
    ``None`` or a complete explanation of the canonical FP64 solve used in
    place of the low route.
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
    solver = str(refinement_solver).lower()
    if solver not in ('minres', 'gmres'):
        raise ValueError('refinement_solver must be minres or gmres')
    inner_tol = min(float(conv_tol), 1.e-10)
    exact_tol = min(float(residual_tol), 1.e-10)
    quantized_tol = 1.e-8
    quantized_residual_max = float('inf')
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
        exact = _ActionCounter(_FP64StorageResponse(
            mf_hf, Co, Cv, raw))
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
        quantized_residual = get_z_residual(
            zvec, mo_energy, mo_occ, fvo, low)
        quantized_residual_max = _maximum(quantized_residual)
        ledger.append({
            'kind': 'initial_low_precision_solve',
            'iterations': int(iterations),
            'solver_info': int(info),
            'quantized_operator_residual_max_abs': quantized_residual_max,
        })
        for _index in range(3):
            if quantized_residual_max <= quantized_tol:
                break
            correction, info, iterations = _iterative_correction(
                mo_energy, mo_occ, -quantized_residual, low,
                inner_tol, max_cycle, solver)
            candidate = zvec + correction
            candidate_residual = get_z_residual(
                candidate, mo_energy, mo_occ, fvo, low)
            candidate_max = _maximum(candidate_residual)
            ledger.append({
                'kind': 'quantized_operator_residual_cleanup',
                'iterations': int(iterations),
                'solver_info': int(info),
                'residual_before_max_abs': quantized_residual_max,
                'quantized_operator_residual_max_abs': candidate_max,
            })
            if (not np.isfinite(candidate_max)
                    or candidate_max >= .95 * quantized_residual_max):
                raise RuntimeError(
                    'quantized physical residual cleanup stagnated')
            zvec = candidate
            quantized_residual = candidate_residual
            quantized_residual_max = candidate_max
        if quantized_residual_max > quantized_tol:
            raise RuntimeError(
                'quantized physical residual exceeds 1e-8')
        physical = get_z_residual(
            zvec, mo_energy, mo_occ, fvo, exact)
        physical_max = _maximum(physical)
        ledger.append({
            'kind': 'initial_true_fp64_residual',
            'fp64_residual_max_abs': physical_max,
        })
        for _index in range(3):
            if physical_max <= exact_tol:
                break
            correction, info, iterations = _iterative_correction(
                mo_energy, mo_occ, -physical, low,
                inner_tol, max_cycle, solver)
            candidate = zvec + correction
            candidate_residual = get_z_residual(
                candidate, mo_energy, mo_occ, fvo, exact)
            candidate_max = _maximum(candidate_residual)
            ledger.append({
                'kind': 'fp64_defect_correction',
                'iterations': int(iterations),
                'solver_info': int(info),
                'residual_before_max_abs': physical_max,
                'fp64_residual_max_abs': candidate_max,
            })
            if (not np.isfinite(candidate_max)
                    or candidate_max >= .95 * physical_max):
                raise RuntimeError('FP64 defect correction stagnated')
            zvec = candidate
            physical = candidate_residual
            physical_max = candidate_max
        if physical_max > exact_tol:
            raise RuntimeError('FP64 physical residual exceeds its gate')
    except (MemoryError, cp.cuda.memory.OutOfMemoryError,
            RuntimeError, FloatingPointError) as error:
        if exact is None:
            exact = _ActionCounter(_FP64StorageResponse(
                mf_hf, Co, Cv, raw))
        zvec, exact_info = solve_z_vector(
            mo_energy, mo_occ, fvo, exact,
            conv_tol=conv_tol, residual_tol=residual_tol,
            max_cycle=max_cycle, refinement_solver=solver,
            initial_guess=None, log=log)
        physical_max = float(exact_info['residual_max_abs'])
        fallback = {
            'reason': f'{type(error).__name__}: {error}',
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
        z_info['response_backend'] = _FP64StorageResponse.backend
    else:
        iteration_count = sum(
            int(item.get('iterations', 0)) for item in ledger)
        z_info = {
            'rhs_max_abs': rhs_max,
            'residual_max_abs': physical_max,
            'residual_relative': (
                physical_max / rhs_max if rhs_max else physical_max),
            'solver': f'mixed_fixed_fp32_{solver}_fp64_defect_correction',
            'dimension': int(cp.asarray(fvo).size),
            'matrix_asymmetry_max_abs': None,
            'primary_wall_seconds': 0.,
            'refinement_wall_seconds': wall,
            'refinement_iterations': iteration_count,
            'requested_conv_tol': float(conv_tol),
            'requested_krylov_tol': None,
            'effective_krylov_tol': inner_tol,
            'krylov_tolerance_policy': 'fixed_fp32_inner_fp64_residual',
            'physical_residual_tol': float(residual_tol),
            'cleanup_krylov_tol': inner_tol,
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
        'storage': storage_profile,
        'low_action': low_profile,
        'fp64_residual_action': exact_profile,
        'quantized_operator_residual_max_abs': quantized_residual_max,
        'physical_residual_max_abs': physical_max,
        'refinement_ledger': ledger,
        'fallback': fallback,
        'wall_seconds': wall,
    }
    return zvec, z_info, profile


__all__ = ['solve_mixed_gdf_z_vector']
