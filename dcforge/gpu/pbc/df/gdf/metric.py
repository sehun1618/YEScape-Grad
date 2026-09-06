'''Provider-owned auxiliary-metric factors for GPU real-Gamma GDF.

GPU4PySCF factorizes the auxiliary Coulomb metric while building the source
CDERI.  Its eigenvalue decomposition removes directions below
``linear_dep_threshold``.  The source Hamiltonian therefore lives in that
retained auxiliary subspace, and its analytic derivative must use the same
subspace rather than solving the full metric again.

This module retains the factor which GPU4PySCF already creates.  It does not
rebuild the metric or CDERI and does not make a second device copy.  The
factor stays provider-owned until ``GDF.reset()`` and is used to construct the
same signed pseudoinverse required by the Gamma derivative.
'''

import hashlib
import threading
from contextlib import contextmanager

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc.df import rsdf_builder

_CAPTURE_LOCK = threading.RLock()


def init_source_metric_factor(provider):
    '''Initialize provider-owned source-metric state.'''
    provider._source_metric_factor = None
    provider.source_metric_factor_metadata = None
    provider.source_metric_factor_use_count = 0


def reset_source_metric_factor(provider):
    '''Release the retained device factor and its representation metadata.'''
    provider._source_metric_factor = None
    provider.source_metric_factor_metadata = None
    provider.source_metric_factor_use_count = 0


@contextmanager
def capture_source_metric_factor(provider):
    r'''Retain the exact Gamma metric factor made by the source CDERI build.

    GPU4PySCF's private ``_precontract_j2c_aux_coeff`` returns the already
    transformed :math:`M^{-1/2}` factor to its CDERI builder and normally lets
    it die when construction finishes.  The narrow transactional adapter below
    gives the same CuPy allocation a second owner on ``provider``.  No metric,
    eigendecomposition, or CDERI operation is repeated.

    The patched upstream entry point is process-global, so the lock covers the
    complete build and restoration occurs even after an exception.  Sampled-k
    factors are deliberately left to the separate R07-k implementation.

    ``ponytail:`` the lock serializes DC-Forge builds but cannot coordinate an
    unrelated direct GPU4PySCF build in another thread.  Replace this adapter
    with an upstream factor-return hook if GPU4PySCF exposes one.
    '''
    if not bool(provider.is_gamma_point):
        yield
        return

    reset_source_metric_factor(provider)
    with _CAPTURE_LOCK:
        original = rsdf_builder._precontract_j2c_aux_coeff

        def capture(auxcell, kpts, omega, rsdf_omega,
                    linear_dep_threshold, kmesh=None):
            factors, negative_metric_size = original(
                auxcell, kpts, omega, rsdf_omega,
                linear_dep_threshold, kmesh)
            if kpts is not None:
                return factors, negative_metric_size
            if len(factors) != 1:
                raise RuntimeError(
                    'real-Gamma source build must produce one metric factor')
            if provider._source_metric_factor is not None:
                raise RuntimeError(
                    'real-Gamma source build produced its metric factor twice')

            factor = factors[0]
            if not isinstance(factor, cp.ndarray):
                raise TypeError(
                    'GPU4PySCF did not return a device metric factor')
            negative_rank = int(negative_metric_size.get(0, 0))
            retained_rank = int(factor.shape[1]) - negative_rank
            if retained_rank < 1 or negative_rank < 0:
                raise RuntimeError('source auxiliary metric has no retained rank')
            if factor.ndim != 2 or factor.dtype != np.float64:
                raise TypeError(
                    'real-Gamma source metric factor must be a FP64 matrix')

            provider._source_metric_factor = factor
            provider.source_metric_factor_metadata = {
                'owner': 'GPU4PySCF source CDERI build',
                'storage': 'device',
                'representation': 'precontracted_inverse_sqrt',
                'shape': [int(value) for value in factor.shape],
                'dtype': np.dtype(factor.dtype).name,
                'bytes': int(factor.nbytes),
                'retained_rank': retained_rank,
                'negative_rank': negative_rank,
                'linear_dep_threshold': float(linear_dep_threshold),
                'physical_omega': float(omega),
                'partition_omega': float(rsdf_omega),
                'auxiliary_signature': _auxiliary_signature(auxcell),
            }
            return factors, negative_metric_size

        rsdf_builder._precontract_j2c_aux_coeff = capture
        try:
            yield
            if provider._source_metric_factor is None:
                raise RuntimeError(
                    'GPU4PySCF source CDERI build did not expose its Gamma '
                    'auxiliary-metric factor')
        except BaseException:
            reset_source_metric_factor(provider)
            raise
        finally:
            rsdf_builder._precontract_j2c_aux_coeff = original


def attach_source_metric_factor(provider, intopt):
    '''Attach one validated provider factor to a derivative optimizer.'''
    factor = provider._source_metric_factor
    metadata = provider.source_metric_factor_metadata
    if factor is None or metadata is None:
        raise RuntimeError(
            'the active GDF provider has no retained source metric factor; '
            'rebuild the provider with mf.kernel() before requesting a Gamma '
            'analytic gradient')
    if metadata['auxiliary_signature'] != _auxiliary_signature(intopt.auxcell):
        raise RuntimeError(
            'source and derivative auxiliary-metric representations differ')
    if metadata['shape'][0] != int(intopt.auxcell.ao_loc[-1]):
        raise RuntimeError(
            'source metric factor does not match the derivative auxiliary axis')
    if metadata['linear_dep_threshold'] != float(
            provider.linear_dep_threshold):
        raise RuntimeError(
            'source metric factor linear-dependence threshold is stale')
    if metadata['partition_omega'] != float(provider.resolved_omega):
        raise RuntimeError('source metric factor range partition is stale')

    intopt._dcforge_source_metric_factor = factor
    intopt._dcforge_source_metric_factor_metadata = metadata
    intopt._dcforge_source_metric_factor_owner = provider
    return intopt


def metric_inverse_from_source(intopt, metric):
    r'''Return :math:`M^+` from the source-CDERI retained metric factor.

    ``metric`` is accepted to validate the derivative representation; it is
    not solved.  If the source factor contains an indefinite auxiliary tail,
    its columns contribute with the same negative sign used by GPU4PySCF's
    positive/negative CDERI split.
    '''
    factor = getattr(intopt, '_dcforge_source_metric_factor', None)
    metadata = getattr(
        intopt, '_dcforge_source_metric_factor_metadata', None)
    owner = getattr(intopt, '_dcforge_source_metric_factor_owner', None)
    if factor is None or metadata is None or owner is None:
        raise RuntimeError(
            'derivative optimizer has no provider-owned source metric factor')

    auxcell = intopt.auxcell
    transformed = auxcell.apply_CT_mat_C(metric)
    if transformed.ndim != 2 or transformed.shape[0] != transformed.shape[1]:
        raise ValueError('auxiliary metric must be square')
    if auxcell.cell.cart:
        raise NotImplementedError('Cartesian auxiliary metrics are unsupported')
    coefficient_shape = tuple(auxcell.ctr_coeff.shape)
    if transformed.shape != (coefficient_shape[1], coefficient_shape[1]):
        raise ValueError(
            'transformed metric and auxiliary contraction space differ')
    if factor.shape[0] != int(auxcell.ao_loc[-1]):
        raise ValueError('source factor and primitive auxiliary axis differ')

    negative_rank = int(metadata['negative_rank'])
    positive_rank = int(factor.shape[1]) - negative_rank
    positive = factor[:, :positive_rank]
    inverse = contract('pr,qr->pq', positive, positive)
    if negative_rank:
        negative = factor[:, positive_rank:]
        contract('pr,qr->pq', negative, negative,
                 alpha=-1., beta=1., out=inverse)
    owner.source_metric_factor_use_count += 1
    return inverse


def _auxiliary_signature(auxcell):
    '''Return a compact exact signature of one sorted auxiliary Cell.'''
    digest = hashlib.sha256()
    for value in (auxcell._atm, auxcell._bas, auxcell._env, auxcell.ao_loc):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update(str(bool(auxcell.cart)).encode())
    digest.update(str(int(auxcell.dimension)).encode())
    return digest.hexdigest()


__all__ = [
    'attach_source_metric_factor',
    'capture_source_metric_factor',
    'init_source_metric_factor',
    'metric_inverse_from_source',
    'reset_source_metric_factor',
]
