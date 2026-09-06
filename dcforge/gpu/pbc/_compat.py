'''Small compatibility helpers for the supported GPU periodic paths.'''

from contextlib import contextmanager
from threading import RLock

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.gto import int1e

_INT1E_LAYOUT_LOCK = RLock()


def gamma_kpts(kpts=None):
    '''Return one Cartesian Gamma point or reject sampled-k input.'''
    if kpts is None:
        return np.zeros((1, 3))
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    if len(kpts) != 1 or np.max(np.abs(kpts)) > 1.e-12:
        raise NotImplementedError(
            'this DC-Forge slice supports one real Gamma point only')
    return kpts


def check_gamma_kpts_equal(left, right, label):
    '''Reject inconsistent provider/derivative Gamma-point identities.'''
    left = gamma_kpts(left)
    right = gamma_kpts(right)
    if not np.allclose(left, right, atol=1.e-12, rtol=0):
        raise ValueError(f'{label} do not match')


@contextmanager
def contiguous_int1e_hermitian(owner):
    '''Localize the GPU4PySCF 1.8 sampled-k Hermitian-layout workaround.

    GPU4PySCF forms a strided complex view after the sampled-k lattice-phase
    contraction.  Its ``hermi_triu`` CUDA kernel requires C-contiguous input.
    Copy only that view, record the boundary on ``owner``, and restore the
    upstream module symbol even when the integral call raises.

    Notes:
        The process-wide lock is deliberate because GPU4PySCF resolves
        ``hermi_triu`` as a module global.  ponytail: replace this context
        when upstream exposes a per-call contiguous conversion or hook.
    '''
    with _INT1E_LAYOUT_LOCK:
        original = int1e.hermi_triu

        def hermi_triu_contiguous(matrix, *args, **kwargs):
            if not matrix.flags.c_contiguous:
                owner.int1e_layout_copy_count += 1
                owner.int1e_layout_copy_bytes += int(matrix.nbytes)
                matrix = cp.ascontiguousarray(matrix)
            return original(matrix, *args, **kwargs)

        int1e.hermi_triu = hermi_triu_contiguous
        try:
            yield
        finally:
            int1e.hermi_triu = original
