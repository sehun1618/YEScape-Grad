'''Version-pinned GPU4PySCF sampled-k compatibility contexts.'''

from contextlib import contextmanager
from threading import RLock

import cupy as cp
from gpu4pyscf.pbc.gto import int1e

_INT1E_LAYOUT_LOCK = RLock()


@contextmanager
def contiguous_int1e_hermitian(owner):
    '''Localize the GPU4PySCF 1.8 Hermitian-layout workaround.

    GPU4PySCF ``PBCIntOpt.intor`` can create a strided complex view after a
    sampled-k lattice-phase contraction and pass it to ``hermi_triu``, whose
    CUDA kernel requires C-contiguous storage.  Only that device view is
    copied; the module symbol is restored in ``finally`` and a process lock
    prevents overlapping compatibility contexts.
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
