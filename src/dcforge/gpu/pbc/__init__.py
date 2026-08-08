'''GPU periodic-boundary DC-DFT operations.

This namespace is intentionally imported explicitly.  It requires the CUDA
runtime and GPU4PySCF, whereas ``import dcforge`` and ``import dcforge.cpu``
must remain usable in a CPU-only environment.
'''

from dcforge.gpu.pbc import df, energy, jk

__all__ = ['df', 'energy', 'jk']
