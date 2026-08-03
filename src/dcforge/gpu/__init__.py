'''GPU operations backed by GPU4PySCF.

This namespace is imported explicitly so that ordinary ``import dcforge``
does not require an installed CUDA runtime or GPU4PySCF distribution.
'''

__all__ = ['df', 'energy', 'grad', 'pbc']
