'''Contracted GPU Gamma overlap and kinetic strain primitives.'''

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.gto import int1e


def contract_overlap_strain(cell, density):
    r'''Return ``Tr[P S^(strain)]`` as a host ``(3,3)`` array.'''
    return _contract_strain(int1e.ovlp_strain_deriv, cell, density)


def contract_kinetic_strain(cell, density):
    r'''Return ``Tr[P T^(strain)]`` as a host ``(3,3)`` array.'''
    return _contract_strain(int1e.kin_strain_deriv, cell, density)


def _contract_strain(function, cell, density):
    density = cp.asarray(density)
    if density.ndim != 2 or density.shape != (cell.nao_nr(),) * 2:
        raise ValueError('GPU one-electron strain density must have shape (nao, nao)')
    if cp.iscomplexobj(density):
        raise NotImplementedError(
            'the first real-Gamma GPU strain route requires a real density')
    value = np.asarray(function(cell, density))
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise RuntimeError('GPU one-electron strain returned an invalid tensor')
    return value
