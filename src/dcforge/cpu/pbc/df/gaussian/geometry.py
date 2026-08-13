'''Geometry copies which preserve an already built periodic AO gauge.'''

import numpy as np
from pyscf import gto


def copy_with_coordinates(cell, coordinates, lattice=None, mesh=None):
    r'''Copy a built Cell and replace only its Cartesian center coordinates.

    PySCF stores AO exponents, contraction coefficients, normalization
    factors, and nuclear coordinates together in ``cell._env``.  Calling
    :meth:`Cell.set_geom_` rebuilds that environment.  In particular, changing
    an Angstrom Cell to ``unit='Bohr'`` can normalize an already transformed
    GPU4PySCF auxiliary basis a second time.  A strain or lattice-image copy
    must instead retain the exact central AO gauge and change only the three
    coordinate slots owned by each atom.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built primary or auxiliary periodic Cell.
        coordinates : ndarray of shape (natm,3)
            New Cartesian nuclear coordinates in Bohr.

    Kwargs:
        lattice : ndarray of shape (3,3) or None
            Optional row-lattice vectors in Bohr.  Supplying a lattice also
            changes the copy's input-unit label to Bohr without rebuilding
            its AO environment.
        mesh : array_like of length 3 or None
            Optional resolved integer mesh copied without re-rounding a
            kinetic-energy cutoff.

    Returns:
        pyscf.pbc.gto.Cell
            A deep copy with unchanged ``_atm``/``_bas`` and unchanged
            non-coordinate ``_env`` entries.
    '''
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (cell.natm, 3):
        raise ValueError(
            'coordinates must have shape '
            f'({cell.natm},3), got {coordinates.shape}')
    if not np.all(np.isfinite(coordinates)):
        raise ValueError('coordinates must be finite')

    copied = cell.copy(deep=True)
    for atom, coordinate in enumerate(coordinates):
        pointer = int(copied._atm[atom, gto.PTR_COORD])
        copied._env[pointer:pointer + 3] = coordinate

    if lattice is not None:
        lattice = np.asarray(lattice, dtype=float)
        if lattice.shape != (3, 3):
            raise ValueError(
                f'lattice must have shape (3,3), got {lattice.shape}')
        if not np.all(np.isfinite(lattice)):
            raise ValueError('lattice must be finite')
        copied.unit = 'Bohr'
        copied.a = lattice.copy()
    if mesh is not None:
        mesh = np.asarray(mesh, dtype=int)
        if mesh.shape != (3,) or np.any(mesh <= 0):
            raise ValueError('mesh must contain three positive integers')
        copied.mesh = mesh.copy()
    return copied


__all__ = ['copy_with_coordinates']
