'''PySCF-compatible fake nuclear Cells for Gaussian-DF derivatives.'''

import numpy as np
from pyscf.pbc.df import aft


def make_fake_nuclear_cell(cell, with_pseudo=True):
    r'''Return PySCF's fake nuclear Cell with coherent basis metadata.

    :func:`pyscf.pbc.df.aft._fake_nuc` deliberately makes a shallow copy of
    ``cell`` and replaces only its libcint ``_bas`` and ``_env`` arrays.  The
    copied Python ``_basis`` dictionary therefore still describes the orbital
    basis.  This is harmless in PySCF's native nuclear-attraction builder,
    which reads the libcint arrays directly, but it is inconsistent with the
    private RSDF screening helper ``_get_refuniq_map``: that helper assumes
    that ``_basis`` and ``_bas`` describe the same shell sequence.

    The fake nuclear representation has exactly one uncontracted s Gaussian
    per atom, independent of whether the orbital basis is SZV, DZVP, TZVP, or
    a larger basis.  This function preserves PySCF's integral arrays exactly
    and rebuilds only the descriptive Python metadata from those arrays.

    Args:
        cell : pyscf.pbc.gto.Cell
            Periodic Cell from which the GTH nuclear Gaussians are built.

    Kwargs:
        with_pseudo : bool
            Use the GTH local radius for pseudized atoms, exactly as in
            :func:`pyscf.pbc.df.aft._fake_nuc`.

    Returns:
        pyscf.pbc.gto.Cell
            Shallow fake nuclear Cell with unchanged ``_atm``, ``_bas``, and
            ``_env`` integral data and shell-consistent ``_basis`` metadata.
    '''
    fake_nuclear = aft._fake_nuc(cell, with_pseudo=with_pseudo)
    shell_slices = np.asarray(
        fake_nuclear.aoslice_nr_by_atom(), dtype=int)[:, :2]
    if np.any(shell_slices[:, 1] - shell_slices[:, 0] != 1):
        raise RuntimeError(
            'PySCF fake nuclear Cell must contain one shell per atom')

    basis = {}
    signatures = {}
    for atom_id, (shell0, _shell1) in enumerate(shell_slices):
        symbol = fake_nuclear.atom_symbol(atom_id)
        shell = _shell_metadata(fake_nuclear, int(shell0))
        signature = _shell_signature(fake_nuclear, int(shell0))
        if symbol in signatures:
            if signature != signatures[symbol]:
                raise RuntimeError(
                    'fake nuclear shells with the same atom symbol must '
                    'have identical exponents and coefficients')
        else:
            basis[symbol] = [shell]
            signatures[symbol] = signature

    fake_nuclear._basis = basis
    fake_nuclear._dcforge_fake_nuclear = True
    return fake_nuclear


def fake_nuclear_refuniq(fake_nuclear):
    r'''Return exact shell-uniqueness data for a fake nuclear Cell.

    The representative Cell returned here retains the original libcint
    coefficients.  This matters because PySCF's fake nuclear Gaussians are
    charge-normalized rather than ordinary normalized orbital functions.
    Rebuilding a temporary Mole from ``_basis`` would renormalize a
    one-primitive shell and could underestimate the short-range screening
    radius.

    Args:
        fake_nuclear : pyscf.pbc.gto.Cell
            Cell returned by :func:`make_fake_nuclear_cell`.

    Returns:
        tuple
            ``(shell_map, unique_atoms, unique_cell, unique_locations)``.
            ``shell_map`` maps every fake shell to its element representative;
            ``unique_cell`` contains one exact libcint shell per unique atom
            symbol; and ``unique_locations`` is the corresponding shell
            offset array.
    '''
    if not getattr(fake_nuclear, '_dcforge_fake_nuclear', False):
        raise TypeError(
            'fake_nuclear_refuniq requires make_fake_nuclear_cell output')
    shell_slices = np.asarray(
        fake_nuclear.aoslice_nr_by_atom(), dtype=int)[:, :2]
    if np.any(shell_slices[:, 1] - shell_slices[:, 0] != 1):
        raise RuntimeError(
            'fake nuclear Cell must contain one shell per atom')

    unique_atoms = []
    unique_shells = []
    unique_indices = {}
    shell_map = np.empty(fake_nuclear.nbas, dtype=np.int32)
    for atom_id, (shell0, _shell1) in enumerate(shell_slices):
        symbol = fake_nuclear.atom_symbol(atom_id)
        if symbol not in unique_indices:
            unique_indices[symbol] = len(unique_atoms)
            unique_atoms.append(symbol)
            unique_shells.append(int(shell0))
        shell_map[int(shell0)] = unique_indices[symbol]

    unique_cell = fake_nuclear.copy(deep=False)
    unique_cell._bas = np.asarray(
        fake_nuclear._bas[unique_shells], dtype=np.int32).copy()
    unique_cell._basis = {
        symbol: fake_nuclear._basis[symbol]
        for symbol in unique_atoms
    }
    unique_cell._dcforge_fake_nuclear = True
    unique_locations = np.arange(len(unique_atoms) + 1, dtype=np.int32)
    return shell_map, unique_atoms, unique_cell, unique_locations


def _shell_metadata(cell, shell_id):
    '''Reconstruct one descriptive PySCF shell from libcint arrays.'''
    angular = int(cell.bas_angular(shell_id))
    kappa = int(cell.bas_kappa(shell_id))
    exponents = np.asarray(cell.bas_exp(shell_id), dtype=float)
    coefficients = np.asarray(cell.bas_ctr_coeff(shell_id), dtype=float)
    primitives = [
        [float(exponents[index]), *coefficients[index].tolist()]
        for index in range(len(exponents))
    ]
    if kappa:
        return [angular, kappa, *primitives]
    return [angular, *primitives]


def _shell_signature(cell, shell_id):
    '''Return an exact signature of one shell's integral representation.'''
    return (
        int(cell.bas_angular(shell_id)),
        int(cell.bas_kappa(shell_id)),
        int(cell.bas_nprim(shell_id)),
        int(cell.bas_nctr(shell_id)),
        np.asarray(cell.bas_exp(shell_id), dtype=float).tobytes(),
        np.asarray(
            cell._libcint_ctr_coeff(shell_id), dtype=float).tobytes(),
    )


__all__ = [
    'fake_nuclear_refuniq',
    'make_fake_nuclear_cell',
]
