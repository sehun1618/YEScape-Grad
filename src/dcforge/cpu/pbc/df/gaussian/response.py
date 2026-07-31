'''Gaussian-DF contracted/reference blocking shared by GDF and RSDF.'''

import numpy as np
from pyscf import lib


def response_blocks(cell, atmlst, blksize, nao, naux, dtype,
                    max_memory, storage='atom_stream'):
    r'''Return validated atom indices and the resolved response block size.

    Args:
        cell : pyscf.pbc.gto.Cell
            Primary periodic cell.
        atmlst : sequence of int or None
            Requested nuclear centers.  ``None`` means every atom.
        blksize : int or None
            Explicit number of atoms per block.  ``None`` selects an
            AO/auxiliary-dimension-aware block between one and eight atoms.
        nao, naux : int
            Primary and auxiliary AO dimensions.
        dtype : numpy dtype
            Response dtype used to estimate explicit tensor storage.
        max_memory : float
            PySCF-style memory budget in MB.
        storage : str
            ``'contracted'`` counts source and metric derivative blocks;
            ``'atom_stream'`` additionally counts CDERI and J/K derivatives.

    Returns:
        tuple
            Validated atom list and positive resolved block size.

    Notes:
        An ``atom_stream`` block simultaneously holds source, metric, two
        CDERI, and four J/K response tensors.  Automatic selection also
        reserves auxiliary-metric work arrays for the batched Cholesky
        derivative.  Its conservative byte model is

        ``3 itemsize (3 naux nao**2 + 5 naux**2 + 4 nao**2)``.

        The contracted model is
        ``3 itemsize (naux nao**2 + naux**2)`` because ``C1/J1/K1`` are absent.
        Automatic blocking gives explicit tensors at most one eighth of the
        currently unused ``max_memory`` budget and caps the block at eight
        atoms.  As in PySCF, this is a best-effort budget rather than a hard
        process-RSS limit.
    '''
    if atmlst is None:
        atmlst = range(cell.natm)
    atmlst = [int(ia) for ia in atmlst]
    if any(ia < 0 or ia >= cell.natm for ia in atmlst):
        raise ValueError('atom index is outside the periodic cell')

    if blksize is None:
        max_memory = float(max_memory)
        if max_memory <= 0:
            raise ValueError('Gaussian response max_memory must be positive')
        itemsize = np.dtype(dtype).itemsize
        if storage == 'contracted':
            # One atom block holds B1 and M1 only.  CDERI and J/K derivatives
            # have been replaced by geometry-independent adjoints.
            bytes_per_atom = 3 * itemsize * (
                naux * nao ** 2 + naux ** 2)
        elif storage == 'atom_stream':
            bytes_per_atom = 3 * itemsize * (
                3 * naux * nao ** 2 + 5 * naux ** 2 + 4 * nao ** 2)
        else:
            raise ValueError("storage must be 'contracted' or 'atom_stream'")
        remaining = max(0., max_memory - lib.current_memory()[0])
        budget = remaining * 1e6 / 8
        blksize = max(1, min(8, int(budget // bytes_per_atom)))
    else:
        blksize = int(blksize)
        if blksize < 1:
            raise ValueError('cderi response blksize must be positive')
    return atmlst, min(blksize, max(1, len(atmlst)))
