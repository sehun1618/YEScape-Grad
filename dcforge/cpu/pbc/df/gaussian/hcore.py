'''One-electron derivatives shared by PySCF GDF and RSGDF providers.'''

import numpy as np
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import gamma_point

from dcforge.cpu.pbc.df.gaussian.local_part1 import LocalPart1Derivative
from dcforge.cpu.pbc.df.gaussian.local_part2 import (
    LocalPart2Derivative,
    contract_vpploc_part2_deriv,
)
from dcforge.cpu.pbc.df.gaussian.pseudo import (
    contract_vppnl_deriv,
    get_vppnl_deriv,
)


def get_hcore_deriv(cell, kpts):
    r'''Return kinetic plus Gaussian-DF-consistent GTH derivatives.

    The ordinary periodic gradient helper differentiates an FFT
    representation of the local potential.  PySCF GDF and RSGDF evaluate
    ``get_pp`` with ``_RSNucBuilder`` and ``_IntPPBuilder`` instead.  This
    function uses those same analytic-integral representations so the
    derivative belongs to the Hamiltonian used by the source RHF and target
    fixed-density energy.

    Returns:
        ndarray of shape (natm, 3, nkpts, nao, nao)
            Full one-electron matrix derivative in Ha/Bohr.
    '''
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    nao = cell.nao_nr()
    kinetic_rows = np.asarray(cell.pbc_intor('int1e_ipkin', kpts=kpts))
    local1 = LocalPart1Derivative(cell, kpts)
    local2 = LocalPart2Derivative(cell, kpts)
    nonlocal1 = get_vppnl_deriv(cell, kpts).transpose(0, 2, 1, 3, 4)
    h1ao = np.empty(
        (cell.natm, 3, len(kpts), nao, nao), dtype=np.complex128)
    aoslices = cell.aoslice_by_atom()
    for ia in range(cell.natm):
        p0, p1 = aoslices[ia, 2:]
        for xyz in range(3):
            kinetic = np.zeros((len(kpts), nao, nao), dtype=kinetic_rows.dtype)
            kinetic[:, p0:p1] -= kinetic_rows[:, xyz, p0:p1]
            kinetic[:, :, p0:p1] -= kinetic_rows[
                :, xyz, p0:p1].transpose(0, 2, 1).conj()
            h1ao[ia, xyz] = kinetic
            h1ao[ia, xyz] += local1.matrix(ia, xyz)
            h1ao[ia, xyz] += local2.matrix(ia, xyz)
        h1ao[ia] += nonlocal1[ia]
    return _real_if_close(h1ao)


def contract_hcore_deriv(cell, kpts, densities, atmlst=None):
    r'''Contract Gaussian-DF one-electron derivatives with density sets.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built periodic cell.
        kpts : ndarray of shape (nkpts, 3)
            Explicit Gamma or full-BZ sampled k-points.
        densities : ndarray of shape (nset,nkpts,nao,nao)
            Hermitian AO density sources.

    Kwargs:
        atmlst : sequence of int or None
            Requested nuclear centers.

    Returns:
        tuple
            Total contractions and a component dictionary.  Every array has
            shape ``(nset,len(atmlst),3)`` and is an energy derivative in
            Ha/Bohr.
    '''
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    nao = cell.nao_nr()
    batch = np.asarray(densities)
    if batch.ndim == 2:
        batch = batch[None, None]
    elif batch.ndim == 3:
        batch = batch[:, None] if len(kpts) == 1 else batch[None]
    if batch.shape[1:] != (len(kpts), nao, nao):
        raise ValueError(
            'hcore density batch must have shape '
            f'(nset,{len(kpts)},{nao},{nao}), got {batch.shape}')
    if atmlst is None:
        atmlst = range(cell.natm)
    atmlst = [int(atom_id) for atom_id in atmlst]

    kinetic_rows = np.asarray(cell.pbc_intor('int1e_ipkin', kpts=kpts))
    kinetic = np.zeros((len(batch), len(atmlst), 3), dtype=float)
    aoslices = cell.aoslice_by_atom()
    for position, atom_id in enumerate(atmlst):
        p0, p1 = aoslices[atom_id, 2:]
        kinetic[:, position] = -2 * lib.einsum(
            'kxpj,skjp->sx', kinetic_rows[:, :, p0:p1],
            batch[:, :, :, p0:p1]).real

    local1 = LocalPart1Derivative(cell, kpts).contract_batch(batch, atmlst)
    # PySCF contracts local part 2 directly in its periodic integral driver.
    # The full LocalPart2Derivative route remains in get_hcore_deriv as a
    # matrix reference, while production avoids its atom-by-AO-square tensors.
    if len(kpts) == 1 and gamma_point(kpts):
        local2 = contract_vpploc_part2_deriv(
            cell, batch, kpts=kpts, atmlst=atmlst)
    else:
        # PySCF's density-contracted C endpoint is Gamma-only.  The explicit
        # sampled-k integral driver caches its three-center primitives and
        # contracts every requested density immediately, retaining scalar
        # ``(nset,natm,3)`` output rather than the full Hcore derivative.
        local2 = LocalPart2Derivative(
            cell, kpts).contract_batch(batch, atmlst)
    nonlocal1 = contract_vppnl_deriv(
        cell, batch, kpts=kpts, atmlst=atmlst)
    components = {
        'kinetic': kinetic,
        'local_part1': local1,
        'local_part2': local2,
        'nonlocal': nonlocal1,
    }
    total = kinetic + local1 + local2 + nonlocal1
    return total, components


def _real_if_close(value, tol=1e-10):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tol:
        return value.real
    return value
