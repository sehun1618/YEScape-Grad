'''Density-contracted real-Gamma Gaussian-provider Hcore derivatives.'''

import time

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.gto import int1e

from dcforge.cpu.pbc.df.gaussian.local_part2 import (
    contract_vpploc_part2_deriv,
)
from dcforge.gpu.pbc.df.gaussian.local_part1 import LocalPart1Derivative
from dcforge.gpu.pbc.df.gaussian.nonlocal_pseudo import (
    contract_nonlocal_pseudo_batch,
)


def contract_hcore_pair(with_df, dm0, dz, atmlst, profile=None):
    r'''Contract provider-consistent ``h^lambda`` with ``P`` and ``D_Z``.

    The one-electron Hamiltonian is kept term-resolved as

    .. math::

       h^\lambda = T^\lambda + V_{loc,1}^\lambda
                    + V_{loc,2}^\lambda + V_{nl}^\lambda.

    Kinetic, local-part-1 and nonlocal terms are density-contracted on the
    GPU.  PySCF's Gamma-only contracted local-part-2 C integral remains the
    authoritative provider primitive and returns only ``(nset,natm,3)``
    scalars through a small explicit host boundary.  No all-atom H1 tensor or
    target-XC mesh is used.

    Args:
        with_df : dcforge GPU PBC GDF or RSDF
            Built source/target Gaussian provider owning one local-part-1
            plan shared by zero-order Hcore and this derivative.
        dm0, dz : cupy.ndarray of shape (nao,nao)
            Source RHF density and Hermitian Z transition density.
        atmlst : sequence of int
            Requested nuclear centers.

    Kwargs:
        profile : dict or None
            Validation-only synchronized timing destination.

    Returns:
        tuple
            ``(de_hcore,de_fock_hcore,components)``.  The first two arrays
            have shape ``(len(atmlst),3)`` in Ha/Bohr; the second already
            includes the DC-DFT coefficient ``-2``.  ``components`` retains
            the two unscaled density contractions for every Hcore subterm.
    '''
    cell = with_df.cell
    if cell.dimension != 3:
        raise NotImplementedError('analytic Gaussian Hcore requires 3D PBC')
    kpts = np.asarray(with_df.kpts, dtype=float).reshape(-1, 3)
    if len(kpts) != 1 or np.max(np.abs(kpts)) > 1e-12:
        raise NotImplementedError('analytic Gaussian Hcore requires Gamma point')
    atmlst = [int(ia) for ia in atmlst]
    if atmlst != list(range(cell.natm)):
        raise NotImplementedError(
            'analytic Gaussian Hcore currently requires all atoms')

    total_start = _profile_start(profile)
    dms = cp.stack((cp.asarray(dm0), cp.asarray(dz)))

    stage_start = _profile_start(profile)
    kinetic = _contract_kinetic(cell, dms, atmlst)
    _profile_stop(profile, 'kinetic_seconds', stage_start)

    stage_start = _profile_start(profile)
    local1_plan = with_df.get_pp_loc_part1_plan()
    local1_driver = LocalPart1Derivative(
        cell, omega=local1_plan['omega'], mesh=local1_plan['mesh'])
    local1, local1_components = local1_driver.contract_batch(dms)
    _profile_stop(profile, 'local_part1_seconds', stage_start)

    stage_start = _profile_start(profile)
    # PySCF's C driver owns this analytic Gamma contraction.  Only the two
    # AO density matrices cross to the host; derivative matrices do not.
    local2 = cp.asarray(contract_vpploc_part2_deriv(
        cell, cp.asnumpy(dms)[:, None], kpts=kpts, atmlst=atmlst))
    _profile_stop(profile, 'local_part2_seconds', stage_start)

    stage_start = _profile_start(profile)
    nonlocal1 = contract_nonlocal_pseudo_batch(cell, dms)[:, atmlst]
    _profile_stop(profile, 'nonlocal_seconds', stage_start)

    components = {
        'kinetic': kinetic,
        'local_part1': local1[:, atmlst],
        'local_part2': local2,
        'nonlocal': nonlocal1,
    }
    total = sum(components.values())
    _profile_stop(profile, 'total_seconds', total_start)
    if profile is not None:
        profile['local_part1_reciprocal'] = (
            local1_driver.last_reciprocal_execution)
        profile['local_part1_components'] = tuple(local1_components)
        profile['local_part1_plan'] = {
            'omega': local1_plan['omega'],
            'ke_cutoff': local1_plan['ke_cutoff'],
            'mesh': local1_plan['mesh'].tolist(),
            'mode': local1_plan['mode'],
        }
        profile['local_part2_host_boundary'] = True
    return total[0], -2 * total[1], components


def _contract_kinetic(cell, densities, atmlst):
    '''Contract compact kinetic AO rows with several Gamma densities.'''
    hkin = int1e.int1e_ipkin(cell, np.zeros((1, 3)))[0]
    aoslices = np.asarray(cell.aoslice_by_atom(), dtype=int)
    de = cp.empty((len(densities), len(atmlst), 3), dtype=cp.float64)
    for iatom, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        hrow = hkin[:, p0:p1]
        de[:, iatom] = -(
            cp.einsum('xpq,dqp->dx', hrow, densities[:, :, p0:p1])
            + cp.einsum(
                'xpq,dpq->dx', hrow.conj(), densities[:, p0:p1])).real
    return de


def _profile_start(profile):
    '''Synchronize and start one optional GDF Hcore substage timer.'''
    if profile is None:
        return None
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


def _profile_stop(profile, name, start):
    '''Synchronize and retain one optional GDF Hcore substage time.'''
    if start is None:
        return
    cp.cuda.Stream.null.synchronize()
    profile[name] = time.perf_counter() - start
