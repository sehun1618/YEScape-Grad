'''Provider-owned real-Gamma GTH pseudopotential construction.'''

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.gto.pseudo.pp_int import get_pp_nl_gpu
from pyscf.pbc.df import aft as cpu_aft

from dcforge.gpu.pbc.df.gaussian.local_part1 import (
    get_pp_loc_part1,
    resolve_pp_loc_part1,
)

PP_LOC_PART1_KEYS = {
    'pp_loc_part1_omega', 'pp_loc_part1_omega_resolved',
    'pp_loc_part1_ke_cutoff', 'pp_loc_part1_mesh',
    'pp_loc_part1_plan_mode',
}


def init_pp_loc_part1(with_df):
    '''Initialize one provider-owned local-part-1 numerical plan.

    Args:
        with_df : dcforge GPU PBC GDF or RSDF
            Provider that owns the active Cell and zero-order Hcore.

    Notes:
        ``None`` follows GPU4PySCF's system-dependent zero-order policy.  A
        positive explicit omega is a diagnostic work split, not an accuracy
        control or a physical range-separated approximation.
    '''
    with_df.pp_loc_part1_omega = None
    with_df.pp_loc_part1_omega_resolved = None
    with_df.pp_loc_part1_ke_cutoff = None
    with_df.pp_loc_part1_mesh = None
    with_df.pp_loc_part1_plan_mode = None
    with_df._pp_loc_part1_omega_requested = None
    with_df._pp_matrix_cache = None
    with_df._pp_matrix_cache_key = None


def get_pp_loc_part1_plan(with_df):
    '''Return and cache the provider-owned local-part-1 SR/LR plan.

    ``cell.precision`` is the public accuracy control.  Mutating or rebuilding
    the Cell requires the ordinary PySCF ``reset(cell)`` protocol so energy
    and gradient cannot silently retain a plan from another representation.
    '''
    requested = with_df.pp_loc_part1_omega
    requested = None if requested is None else float(requested)
    if (with_df.pp_loc_part1_omega_resolved is None
            or requested != with_df._pp_loc_part1_omega_requested):
        plan = resolve_pp_loc_part1(with_df.cell, requested)
        with_df.pp_loc_part1_omega_resolved = plan['omega']
        with_df.pp_loc_part1_ke_cutoff = plan['ke_cutoff']
        with_df.pp_loc_part1_mesh = plan['mesh'].copy()
        with_df.pp_loc_part1_plan_mode = plan['mode']
        with_df._pp_loc_part1_omega_requested = requested
        with_df._pp_matrix_cache = None
        with_df._pp_matrix_cache_key = None
    return {
        'omega': float(with_df.pp_loc_part1_omega_resolved),
        'ke_cutoff': float(with_df.pp_loc_part1_ke_cutoff),
        'mesh': np.asarray(with_df.pp_loc_part1_mesh, dtype=int).copy(),
        'mode': with_df.pp_loc_part1_plan_mode,
    }


def get_pp(with_df, kpts=None):
    r'''Return the GTH pseudopotential from one provider-owned SR/LR plan.

    Local part 1 uses the shared analytic short-range three-center plus
    long-range Fourier representation.  PySCF local part 2 and GPU4PySCF
    nonlocal projectors remain their authoritative numerical primitives.

    Args:
        with_df : dcforge GPU PBC GDF or RSDF
            Provider that owns the active Cell.

    Kwargs:
        kpts : ndarray or None
            One Gamma point or ``None``.  Sampled-k remains unsupported.

    Returns:
        cupy.ndarray
            GTH pseudopotential AO matrix with the ordinary single-k-point
            shape convention.
    '''
    plan = get_pp_loc_part1_plan(with_df)
    kpts_key = (
        None if kpts is None
        else np.asarray(kpts, dtype=float).shape
        + tuple(np.asarray(kpts, dtype=float).ravel()))
    cache_key = plan['omega'], tuple(plan['mesh']), kpts_key
    if with_df._pp_matrix_cache_key == cache_key:
        return with_df._pp_matrix_cache.copy()

    is_single_kpt = kpts is not None and np.asarray(kpts).ndim == 1
    pp2builder = cpu_aft._IntPPBuilder(with_df.cell, kpts)
    value = cp.asarray(pp2builder.get_pp_loc_part2())
    value += cp.asarray(get_pp_nl_gpu(with_df.cell, kpts))
    value += get_pp_loc_part1(
        with_df.cell, kpts, omega=plan['omega'], mesh=plan['mesh'])
    if is_single_kpt and value.ndim == 3:
        value = value[0]
    with_df._pp_matrix_cache = value.copy()
    with_df._pp_matrix_cache_key = cache_key
    return value


def reset_pp_loc_part1(with_df):
    '''Invalidate the provider-owned plan and pseudopotential matrix cache.'''
    with_df.pp_loc_part1_omega_resolved = None
    with_df.pp_loc_part1_ke_cutoff = None
    with_df.pp_loc_part1_mesh = None
    with_df.pp_loc_part1_plan_mode = None
    with_df._pp_loc_part1_omega_requested = None
    with_df._pp_matrix_cache = None
    with_df._pp_matrix_cache_key = None
