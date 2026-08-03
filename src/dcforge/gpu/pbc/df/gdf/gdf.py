'''GPU4PySCF GDF provider with the DC-Forge PBC contract.'''

import numpy as np
from gpu4pyscf.pbc import df

from dcforge.gpu.pbc.df.gaussian.pseudo import (
    PP_LOC_PART1_KEYS,
    get_pp,
    get_pp_loc_part1_plan,
    init_pp_loc_part1,
    reset_pp_loc_part1,
)
from dcforge.gpu.pbc.df.gaussian.reciprocal import (
    DEFAULT_BLOCK_ALIGNMENT,
    DEFAULT_MAX_BLOCK_SIZE,
    DEFAULT_MEMORY_FRACTION,
    DEFAULT_MINIMUM_RESERVE_BYTES,
)


class GDF(df.GDF):
    '''Gamma-point GPU Gaussian density-fitting provider.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built three-dimensional periodic cell.

    Kwargs:
        auxbasis : str, dict, or None
            Auxiliary basis passed to GPU4PySCF.  ``None`` retains its normal
            periodic matching/even-tempered construction policy.

    Notes:
        This Stage-1 provider uses GPU4PySCF's native GDF build and device
        J/K contractions.  The later Gaussian-response stages extend this
        class rather than substituting a host-side CPU provider.
    '''

    backend_name = 'gdf'
    device = 'gpu'
    _keys = df.GDF._keys | {
        'reciprocal_memory_fraction', 'reciprocal_minimum_reserve_bytes',
        'reciprocal_max_block_size', 'reciprocal_block_alignment',
        'reciprocal_gamma_conjugate_symmetry',
        'native_multi_observable_library',
    } | PP_LOC_PART1_KEYS

    def __init__(self, cell, auxbasis=None):
        df.GDF.__init__(self, cell, kpts=np.zeros((1, 3)))
        # These controls affect only the native analytic-response reciprocal
        # workspaces.  They never alter the GDF integrals, source RHF, or
        # fixed-density target energy.  Retain the resolved plan on the
        # gradient driver rather than assuming a device has fixed free memory.
        self.reciprocal_memory_fraction = DEFAULT_MEMORY_FRACTION
        self.reciprocal_minimum_reserve_bytes = (
            DEFAULT_MINIMUM_RESERVE_BYTES)
        self.reciprocal_max_block_size = DEFAULT_MAX_BLOCK_SIZE
        self.reciprocal_block_alignment = DEFAULT_BLOCK_ALIGNMENT
        # Real Gamma reciprocal integrands occur in exact G,-G conjugate
        # pairs.  Production evaluates one representative and applies its
        # multiplicity.  Setting this False retains the full-grid numerical
        # oracle; it changes work only, never the GDF mesh or cutoff.
        self.reciprocal_gamma_conjugate_symmetry = True
        # The installed GPU4PySCF 1.8.0 short-range derivative ABI has one
        # packed source/output channel.  A separately built, audited CUDA
        # extension may opt into a two-channel target-J/Z-Fock ledger through
        # this absolute path.  ``None`` deliberately preserves the stable
        # one-traversal fused production route.
        self.native_multi_observable_library = None
        init_pp_loc_part1(self)
        if auxbasis is not None:
            self.auxbasis = auxbasis

    get_pp_loc_part1_plan = get_pp_loc_part1_plan
    get_pp = get_pp

    def reset(self, cell=None):
        '''Reset GDF intermediates and invalidate the local-part-1 plan.'''
        df.GDF.reset(self, cell)
        # GPU4PySCF clears device CDERI but, unlike PySCF GDF, retains the old
        # auxiliary Cell.  A scanner geometry must rebuild that geometry-
        # dependent object even when its basis specification is unchanged.
        self.auxcell = None
        reset_pp_loc_part1(self)
        return self

    def capability(self):
        '''Return the validated GPU GDF capability boundary.'''
        return {
            'backend': self.backend_name,
            'device': self.device,
            'sampling': 'gamma',
            'reference': 'RHF',
            'target': ['PBE', 'R2SCAN'],
            'energy': True,
            'analytic_gradient': True,
            'analytic_gradient_process_scope': 'same_process_reentrant',
            'native_multi_observable': (
                self.native_multi_observable_library is not None),
            'stress': False,
            'auxbasis': self.auxbasis,
            'reciprocal_memory_fraction': self.reciprocal_memory_fraction,
            'reciprocal_max_block_size': self.reciprocal_max_block_size,
            'reciprocal_gamma_conjugate_symmetry': (
                self.reciprocal_gamma_conjugate_symmetry),
            'pp_loc_part1_omega': self.pp_loc_part1_omega,
            'pp_loc_part1_omega_resolved': (
                self.pp_loc_part1_omega_resolved),
            'pp_loc_part1_ke_cutoff': self.pp_loc_part1_ke_cutoff,
            'pp_loc_part1_mesh': (
                None if self.pp_loc_part1_mesh is None
                else np.asarray(self.pp_loc_part1_mesh, dtype=int).tolist()),
            'pp_loc_part1_plan_mode': self.pp_loc_part1_plan_mode,
        }
