'''GPU4PySCF GDF provider with the DC-Forge PBC contract.'''

import cupy as cp
import numpy as np
from gpu4pyscf.pbc import df
from pyscf.pbc import df as cpu_df

from dcforge.cpu.pbc.kpts import normalize_kpts
from dcforge.gpu.pbc.df.cderi import (
    CDERI_KEYS,
    cderi_memory_report,
    cleanup_cderi_storage,
    finalize_cderi_storage,
    init_cderi_options,
    preflight_cderi_memory,
    release_cderi,
)
from dcforge.gpu.pbc.df.gaussian.pseudo import (
    PP_LOC_PART1_KEYS,
    get_pp_loc_part1_plan,
    init_pp_loc_part1,
    reset_pp_loc_part1,
)
from dcforge.gpu.pbc.df.gaussian.pseudo import (
    get_pp as get_pp_gamma,
)
from dcforge.gpu.pbc.df.gaussian.reciprocal import (
    DEFAULT_BLOCK_ALIGNMENT,
    DEFAULT_MAX_BLOCK_SIZE,
    DEFAULT_MEMORY_FRACTION,
    DEFAULT_MINIMUM_RESERVE_BYTES,
)
from dcforge.gpu.pbc.df.gdf.adjoint import (
    DEFAULT_ADJOINT_AUXILIARY_BLOCK_SIZE,
    DEFAULT_ADJOINT_HOST_CACHE,
    DEFAULT_ADJOINT_HOST_CACHE_FALLBACK,
    DEFAULT_ADJOINT_HOST_MAX_BYTES,
    DEFAULT_ADJOINT_HOST_MEMORY_FRACTION,
    DEFAULT_ADJOINT_HOST_MINIMUM_RESERVE_BYTES,
    DEFAULT_ADJOINT_STORAGE,
)
from dcforge.gpu.pbc.df.gdf.build_memory import (
    DEFAULT_NATIVE_BUILD_MAX_WORKING_BYTES,
    DEFAULT_NATIVE_BUILD_MEMORY_FRACTION,
    DEFAULT_NATIVE_BUILD_MINIMUM_RESERVE_BYTES,
    bounded_native_build_memory,
)


class GDF(df.GDF):
    '''GPU Gaussian density-fitting provider for Gamma or explicit full BZ.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built three-dimensional periodic cell.

    Kwargs:
        kpts : array_like or None
            Explicit Cartesian full-BZ k-points.  ``None`` retains the
            existing one-Gamma provider.
        auxbasis : str, dict, or None
            Auxiliary basis passed to GPU4PySCF.  ``None`` retains its normal
            periodic matching/even-tempered construction policy.

    Notes:
        Central factors and J/K contractions use GPU4PySCF.  The sampled-k
        nuclear response uses its native density-contracted derivative; the
        sampled-k zero-order GTH Hcore remains an explicitly recorded PySCF
        compatibility primitive.
    '''

    backend_name = 'gdf'
    device = 'gpu'
    _keys = df.GDF._keys | {
        'native_build_memory_fraction',
        'native_build_minimum_reserve_bytes',
        'native_build_max_working_bytes', 'native_build_memory_plan',
        'reciprocal_memory_fraction', 'reciprocal_minimum_reserve_bytes',
        'reciprocal_max_block_size', 'reciprocal_block_alignment',
        'reciprocal_gamma_conjugate_symmetry',
        'adjoint_auxiliary_block_size', 'adjoint_storage',
        'adjoint_host_cache', 'adjoint_host_memory_fraction',
        'adjoint_host_minimum_reserve_bytes', 'adjoint_host_max_bytes',
        'adjoint_host_cache_fallback', 'adjoint_host_cache_plan',
        'native_multi_observable_library',
        'sampled_k_hcore_host_calls', 'sampled_k_hcore_host_bytes',
        'sampled_k_hcore_owner',
        'sampled_k_response_layout_copy_count',
        'sampled_k_response_layout_copy_bytes',
        'kpts_reference', 'kpts_roundtrip_max_abs',
    } | PP_LOC_PART1_KEYS | CDERI_KEYS

    def __init__(self, cell, kpts=None, auxbasis=None):
        kpts = normalize_kpts(kpts, required=False)
        df.GDF.__init__(self, cell, kpts=kpts)
        self.kpts_reference = kpts.copy()
        self.kpts_roundtrip_max_abs = float(np.max(np.abs(
            np.asarray(self.kpts) - self.kpts_reference)))
        # GPU4PySCF's native central-factor builder derives one AO-pair batch
        # from globally available VRAM.  On a large-memory device that can
        # create a single multi-GiB native workspace even though smaller
        # batches give the same CDERI.  Resolve the planning budget from the
        # actual free VRAM, retain room for later SCF/XC arrays, and cap only
        # this one builder workspace.  These are performance/safety controls;
        # they do not change the GDF Hamiltonian or auxiliary basis.
        self.native_build_memory_fraction = (
            DEFAULT_NATIVE_BUILD_MEMORY_FRACTION)
        self.native_build_minimum_reserve_bytes = (
            DEFAULT_NATIVE_BUILD_MINIMUM_RESERVE_BYTES)
        self.native_build_max_working_bytes = (
            DEFAULT_NATIVE_BUILD_MAX_WORKING_BYTES)
        self.native_build_memory_plan = None
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
        # The direct real-Gamma reverse source is exact under the enforced
        # real-symmetric density contract.  A 128-row dense AO block reuses
        # the transpose-related exchange product.  By default, the first-pass
        # packed source is retained in mapped host memory so the expensive
        # source algebra is evaluated once.  The live host-memory planner
        # preserves a reserve and falls back to the exact double-pass route
        # when that cache does not fit.  Sampled-k response does not enter
        # this real-Gamma-only path.
        self.adjoint_auxiliary_block_size = (
            DEFAULT_ADJOINT_AUXILIARY_BLOCK_SIZE)
        self.adjoint_storage = DEFAULT_ADJOINT_STORAGE
        self.adjoint_host_cache = DEFAULT_ADJOINT_HOST_CACHE
        self.adjoint_host_memory_fraction = (
            DEFAULT_ADJOINT_HOST_MEMORY_FRACTION)
        self.adjoint_host_minimum_reserve_bytes = (
            DEFAULT_ADJOINT_HOST_MINIMUM_RESERVE_BYTES)
        self.adjoint_host_max_bytes = DEFAULT_ADJOINT_HOST_MAX_BYTES
        self.adjoint_host_cache_fallback = (
            DEFAULT_ADJOINT_HOST_CACHE_FALLBACK)
        self.adjoint_host_cache_plan = None
        # The installed GPU4PySCF 1.8.0 short-range derivative ABI has one
        # packed source/output channel.  A separately built, audited CUDA
        # extension may opt into a two-channel target-J/Z-Fock ledger through
        # this absolute path.  ``None`` deliberately preserves the stable
        # one-traversal fused production route.
        self.native_multi_observable_library = None
        self.sampled_k_hcore_host_calls = 0
        self.sampled_k_hcore_host_bytes = 0
        self.sampled_k_hcore_owner = None
        self.sampled_k_response_layout_copy_count = 0
        self.sampled_k_response_layout_copy_bytes = 0
        init_cderi_options(self)
        init_pp_loc_part1(self)
        if auxbasis is not None:
            self.auxbasis = auxbasis

    get_pp_loc_part1_plan = get_pp_loc_part1_plan

    def build(self, j_only=None, kpts_band=None):
        r'''Build native GDF factors with an adaptive device-memory budget.

        Args:
            j_only : bool or None
                Passed unchanged to GPU4PySCF ``GDF.build``.  ``None`` keeps
                the provider's current ``_j_only`` setting.
            kpts_band : array_like or None
                Passed unchanged to GPU4PySCF.  Its current native GDF
                implementation supports only ``None``.

        Returns:
            GDF
                This provider with the native central factors attached.

        Notes:
            The adapter changes only the free-memory value used by
            GPU4PySCF's AO-pair batch planner.  Integral definitions,
            factorization algebra, J/K contractions, and final CDERI layout
            remain the inherited implementation.  The resolved query history
            is available in ``native_build_memory_plan``.
        '''
        preflight_cderi_memory(self)
        with bounded_native_build_memory(self):
            df.GDF.build(self, j_only=j_only, kpts_band=kpts_band)
        finalize_cderi_storage(self)
        return self

    def get_pp(self, kpts=None):
        '''Return provider-consistent GTH matrices for Gamma or sampled k.

        The optimized DC-Forge SR/LR implementation is currently real-Gamma
        only.  For sampled k-points, PySCF's Gaussian ``GDF.get_pp`` is the
        authoritative zero-order primitive used by the matched CPU route.
        The complete small ``(nkpts,nao,nao)`` matrix is copied to the device
        and the boundary is recorded explicitly on this provider.
        '''
        requested = self.kpts if kpts is None else np.asarray(kpts)
        sampled_k = (
            np.asarray(requested).reshape(-1, 3).shape[0] > 1
            or np.any(np.asarray(requested) != 0))
        if not sampled_k:
            return get_pp_gamma(self, kpts)

        witness = cpu_df.GDF(self.cell, kpts=self.kpts)
        witness.auxbasis = self.auxbasis
        witness.linear_dep_threshold = self.linear_dep_threshold
        witness.exp_to_discard = self.exp_to_discard
        host = np.asarray(witness.get_pp(requested))
        self.sampled_k_hcore_host_calls += 1
        self.sampled_k_hcore_host_bytes += int(host.nbytes)
        self.sampled_k_hcore_owner = 'PySCF GDF.get_pp sampled-k'
        return cp.asarray(host)

    def reset(self, cell=None):
        '''Reset GDF intermediates and invalidate the local-part-1 plan.'''
        cleanup_cderi_storage(self)
        df.GDF.reset(self, cell)
        # GPU4PySCF clears device CDERI but, unlike PySCF GDF, retains the old
        # auxiliary Cell.  A scanner geometry must rebuild that geometry-
        # dependent object even when its basis specification is unchanged.
        self.auxcell = None
        self.sampled_k_hcore_host_calls = 0
        self.sampled_k_hcore_host_bytes = 0
        self.sampled_k_hcore_owner = None
        self.sampled_k_response_layout_copy_count = 0
        self.sampled_k_response_layout_copy_bytes = 0
        self.native_build_memory_plan = None
        self.adjoint_host_cache_plan = None
        reset_pp_loc_part1(self)
        return self

    def cderi_memory_report(self, response_observables=2):
        '''Return exact central-factor and dense-response memory telemetry.'''
        return cderi_memory_report(self, response_observables)

    def release_cderi(self):
        '''Release central factors after all source-HF response actions.'''
        return release_cderi(self)

    def capability(self):
        '''Return the validated GPU GDF capability boundary.'''
        sampled_k = len(self.kpts) > 1 or np.any(np.asarray(self.kpts) != 0)
        return {
            'backend': self.backend_name,
            'device': self.device,
            'sampling': 'full_bz' if sampled_k else 'gamma',
            'reference': 'KRHF' if sampled_k else 'RHF',
            'target': ['PBE', 'R2SCAN'],
            'energy': True,
            'analytic_gradient': True,
            'analytic_gradient_process_scope': 'same_process_reentrant',
            'native_multi_observable': (
                self.native_multi_observable_library is not None),
            'stress': False,
            'sampled_k_hcore_host_boundary': sampled_k,
            'kpts_roundtrip_max_abs': self.kpts_roundtrip_max_abs,
            'auxbasis': self.auxbasis,
            'native_build_memory_fraction': (
                self.native_build_memory_fraction),
            'native_build_minimum_reserve_bytes': (
                self.native_build_minimum_reserve_bytes),
            'native_build_max_working_bytes': (
                self.native_build_max_working_bytes),
            'native_build_memory_plan': self.native_build_memory_plan,
            'cderi_storage': self.cderi_storage,
            'cderi_memory_metadata': self.cderi_memory_metadata,
            'release_cderi_after_response': self.release_cderi_after_response,
            'reciprocal_memory_fraction': self.reciprocal_memory_fraction,
            'reciprocal_max_block_size': self.reciprocal_max_block_size,
            'reciprocal_gamma_conjugate_symmetry': (
                self.reciprocal_gamma_conjugate_symmetry),
            'adjoint_auxiliary_block_size': (
                self.adjoint_auxiliary_block_size),
            'adjoint_storage': self.adjoint_storage,
            'adjoint_host_cache': self.adjoint_host_cache,
            'adjoint_host_memory_fraction': (
                self.adjoint_host_memory_fraction),
            'adjoint_host_minimum_reserve_bytes': (
                self.adjoint_host_minimum_reserve_bytes),
            'adjoint_host_max_bytes': self.adjoint_host_max_bytes,
            'adjoint_host_cache_fallback': (
                self.adjoint_host_cache_fallback),
            'adjoint_host_cache_plan': self.adjoint_host_cache_plan,
            'pp_loc_part1_omega': self.pp_loc_part1_omega,
            'pp_loc_part1_omega_resolved': (
                self.pp_loc_part1_omega_resolved),
            'pp_loc_part1_ke_cutoff': self.pp_loc_part1_ke_cutoff,
            'pp_loc_part1_mesh': (
                None if self.pp_loc_part1_mesh is None
                else np.asarray(self.pp_loc_part1_mesh, dtype=int).tolist()),
            'pp_loc_part1_plan_mode': self.pp_loc_part1_plan_mode,
        }
