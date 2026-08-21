'''GPU4PySCF GDF provider with the DC-Forge PBC contract.'''

import threading
from contextlib import contextmanager, nullcontext

import cupy as cp
import numpy as np
from gpu4pyscf.pbc import df
from gpu4pyscf.pbc.df import rsdf_builder
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
from dcforge.gpu.pbc.df.compatibility import (
    derivative_int3c2e_scheme_metadata,
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
from dcforge.gpu.pbc.df.gdf.build_profile import profile_native_gamma_build
from dcforge.gpu.pbc.df.gdf.precision import (
    build_context as gdf_precision_build_context,
    install_fp32_view,
    normalize_profile as normalize_gdf_precision_profile,
)

_FIXED_OMEGA_LOCK = threading.RLock()


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
        'native_build_profile', 'native_build_profile_sample_interval',
        'native_build_substage_profile',
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
        'omega',
        'resolved_omega', 'resolved_source_mesh', 'resolved_metric_mesh',
        'gdf_precision_profile', 'gdf_precision_profile_effective',
        'gdf_precision_report', 'gdf_derivative_precision',
    } | PP_LOC_PART1_KEYS | CDERI_KEYS

    def __init__(self, cell, kpts=None, auxbasis=None):
        kpts = normalize_kpts(kpts, required=False)
        df.GDF.__init__(self, cell, kpts=kpts)
        # GPU4PySCF normally sets this instance flag in
        # ``RHF.density_fit``.  DC-Forge installs a shared provider directly
        # on the source and target objects, so the upstream decorator is not
        # called.  Resolve the same flag here; otherwise one zero k-point is
        # built correctly but every source J/K contraction is dispatched to
        # the slower sampled-k complex path.
        self.is_gamma_point = bool(
            len(kpts) == 1 and np.max(np.abs(kpts)) < 1.e-12)
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
        # Diagnostics-only exclusive substage profile.  The enabled path
        # synchronizes CUDA at nested builder boundaries and therefore is not
        # a performance default.
        self.native_build_profile = False
        self.native_build_profile_sample_interval = .02
        self.native_build_substage_profile = None
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
        # Optional G7 validation route.  The zero-order and LR derivative
        # traverse the same real-Gamma AO-pair Fourier values, but the global
        # metric solve lies between them.  ``cache``/``auto`` may retain only
        # the selected unique-pair values in bounded host memory; the full
        # dense AO-pair Fourier tensor is never persistent.  Recompute remains
        # the default because the exact cache slowed both Water20 and Water64
        # complete-gradient gates; this policy is independent of the promoted
        # MO-direct primitive-source representation.
        self._experimental_reciprocal_pair_cache = 'recompute'
        self._experimental_reciprocal_pair_cache_host = 'pageable'
        self._experimental_reciprocal_pair_cache_host_memory_fraction = .5
        self._experimental_reciprocal_pair_cache_host_minimum_reserve_bytes = (
            8 * 1024**3)
        self._experimental_reciprocal_pair_cache_host_max_bytes = None
        self._experimental_reciprocal_pair_cache_fallback = True
        self._experimental_reciprocal_pair_cache_plan = None
        # The direct real-Gamma reverse source is exact under the enforced
        # real-symmetric density contract.  A 128-row dense AO block reuses
        # the transpose-related exchange product.  By default, the first-pass
        # packed source is retained in mapped host memory so the expensive
        # source algebra is evaluated once.  These controls apply only within
        # the explicit/fallback packed route; production ``auto`` normally
        # selects MO-direct.  The live host-memory planner preserves a reserve
        # and falls back to the exact double-pass route when that cache does
        # not fit.  Sampled-k response does not enter this real-Gamma path.
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
        self._experimental_reciprocal_pair_cache_plan = None
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
        # These values describe the numerical SR/LR partition actually used
        # by GPU4PySCF's native central-factor builder.  They are not user
        # accuracy parameters and do not change the full Coulomb operator.
        # Stress finite differences nevertheless require the resolved split
        # and both integer meshes to remain identical at +strain and -strain.
        self.omega = None
        self.resolved_omega = None
        self.resolved_source_mesh = None
        self.resolved_metric_mesh = None
        # FP64 is the immutable default.  The two opt-in profiles are limited
        # to the positive-metric, single-device, real-Gamma route.  The
        # provider owns the CDERI construction/storage boundary, while the
        # RHFDFT and gradient drivers own the downstream arithmetic schedule.
        self.gdf_precision_profile = 'fp64'
        self.gdf_precision_profile_effective = 'fp64'
        self.gdf_precision_report = None
        self.gdf_derivative_precision = 'fp64'
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
        precision_profile = normalize_gdf_precision_profile(
            self.gdf_precision_profile)
        if precision_profile != 'fp64':
            if not self.is_gamma_point:
                raise NotImplementedError(
                    'GDF mixed-precision profiles require real Gamma')
            if int(self.cell.dimension) != 3:
                raise NotImplementedError(
                    'GDF mixed-precision profiles require a 3D Cell')
            if int(rsdf_builder.num_devices) != 1:
                raise NotImplementedError(
                    'GDF mixed-precision profiles are single-GPU only')
            if str(self.cderi_storage).lower() not in (
                    'mapped_host', 'auto'):
                raise NotImplementedError(
                    'GDF mixed-precision profiles require mapped-host CDERI')
            # The selected profile already stores the only persistent factor
            # in FP32, so no additional FP32 sidecar must enter the auto
            # storage estimate.
            self.cderi_auto_sidecar_factor = 0.
        preflight_cderi_memory(self)
        if (precision_profile != 'fp64'
                and self.cderi_storage_resolved != 'mapped_host'):
            raise MemoryError(
                'GDF mixed-precision profiles currently require mapped-host '
                'storage; increase available host memory or explicitly use '
                "gdf_precision_profile='fp64' with HDF5 storage")
        profile_context = (
            profile_native_gamma_build(self)
            if self.native_build_profile else nullcontext())
        with (bounded_native_build_memory(self),
              _fixed_partition_omega(self.omega), profile_context,
              gdf_precision_build_context(precision_profile) as ledger):
            df.GDF.build(self, j_only=j_only, kpts_band=kpts_band)
        if precision_profile != 'fp64':
            storage = install_fp32_view(self)
            ledger = dict(ledger)
            ledger.update({
                'storage_shape': [int(value) for value in storage.shape],
                'storage_bytes': int(storage.nbytes),
                'storage_dtype': np.dtype(storage.dtype).name,
            })
        else:
            ledger = dict(ledger)
        self.gdf_precision_profile_effective = precision_profile
        self.gdf_precision_report = {
            'requested_profile': precision_profile,
            'effective_profile': precision_profile,
            'build': ledger,
            'warning': (
                None if precision_profile == 'fp64' else
                'The active Hamiltonian contains a quantized FP32 CDERI. '
                'A later FP64 contraction cannot restore discarded bits.'),
        }
        self._record_resolved_response_settings()
        finalize_cderi_storage(self)
        return self

    def _record_resolved_response_settings(self):
        r'''Record the native GDF split and integer reciprocal meshes.

        GPU4PySCF 1.8.1 resolves an internal range-separation parameter for
        its short-range three-center integrals.  The source mesh follows the
        primary Cell cutoff for that parameter, while the metric mesh uses
        the auxiliary Cell and its stricter ``precision*1e-6`` threshold.
        Re-evaluating these public builder formulae after ``build`` records
        the representation which produced the active CDERI; it does not
        rebuild or replace that CDERI.

        The attributes are consumed by the common GDF response code and by
        strain-state fingerprints.  A relaxed-energy finite-difference run
        therefore fails closed if a strained Cell crosses an automatic mesh
        or omega boundary.
        '''
        cell = self.cell
        auxcell = self.auxcell
        if auxcell is None:
            raise RuntimeError(
                'native GDF build did not create an auxiliary Cell')
        physical_omega = abs(float(getattr(self, '_omega', 0.)))
        partition_omega = resolve_partition_omega(cell, self.omega)
        with_long_range = physical_omega < partition_omega
        if with_long_range:
            source_cutoff = rsdf_builder.estimate_ke_cutoff_for_omega(
                cell, partition_omega)
            source_mesh = cell.symmetrize_mesh(
                cell.cutoff_to_mesh(source_cutoff))
            metric_cutoff = rsdf_builder.estimate_ke_cutoff_for_omega(
                auxcell, partition_omega, auxcell.precision * 1.e-6)
            metric_mesh = auxcell.cutoff_to_mesh(metric_cutoff)
        else:
            source_mesh = metric_mesh = np.ones(3, dtype=int)
        self.resolved_omega = float(partition_omega)
        self.resolved_source_mesh = np.asarray(
            source_mesh, dtype=int).copy()
        self.resolved_metric_mesh = np.asarray(
            metric_mesh, dtype=int).copy()

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
        self.resolved_omega = None
        self.resolved_source_mesh = None
        self.resolved_metric_mesh = None
        self.native_build_memory_plan = None
        self.native_build_substage_profile = None
        self.gdf_precision_profile_effective = (
            'fp64' if self.gdf_precision_profile == 'fp64' else None)
        self.gdf_precision_report = None
        self.adjoint_host_cache_plan = None
        self.cderi_storage_resolved = (
            'mapped_host' if self.cderi_storage == 'auto'
            else str(self.cderi_storage).lower())
        self.cderi_auto_plan = None
        self._cderi_preflight_memory_metadata = None
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
            'derivative_int3c2e_scheme': (
                derivative_int3c2e_scheme_metadata()),
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
            'native_build_profile': self.native_build_profile,
            'native_build_profile_sample_interval': (
                self.native_build_profile_sample_interval),
            'native_build_substage_profile': (
                self.native_build_substage_profile),
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
            'omega': self.omega,
            'resolved_omega': self.resolved_omega,
            'resolved_source_mesh': (
                None if self.resolved_source_mesh is None
                else self.resolved_source_mesh.tolist()),
            'resolved_metric_mesh': (
                None if self.resolved_metric_mesh is None
                else self.resolved_metric_mesh.tolist()),
            'gdf_precision_profile': self.gdf_precision_profile,
            'gdf_precision_profile_effective': (
                self.gdf_precision_profile_effective),
            'gdf_precision_report': self.gdf_precision_report,
            'gdf_derivative_precision': self.gdf_derivative_precision,
        }


def resolve_partition_omega(cell, omega=None):
    r'''Return the GPU4PySCF GDF algorithmic SR/LR partition.

    ``omega=None`` delegates to GPU4PySCF's native system-dependent choice.
    A positive explicit value freezes only the auxiliary real/reciprocal
    decomposition used to build CDERI.  It does not set ``cell.omega`` and
    therefore does not turn the physical Coulomb operator into a
    range-separated Hamiltonian.
    '''
    physical_omega = abs(float(getattr(cell, 'omega', 0.)))
    if omega is None:
        partition = float(rsdf_builder._guess_omega(cell))
    else:
        partition = float(omega)
        if not np.isfinite(partition) or partition <= 0:
            raise ValueError('GDF partition omega must be positive and finite')
    return max(physical_omega, partition)


@contextmanager
def _fixed_partition_omega(omega):
    r'''Temporarily freeze GPU4PySCF's internal GDF partition resolver.

    GPU4PySCF 1.8.1 exposes no argument for its nonphysical ``rsdf_omega``;
    its builder calls the module-level ``_guess_omega`` internally.  The
    process lock makes this opt-in adapter transactional and restores the
    original resolver even if CDERI construction fails.  Normal providers
    with ``omega=None`` do not touch module state.
    '''
    if omega is None:
        yield
        return
    omega = float(omega)
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('GDF partition omega must be positive and finite')
    with _FIXED_OMEGA_LOCK:
        original = rsdf_builder._guess_omega

        def fixed(_cell, kmesh=None):
            del kmesh
            return omega

        rsdf_builder._guess_omega = fixed
        try:
            yield
        finally:
            rsdf_builder._guess_omega = original
