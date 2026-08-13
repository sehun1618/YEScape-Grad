'''GPU RSDF provider with explicit source and metric ownership.

GPU4PySCF exposes GPU range-separated three-center and factorization kernels
through :mod:`gpu4pyscf.pbc.df.rsdf_builder`, but it has no public ``RSGDF``
provider class.  PySCF ``RSGDF`` independently resolves the source and metric
range-separation parameters.  This adapter preserves that distinction:
GPU4PySCF builds the large three-center source and factorizes the metric,
while PySCF supplies only the small screened two-center metric primitive.
'''

from contextlib import contextmanager, nullcontext
from threading import RLock
from time import perf_counter

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import get_avail_mem
from gpu4pyscf.pbc import df as gpu_df
from gpu4pyscf.pbc.df import rsdf_builder
from pyscf.pbc.df import rsdf as cpu_rsdf
from pyscf.pbc.df import rsdf_helper
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from dcforge.cpu.pbc.kpts import normalize_kpts
from dcforge.gpu.pbc.df.cderi import (
    CDERI_KEYS,
    cderi_memory_report,
    cleanup_cderi_storage,
    finalize_cderi_storage,
    init_cderi_options,
    load_unique_gamma_cderi,
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
from dcforge.gpu.pbc.df.rsdf.build_profile import profile_native_gamma_build

_BUILDER_ADAPTER_LOCK = RLock()
_METRIC_MEMORY_FRACTION = .25
_METRIC_MINIMUM_RESERVE_BYTES = 512 * 1024**2
_METRIC_MAX_GRID_BLOCK_SIZE = 4096
_SOURCE_RECIPROCAL_MEMORY_FRACTION = .15
_SOURCE_RECIPROCAL_MINIMUM_RESERVE_BYTES = 512 * 1024**2
_SOURCE_RECIPROCAL_MAX_BLOCK_SIZE = 512
_SOURCE_RECIPROCAL_BLOCK_ALIGNMENT = 32


class RSDF(gpu_df.GDF):
    r'''Range-separated GPU provider for Gamma or explicit full BZ.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built three-dimensional GTH periodic cell.

    Kwargs:
        kpts : array_like or None
            Explicit Cartesian full-BZ k-points.  ``None`` selects Gamma.
        auxbasis : str, dict, or None
            Auxiliary basis passed to the PySCF RSDF parameter witness.

    Notes:
        ``omega``/``mesh_compact`` define the three-center source while
        ``omega_j2c``/``mesh_j2c`` define the two-center metric.  They are
        resolved by a temporary PySCF ``RSGDF`` witness before the GPU
        builder starts; neither is inferred from the other.  The host metric
        primitive is an explicit, bounded ``O(naux^2)`` compatibility
        boundary, not a host J/K or host three-center fallback.  For sampled
        k-point nuclear gradients, the production response uses the resolved
        provider-authoritative CPU source/metric derivative.  A one-split
        GPU4PySCF full-Coulomb response remains a diagnostic probe because it
        does not differentiate RSDF's independent finite metric settings.
    '''

    backend_name = 'rsdf'
    device = 'gpu'
    _keys = gpu_df.GDF._keys | {
        'omega', 'ke_cutoff', 'mesh_compact', 'omega_j2c', 'mesh_j2c',
        'precision_R', 'precision_G', 'precision_j2c', 'npw_max',
        '_omega_min', 'use_bvk', 'j2c_eig_always',
        'central_build_metadata', '_central_metric',
        '_central_metric_factor', '_parameter_witness_cache',
        '_parameter_witness_signature',
        'native_build_profile', 'native_build_profile_sample_interval',
        'native_build_substage_profile',
        'source_reciprocal_memory_fraction',
        'source_reciprocal_minimum_reserve_bytes',
        'source_reciprocal_max_block_size',
        'source_reciprocal_block_alignment',
        'source_reciprocal_gamma_conjugate_symmetry',
        'response_release_cached_blocks', 'response_profile_components',
        'response_mode', 'response_pair_block_size',
        'response_auxiliary_block_size', 'response_memory_fraction',
        'response_minimum_reserve_bytes', 'response_fused_max_bytes',
        'sampled_k_hcore_host_calls', 'sampled_k_hcore_host_bytes',
        'sampled_k_hcore_owner',
        'sampled_k_response_layout_copy_count',
        'sampled_k_response_layout_copy_bytes',
        'kpts_reference', 'kpts_roundtrip_max_abs',
    } | PP_LOC_PART1_KEYS | CDERI_KEYS

    def __init__(self, cell, kpts=None, auxbasis=None):
        kpts = normalize_kpts(kpts, required=False)
        gpu_df.GDF.__init__(self, cell, kpts=kpts)
        # Match GPU4PySCF ``RHF.density_fit`` dispatch even though DC-Forge
        # installs this shared provider without invoking that decorator.
        self.is_gamma_point = bool(
            len(kpts) == 1 and np.max(np.abs(kpts)) < 1.e-12)
        self.kpts_reference = kpts.copy()
        self.kpts_roundtrip_max_abs = float(np.max(np.abs(
            np.asarray(self.kpts) - self.kpts_reference)))
        witness = cpu_rsdf.RSGDF(cell, kpts=kpts)
        if auxbasis is not None:
            witness.auxbasis = auxbasis
        self.auxbasis = witness.auxbasis
        for name in (
                'use_bvk', 'precision_R', 'precision_G', 'npw_max',
                '_omega_min', 'omega', 'ke_cutoff', 'mesh_compact',
                'omega_j2c', 'mesh_j2c', 'precision_j2c',
                'j2c_eig_always'):
            setattr(self, name, getattr(witness, name))
        self.central_build_metadata = None
        # Diagnostics-only build profiler.  The enabled route introduces
        # synchronized CUDA boundaries and must therefore never be folded
        # into production timing.
        self.native_build_profile = False
        self.native_build_profile_sample_interval = .02
        self.native_build_substage_profile = None
        # ``_central_metric`` is the exact Gamma metric in the same AO
        # auxiliary basis as the compressed device CDERI.  The RSDF builder
        # already needs this small O(naux**2) object to construct ``C``;
        # retaining it avoids rebuilding a potentially different metric when
        # the response code pushes an observable through the Cholesky gauge.
        # It is deliberately not a dense three-center or derivative cache.
        self._central_metric = None
        self._central_metric_factor = None
        # The PySCF RSGDF witness owns resolved source/metric settings and the
        # authoritative auxiliary Cell.  It is immutable for one provider
        # representation and can therefore be reused until ``reset`` or an
        # explicit parameter change alters its signature.
        self._parameter_witness_cache = None
        self._parameter_witness_signature = None
        # Controls affect only the native source-response workspace.  They
        # neither change the RSDF source/metric integrals nor the central
        # energy factorization.  The final plan is attached to response
        # metadata so a Water64 calculation is reproducible.
        self.source_reciprocal_memory_fraction = (
            _SOURCE_RECIPROCAL_MEMORY_FRACTION)
        self.source_reciprocal_minimum_reserve_bytes = (
            _SOURCE_RECIPROCAL_MINIMUM_RESERVE_BYTES)
        self.source_reciprocal_max_block_size = _SOURCE_RECIPROCAL_MAX_BLOCK_SIZE
        self.source_reciprocal_block_alignment = (
            _SOURCE_RECIPROCAL_BLOCK_ALIGNMENT)
        # Real Gamma reciprocal source integrands occur in exact G,-G pairs.
        # Production evaluates one representative with integer multiplicity;
        # False retains the complete-grid numerical oracle.
        self.source_reciprocal_gamma_conjugate_symmetry = True
        # Native RSDF response consists of three long, non-overlapping stages:
        # reverse auxiliary propagation, source derivatives, and metric
        # derivatives.  Releasing only *unused* CuPy pool blocks at those
        # ownership boundaries prevents a completed stage from inflating the
        # resident high-water mark of the next one.  Live arrays are retained
        # by CuPy and are therefore unaffected mathematically.
        self.response_release_cached_blocks = True
        # Diagnostics-only component timing adds CUDA synchronizations around
        # adjoint/reverse/source/metric boundaries.  It is disabled in the
        # production route so profiler overhead cannot contaminate benchmark
        # wall times.
        self.response_profile_components = False
        # Production ``auto`` uses one direct-packed dual-observable route
        # when its bounded preflight fits and otherwise retains the
        # one-observable-at-a-time sequential fallback.  ``dense`` remains
        # the algebraically independent validation oracle.
        self.response_mode = 'auto'
        self.response_pair_block_size = 8192
        self.response_auxiliary_block_size = 32
        self.response_memory_fraction = .5
        self.response_minimum_reserve_bytes = 4 * 1024**3
        # R6 Water20/DZVP and Water64/SZV benchmarks show that stacked
        # observables are worthwhile while their predicted persistent work
        # is a few GiB (6.17% gradient speedup at 3.12 GB), but not when the
        # same choice consumes 10.28 GB (1.53% speedup and 46.1% more peak
        # VRAM).  This storage-only ceiling keeps ``auto`` size-aware without
        # changing source/metric parameters or numerical accuracy.  Explicit
        # ``response_mode='fused_packed'`` remains an opt-in override.
        self.response_fused_max_bytes = 4 * 1024**3
        self.sampled_k_hcore_host_calls = 0
        self.sampled_k_hcore_host_bytes = 0
        self.sampled_k_hcore_owner = None
        self.sampled_k_response_layout_copy_count = 0
        self.sampled_k_response_layout_copy_bytes = 0
        init_cderi_options(self)
        # The one-electron GTH operator is independent of the RSDF source and
        # metric range separations.  It owns a separate PySCF-derived local
        # part-1 plan that is shared by zero-order Hcore and its derivative.
        init_pp_loc_part1(self)

    get_pp_loc_part1_plan = get_pp_loc_part1_plan

    def get_pp(self, kpts=None):
        '''Return provider-consistent GTH matrices for Gamma or sampled k.'''
        requested = self.kpts if kpts is None else np.asarray(kpts)
        sampled_k = (
            np.asarray(requested).reshape(-1, 3).shape[0] > 1
            or np.any(np.asarray(requested) != 0))
        if not sampled_k:
            return get_pp_gamma(self, kpts)

        witness = cpu_rsdf.RSGDF(self.cell, kpts=self.kpts)
        witness.auxbasis = self.auxbasis
        for name in (
                'use_bvk', 'precision_R', 'precision_G', 'npw_max',
                '_omega_min', 'omega', 'ke_cutoff', 'mesh_compact',
                'omega_j2c', 'mesh_j2c', 'precision_j2c',
                'j2c_eig_always'):
            setattr(witness, name, getattr(self, name))
        host = np.asarray(witness.get_pp(requested))
        self.sampled_k_hcore_host_calls += 1
        self.sampled_k_hcore_host_bytes += int(host.nbytes)
        self.sampled_k_hcore_owner = 'PySCF RSGDF.get_pp sampled-k'
        return cp.asarray(host)

    def build(self, j_only=None, kpts_band=None):
        r'''Build RSDF factors with PySCF-resolved source/metric data.

        The compressed GPU factors have the ordinary GPU4PySCF ``GDF``
        layout.  Consequently inherited GPU J/K contractions remain device
        resident.  Only the screened two-center metric used inside the
        factorization is obtained from PySCF to match the RSDF convention.
        '''
        if kpts_band is not None:
            raise NotImplementedError('Gamma GPU RSDF does not support kpts_band')
        if j_only is not None:
            self._j_only = bool(j_only)
        if self._j_only:
            raise NotImplementedError(
                'GPU RSDF currently builds exchange-capable Gamma factors only')
        _check_cell(self)
        profile_context = (
            profile_native_gamma_build(self)
            if self.native_build_profile else nullcontext(None))
        with profile_context as profiler:
            with _profile_stage(profiler, 'cderi_preflight'):
                preflight_cderi_memory(self)
            with _profile_stage(profiler, 'parameter_witness'):
                witness = self._build_parameter_witness()
            if witness.j2c_eig_always:
                raise NotImplementedError(
                    'GPU RSDF requires Cholesky metric factorization; '
                    'j2c_eig_always=True is unsupported')
            self._adopt_witness_settings(witness)
            state = _BuildState(profiler=profiler)
            start = perf_counter()
            kmesh = np.asarray(kpts_to_kmesh(
                self.cell, self.kpts, precision=self.cell.precision), dtype=int)
            with (_cpu_equivalent_builder(self, witness.auxcell, state),
                  _profile_stage(profiler, 'central_builder_unclassified')):
                cderi, cderip, cderi_idx = rsdf_builder.build_cderi(
                    self.cell, witness.auxcell, kpts=self.kpts,
                    kmesh=kmesh, j_only=False, omega=None,
                    linear_dep_threshold=float(self.linear_dep_threshold),
                    compress=True)
            cp.cuda.Stream.null.synchronize()
            self.auxcell = witness.auxcell
            self.nao = int(self.cell.nao_nr())
            self.kmesh = kmesh
            self._omega = None
            self._cderi = cderi
            self._cderip = cderip
            self._cderi_idx = tuple(
                cp.asarray(value, dtype=cp.int32) for value in cderi_idx)
            self._j_only = False
            self._central_metric = state.central_metric
            self._central_metric_factor = None
            self.central_build_metadata = {
                'builder': 'dcforge sampled-k RSDF exact central adapter',
                'source_owner': 'GPU4PySCF CUDA three-center builder',
                'short_range_metric_owner': 'PySCF rsdf_helper.intor_j2c',
                'long_range_metric_owner': 'GPU4PySCF reciprocal kernels',
                'factorization_owner': 'GPU4PySCF CuPy Cholesky',
                'source_omega': float(self.omega),
                'source_mesh': [int(value) for value in self.mesh_compact],
                'metric_omega': float(self.omega_j2c),
                'metric_mesh': [int(value) for value in self.mesh_j2c],
                'host_short_range_metric_seconds': state.host_metric_seconds,
                'device_long_range_metric_seconds': state.device_metric_seconds,
                'metric_block_plans': state.metric_block_plans,
                'total_build_seconds': perf_counter() - start,
            }
            with _profile_stage(profiler, 'storage_checkpoint'):
                finalize_cderi_storage(self)
        return self

    def reset(self, cell=None):
        '''Reset geometry-dependent GPU factors and RSDF build diagnostics.'''
        cleanup_cderi_storage(self)
        gpu_df.GDF.reset(self, cell)
        # Match PySCF GDF/RSGDF reset semantics.  GPU4PySCF's GDF reset does
        # not clear ``auxcell`` even though its centers and environment belong
        # to the previous geometry.
        self.auxcell = None
        self.central_build_metadata = None
        self._central_metric = None
        self._central_metric_factor = None
        self._parameter_witness_cache = None
        self._parameter_witness_signature = None
        self.native_build_substage_profile = None
        self.sampled_k_hcore_host_calls = 0
        self.sampled_k_hcore_host_bytes = 0
        self.sampled_k_hcore_owner = None
        self.sampled_k_response_layout_copy_count = 0
        self.sampled_k_response_layout_copy_bytes = 0
        reset_pp_loc_part1(self)
        return self

    def cderi_memory_report(self, response_observables=2):
        '''Return exact central-factor and dense-response memory telemetry.'''
        return cderi_memory_report(self, response_observables)

    def release_cderi(self):
        '''Release central factors after all source-HF response actions.'''
        return release_cderi(self)

    def packed_cderi(self, auxiliary_block_size=256):
        '''Return unique real-Gamma AO-pair CDERI and its symmetry layout.'''
        return load_unique_gamma_cderi(self, auxiliary_block_size)

    def canonical_cderi(self):
        r'''Expand the compressed Gamma CDERI into its device tensor form.

        Returns:
            cupy.ndarray
                Real Gamma CDERI ``C[P,i,j]`` with shape
                ``(naux, nao, nao)``.  Its AO-pair symmetrisation is applied
                by a real-Gamma triangular expansion on the device.

        Notes:
            The ordinary energy route must keep the provider's compressed
            CDERI representation.  This method is intentionally a response
            primitive: reverse auxiliary adjoints require a common
            ``(P,i,j)`` representation before they can be pushed through
            ``C = L^{-1} B``.  It creates no nuclear derivative tensor and
            is not retained by the provider, so its storage can be released
            before the native source contraction starts on a large system.
        '''
        kpts = np.asarray(self.kpts, dtype=float).reshape(-1, 3)
        if len(kpts) != 1 or not is_zero(kpts[0]):
            raise NotImplementedError(
                'canonical_cderi is the real-Gamma RSDF response primitive')
        if self._cderi is None:
            self.build(j_only=False)
        if self._cderip is not None:
            raise NotImplementedError(
                'GPU RSDF response does not support an indefinite metric')
        packed, layout = self.packed_cderi()
        nao = int(layout['nao'])
        # This is the real-Gamma triangular expansion of GPU4PySCF's
        # compressed central factor.  It is deliberately separate from the
        # provider's fast compressed J/K route: source-HF SCF and CPHF keep
        # their native GPU4PySCF contractions, while reverse adjoints need
        # one common ``(P,i,j)`` factor representation.
        row = cp.asarray(layout['row'], dtype=cp.int32)
        column = cp.asarray(layout['column'], dtype=cp.int32)
        expanded = cp.zeros((packed.shape[0], nao, nao), dtype=packed.dtype)
        expanded[:, row, column] = packed
        expanded[:, column, row] = packed
        return expanded

    def central_metric_factor(self):
        r'''Return the lower Cholesky factor of the authoritative metric.

        ``canonical_cderi`` and this factor share the exact source/metric
        settings used by :meth:`build`.  The factor therefore defines the
        same response identity as the CPU RSDF convention,
        ``C = L^{-1} B`` and ``M = L L^T``.
        '''
        if self._central_metric is None:
            raise RuntimeError('build GPU RSDF before requesting its metric factor')
        if self._central_metric_factor is None:
            self._central_metric_factor = cp.linalg.cholesky(
                self._central_metric)
        return self._central_metric_factor

    def capability(self):
        '''Return the resolved GPU RSDF capability boundary.

        The compact source mesh is an accuracy/performance setting resolved by
        PySCF's RSGDF policy.  It is not a CUDA safety discriminator: small
        meshes can imply many real-space images and therefore long execution,
        but the same native source endpoint remains mathematically valid.
        '''
        mesh = getattr(self, 'mesh_compact', None)
        sampled_k = len(self.kpts) > 1 or np.any(np.asarray(self.kpts) != 0)
        source_mesh = None if mesh is None else [int(value) for value in mesh]
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
            'analytic_gradient_requirements': (
                'sampled-k provider-exact resolved CPU response'
                if sampled_k else
                'Gamma real-orbital RSDF with CUDA task batches <= POOL_SIZE'),
            'sampled_k_response': (
                'DC-Forge CPU provider-authoritative source/metric response'
                if sampled_k else None),
            'sampled_k_full_coulomb_probe': (
                'GPU4PySCF native combined; not production eligible'
                if sampled_k else None),
            'source_mesh': source_mesh,
            'stress': not sampled_k,
            'sampled_k_hcore_host_boundary': sampled_k,
            'kpts_roundtrip_max_abs': self.kpts_roundtrip_max_abs,
            'auxbasis': self.auxbasis,
            'source_omega': self.omega,
            'metric_omega': self.omega_j2c,
            'source_reciprocal_gamma_conjugate_symmetry': (
                self.source_reciprocal_gamma_conjugate_symmetry),
            'response_release_cached_blocks': (
                self.response_release_cached_blocks),
            'response_profile_components': self.response_profile_components,
            'native_build_profile': self.native_build_profile,
            'native_build_substage_profile': (
                self.native_build_substage_profile),
            'response_mode': self.response_mode,
            'response_pair_block_size': self.response_pair_block_size,
            'response_auxiliary_block_size': (
                self.response_auxiliary_block_size),
            'response_memory_fraction': self.response_memory_fraction,
            'response_minimum_reserve_bytes': (
                self.response_minimum_reserve_bytes),
            'response_fused_max_bytes': self.response_fused_max_bytes,
            'cderi_storage': self.cderi_storage,
            'cderi_memory_metadata': self.cderi_memory_metadata,
            'release_cderi_after_response': self.release_cderi_after_response,
            'pp_loc_part1_omega': self.pp_loc_part1_omega,
            'pp_loc_part1_omega_resolved': self.pp_loc_part1_omega_resolved,
            'pp_loc_part1_ke_cutoff': self.pp_loc_part1_ke_cutoff,
            'pp_loc_part1_mesh': (
                None if self.pp_loc_part1_mesh is None
                else np.asarray(self.pp_loc_part1_mesh, dtype=int).tolist()),
            'pp_loc_part1_plan_mode': self.pp_loc_part1_plan_mode,
        }

    def response_settings(self):
        '''Return independently resolved RSDF source and metric settings.'''
        required = ('omega', 'mesh_compact', 'omega_j2c', 'mesh_j2c')
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise RuntimeError(
                'build GPU RSDF before requesting response settings: '
                + ', '.join(missing))
        return {
            'source_omega': float(self.omega),
            'source_mesh': [int(value) for value in self.mesh_compact],
            'metric_omega': float(self.omega_j2c),
            'metric_mesh': [int(value) for value in self.mesh_j2c],
        }

    def _build_parameter_witness(self):
        signature = self._make_parameter_witness_signature()
        if (self._parameter_witness_cache is not None
                and self._parameter_witness_signature == signature):
            return self._parameter_witness_cache
        witness = cpu_rsdf.RSGDF(self.cell, kpts=self.kpts)
        witness.auxbasis = self.auxbasis
        for name in (
                'use_bvk', 'precision_R', 'precision_G', 'npw_max',
                '_omega_min', 'omega', 'ke_cutoff', 'mesh_compact',
                'omega_j2c', 'mesh_j2c', 'precision_j2c',
                'j2c_eig_always'):
            setattr(witness, name, getattr(self, name))
        witness.linear_dep_threshold = float(self.linear_dep_threshold)
        witness._rs_build()
        self._parameter_witness_cache = witness
        self._parameter_witness_signature = signature
        return witness

    def _adopt_witness_settings(self, witness):
        for name in (
                'omega', 'ke_cutoff', 'mesh_compact', 'omega_j2c',
                'mesh_j2c', 'precision_R', 'precision_G', 'precision_j2c'):
            value = getattr(witness, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            setattr(self, name, value)
        # Resolution changes ``None``/automatic inputs into their explicit
        # values.  The cached witness now belongs to this resolved signature.
        if self._parameter_witness_cache is not None:
            self._parameter_witness_signature = (
                self._make_parameter_witness_signature())

    def _make_parameter_witness_signature(self):
        '''Return the provider-lifetime identity of PySCF RSDF resolution.'''
        values = []
        for name in (
                'use_bvk', 'precision_R', 'precision_G', 'npw_max',
                '_omega_min', 'omega', 'ke_cutoff', 'mesh_compact',
                'omega_j2c', 'mesh_j2c', 'precision_j2c',
                'j2c_eig_always', 'linear_dep_threshold'):
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                value = (str(value.dtype), tuple(value.shape), value.tobytes())
            values.append(value)
        return (
            self.cell.dumps(), repr(self.auxbasis),
            np.asarray(self.kpts, dtype=float).tobytes(), tuple(values))


class _BuildState:
    '''Per-build metric timings and bounded reciprocal execution metadata.'''

    def __init__(self, profiler=None):
        self.host_metric_seconds = 0.
        self.device_metric_seconds = 0.
        self.metric_block_plans = []
        self.central_metric = None
        self.profiler = profiler


def _profile_stage(profiler, name):
    '''Return one diagnostics-only stage or a zero-cost null context.'''
    if profiler is None:
        return nullcontext()
    return profiler.stage(name)


@contextmanager
def _cpu_equivalent_builder(provider, reference_auxcell, state):
    r'''Install a narrow, synchronous adapter for the GPU RSDF builder.

    GPU4PySCF 1.8.1 does not expose independent source/metric settings in
    ``build_cderi``.  The process-global functions are changed only while
    that synchronous call is active and are restored in ``finally``.  The
    lock prevents two RSDF builds from interleaving incompatible settings.
    '''
    with _BUILDER_ADAPTER_LOCK:
        original_optimizer = rsdf_builder.SRInt3c2eOpt
        original_guess = rsdf_builder._guess_omega
        original_metric = rsdf_builder._get_2c2e
        original_precontract = rsdf_builder._precontract_j2c_aux_coeff
        original_prefer_ed = rsdf_builder.PREFER_ED
        source_mesh = np.asarray(provider.mesh_compact, dtype=int)

        class SourceOptimizer(original_optimizer):
            def __init__(self, *args, **kwargs):
                original_optimizer.__init__(self, *args, **kwargs)
                self.mesh = source_mesh.copy()

        def source_omega(_cell, kmesh=None):
            del kmesh
            return float(provider.omega)

        def metric(auxcell, unique_kpts, omega, metric_omega,
                   bvk_kmesh=None):
            del bvk_kmesh
            return _build_metric(
                auxcell, unique_kpts, physical_omega=float(omega),
                metric_omega=float(metric_omega), provider=provider,
                reference_auxcell=reference_auxcell, state=state)

        def precontract(auxcell, kpts, omega, _source_omega,
                        linear_dep_threshold, kmesh=None):
            return original_precontract(
                auxcell, kpts, omega, float(provider.omega_j2c),
                linear_dep_threshold, kmesh)

        rsdf_builder.SRInt3c2eOpt = SourceOptimizer
        rsdf_builder._guess_omega = source_omega
        rsdf_builder._get_2c2e = metric
        rsdf_builder._precontract_j2c_aux_coeff = precontract
        # CPU RSDF's ordinary factorization is Cholesky.  GPU4PySCF's ED
        # default gives a different auxiliary rank and cannot be compared to
        # the provider contract used by the CPU reference implementation.
        rsdf_builder.PREFER_ED = False
        try:
            yield
        finally:
            rsdf_builder.SRInt3c2eOpt = original_optimizer
            rsdf_builder._guess_omega = original_guess
            rsdf_builder._get_2c2e = original_metric
            rsdf_builder._precontract_j2c_aux_coeff = original_precontract
            rsdf_builder.PREFER_ED = original_prefer_ed


def _build_metric(auxcell, unique_kpts, physical_omega, metric_omega,
                  provider, reference_auxcell, state):
    '''Build PySCF-screened metric plus GPU long-range reciprocal correction.'''
    if unique_kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = np.asarray(unique_kpts, dtype=float).reshape(-1, 3)
    host_auxcell = reference_auxcell.copy()
    host_auxcell.precision = float(provider.precision_j2c)
    start = perf_counter()
    with _profile_stage(state.profiler, 'metric_short_range'):
        metric = cp.asarray(rsdf_helper.intor_j2c(
            host_auxcell, float(metric_omega), kpts=kpts))
    state.host_metric_seconds += perf_counter() - start

    work_auxcell = auxcell.copy()
    work_auxcell.precision = float(provider.precision_j2c)
    mesh = (np.asarray(provider.mesh_j2c, dtype=int)
            if physical_omega < metric_omega else np.ones(3, dtype=int))
    gv = work_auxcell.get_Gv(mesh)
    coulomb = rsdf_builder._weighted_coulG_kpts(
        work_auxcell, mesh, physical_omega, metric_omega,
        None if unique_kpts is None else kpts)
    block_size, plan = _metric_block_plan(len(gv), int(work_auxcell.nao))
    state.metric_block_plans.append(plan)
    event_start = cp.cuda.Event()
    event_stop = cp.cuda.Event()
    event_start.record()
    with _profile_stage(state.profiler, 'metric_long_range'):
        for ik, kpt in enumerate(kpts):
            for p0 in range(0, len(gv), block_size):
                p1 = min(len(gv), p0 + block_size)
                aoaux = rsdf_builder.ft_ao.ft_ao(
                    work_auxcell, gv[p0:p1], kpt=kpt, sort_output=True)
                weighted = aoaux.conj() * coulomb[ik, p0:p1, None]
                correction = weighted.T.dot(aoaux)
                if is_zero(kpt):
                    correction = correction.real
                metric[ik] += correction
    event_stop.record()
    event_stop.synchronize()
    state.device_metric_seconds += cp.cuda.get_elapsed_time(
        event_start, event_stop) / 1000.
    # The builder consumes this same Gamma metric immediately to construct
    # its precontracted Cholesky transformation.  A retained device copy is
    # the authoritative metric for the later reverse-Cholesky response; it
    # is small compared with either compressed CDERI or any source response.
    # ``original_precontract`` may normalise the Gamma input as either
    # ``None`` or one explicit zero k-point.  The builder accepts both, but
    # the response factor must always be the rank-two Gamma metric.
    if len(kpts) == 1 and is_zero(kpts[0]):
        state.central_metric = (
            metric[0].copy() if metric.ndim == 3 else metric.copy())
    return metric[0] if unique_kpts is None else metric


def _metric_block_plan(ngrids, naux):
    '''Choose a bounded reciprocal metric block without changing the metric.'''
    free_bytes = int(get_avail_mem(exclude_memory_pool=True))
    budget = min(
        int(free_bytes * _METRIC_MEMORY_FRACTION),
        max(0, free_bytes - _METRIC_MINIMUM_RESERVE_BYTES))
    bytes_per_grid = 2 * int(naux) * np.dtype(np.complex128).itemsize
    block_size = min(
        int(ngrids), _METRIC_MAX_GRID_BLOCK_SIZE,
        budget // max(1, bytes_per_grid))
    if block_size < 1:
        raise MemoryError(
            'insufficient GPU memory for one RSDF metric reciprocal vector: '
            f'budget={budget} bytes, required={bytes_per_grid} bytes')
    return int(block_size), {
        'ngrids': int(ngrids),
        'naux': int(naux),
        'free_device_bytes_at_plan': free_bytes,
        'memory_fraction': _METRIC_MEMORY_FRACTION,
        'minimum_reserve_bytes': _METRIC_MINIMUM_RESERVE_BYTES,
        'budget_bytes': budget,
        'bytes_per_grid': bytes_per_grid,
        'block_size': int(block_size),
    }


def _check_cell(provider):
    '''Reject unsupported dimensions before mutating the builder state.'''
    if provider.cell.dimension != 3:
        raise NotImplementedError('GPU RSDF currently requires a 3D Cell')
