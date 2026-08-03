'''Gamma-point GPU RSDF provider with explicit source and metric ownership.

GPU4PySCF exposes GPU range-separated three-center and factorization kernels
through :mod:`gpu4pyscf.pbc.df.rsdf_builder`, but it has no public ``RSGDF``
provider class.  PySCF ``RSGDF`` independently resolves the source and metric
range-separation parameters.  This adapter preserves that distinction:
GPU4PySCF builds the large three-center source and factorizes the metric,
while PySCF supplies only the small screened two-center metric primitive.
'''

from contextlib import contextmanager
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

from dcforge.gpu.pbc.df.gaussian.pseudo import (
    PP_LOC_PART1_KEYS,
    get_pp,
    get_pp_loc_part1_plan,
    init_pp_loc_part1,
    reset_pp_loc_part1,
)

_BUILDER_ADAPTER_LOCK = RLock()
_METRIC_MEMORY_FRACTION = .25
_METRIC_MINIMUM_RESERVE_BYTES = 512 * 1024**2
_METRIC_MAX_GRID_BLOCK_SIZE = 4096
_SOURCE_RECIPROCAL_MEMORY_FRACTION = .15
_SOURCE_RECIPROCAL_MINIMUM_RESERVE_BYTES = 512 * 1024**2
_SOURCE_RECIPROCAL_MAX_BLOCK_SIZE = 512
_SOURCE_RECIPROCAL_BLOCK_ALIGNMENT = 32


class RSDF(gpu_df.GDF):
    r'''Gamma-point range-separated Gaussian density-fitting provider.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built three-dimensional GTH periodic cell.

    Kwargs:
        auxbasis : str, dict, or None
            Auxiliary basis passed to the PySCF RSDF parameter witness.

    Notes:
        ``omega``/``mesh_compact`` define the three-center source while
        ``omega_j2c``/``mesh_j2c`` define the two-center metric.  They are
        resolved by a temporary PySCF ``RSGDF`` witness before the GPU
        builder starts; neither is inferred from the other.  The host metric
        primitive is an explicit, bounded ``O(naux^2)`` compatibility
        boundary, not a host J/K or host three-center fallback.
    '''

    backend_name = 'rsdf'
    device = 'gpu'
    _keys = gpu_df.GDF._keys | {
        'omega', 'ke_cutoff', 'mesh_compact', 'omega_j2c', 'mesh_j2c',
        'precision_R', 'precision_G', 'precision_j2c', 'npw_max',
        '_omega_min', 'use_bvk', 'j2c_eig_always',
        'central_build_metadata', '_central_metric',
        'source_reciprocal_memory_fraction',
        'source_reciprocal_minimum_reserve_bytes',
        'source_reciprocal_max_block_size',
        'source_reciprocal_block_alignment',
        'source_reciprocal_gamma_conjugate_symmetry',
        'response_release_cached_blocks',
    } | PP_LOC_PART1_KEYS

    def __init__(self, cell, auxbasis=None):
        gpu_df.GDF.__init__(self, cell, kpts=np.zeros((1, 3)))
        witness = cpu_rsdf.RSGDF(cell, kpts=np.zeros((1, 3)))
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
        # ``_central_metric`` is the exact Gamma metric in the same AO
        # auxiliary basis as the compressed device CDERI.  The RSDF builder
        # already needs this small O(naux**2) object to construct ``C``;
        # retaining it avoids rebuilding a potentially different metric when
        # the response code pushes an observable through the Cholesky gauge.
        # It is deliberately not a dense three-center or derivative cache.
        self._central_metric = None
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
        # The one-electron GTH operator is independent of the RSDF source and
        # metric range separations.  It owns a separate PySCF-derived local
        # part-1 plan that is shared by zero-order Hcore and its derivative.
        init_pp_loc_part1(self)

    get_pp_loc_part1_plan = get_pp_loc_part1_plan
    get_pp = get_pp

    def build(self, j_only=None, kpts_band=None):
        r'''Build Gamma RSDF factors with PySCF-resolved source/metric data.

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
        _check_gamma_cell(self)
        witness = self._build_parameter_witness()
        if witness.j2c_eig_always:
            raise NotImplementedError(
                'GPU RSDF requires Cholesky metric factorization; '
                'j2c_eig_always=True is unsupported')
        self._adopt_witness_settings(witness)
        state = _BuildState()
        start = perf_counter()
        with _cpu_equivalent_builder(
                self, witness.auxcell, state):
            cderi, cderip, cderi_idx = rsdf_builder.build_cderi(
                self.cell, witness.auxcell, kpts=None,
                kmesh=np.ones(3, dtype=int), j_only=False, omega=None,
                linear_dep_threshold=float(self.linear_dep_threshold),
                compress=True)
        cp.cuda.Stream.null.synchronize()
        self.auxcell = witness.auxcell
        self.nao = int(self.cell.nao_nr())
        self.kmesh = np.ones(3, dtype=int)
        self._omega = None
        self._cderi = cderi
        self._cderip = cderip
        self._cderi_idx = tuple(
            cp.asarray(value, dtype=cp.int32) for value in cderi_idx)
        self._j_only = False
        self._central_metric = state.central_metric
        self.central_build_metadata = {
            'builder': 'dcforge Gamma RSDF exact central adapter',
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
        return self

    def reset(self, cell=None):
        '''Reset geometry-dependent GPU factors and RSDF build diagnostics.'''
        gpu_df.GDF.reset(self, cell)
        # Match PySCF GDF/RSGDF reset semantics.  GPU4PySCF's GDF reset does
        # not clear ``auxcell`` even though its centers and environment belong
        # to the previous geometry.
        self.auxcell = None
        self.central_build_metadata = None
        self._central_metric = None
        reset_pp_loc_part1(self)
        return self

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
        if self._cderi is None:
            self.build(j_only=False)
        if self._cderip is not None:
            raise NotImplementedError(
                'GPU RSDF response does not support an indefinite metric')
        packed = self._cderi.get(0)
        if packed is None:
            raise RuntimeError('Gamma RSDF compressed CDERI block is missing')
        pair_address = self._cderi_idx[0]
        nao = int(self.cell.nao_nr())
        # This is the real-Gamma triangular expansion of GPU4PySCF's
        # compressed central factor.  It is deliberately separate from the
        # provider's fast compressed J/K route: source-HF SCF and CPHF keep
        # their native GPU4PySCF contractions, while reverse adjoints need
        # one common ``(P,i,j)`` factor representation.
        packed = cp.asarray(packed)
        row, column = divmod(cp.asarray(pair_address), nao)
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
        return cp.linalg.cholesky(self._central_metric)

    def capability(self):
        '''Return the resolved Gamma GPU RSDF capability boundary.

        The compact source mesh is an accuracy/performance setting resolved by
        PySCF's RSGDF policy.  It is not a CUDA safety discriminator: small
        meshes can imply many real-space images and therefore long execution,
        but the same native source endpoint remains mathematically valid.
        '''
        mesh = getattr(self, 'mesh_compact', None)
        source_mesh = None if mesh is None else [int(value) for value in mesh]
        return {
            'backend': self.backend_name,
            'device': self.device,
            'sampling': 'gamma',
            'reference': 'RHF',
            'target': ['PBE', 'R2SCAN'],
            'energy': True,
            'analytic_gradient': True,
            'analytic_gradient_process_scope': 'same_process_reentrant',
            'analytic_gradient_requirements': (
                'Gamma real-orbital RSDF with CUDA task batches <= POOL_SIZE'),
            'source_mesh': source_mesh,
            'stress': False,
            'auxbasis': self.auxbasis,
            'source_omega': self.omega,
            'metric_omega': self.omega_j2c,
            'source_reciprocal_gamma_conjugate_symmetry': (
                self.source_reciprocal_gamma_conjugate_symmetry),
            'response_release_cached_blocks': (
                self.response_release_cached_blocks),
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
        witness = cpu_rsdf.RSGDF(self.cell, kpts=np.zeros((1, 3)))
        witness.auxbasis = self.auxbasis
        for name in (
                'use_bvk', 'precision_R', 'precision_G', 'npw_max',
                '_omega_min', 'omega', 'ke_cutoff', 'mesh_compact',
                'omega_j2c', 'mesh_j2c', 'precision_j2c',
                'j2c_eig_always'):
            setattr(witness, name, getattr(self, name))
        witness.linear_dep_threshold = float(self.linear_dep_threshold)
        witness._rs_build()
        return witness

    def _adopt_witness_settings(self, witness):
        for name in (
                'omega', 'ke_cutoff', 'mesh_compact', 'omega_j2c',
                'mesh_j2c', 'precision_R', 'precision_G', 'precision_j2c'):
            value = getattr(witness, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            setattr(self, name, value)


class _BuildState:
    '''Per-build metric timings and bounded reciprocal execution metadata.'''

    def __init__(self):
        self.host_metric_seconds = 0.
        self.device_metric_seconds = 0.
        self.metric_block_plans = []
        self.central_metric = None


@contextmanager
def _cpu_equivalent_builder(provider, reference_auxcell, state):
    r'''Install a narrow, synchronous adapter for the GPU RSDF builder.

    GPU4PySCF 1.8.0 does not expose independent source/metric settings in
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


def _check_gamma_cell(provider):
    '''Reject unsupported representations before mutating the builder state.'''
    if provider.cell.dimension != 3:
        raise NotImplementedError('GPU RSDF currently requires a 3D Cell')
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    if len(kpts) != 1 or not is_zero(kpts[0]):
        raise NotImplementedError('GPU RSDF currently requires Gamma point')
