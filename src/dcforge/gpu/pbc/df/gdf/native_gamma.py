'''Gamma-point signed GDF derivative assembled from GPU4PySCF primitives.

GPU4PySCF exposes a quadratic GDF derivative endpoint.  A DC-DFT Z-vector,
however, needs the difference between two *signed* quadratic densities.  The
ordinary fused route contracts that difference into one coherent source before
the native primitive traversal.  An explicit optional two-channel route shares
the zero-order and short-range work for target-J and Z-Fock, while repeating
the upstream single-channel reciprocal and metric primitives for each physical
channel.  It deliberately uses GPU4PySCF primitives directly rather than
importing any legacy implementation.

The real-Gamma kernel uses a bounded, transactional reciprocal scheduler.  It
preflights the largest live Fourier tensors, records the resolved plan, and
retries an uncommitted block at a smaller size after a CUDA allocation error.
Production also evaluates one representative of each exact ``G,-G``
conjugate pair with its integer multiplicity.  This preserves the complete
reciprocal weight and algebraic source; the unreduced full-grid route remains
an explicit numerical oracle.  Expanding a unique-pair primitive adjoint back
to the raw libpbc AO-pair order is likewise blocked: production never
materialises the complete ``selected[:, inverse]`` advanced-indexing result.
'''

import ctypes
import math
import time

import cupy as cp
import numpy as np
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.df.grad.rhf import factorize_dm
from gpu4pyscf.gto.mole import groupby
from gpu4pyscf.lib.cupy_helper import (
    asarray,
    contract,
    empty_aligned,
    empty_mapped,
    get_avail_mem,
)
from gpu4pyscf.pbc.df import aft_jk, ft_ao
from gpu4pyscf.pbc.df.grad.rhf import (
    _split_l_ctr_pattern,
    get_ao_pair_loc,
)
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.df.int3c2e import MAX_IMGS_PER_TASK, POOL_SIZE, SRInt3c2eOpt, libpbc
from gpu4pyscf.pbc.df.rsdf_builder import _guess_omega, _weighted_coulG_kpts
from pyscf import lib
from pyscf.gto import ATOM_OF

from dcforge.gpu.pbc.df.cderi import canonical_pair_layout
from dcforge.gpu.pbc.df.compatibility import derivative_int3c2e_scheme
from dcforge.gpu.pbc.df.gaussian.ewald import (
    cross_derivative as ewald_cross_derivative,
)
from dcforge.gpu.pbc.df.gaussian.ewald import (
    signed_derivative as ewald_signed_derivative,
)
from dcforge.gpu.pbc.df.gaussian.reciprocal import (
    execute_gamma_reciprocal_blocks,
    gamma_conjugate_reciprocal_indices,
    plan_gamma_reciprocal_blocks,
)
from dcforge.gpu.pbc.df.gdf.adjoint import (
    build_rhf_packed_primitive_adjoint,
    build_rhf_primitive_adjoint_channels,
    plan_adjoint_host_cache,
)
from dcforge.gpu.pbc.df.gdf.rhf_factorized import (
    build_rhf_mo_factorized_packed_adjoint,
)
from dcforge.gpu.pbc.df.gdf.rhf_factorized_consumers import (
    DEFAULT_RAW_PAIR_BLOCK_SIZE,
    build_rhf_mo_response_short_range_source_block,
    contract_rhf_mo_response_long_range_block,
    prepare_rhf_mo_short_range_pair_indices,
)
from dcforge.gpu.pbc.df.gdf.rhf_mo_response import (
    add_reciprocal_rhf_mo_projections,
    build_short_range_rhf_mo_projections,
    make_rhf_mo_response_factors,
    solve_rhf_mo_projections_inplace,
)
from dcforge.gpu.pbc.df.gdf.two_channel import get_short_range_kernel

# A power-of-two AO-pair tile keeps the largest gather bounded while retaining
# enough work per CuPy kernel.  Water64/GTH-DZVP/ETB was validated at this
# value: the blocked route removed the full 12.8-GiB gather without measurable
# end-to-end slowdown.  This is an implementation batch, not a scientific
# approximation or a user convergence parameter.
_PACKED_SOURCE_PAIR_BLOCK_SIZE = 1 << 15


class _SignedState:
    '''One weighted quadratic density in the primitive sorted-AO basis.'''

    def __init__(self, coulomb_weight, exchange_weight, factor_l, factor_r,
                 density_physical, density_sorted):
        self.coulomb_weight = float(coulomb_weight)
        self.exchange_weight = float(exchange_weight)
        self.factor_l = factor_l
        self.factor_r = factor_r
        self.density_physical = density_physical
        self.density_sorted = density_sorted
        self.three_center_oo = None
        self.aux_vector = None

    @property
    def rank(self):
        '''Return the low-rank factor width used by the native contractions.'''
        return int(self.factor_l.shape[1])


class _PackedPrimitiveSource:
    '''Unique real-Gamma AO-pair source plus its symmetry metadata.'''

    def __init__(self, values, layout):
        self.values = cp.asarray(values)
        self.layout = layout


def signed_hf_bilinear_derivative(mf_hf, dm0, dz, atmlst=None,
                                  return_metadata=False):
    r'''Return ``Q[P+D_Z]^lambda-Q[P-D_Z]^lambda`` in Ha/Bohr.

    Args:
        mf_hf : gpu4pyscf.pbc.scf.hf.RHF
            Gamma-point source RHF with an active GPU GDF provider.
        dm0 : cupy.ndarray
            Closed-shell source density ``P`` in the physical AO basis.
        dz : cupy.ndarray
            Real symmetric adjoint transition density ``D_Z``.

    Kwargs:
        atmlst : sequence of int or None
            Requested nuclear centers.  ``None`` selects all atoms.
        return_metadata : bool
            If true, also return the resolved reciprocal memory plan and
            completed zero-order/derivative schedules.  The derivative value
            itself is unchanged.

    Returns:
        cupy.ndarray
            The signed restricted-HF two-electron derivative.  The caller
            contributes its negative to ``de_fock``.

    Notes:
        ``Q[D]=Tr[D(J[D]-K[D]/2)]/2``.  The two states have weights ``+1``
        and ``-1`` and are contracted before each native derivative kernel;
        no full derivative CDERI or AO J/K matrix is materialised.
    '''
    provider = mf_hf.with_df
    _check_gamma_gdf(provider, dm0, dz)
    if provider.auxcell is None:
        provider.build(j_only=False)

    cell = provider.cell
    auxcell = provider.auxcell
    omega = float(_guess_omega(cell))
    intopt = SRInt3c2eOpt(cell, auxcell, omega, None).build()
    states = (
        _make_state(intopt.cell, dm0 + dz, +1., +1.),
        _make_state(intopt.cell, dm0 - dz, -1., -1.),
    )
    value, metadata = _signed_derivative(
        intopt, states, exxdiv=mf_hf.exxdiv,
        reciprocal_options=_reciprocal_options(provider))
    atoms = _atmlst(cell, atmlst)
    value = cp.asarray(value[atoms])
    if return_metadata:
        return value, metadata
    return value


def coulomb_energy_derivative(mf_hf, dm, atmlst=None, return_metadata=False):
    r'''Return native ``d[Tr(P J[P])/2]/dR`` in Ha/Bohr.

    This is the target-Hartree counterpart to
    :func:`signed_hf_bilinear_derivative`.  Keeping both on the same primitive
    route avoids GPU4PySCF's mutable quadratic-dispatch state and guarantees
    that target J and source-HF response use identical Gamma GDF conventions.
    '''
    provider = mf_hf.with_df
    zero = cp.zeros_like(cp.asarray(dm))
    _check_gamma_gdf(provider, dm, zero)
    if provider.auxcell is None:
        provider.build(j_only=False)
    cell = provider.cell
    intopt = SRInt3c2eOpt(
        cell, provider.auxcell, float(_guess_omega(cell)), None).build()
    state = _make_state(intopt.cell, dm, +1., 0.)
    value, metadata = _signed_derivative(
        intopt, (state,), exxdiv=None,
        reciprocal_options=_reciprocal_options(provider))
    value = cp.asarray(value[_atmlst(cell, atmlst)])
    if return_metadata:
        return value, metadata
    return value


def target_j_fock_derivative(
        mf_hf, dm0, dz, atmlst=None, return_metadata=False,
        profile_detail=False, occupied_coefficients=None,
        response_orbitals=None, adjoint_backend='auto'):
    r'''Return the fused target-J and signed source-Fock two-electron term.

    The result is

    .. math::

       E_J[P]^\lambda-
       \{Q[P+D_Z]^\lambda-Q[P-D_Z]^\lambda\}.

    The ``packed`` and ``mo_factorized`` routes build packed ``B`` and
    ``W=M^{-1}B`` once before applying their reverse adjoint.  Production
    ``mo_direct`` instead projects each primitive transaction to
    ``Boo=C.T@B@C`` and ``Bou=C.T@B@U`` and solves the same metric only for
    those occupied-space right-hand sides.  Both are algebraically identical
    to a target ``P`` state plus the signed ``P+D_Z`` and ``P-D_Z``
    polarization states, but avoid their two full-rank signed three-center
    tensors.  The coupled traversal also avoids the mutable native derivative
    state that makes separately dispatched GDF observables unsafe.

    The returned target-J plus Z-Fock observable is intentionally coupled and
    must remain ``de_j_fock_native``.  With ``return_metadata=True``, the
    same traversal additionally exposes its exact primitive ``dB`` source,
    auxiliary ``dM`` metric, and Ewald components.  Those components audit
    the fused observable; they are not a claim that target J and source Fock
    were dispatched as separate native derivatives.

    ``profile_detail=True`` is a validation-only timing request.  It records
    backend-specific zero-order projection, metric solve, adjoint and
    long-range Fourier/contraction/reduction work; it changes no GDF tensor,
    block size or derivative term.
    '''
    provider = mf_hf.with_df
    _check_gamma_gdf(provider, dm0, dz)
    if provider.auxcell is None:
        provider.build(j_only=False)
    cell = provider.cell
    optimizer_profile = {} if profile_detail else None
    stage_start = time.perf_counter() if profile_detail else None
    intopt = SRInt3c2eOpt(
        cell, provider.auxcell, float(_guess_omega(cell)), None).build()
    if profile_detail:
        _finish_profile_stage(
            optimizer_profile, 'optimizer_seconds', stage_start)
    value, metadata = _direct_adjoint_derivative(
        intopt, dm0, dz, exxdiv=mf_hf.exxdiv,
        reciprocal_options=_reciprocal_options(provider),
        adjoint_options=_adjoint_options(provider),
        occupied_coefficients=occupied_coefficients,
        response_orbitals=response_orbitals,
        adjoint_backend=adjoint_backend,
        profile_detail=profile_detail)
    provider.adjoint_host_cache_plan = metadata['adjoint'].get(
        'host_cache_plan')
    provider._experimental_reciprocal_pair_cache_plan = metadata.get(
        'reciprocal_pair_cache')
    if profile_detail:
        metadata['profile']['optimizer_seconds'] = optimizer_profile[
            'optimizer_seconds']
    atoms = _atmlst(cell, atmlst)
    value = cp.asarray(value[atoms])
    if return_metadata:
        # The native traversal produces all-atom source/metric/Ewald pieces
        # alongside its final reduction.  Keep the metadata shape consistent
        # with the requested atom subset just as the public return value is.
        metadata['term_components'] = {
            name: np.asarray(component, dtype=float)[atoms]
            for name, component in metadata['term_components'].items()
        }
        return value, metadata
    return value


def fock_bilinear_derivative(
        mf_hf, dm0, dz, atmlst=None, return_metadata=False,
        profile_detail=False, occupied_coefficients=None,
        response_orbitals=None, adjoint_backend='auto'):
    r'''Return ``Q[P+D_Z]^lambda-Q[P-D_Z]^lambda`` by direct adjoint.

    This is the source-Fock-only counterpart of
    :func:`target_j_fock_derivative`.  It is used when target Hartree belongs
    to multigrid and therefore must not be included in the Gaussian response.
    The direct ``P,D_Z`` adjoint is algebraically identical to signed
    polarization but builds one primitive three-center tensor instead of the
    two full-rank ``P+D_Z`` and ``P-D_Z`` states.

    ``profile_detail=True`` enables the same validation-only metric/adjoint
    timing decomposition as :func:`target_j_fock_derivative`.
    '''
    provider = mf_hf.with_df
    _check_gamma_gdf(provider, dm0, dz)
    if provider.auxcell is None:
        provider.build(j_only=False)
    cell = provider.cell
    intopt = SRInt3c2eOpt(
        cell, provider.auxcell, float(_guess_omega(cell)), None).build()
    value, metadata = _direct_adjoint_derivative(
        intopt, dm0, dz, exxdiv=mf_hf.exxdiv,
        reciprocal_options=_reciprocal_options(provider),
        adjoint_options=_adjoint_options(provider),
        include_target_j=False, response_sign=1.,
        occupied_coefficients=occupied_coefficients,
        response_orbitals=response_orbitals,
        adjoint_backend=adjoint_backend,
        profile_detail=profile_detail)
    provider.adjoint_host_cache_plan = metadata['adjoint'].get(
        'host_cache_plan')
    provider._experimental_reciprocal_pair_cache_plan = metadata.get(
        'reciprocal_pair_cache')
    atoms = _atmlst(cell, atmlst)
    value = cp.asarray(value[atoms])
    if return_metadata:
        metadata['term_components'] = {
            name: np.asarray(component, dtype=float)[atoms]
            for name, component in metadata['term_components'].items()
        }
        return value, metadata
    return value


def target_j_fock_signed_derivative(mf_hf, dm0, dz, atmlst=None,
                                    return_metadata=False):
    r'''Return fused target-J/Z-Fock response through signed quadratic states.

    The direct primitive-adjoint route is the PBE performance default.  This
    algebraically equivalent route retains the three physical quadratic
    states used by the original one-pass formulation,

    .. math::

       E_J[P]^\lambda-Q[P+D_Z]^\lambda+Q[P-D_Z]^\lambda.

    It is retained as an algebraically independent production oracle and
    fallback for the faster direct primitive-adjoint source.  All three
    states share one short-range, reciprocal, metric, and Ewald traversal;
    no repeated native derivative dispatch is made.

    Args:
        mf_hf : gpu4pyscf.pbc.scf.hf.RHF
            Converged real-Gamma source RHF with the active GDF provider.
        dm0, dz : cupy.ndarray
            Source density ``P`` and Hermitian Z transition density ``D_Z``.

    Kwargs:
        atmlst : sequence of int or None
            Requested nuclear centers.
        return_metadata : bool
            Return the reciprocal schedule and fused source/metric/Ewald
            ledger together with the derivative.
    '''
    provider = mf_hf.with_df
    _check_gamma_gdf(provider, dm0, dz)
    if provider.auxcell is None:
        provider.build(j_only=False)
    cell = provider.cell
    intopt = SRInt3c2eOpt(
        cell, provider.auxcell, float(_guess_omega(cell)), None).build()
    states = (
        _make_state(intopt.cell, dm0, +1., 0.),
        _make_state(intopt.cell, dm0 + dz, -1., -1.),
        _make_state(intopt.cell, dm0 - dz, +1., +1.),
    )
    value, metadata = _signed_derivative(
        intopt, states, exxdiv=mf_hf.exxdiv,
        reciprocal_options=_reciprocal_options(provider))
    value = cp.asarray(value[_atmlst(cell, atmlst)])
    metadata['route'] = 'signed_quadratic_target_j_and_z_fock'
    if return_metadata:
        metadata['term_components'] = {
            name: np.asarray(component)[_atmlst(cell, atmlst)]
            for name, component in metadata['term_components'].items()
        }
        return value, metadata
    return value


def target_j_and_fock_derivative(mf_hf, dm0, dz, atmlst=None):
    r'''Return separate target-J and Z-Fock GDF derivative observables.

    This optional production route uses one patched two-channel short-range
    CUDA traversal.  It is not a numerical split of
    :func:`target_j_fock_derivative`: the primitive and metric reverse
    adjoints are independently constructed for

    .. math::

       de_J = E_J[P]^\lambda,
       \qquad
       de_{\rm Fock} =
       -\{Q[P+D_Z]^\lambda-Q[P-D_Z]^\lambda\}.

    The reciprocal Fourier derivative has no non-reentrancy defect in the
    audited GPU4PySCF release and is evaluated once per physical channel.
    The short-range endpoint is instead invoked exactly once with both packed
    sources.  This is why a plain Python sequence of existing one-channel
    calls is not an acceptable replacement.

    Args:
        mf_hf : gpu4pyscf.pbc.scf.hf.RHF
            Converged source RHF with a real-Gamma native GDF provider.
        dm0, dz : cupy.ndarray
            Physical-AO source density ``P`` and symmetric Z transition
            density ``D_Z``.

    Kwargs:
        atmlst : sequence of int or None
            Requested atom centers.  ``None`` returns all atoms.

    Returns:
        tuple
            ``(terms, metadata)``.  ``terms`` has device arrays ``de_j``,
            ``de_fock`` and their source/metric/Ewald term ledgers.
    '''
    provider = mf_hf.with_df
    _check_gamma_gdf(provider, dm0, dz)
    kernel = get_short_range_kernel(provider)
    if provider.auxcell is None:
        provider.build(j_only=False)
    cell = provider.cell
    profile = {}
    total_start = time.perf_counter()
    intopt = SRInt3c2eOpt(
        cell, provider.auxcell, float(_guess_omega(cell)), None).build()
    profile['optimizer_seconds'] = time.perf_counter() - total_start
    value, metadata = _direct_adjoint_derivative_channels(
        intopt, dm0, dz, mf_hf.exxdiv, _reciprocal_options(provider), kernel)
    metadata['profile']['optimizer_seconds'] = profile['optimizer_seconds']
    atoms = _atmlst(cell, atmlst)
    terms = {
        name: cp.asarray(component)[atoms]
        for name, component in value.items()
    }
    return terms, metadata


def _check_gamma_gdf(provider, dm0, dz):
    '''Reject representations outside the real Gamma native kernel contract.'''
    kpts = np.asarray(provider.kpts).reshape(-1, 3)
    if len(kpts) != 1 or np.max(np.abs(kpts)) > 1e-12:
        raise NotImplementedError('native signed GDF response requires Gamma')
    if provider.cell.dimension != 3:
        raise NotImplementedError('native signed GDF response requires 3D PBC')
    for name, value in (('dm0', dm0), ('dz', dz)):
        value = cp.asarray(value)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError(f'{name} must be a square Gamma AO density')
        imaginary = float(cp.max(cp.abs(value.imag)).get())
        asymmetry = float(cp.max(cp.abs(value - value.conj().T)).get())
        if imaginary > 1e-10:
            raise NotImplementedError(
                f'native signed GDF response requires real {name}')
        if asymmetry > 1e-9:
            raise ValueError(f'{name} must be Hermitian')


def _reciprocal_options(provider):
    '''Read and validate the provider-owned reciprocal workspace controls.'''
    options = {
        'memory_fraction': float(provider.reciprocal_memory_fraction),
        'minimum_reserve_bytes': int(provider.reciprocal_minimum_reserve_bytes),
        'block_alignment': int(provider.reciprocal_block_alignment),
        'maximum_block_size': provider.reciprocal_max_block_size,
        'gamma_conjugate_symmetry': bool(
            provider.reciprocal_gamma_conjugate_symmetry),
        'pair_cache': str(provider._experimental_reciprocal_pair_cache),
        'pair_cache_host': str(
            provider._experimental_reciprocal_pair_cache_host),
        'pair_cache_host_memory_fraction': float(
            provider._experimental_reciprocal_pair_cache_host_memory_fraction),
        'pair_cache_host_minimum_reserve_bytes': int(
            provider._experimental_reciprocal_pair_cache_host_minimum_reserve_bytes),
        'pair_cache_host_max_bytes': (
            provider._experimental_reciprocal_pair_cache_host_max_bytes),
        'pair_cache_fallback': bool(
            provider._experimental_reciprocal_pair_cache_fallback),
    }
    if options['maximum_block_size'] is not None:
        options['maximum_block_size'] = int(options['maximum_block_size'])
    if options['pair_cache_host_max_bytes'] is not None:
        options['pair_cache_host_max_bytes'] = int(
            options['pair_cache_host_max_bytes'])
    if options['pair_cache'] not in ('recompute', 'cache', 'auto'):
        raise ValueError(
            'reciprocal_pair_cache must be recompute, cache, or auto')
    if options['pair_cache_host'] not in ('pageable', 'mapped'):
        raise ValueError(
            'reciprocal_pair_cache_host must be pageable or mapped')
    return options


def _adjoint_options(provider):
    '''Read provider-owned real-Gamma packed-adjoint execution controls.'''
    options = {
        'auxiliary_block_size': int(provider.adjoint_auxiliary_block_size),
        'storage': str(provider.adjoint_storage),
        'host_cache': str(provider.adjoint_host_cache),
        'host_memory_fraction': float(
            provider.adjoint_host_memory_fraction),
        'host_minimum_reserve_bytes': int(
            provider.adjoint_host_minimum_reserve_bytes),
        'host_max_bytes': provider.adjoint_host_max_bytes,
        'host_cache_fallback': bool(provider.adjoint_host_cache_fallback),
    }
    if options['host_max_bytes'] is not None:
        options['host_max_bytes'] = int(options['host_max_bytes'])
    return options


def _is_gpu_out_of_memory(error):
    '''Identify CUDA allocation failures that can be safely rebached.'''
    if isinstance(error, cp.cuda.memory.OutOfMemoryError):
        return True
    message = str(error).lower()
    return (
        error.__class__.__name__ == 'CUDARuntimeError'
        and ('out of memory' in message
             or 'cudaerrormemoryallocation' in message)
    )


def _resolve_adjoint_backend(adjoint_backend):
    '''Resolve the real-Gamma primitive-source representation.'''
    requested = str(adjoint_backend).lower()
    if requested not in ('auto', 'packed', 'mo_factorized', 'mo_direct'):
        raise ValueError(
            'adjoint_backend must be auto, packed, mo_factorized, or '
            'mo_direct')
    if requested == 'auto':
        return requested, 'mo_direct', None
    return requested, requested, None


def _sorted_rhf_mo_factors(
        cell, occupied_coefficients, response_orbitals,
        backend='mo_factorized'):
    r'''Map physical ``C_o`` and ``U=C_v Z`` to native sorted real AOs.

    The factorized source is exact only when these factors define the same
    physical densities passed to the rest of the derivative,

    ``P=2 C_o C_o^T`` and ``D_Z=U C_o^T+C_o U^T``.

    Missing or genuinely complex factors are rejected for an explicitly
    selected factorized backend.  There is intentionally no silent fallback
    to the packed oracle under an explicit request; only ``auto`` may record
    and take that exact representation fallback.
    '''
    if occupied_coefficients is None or response_orbitals is None:
        raise ValueError(
            f"adjoint_backend='{backend}' requires "
            'occupied_coefficients and response_orbitals')
    occupied = cp.asarray(occupied_coefficients)
    response = cp.asarray(response_orbitals)
    if occupied.ndim != 2 or response.ndim != 2:
        raise ValueError('occupied_coefficients and response_orbitals must be 2D')
    if occupied.shape != response.shape:
        raise ValueError(
            'occupied_coefficients and response_orbitals must have the same '
            f'shape, got {occupied.shape} and {response.shape}')
    physical_nao = int(cell.cell.nao_nr())
    if occupied.shape[0] != physical_nao:
        raise ValueError(
            'MO factors must use the physical AO dimension '
            f'{physical_nao}, got {occupied.shape[0]}')
    real_factors = []
    for name, factor in (
            ('occupied_coefficients', occupied),
            ('response_orbitals', response)):
        if cp.iscomplexobj(factor):
            imaginary = float(cp.max(cp.abs(factor.imag)).get())
            if imaginary > 1.e-10:
                raise NotImplementedError(
                    f"adjoint_backend='{backend}' requires real Gamma "
                    f'{name}; maximum imaginary component is {imaginary:.3e}')
            factor = factor.real
        real_factors.append(cp.asarray(factor, dtype=cp.float64))
    return tuple(
        cell.apply_C_dot(factor, axis=0) for factor in real_factors)


def _release_cached_gpu_blocks(_start, _stop, _error):
    '''Return failed transaction buffers to CuPy before retrying the interval.'''
    cp.get_default_memory_pool().free_all_blocks()


def _finish_profile_stage(profile, name, start):
    '''Synchronize one validation profile stage and return its elapsed time.

    Native libpbc and CuPy launches are asynchronous with respect to the
    host.  The explicit synchronization is deliberately confined to the
    diagnostic metadata path of the complete GDF derivative: it establishes
    non-overlapping stage timings but never changes a tensor, approximation,
    or scheduler decision.
    '''
    cp.cuda.Stream.null.synchronize()
    profile[name] = time.perf_counter() - start
    return time.perf_counter()


class _LongRangeDetailProfiler:
    '''Collect opt-in LR derivative GPU events without per-stage barriers.

    The reciprocal derivative runs many small G-vector transactions.  A host
    synchronization around every operation would make the diagnostic change
    the launch schedule it is intended to measure.  Instead, this helper
    records pairs of CUDA events for each logical operation and resolves all
    pairs at the end of the traversal.  Host/device atom reduction is timed
    only after that resolution, so its blocking transfers do not double count
    queued Fourier or contraction work.

    This object is constructed only for ``profile_detail=True``.  The normal
    path therefore creates no events and adds no synchronization.
    '''

    _GPU_CATEGORIES = (
        'ao_pair_ft',
        'auxiliary_ft',
        'source_contraction',
        'metric_contraction',
        'atom_reduction',
    )

    def __init__(self):
        cp.cuda.Stream.null.synchronize()
        self._wall_start = time.perf_counter()
        self._events = {name: [] for name in self._GPU_CATEGORIES}
        self._gpu_seconds = None
        self._atom_reduction_host_seconds = 0.

    def begin(self, category):
        '''Record the start event for one queued GPU operation group.'''
        if category not in self._events:
            raise KeyError(f'unknown long-range profile category {category!r}')
        start = cp.cuda.Event()
        start.record()
        return start

    def end(self, category, start):
        '''Record an end event paired with ``start`` in ``category``.'''
        stop = cp.cuda.Event()
        stop.record()
        self._events[category].append((start, stop))

    def resolve_gpu(self):
        '''Synchronize once and convert every queued event pair to seconds.'''
        if self._gpu_seconds is not None:
            return
        cp.cuda.Stream.null.synchronize()
        self._gpu_seconds = {
            name: sum(
                cp.cuda.get_elapsed_time(start, stop)
                for start, stop in events) * 1.e-3
            for name, events in self._events.items()
        }

    def add_atom_reduction_host_seconds(self, seconds):
        '''Add D2H/grouping wall time after queued GPU work has completed.'''
        self._atom_reduction_host_seconds += float(seconds)

    def finish(self, block_count):
        '''Return the flat response-profile fields for the completed LR pass.'''
        self.resolve_gpu()
        seconds = dict(self._gpu_seconds)
        seconds['atom_reduction'] += self._atom_reduction_host_seconds
        wall_seconds = time.perf_counter() - self._wall_start
        accounted = sum(seconds.values())
        allocation_other = max(0., wall_seconds - accounted)
        detail_total = accounted + allocation_other
        return {
            'long_range_ao_pair_ft_seconds': seconds['ao_pair_ft'],
            'long_range_auxiliary_ft_seconds': seconds['auxiliary_ft'],
            'long_range_source_contraction_seconds': (
                seconds['source_contraction']),
            'long_range_metric_contraction_seconds': (
                seconds['metric_contraction']),
            'long_range_atom_reduction_seconds': seconds['atom_reduction'],
            'long_range_allocation_other_seconds': allocation_other,
            'long_range_detail_accounted_seconds': accounted,
            'long_range_detail_total_seconds': detail_total,
            'long_range_detail_wall_seconds': wall_seconds,
            'long_range_detail_block_count': int(block_count),
        }


def _make_state(cell, density, coulomb_weight, exchange_weight):
    '''Map one physical signed density into GPU4PySCF's sorted AO basis.'''
    density = cp.asarray(density.real, dtype=cp.float64)
    factor_l, factor_r = factorize_dm(density, hermi=1)
    factor_l = cell.apply_C_dot(factor_l, axis=0)
    if factor_r is None:
        factor_r = factor_l
    else:
        factor_r = cell.apply_C_dot(factor_r, axis=0)
    if factor_l.dtype != np.float64 or factor_r.dtype != np.float64:
        raise NotImplementedError('native Gamma signed factors must be real')
    return _SignedState(
        coulomb_weight, exchange_weight, factor_l, factor_r, density,
        factor_l @ factor_r.T)


def _signed_derivative(intopt, states, exxdiv, reciprocal_options):
    '''Apply the complete signed GDF nuclear derivative to prepared states.'''
    profile = {}
    total_start = time.perf_counter()
    stage_start = total_start
    cell = intopt.cell
    auxcell = intopt.auxcell
    nao = _state_nao(states)
    pair_addresses = intopt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    naux = int(auxcell.ao_loc[-1])
    batch_size, aux_block, three_center_plan = _work_blocks(
        states, len(pair_addresses), naux, len(intopt.bvkmesh_Ls))

    _build_three_center_occupied(
        intopt, states, pair_addresses, i_addr, j_addr,
        batch_size=batch_size, aux_block=aux_block)
    stage_start = _finish_profile_stage(
        profile, 'three_center_short_range_seconds', stage_start)

    auxcell.rcut = _estimate_sr_2c2e_rcut(
        auxcell, intopt.omega, auxcell.precision * 1e-6)
    int2opt = Int2c2eOpt(auxcell)
    metric = int2opt.int2c2e(sort_output=False)
    reciprocal = _add_zero_order_reciprocal(
        intopt, states, metric, i_addr, j_addr, reciprocal_options)
    stage_start = _finish_profile_stage(
        profile, 'three_center_reciprocal_seconds', stage_start)
    metric_inverse = _metric_inverse(auxcell, metric)
    del metric
    solve_plans = []
    for state in states:
        solve_plans.append(_solve_three_center_inplace(
            metric_inverse, state.three_center_oo))
        state.aux_vector = state.three_center_oo.trace(axis1=1, axis2=2)
    del metric_inverse

    metric_source = _metric_source(states, naux)
    stage_start = _finish_profile_stage(
        profile, 'metric_solve_and_adjoint_seconds', stage_start)
    metric_derivative = np.asarray(
        -int2opt.energy_ip1_per_atom(metric_source), dtype=float)
    stage_start = _finish_profile_stage(
        profile, 'metric_derivative_seconds', stage_start)
    long_range, derivative_execution, long_components = (
        _long_range_derivative(
            intopt, reciprocal, states, metric_source, nao, naux,
            return_components=True))
    stage_start = _finish_profile_stage(
        profile, 'long_range_derivative_seconds', stage_start)
    short_range, short_components = _short_range_derivative(
        intopt, states, pair_addresses, batch_size, aux_block, nao, naux,
        return_components=True)
    stage_start = _finish_profile_stage(
        profile, 'short_range_derivative_seconds', stage_start)
    ewald = np.asarray(
        ewald_signed_derivative(cell, states, exxdiv), dtype=float)
    stage_end = _finish_profile_stage(
        profile, 'ewald_derivative_seconds', stage_start)
    profile['total_seconds'] = stage_end - total_start
    term_components = {
        'source': long_components['source'] + short_components['source'],
        'metric': (metric_derivative + long_components['metric']
                   + short_components['metric']),
        'ewald': ewald,
    }
    # Preserve two independently assembled views of the same result.  The
    # direct accumulator follows the physical kernel stages, whereas
    # ``component_sum`` follows the public source/metric/Ewald ledger.  Their
    # residual therefore detects a bookkeeping regression instead of
    # comparing one expression with itself.
    value = metric_derivative + long_range + short_range + ewald
    component_sum = sum(term_components.values())
    metadata = {
        'reciprocal_plan': reciprocal.plan_metadata(),
        'reciprocal_zero_order': reciprocal.zero_order_execution.as_dict(),
        'reciprocal_derivative': derivative_execution.as_dict(),
        'three_center_workspace_plan': three_center_plan,
        'metric_solve_workspace_plans': solve_plans,
        'profile': profile,
        'term_components': term_components,
        'term_component_residual_max_abs': float(np.max(np.abs(
            np.asarray(value, dtype=float) - component_sum))),
    }
    return np.asarray(value, dtype=float), metadata


def _direct_adjoint_derivative(
        intopt, reference_density, response_density, exxdiv,
        reciprocal_options, include_target_j=True, response_sign=-1.,
        profile_detail=False, adjoint_options=None,
        occupied_coefficients=None, response_orbitals=None,
        adjoint_backend='auto'):
    r'''Apply the fused target-J/Z term through one primitive reverse adjoint.

    ``_signed_derivative`` is retained as the polarization reference route for
    the separately exposed quadratic helpers above.  This production path
    instead computes the same quantity from one metric-solved primitive
    tensor ``W=M^{-1}B`` and the direct cross-density source:

    .. math::

       E_J[P]^\lambda-
       \{Q[P+D_Z]^\lambda-Q[P-D_Z]^\lambda\}.

    The long-range, short-range, metric, and Ewald kernels are the same as
    the signed route.  Only their source changes, so no numerical derivative
    or changed DF approximation is introduced.

    Args:
        intopt : gpu4pyscf.pbc.df.int3c2e.SRInt3c2eOpt
            Built real-Gamma three-center primitive optimizer.
        reference_density, response_density : cupy.ndarray
            Physical-AO source density ``P`` and Z transition density
            ``D_Z``.
        exxdiv : str or None
            Source exchange divergence convention.
        reciprocal_options : dict
            Provider-owned bounded reciprocal workspace policy.
        adjoint_options : dict or None
            Provider-owned real-Gamma source block, host-cache and fallback
            policy.  ``None`` retains the low-level two-pass defaults used by
            algebra tests and internal oracles.
        occupied_coefficients, response_orbitals : cupy.ndarray or None
            Physical-AO ``C_o`` and ``U=C_v Z`` factors.  They are required
            by explicit ``mo_factorized``/``mo_direct`` and by production
            ``auto`` when it resolves to ``mo_direct``.  They are mapped to
            the same sorted AO basis as the native three-center tensor.
        adjoint_backend : {'auto', 'packed', 'mo_factorized', 'mo_direct'}
            Primitive-source representation.  ``auto`` selects exact
            ``mo_direct`` for the supported closed-shell real-Gamma factors
            and falls back to the established density-packed oracle only
            when those representation inputs are unavailable or genuinely
            complex.  ``mo_factorized`` changes the adjoint algebra but
            retains the packed zero-order ``B/W`` tensor.  ``mo_direct`` also
            projects zero-order and derivative transactions directly through
            ``C`` and ``U``, so no persistent packed AO-pair tensor is built.
        include_target_j : bool
            Include the target ``E_J[P]^lambda`` derivative.  False selects
            the source-Fock polarization alone for an MG-owned target J.
        response_sign : {+1., -1.}
            Sign of ``Q[P+D_Z]^lambda-Q[P-D_Z]^lambda``.
        profile_detail : bool
            Split backend-specific zero-order, metric/adjoint and derivative
            wall times.  Packed routes additionally record packed-adjoint
            pass timings.  This adds synchronization and CUDA events for
            profiling only; the mathematical and memory schedulers are
            unchanged.

    Returns:
        tuple
            All-atom Ha/Bohr derivative and resolved scheduler/adjoint
            metadata.
    '''
    profile = {} if profile_detail else None
    adjoint_options = {} if adjoint_options is None else dict(adjoint_options)
    backend_requested, backend_resolved, backend_fallback_reason = (
        _resolve_adjoint_backend(adjoint_backend))
    response_sign = float(response_sign)
    if response_sign not in (-1., 1.):
        raise ValueError('response_sign must be +1 or -1')
    if backend_resolved == 'mo_direct':
        try:
            sorted_mo_factors = _sorted_rhf_mo_factors(
                intopt.cell, occupied_coefficients, response_orbitals,
                backend='mo_direct')
        except (ValueError, NotImplementedError) as error:
            if backend_requested != 'auto':
                raise
            # ``auto`` may be reached by an internal caller that does not
            # provide the closed-shell real-Gamma MO factors, or by a Gamma
            # calculation whose orbitals are genuinely complex.  Those are
            # representation limitations rather than changed physics, so the
            # established density-packed oracle remains the exact fallback.
            backend_resolved = 'packed'
            backend_fallback_reason = f'{type(error).__name__}: {error}'
        else:
            return _direct_rhf_mo_adjoint_derivative(
                intopt, reference_density, response_density, exxdiv,
                reciprocal_options, include_target_j=include_target_j,
                response_sign=response_sign, profile_detail=profile_detail,
                sorted_mo_factors=sorted_mo_factors,
                backend_requested=backend_requested,
                backend_fallback_reason=backend_fallback_reason)
    total_start = time.perf_counter() if profile_detail else None
    stage_start = total_start
    cell = intopt.cell
    auxcell = intopt.auxcell
    reference_state = _make_state(cell, reference_density, 0., 0.)
    response_state = _make_state(cell, response_density, 0., 0.)
    nao = int(reference_state.density_sorted.shape[0])
    pair_addresses = intopt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    pair_layout = canonical_pair_layout(pair_addresses, nao)
    naux = int(auxcell.ao_loc[-1])
    batch_size, aux_block, three_center_plan = _work_blocks_packed(
        len(pair_addresses), int(pair_layout['unique_pair_count']), naux,
        len(intopt.bvkmesh_Ls), nao)
    solved = _build_three_center_packed(
        intopt, pair_layout, batch_size=batch_size, aux_block=aux_block)
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'three_center_short_range_seconds', stage_start)

    auxcell.rcut = _estimate_sr_2c2e_rcut(
        auxcell, intopt.omega, auxcell.precision * 1e-6)
    int2opt = Int2c2eOpt(auxcell)
    metric = int2opt.int2c2e(sort_output=False)
    reciprocal = _add_zero_order_reciprocal_packed(
        intopt, solved, pair_layout, metric, i_addr, j_addr,
        reciprocal_options, profile_detail=profile_detail)
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'three_center_reciprocal_seconds', stage_start)
    metric_inverse_start = stage_start
    metric_inverse = _metric_inverse(auxcell, metric)
    del metric
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'metric_inverse_seconds', metric_inverse_start)
    metric_solve_start = stage_start
    metric_solve_plan = _solve_three_center_inplace(metric_inverse, solved)
    del metric_inverse
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'metric_solve_seconds', metric_solve_start)
    adjoint_start = stage_start
    if backend_resolved == 'mo_factorized':
        occupied_sorted, response_sorted = _sorted_rhf_mo_factors(
            cell, occupied_coefficients, response_orbitals)
        # The density-packed block/cache controls describe a different
        # two-pass algorithm.  Keep the independently benchmarked MO defaults
        # (32 auxiliary rows and 8192 AO pairs) until this experimental route
        # receives its own provider-level tuning controls.
        source, metric_source, adjoint_metadata = (
            build_rhf_mo_factorized_packed_adjoint(
                solved, pair_layout, occupied_sorted, response_sorted,
                include_target_j=include_target_j,
                response_sign=response_sign,
                overwrite_solved=True))
    else:
        source, metric_source, adjoint_metadata = (
            build_rhf_packed_primitive_adjoint(
                solved, pair_layout, reference_state.density_sorted,
                response_state.density_sorted,
                include_target_j=include_target_j,
                response_sign=response_sign, overwrite_solved=True,
                profile_detail=profile_detail, **adjoint_options))
    adjoint_metadata.update({
        'backend_requested': backend_requested,
        'backend_resolved': backend_resolved,
        'backend_fallback_reason': backend_fallback_reason,
        'gradient_integration': (
            'explicit_experimental' if backend_resolved == 'mo_factorized'
            else 'production_packed'),
    })
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'adjoint_seconds', adjoint_start)
        profile.update({
            f'adjoint_{name}': value
            for name, value in (adjoint_metadata.get('profile') or {}).items()
        })
        profile['metric_solve_and_adjoint_seconds'] = sum(
            profile[name] for name in (
                'metric_inverse_seconds', 'metric_solve_seconds',
                'adjoint_seconds'))
    solved_bytes = int(solved.nbytes)
    del solved
    # Returning ``solved`` to CuPy's pool is enough: the ensuing LR/SR
    # transactions can reuse its workspace.  Do not call
    # ``free_all_blocks()`` here.  A pool-wide ``cudaFree`` synchronizes and
    # deallocates every cached block, which was measured to dominate Water64
    # wall time while changing neither the live-memory peak nor the result.
    # The reciprocal scheduler remains responsible for bounded working sets.

    # Materialize the small per-atom metric contraction on the host before
    # accumulating the remaining components.  CuPy ``+=`` is in-place; using
    # the device result itself as ``value`` would overwrite the independently
    # reported two-center metric term below.
    metric_derivative = np.asarray(
        -int2opt.energy_ip1_per_atom(metric_source), dtype=float)
    value = metric_derivative.copy()
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'metric_derivative_seconds', stage_start)
    long_range, derivative_execution, long_range_components = _long_range_derivative(
        intopt, reciprocal, (), metric_source, nao, naux,
        primitive_source=_PackedPrimitiveSource(source, pair_layout),
        return_components=True,
        detail_profile=profile if profile_detail else None)
    reciprocal_pair_cache = reciprocal.release_pair_cache()
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'long_range_derivative_seconds', stage_start)
        # Reconcile the event-accounted inner traversal with the existing
        # public LR wall stage.  The difference contains allocations, Python
        # scheduling, cache release and timing overhead, so the six reported
        # categories partition ``long_range_derivative_seconds`` whenever
        # ordinary timing noise leaves a non-negative residual.
        long_range_accounted = sum(profile[name] for name in (
            'long_range_ao_pair_ft_seconds',
            'long_range_auxiliary_ft_seconds',
            'long_range_source_contraction_seconds',
            'long_range_metric_contraction_seconds',
            'long_range_atom_reduction_seconds'))
        profile['long_range_detail_accounted_seconds'] = long_range_accounted
        profile['long_range_allocation_other_seconds'] = max(
            0., profile['long_range_derivative_seconds']
            - long_range_accounted)
        profile['long_range_detail_total_seconds'] = (
            long_range_accounted
            + profile['long_range_allocation_other_seconds'])
    value += long_range
    short_range, short_range_components = _short_range_derivative(
        intopt, (), pair_addresses, batch_size, aux_block, nao, naux,
        primitive_source=_PackedPrimitiveSource(source, pair_layout),
        return_components=True)
    value += short_range
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'short_range_derivative_seconds', stage_start)
    ewald = -response_sign * ewald_cross_derivative(
        cell, reference_state.density_physical,
        response_state.density_physical, exxdiv)
    value += ewald
    if profile_detail:
        profile_end = _finish_profile_stage(
            profile, 'ewald_derivative_seconds', stage_start)
        profile['total_seconds'] = profile_end - total_start
    term_components = {
        # ``source`` contracts the primitive three-center derivative ``dB``.
        # ``metric`` contracts ``dM`` and therefore includes the direct
        # two-center term and the auxiliary-center portions of the SR/LR
        # three-center kernels.  These are components of the *fused*
        # target-J plus Z-Fock observable; they do not pretend that target J
        # and source Fock were dispatched as independent native derivatives.
        'source': (long_range_components['source']
                   + short_range_components['source']),
        'metric': (np.asarray(metric_derivative, dtype=float)
                   + long_range_components['metric']
                   + short_range_components['metric']),
        'ewald': np.asarray(ewald, dtype=float),
    }
    component_sum = (
        term_components['source'] + term_components['metric']
        + term_components['ewald'])
    metadata = {
        'reciprocal_plan': reciprocal.plan_metadata(),
        'reciprocal_zero_order': reciprocal.zero_order_execution.as_dict(),
        'reciprocal_derivative': derivative_execution.as_dict(),
        'reciprocal_pair_cache': reciprocal_pair_cache,
        'three_center_workspace_plan': three_center_plan,
        'metric_solve_workspace_plan': metric_solve_plan,
        'adjoint': adjoint_metadata,
        'include_target_j': bool(include_target_j),
        'response_sign': response_sign,
        'profile': profile,
        'solved_three_center_bytes': solved_bytes,
        'dense_solved_three_center_bytes_avoided': int(
            naux * (nao ** 2 - int(pair_layout['unique_pair_count'])) * 8),
        # Two full-rank polarization tensors of this shape are absent.
        'signed_state_full_rank_bytes_avoided': 2 * naux * nao ** 2 * 8,
        'term_components': term_components,
        'term_component_residual_max_abs': float(np.max(np.abs(
            np.asarray(value, dtype=float) - component_sum))),
    }
    return np.asarray(value, dtype=float), metadata


def _direct_rhf_mo_adjoint_derivative(
        intopt, reference_density, response_density, exxdiv,
        reciprocal_options, include_target_j=True, response_sign=-1.,
        profile_detail=False, sorted_mo_factors=None,
        backend_requested='mo_direct',
        backend_fallback_reason=None):
    r'''Evaluate the exact RHF adjoint without persistent packed AO pairs.

    This is the promoted production NR3/NR4 path.  Short-range and
    reciprocal zero-order three-center blocks are projected immediately to
    ``Boo=C.T@B@C`` and ``Bou=C.T@B@U``.  The native primitive metric inverse
    is then applied only to these occupied-space right-hand sides.  Long- and
    short-range derivative kernels consume ``C,U,A,K,Gamma`` directly and
    construct only their transaction-local AO source.

    The packed implementation in :func:`_direct_adjoint_derivative` remains
    the numerical oracle and unsupported-representation fallback.  After the
    Water1/8/20/64 accuracy, scale, OOM, reentry and scanner gates, this
    routine is the supported real-Gamma RHF ``auto`` route as well as the
    explicit ``adjoint_backend='mo_direct'`` route.
    '''
    profile = {} if profile_detail else None
    total_start = time.perf_counter() if profile_detail else None
    stage_start = total_start
    cell = intopt.cell
    auxcell = intopt.auxcell
    if sorted_mo_factors is None or len(sorted_mo_factors) != 2:
        raise ValueError(
            'mo_direct requires two prevalidated sorted MO factors')
    occupied, response = sorted_mo_factors
    reference_physical = cp.asarray(
        cp.asarray(reference_density).real, dtype=cp.float64)
    response_physical = cp.asarray(
        cp.asarray(response_density).real, dtype=cp.float64)
    nao = int(occupied.shape[0])
    naux = int(auxcell.ao_loc[-1])
    pair_addresses = intopt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]

    occupied_projection, mixed_projection, short_range_plan = (
        build_short_range_rhf_mo_projections(
            intopt, occupied, response))
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'three_center_short_range_seconds', stage_start)

    auxcell.rcut = _estimate_sr_2c2e_rcut(
        auxcell, intopt.omega, auxcell.precision * 1.e-6)
    int2opt = Int2c2eOpt(auxcell)
    metric = int2opt.int2c2e(sort_output=False)
    reciprocal = add_reciprocal_rhf_mo_projections(
        intopt, occupied_projection, mixed_projection, metric,
        occupied, response, reciprocal_options)
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'three_center_reciprocal_seconds', stage_start)

    metric_inverse_start = stage_start
    metric_inverse = _metric_inverse(auxcell, metric)
    del metric
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'metric_inverse_seconds', metric_inverse_start)
    metric_solve_start = stage_start
    occupied_solve_plan, mixed_solve_plan = (
        solve_rhf_mo_projections_inplace(
            metric_inverse, occupied_projection, mixed_projection))
    del metric_inverse
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'metric_solve_seconds', metric_solve_start)

    adjoint_start = stage_start
    factors = make_rhf_mo_response_factors(
        occupied, response, occupied_projection, mixed_projection,
        include_target_j=include_target_j, response_sign=response_sign,
        overwrite_projections=True)
    projection_bytes = int(
        occupied_projection.nbytes + mixed_projection.nbytes)
    factor_bytes = int(
        factors.source_factor.nbytes + factors.response_factor.nbytes
        + factors.metric_source.nbytes)
    # O/R values have served their purpose once Gamma is complete.  Their two
    # device buffers are reused in place as K/A rather than allocating a
    # second pair of persistent occupied-space stacks.
    del occupied_projection, mixed_projection
    adjoint_metadata = {
        'route': 'real_gamma_rhf_mo_direct_adjoint',
        'domain': 'real_gamma_closed_shell_rhf',
        'source_storage': 'factorized_mo_transaction_local_ao_source',
        'include_target_j': bool(include_target_j),
        'response_sign': float(response_sign),
        'nao': nao,
        'nocc': int(occupied.shape[1]),
        'naux': naux,
        'projection_storage_bytes_reused_for_factors': projection_bytes,
        'persistent_factor_bytes': factor_bytes,
        'projection_storage_reused_for_factors': True,
        'persistent_packed_source_bytes': 0,
        'production_auto_selected': backend_requested == 'auto',
        'profile': None,
        'backend_requested': backend_requested,
        'backend_resolved': 'mo_direct',
        'backend_fallback_reason': backend_fallback_reason,
        'gradient_integration': (
            'production_mo_direct' if backend_requested == 'auto'
            else 'explicit_validation'),
    }
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'adjoint_seconds', adjoint_start)
        profile['metric_solve_and_adjoint_seconds'] = sum(
            profile[name] for name in (
                'metric_inverse_seconds', 'metric_solve_seconds',
                'adjoint_seconds'))

    metric_source = factors.metric_source
    metric_derivative = np.asarray(
        -int2opt.energy_ip1_per_atom(metric_source), dtype=float)
    value = metric_derivative.copy()
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'metric_derivative_seconds', stage_start)

    long_range, derivative_execution, long_range_components = (
        _long_range_derivative(
            intopt, reciprocal, (), metric_source, nao, naux,
            mo_response_factors=factors, return_components=True,
            detail_profile=profile if profile_detail else None))
    value += long_range
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'long_range_derivative_seconds', stage_start)
        long_range_accounted = sum(profile[name] for name in (
            'long_range_ao_pair_ft_seconds',
            'long_range_auxiliary_ft_seconds',
            'long_range_source_contraction_seconds',
            'long_range_metric_contraction_seconds',
            'long_range_atom_reduction_seconds'))
        profile['long_range_detail_accounted_seconds'] = long_range_accounted
        profile['long_range_allocation_other_seconds'] = max(
            0., profile['long_range_derivative_seconds']
            - long_range_accounted)
        profile['long_range_detail_total_seconds'] = (
            long_range_accounted
            + profile['long_range_allocation_other_seconds'])

    batch_size, auxiliary_block_size, derivative_source_plan = (
        _work_block_rhf_mo_short_range_source(
            len(pair_addresses), naux, nao, occupied.shape[1]))
    short_range, short_range_components = _short_range_derivative(
        intopt, (), pair_addresses, batch_size, auxiliary_block_size,
        nao, naux, mo_response_factors=factors, return_components=True)
    value += short_range
    if profile_detail:
        stage_start = _finish_profile_stage(
            profile, 'short_range_derivative_seconds', stage_start)

    ewald = -response_sign * ewald_cross_derivative(
        cell, reference_physical, response_physical, exxdiv)
    value += ewald
    if profile_detail:
        profile_end = _finish_profile_stage(
            profile, 'ewald_derivative_seconds', stage_start)
        profile['total_seconds'] = profile_end - total_start

    term_components = {
        'source': (long_range_components['source']
                   + short_range_components['source']),
        'metric': (metric_derivative + long_range_components['metric']
                   + short_range_components['metric']),
        'ewald': np.asarray(ewald, dtype=float),
    }
    component_sum = sum(term_components.values())
    reciprocal_pair_cache = _mo_direct_pair_cache_metadata(
        reciprocal_options)
    metadata = {
        'reciprocal_plan': reciprocal.plan_metadata(),
        'reciprocal_zero_order': reciprocal.zero_order_execution.as_dict(),
        'reciprocal_derivative': derivative_execution.as_dict(),
        'reciprocal_pair_cache': reciprocal_pair_cache,
        'three_center_workspace_plan': short_range_plan,
        'short_range_source_workspace_plan': derivative_source_plan,
        'metric_solve_workspace_plans': {
            'occupied_projection': occupied_solve_plan,
            'mixed_projection': mixed_solve_plan,
        },
        'adjoint': adjoint_metadata,
        'include_target_j': bool(include_target_j),
        'response_sign': float(response_sign),
        'profile': profile,
        'solved_three_center_bytes': 0,
        # The direct route intentionally does not construct a canonical
        # unique-pair layout merely to report its size.  Doing so would add a
        # device-to-host copy and NumPy unique pass to the optimized path.
        'raw_derivative_pair_count': len(pair_addresses),
        'dense_solved_three_center_bytes_avoided': int(
            naux * nao ** 2 * np.dtype(np.float64).itemsize),
        'signed_state_full_rank_bytes_avoided': 2 * naux * nao ** 2 * 8,
        'term_components': term_components,
        'term_component_residual_max_abs': float(np.max(np.abs(
            np.asarray(value, dtype=float) - component_sum))),
    }
    return np.asarray(value, dtype=float), metadata


def _mo_direct_pair_cache_metadata(reciprocal_options):
    '''Report why the packed reciprocal host cache is bypassed by MO direct.'''
    return {
        'applicable': False,
        'bypassed': True,
        'requested_mode': str(
            reciprocal_options.get('pair_cache', 'recompute')),
        'resolved_mode': 'not_applicable_mo_direct',
        'reason': (
            'mo_direct consumes the complete transaction-local AO-pair '
            'Fourier matrix; the packed unique-pair host cache is not used'),
    }


def _direct_adjoint_derivative_channels(intopt, reference_density,
                                        response_density, exxdiv,
                                        reciprocal_options, kernel):
    r'''Evaluate physical target-J and Z-Fock channels from one SR traversal.

    ``kernel`` is the explicit two-channel short-range CUDA endpoint returned
    by :func:`two_channel.get_short_range_kernel`.  The zero-order build,
    Cholesky solve and reciprocal-vector schedule are identical to the fused
    direct-adjoint path.  Only the final primitive adjoints remain separate.
    '''
    profile = {}
    total_start = time.perf_counter()
    stage_start = total_start
    cell = intopt.cell
    auxcell = intopt.auxcell
    reference_state = _make_state(cell, reference_density, 0., 0.)
    response_state = _make_state(cell, response_density, 0., 0.)
    nao = int(reference_state.density_sorted.shape[0])
    identity = cp.eye(nao, dtype=cp.float64)
    primitive_state = _SignedState(
        0., 0., identity, identity,
        cp.asarray(reference_density.real, dtype=cp.float64), identity)
    pair_addresses = intopt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    naux = int(auxcell.ao_loc[-1])
    batch_size, aux_block, three_center_plan = _work_blocks(
        (primitive_state,), len(pair_addresses), naux, len(intopt.bvkmesh_Ls))
    _build_three_center_occupied(
        intopt, (primitive_state,), pair_addresses, i_addr, j_addr,
        batch_size=batch_size, aux_block=aux_block)
    stage_start = _finish_profile_stage(
        profile, 'three_center_short_range_seconds', stage_start)

    auxcell.rcut = _estimate_sr_2c2e_rcut(
        auxcell, intopt.omega, auxcell.precision * 1e-6)
    int2opt = Int2c2eOpt(auxcell)
    metric = int2opt.int2c2e(sort_output=False)
    reciprocal = _add_zero_order_reciprocal(
        intopt, (primitive_state,), metric, i_addr, j_addr,
        reciprocal_options)
    stage_start = _finish_profile_stage(
        profile, 'three_center_reciprocal_seconds', stage_start)
    metric_inverse = _metric_inverse(auxcell, metric)
    del metric
    metric_solve_plan = _solve_three_center_inplace(
        metric_inverse, primitive_state.three_center_oo)
    solved = primitive_state.three_center_oo
    del metric_inverse
    target, fock, adjoint_metadata = (
        build_rhf_primitive_adjoint_channels(
            solved, reference_state.density_sorted,
            response_state.density_sorted, response_sign=-1.))
    primitive_state.three_center_oo = None
    solved_bytes = int(solved.nbytes)
    del solved
    stage_start = _finish_profile_stage(
        profile, 'metric_solve_and_adjoint_seconds', stage_start)

    target_metric_derivative = np.asarray(
        -int2opt.energy_ip1_per_atom(target['metric']), dtype=float)
    fock_metric_derivative = np.asarray(
        -int2opt.energy_ip1_per_atom(fock['metric']), dtype=float)
    stage_start = _finish_profile_stage(
        profile, 'metric_derivative_seconds', stage_start)

    target_long_range, target_execution, target_long_components = (
        _long_range_derivative(
            intopt, reciprocal, (), target['metric'], nao, naux,
            primitive_source=target['source'], return_components=True))
    fock_long_range, fock_execution, fock_long_components = (
        _long_range_derivative(
            intopt, reciprocal, (), fock['metric'], nao, naux,
            primitive_source=fock['source'], return_components=True))
    stage_start = _finish_profile_stage(
        profile, 'long_range_derivative_seconds', stage_start)

    (target_short_range, fock_short_range,
     target_short_components, fock_short_components) = (
        _short_range_derivative_channels(
            intopt, pair_addresses, batch_size, aux_block, nao, naux,
            target['source'], fock['source'], kernel))
    stage_start = _finish_profile_stage(
        profile, 'short_range_derivative_seconds', stage_start)
    fock_ewald = ewald_cross_derivative(
        cell, reference_state.density_physical,
        response_state.density_physical, exxdiv)
    stage_end = _finish_profile_stage(
        profile, 'ewald_derivative_seconds', stage_start)
    profile['total_seconds'] = stage_end - total_start

    target_components = {
        'source': (target_long_components['source']
                   + target_short_components['source']),
        'metric': (target_metric_derivative
                   + target_long_components['metric']
                   + target_short_components['metric']),
        'ewald': np.zeros_like(target_metric_derivative),
    }
    fock_components = {
        'source': (fock_long_components['source']
                   + fock_short_components['source']),
        'metric': (fock_metric_derivative
                   + fock_long_components['metric']
                   + fock_short_components['metric']),
        'ewald': np.asarray(fock_ewald, dtype=float),
    }
    target_value = target_metric_derivative + target_long_range + target_short_range
    fock_value = (fock_metric_derivative + fock_long_range + fock_short_range
                  + fock_ewald)
    target_residual = float(np.max(np.abs(
        target_value - sum(target_components.values()))))
    fock_residual = float(np.max(np.abs(
        fock_value - sum(fock_components.values()))))
    metadata = {
        'route': 'direct_real_gamma_two_channel_short_range_adjoint',
        'adjoint': adjoint_metadata,
        'profile': profile,
        'three_center_workspace_plan': three_center_plan,
        'metric_solve_workspace_plan': metric_solve_plan,
        'solved_three_center_bytes': solved_bytes,
        'reciprocal_plan': reciprocal.plan_metadata(),
        'reciprocal_zero_order': reciprocal.zero_order_execution.as_dict(),
        'target_reciprocal_derivative': target_execution.as_dict(),
        'fock_reciprocal_derivative': fock_execution.as_dict(),
        'target_term_component_residual_max_abs': target_residual,
        'fock_term_component_residual_max_abs': fock_residual,
    }
    return {
        'de_j': cp.asarray(target_value),
        'de_fock': cp.asarray(fock_value),
        'de_j_source': cp.asarray(target_components['source']),
        'de_j_metric': cp.asarray(target_components['metric']),
        'de_j_ewald': cp.asarray(target_components['ewald']),
        'de_fock_source': cp.asarray(fock_components['source']),
        'de_fock_metric': cp.asarray(fock_components['metric']),
        'de_fock_ewald': cp.asarray(fock_components['ewald']),
    }, metadata


def _state_nao(states):
    '''Return the common sorted AO dimension of all signed states.'''
    shapes = {tuple(state.density_sorted.shape) for state in states}
    if len(shapes) != 1:
        raise ValueError(f'inconsistent signed-density shapes: {shapes}')
    shape = shapes.pop()
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f'signed density must be square, got {shape}')
    return int(shape[0])


def _work_blocks(states, npair, naux, nbvk):
    '''Choose joint SR evaluator/contraction blocks from free device memory.

    The former planner budgeted the evaluator batch and AO contraction slice
    independently.  Their buffers are live together, so each could consume a
    large fraction of the same reported free memory.  This joint plan first
    reserves the persistent ``B[P,i,j]`` tensors, then assigns disjoint
    budgets to the evaluator and the ``full + temp`` contraction workspaces.
    '''
    available = int(get_avail_mem(exclude_memory_pool=True))
    itemsize = np.dtype(np.float64).itemsize
    persistent = sum(naux * state.rank ** 2 * itemsize for state in states)
    reserve = min(512 * 1024**2, max(0, available // 8))
    usable = max(1, available - persistent - reserve)

    evaluator_bytes_per_aux = max(1, npair * max(1, nbvk) * itemsize)
    batch_budget = int(.35 * usable)
    batch = max(1, min(naux, batch_budget // evaluator_bytes_per_aux))

    nao = _state_nao(states)
    maximum_rank = max(state.rank for state in states)
    contraction_bytes_per_aux = (
        nao * nao + maximum_rank * nao) * itemsize
    contraction_budget = int(.35 * usable)
    block = max(1, min(naux, contraction_budget
                       // max(1, contraction_bytes_per_aux)))
    if block >= 8:
        block = max(8, block // 8 * 8)
    plan = {
        'available_bytes': available,
        'persistent_three_center_bytes': persistent,
        'reserve_bytes': reserve,
        'usable_workspace_bytes': usable,
        'evaluator_budget_bytes': batch_budget,
        'contraction_budget_bytes': contraction_budget,
        'evaluator_aux_batch_size': int(batch),
        'contraction_aux_block_size': int(block),
        'evaluator_estimated_bytes': int(
            batch * evaluator_bytes_per_aux),
        'contraction_estimated_bytes': int(
            block * contraction_bytes_per_aux),
    }
    return int(batch), int(block), plan


def _work_blocks_packed(
        raw_pair_count, unique_pair_count, naux, nbvk, nao):
    '''Plan the direct primitive path with packed persistent AO-pair storage.'''
    available = int(get_avail_mem(exclude_memory_pool=True))
    itemsize = np.dtype(np.float64).itemsize
    persistent = int(naux) * int(unique_pair_count) * itemsize
    reserve = min(512 * 1024**2, max(0, available // 8))
    usable = max(1, available - persistent - reserve)
    evaluator_bytes_per_aux = max(
        1, int(raw_pair_count) * max(1, int(nbvk)) * itemsize)
    batch_budget = int(.35 * usable)
    batch = max(1, min(int(naux), batch_budget // evaluator_bytes_per_aux))
    contraction_budget = int(.35 * usable)
    block = max(1, min(
        int(naux), contraction_budget
        // max(1, int(unique_pair_count) * itemsize)))
    if block >= 8:
        block = max(8, block // 8 * 8)
    source_pair_block = min(
        int(raw_pair_count), _PACKED_SOURCE_PAIR_BLOCK_SIZE)
    return int(batch), int(block), {
        'layout': 'unique_real_gamma_ao_pairs',
        'available_bytes': available,
        'persistent_three_center_bytes': persistent,
        'dense_persistent_bytes_avoided': int(
            naux * (int(nao) ** 2 - int(unique_pair_count)) * itemsize),
        'raw_pair_count': int(raw_pair_count),
        'unique_pair_count': int(unique_pair_count),
        'reserve_bytes': reserve,
        'usable_workspace_bytes': usable,
        'evaluator_budget_bytes': batch_budget,
        'contraction_budget_bytes': contraction_budget,
        'evaluator_aux_batch_size': int(batch),
        'contraction_aux_block_size': int(block),
        'packed_source_reorder': 'blocked_cupy_gather',
        'packed_source_pair_block_size': source_pair_block,
        'packed_source_maximum_gather_bytes': int(
            source_pair_block * block * itemsize),
        'packed_source_full_gather_bytes_avoided': int(
            (int(raw_pair_count) - source_pair_block) * block * itemsize),
    }


def _work_block_rhf_mo_short_range_source(
        raw_pair_count, naux, nao, nocc):
    '''Plan the NR4 raw-pair source independently of zero-order projection.

    The zero-order projection temporary scales mainly as ``nao**2`` per
    auxiliary row.  The SR derivative consumer instead holds
    ``pair_tile*nocc`` values per row while the complete native raw-pair batch
    is live.  Reusing the former block size can therefore oversubscribe large
    occupied spaces even when zero order succeeded.
    '''
    raw_pair_count = int(raw_pair_count)
    naux = int(naux)
    nao = int(nao)
    nocc = int(nocc)
    if min(raw_pair_count, naux, nao, nocc) < 1:
        raise ValueError(
            'raw-pair, auxiliary, AO, and occupied dimensions must be '
            'positive')
    available = int(get_avail_mem(exclude_memory_pool=True))
    itemsize = np.dtype(np.float64).itemsize
    reserve = min(512 * 1024**2, max(0, available // 8))
    usable = max(0, available - reserve)
    packed_bytes_per_aux = raw_pair_count * itemsize
    pair_block = min(raw_pair_count, DEFAULT_RAW_PAIR_BLOCK_SIZE)
    bytes_per_aux = (
        nao * nocc + pair_block * (nocc + 1)) * itemsize
    minimum_required = packed_bytes_per_aux + bytes_per_aux
    if usable < minimum_required:
        raise MemoryError(
            'insufficient GPU memory for one mo_direct Gamma GDF '
            'short-range source row: '
            f'usable={usable} bytes, required={minimum_required} bytes '
            f'(raw_source={packed_bytes_per_aux}, '
            f'factor_contraction={bytes_per_aux}), '
            f'free={available} bytes, reserve={reserve} bytes')
    packed_budget = max(packed_bytes_per_aux, int(.35 * usable))
    batch = max(
        1, min(naux, packed_budget // packed_bytes_per_aux))
    factor_budget = max(bytes_per_aux, int(.20 * usable))
    block = max(1, min(naux, factor_budget // bytes_per_aux))
    # Avoid very large factor contractions even on high-memory devices.  The
    # native raw-pair batch has its own budget above; both final tails are exact.
    block = min(block, 64)
    if block >= 8:
        block = max(8, block // 8 * 8)
    return int(batch), int(block), {
        'layout': 'rhf_mo_raw_pair_transaction',
        'available_bytes': available,
        'reserve_bytes': reserve,
        'usable_workspace_bytes': usable,
        'minimum_raw_source_row_bytes': packed_bytes_per_aux,
        'minimum_factor_contraction_row_bytes': bytes_per_aux,
        'minimum_combined_row_bytes': minimum_required,
        'packed_buffer_budget_bytes': packed_budget,
        'factor_contraction_budget_bytes': factor_budget,
        'raw_pair_count': raw_pair_count,
        'auxiliary_batch_size': int(batch),
        'estimated_packed_buffer_bytes': int(batch * packed_bytes_per_aux),
        'raw_pair_block_size': pair_block,
        'ao_count': int(nao),
        'occupied_count': int(nocc),
        'auxiliary_block_size': int(block),
        'estimated_factor_contraction_bytes': int(block * bytes_per_aux),
    }


def _build_three_center_occupied(intopt, states, pair_addresses, i_addr,
                                 j_addr, batch_size, aux_block):
    '''Build ``B_r,ij`` for both signed states from one SR primitive pass.'''
    nao = states[0].density_sorted.shape[0]
    naux = int(intopt.auxcell.ao_loc[-1])
    evaluator, _, _, aux_offsets = intopt.int3c2e_evaluator(
        aux_batch_size=batch_size, cart=True)
    for state in states:
        state.three_center_oo = cp.empty((naux, state.rank, state.rank))

    full = cp.zeros((nao, nao, aux_block))
    packed = cp.empty((batch_size, len(pair_addresses)))
    maximum_rank = max(state.rank for state in states)
    temp = cp.empty((maximum_rank, nao, aux_block))
    offset = 0
    for batch_id in range(len(aux_offsets) - 1):
        compressed = evaluator(aux_batch_id=batch_id, out=packed)[:, 0, :]
        for p0, p1 in lib.prange(0, compressed.shape[1], aux_block):
            width = p1 - p0
            p2 = offset + width
            three_center = full[:, :, :width]
            three_center[j_addr, i_addr] = three_center[i_addr, j_addr] = (
                compressed[:, p0:p1])
            for state in states:
                work = temp[:state.rank, :, :width]
                contract('pqr,pi->iqr', three_center, state.factor_r, out=work)
                contract(
                    'iqr,qj->rij', work, state.factor_l,
                    out=state.three_center_oo[offset:p2])
            offset = p2


def _build_three_center_packed(
        intopt, layout, batch_size, aux_block):
    '''Build primitive ``B[P,ij]`` directly in unique Gamma-pair storage.'''
    naux = int(intopt.auxcell.ao_loc[-1])
    npair = int(layout['unique_pair_count'])
    first = cp.asarray(layout['first'], dtype=cp.int64)
    evaluator, _, _, aux_offsets = intopt.int3c2e_evaluator(
        aux_batch_size=batch_size, cart=True)
    output = cp.empty((naux, npair), dtype=cp.float64)
    raw = cp.empty((batch_size, int(layout['raw_pair_count'])))
    offset = 0
    for batch_id in range(len(aux_offsets) - 1):
        compressed = evaluator(aux_batch_id=batch_id, out=raw)[:, 0, :]
        for p0, p1 in lib.prange(0, compressed.shape[1], aux_block):
            width = p1 - p0
            p2 = offset + width
            output[offset:p2] = compressed[:, p0:p1].T[:, first]
            offset = p2
    if offset != naux:
        raise RuntimeError(
            f'packed three-center build produced {offset} of {naux} rows')
    return output


def _solve_three_center_inplace(metric_inverse, three_center):
    r'''Overwrite ``B[P,ij]`` with ``M^{-1}B[P,ij]`` in AO-pair blocks.

    The auxiliary metric couples all ``P`` rows but does not couple distinct
    AO-pair columns.  Each column block can therefore be solved into one
    bounded workspace and copied back without retaining a second complete
    ``(naux,nao,nao)`` tensor.
    '''
    naux = int(three_center.shape[0])
    flat = three_center.reshape(naux, -1)
    npair = int(flat.shape[1])
    available = int(get_avail_mem(exclude_memory_pool=True))
    reserve = min(512 * 1024**2, max(0, available // 8))
    budget = max(naux * 8, int(.40 * max(1, available - reserve)))
    block = max(1, min(npair, budget // max(1, naux * 8)))
    if block >= 64:
        block = max(64, block // 64 * 64)
    workspace = cp.empty((naux, block), dtype=three_center.dtype)
    for p0, p1 in lib.prange(0, npair, block):
        width = p1 - p0
        output = workspace[:, :width]
        contract('uv,vk->uk', metric_inverse, flat[:, p0:p1], out=output)
        flat[:, p0:p1] = output
    return {
        'available_bytes': available,
        'reserve_bytes': reserve,
        'budget_bytes': budget,
        'ao_pair_block_size': int(block),
        'block_count': int((npair + block - 1) // block),
        'workspace_bytes': int(workspace.nbytes),
        'full_output_bytes_avoided': int(three_center.nbytes),
    }


class _Reciprocal:
    '''Zero-order reciprocal data reused by the metric and derivative passes.'''

    def __init__(self, ft_opt, evaluator, vectors, coulomb, block_size,
                 memory_plan, zero_order_execution, full_grid_count,
                 gamma_conjugate_symmetry, pair_cache=None,
                 pair_cache_plan=None, profile_detail=False):
        self.ft_opt = ft_opt
        self.evaluator = evaluator
        self.vectors = vectors
        self.coulomb = coulomb
        self.block_size = int(block_size)
        self.memory_plan = memory_plan
        self.zero_order_execution = zero_order_execution
        self.full_grid_count = int(full_grid_count)
        self.gamma_conjugate_symmetry = bool(gamma_conjugate_symmetry)
        self.pair_cache = pair_cache
        self.pair_cache_plan = pair_cache_plan
        self.profile_detail = bool(profile_detail)

    def plan_metadata(self):
        '''Return the memory plan plus the exact reciprocal reduction scope.'''
        metadata = self.memory_plan.as_dict()
        metadata.update({
            'full_grid_count': self.full_grid_count,
            'evaluated_grid_count': len(self.vectors),
            'gamma_conjugate_symmetry': self.gamma_conjugate_symmetry,
        })
        return metadata

    def selected_pair_block(self, p0, p1):
        '''Return cached ``FT[ij,G]`` for one derivative transaction.'''
        if self.pair_cache is None:
            return None
        start = time.perf_counter()
        block = asarray(self.pair_cache[2*p0:2*p1])
        if self.profile_detail:
            cp.cuda.Stream.null.synchronize()
            self.pair_cache_plan['host_read_seconds'] += (
                time.perf_counter() - start)
        self.pair_cache_plan['host_read_bytes'] += int(block.nbytes)
        self.pair_cache_plan['hit_count'] += 1
        return block.T

    def release_pair_cache(self):
        '''Release the zero-order Fourier cache after the LR derivative.'''
        plan = self.pair_cache_plan
        if plan is None:
            return None
        start = time.perf_counter()
        self.pair_cache = None
        plan['release_seconds'] = time.perf_counter() - start
        plan['released'] = True
        return plan


def _add_zero_order_reciprocal(intopt, states, metric, i_addr, j_addr,
                               reciprocal_options):
    '''Add LR three-centre/metric values and retain the derivative workspace.'''
    cell = intopt.cell
    auxcell = intopt.auxcell
    mesh = intopt.mesh
    ft_opt = ft_ao.FTOpt.from_intopt(intopt)
    evaluator = ft_opt.ft_evaluator(
        compressing=False, cart=True, original_ao_order=False)[0]
    vectors = asarray(auxcell.get_Gv(mesh))
    coulomb = asarray(_weighted_coulG_kpts(
        cell, mesh, 0., intopt.omega))
    if coulomb.ndim == 2:
        coulomb = coulomb[0]
    if coulomb.ndim != 1 or coulomb.size != len(vectors):
        raise ValueError('Gamma GDF Coulomb weights do not match the grid')
    full_grid_count = len(vectors)
    gamma_symmetry = bool(reciprocal_options['gamma_conjugate_symmetry'])
    if gamma_symmetry:
        indices, multiplicities = gamma_conjugate_reciprocal_indices(mesh)
        vectors = vectors[indices]
        coulomb = coulomb[indices] * asarray(multiplicities)

    nao = states[0].density_sorted.shape[0]
    naux = int(auxcell.ao_loc[-1])
    nocc = max(state.rank for state in states)
    memory_plan = plan_gamma_reciprocal_blocks(
        len(vectors), nao, naux, nocc,
        free_device_bytes=int(get_avail_mem(exclude_memory_pool=True)),
        **{name: reciprocal_options[name] for name in (
            'memory_fraction', 'minimum_reserve_bytes', 'block_alignment',
            'maximum_block_size')})

    def worker(p0, p1):
        ngrid = p1 - p0
        aux_g = ft_ao.ft_ao(auxcell, vectors[p0:p1]).T
        weighted = aux_g * coulomb[p0:p1]
        pair_g = evaluator(vectors[p0:p1]).view(np.float64).reshape(
            nao, nao, ngrid * 2)
        pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]
        occupied = []
        for state in states:
            temp = cp.empty((state.rank, nao, ngrid * 2))
            occ = cp.empty((state.rank, state.rank, ngrid * 2))
            contract('pqG,pi->iqG', pair_g, state.factor_r, out=temp)
            contract('iqG,qj->ijG', temp, state.factor_l, out=occ)
            occupied.append(occ)

        return aux_g, weighted, occupied

    def accept_result(_p0, _p1, value):
        aux_g, weighted, occupied = value
        contract('iG,jG->ij', aux_g.view(np.float64), weighted.view(np.float64),
                 beta=1, out=metric)
        for state, occ in zip(states, occupied):
            contract('rG,ijG->rij', weighted.view(np.float64), occ,
                     beta=1, out=state.three_center_oo)

    execution = execute_gamma_reciprocal_blocks(
        len(vectors), memory_plan.initial_block_size, worker, accept_result,
        phase='zero_order', is_oom=_is_gpu_out_of_memory,
        block_alignment=memory_plan.block_alignment,
        on_oom=_release_cached_gpu_blocks)
    return _Reciprocal(
        ft_opt, evaluator, vectors, coulomb, execution.final_block_size,
        memory_plan, execution, full_grid_count, gamma_symmetry)


def _add_zero_order_reciprocal_packed(
        intopt, three_center, layout, metric, i_addr, j_addr,
        reciprocal_options, profile_detail=False):
    '''Add LR values to packed primitive B and retain derivative evaluators.'''
    cell = intopt.cell
    auxcell = intopt.auxcell
    mesh = intopt.mesh
    ft_opt = ft_ao.FTOpt.from_intopt(intopt)
    evaluator = ft_opt.ft_evaluator(
        compressing=False, cart=True, original_ao_order=False)[0]
    vectors = asarray(auxcell.get_Gv(mesh))
    coulomb = asarray(_weighted_coulG_kpts(
        cell, mesh, 0., intopt.omega))
    if coulomb.ndim == 2:
        coulomb = coulomb[0]
    if coulomb.ndim != 1 or coulomb.size != len(vectors):
        raise ValueError('Gamma GDF Coulomb weights do not match the grid')
    full_grid_count = len(vectors)
    gamma_symmetry = bool(reciprocal_options['gamma_conjugate_symmetry'])
    if gamma_symmetry:
        indices, multiplicities = gamma_conjugate_reciprocal_indices(mesh)
        vectors = vectors[indices]
        coulomb = coulomb[indices] * asarray(multiplicities)

    nao = int(layout['nao'])
    naux = int(auxcell.ao_loc[-1])
    unique_address = cp.asarray(layout['address'], dtype=cp.int64)
    memory_plan = plan_gamma_reciprocal_blocks(
        len(vectors), nao, naux, nao,
        free_device_bytes=int(get_avail_mem(exclude_memory_pool=True)),
        **{name: reciprocal_options[name] for name in (
            'memory_fraction', 'minimum_reserve_bytes', 'block_alignment',
            'maximum_block_size')})

    pair_cache, pair_cache_plan = _make_reciprocal_pair_cache(
        len(vectors), int(layout['unique_pair_count']),
        reciprocal_options, profile_detail)

    def worker(p0, p1):
        ngrid = p1 - p0
        aux_g = ft_ao.ft_ao(auxcell, vectors[p0:p1]).T
        weighted = aux_g * coulomb[p0:p1]
        pair_g = evaluator(vectors[p0:p1]).view(np.float64).reshape(
            nao, nao, ngrid * 2)
        pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]
        selected = pair_g.reshape(nao * nao, ngrid * 2)[unique_address]
        return aux_g, weighted, selected

    def accept_result(_p0, _p1, value):
        aux_g, weighted, selected = value
        contract('iG,jG->ij', aux_g.view(np.float64),
                 weighted.view(np.float64), beta=1., out=metric)
        contract('rG,kG->rk', weighted.view(np.float64), selected,
                 beta=1., out=three_center)
        if pair_cache is not None:
            start = time.perf_counter()
            contiguous = cp.ascontiguousarray(selected.T)
            contiguous.get(
                out=pair_cache[2*_p0:2*_p1], blocking=True)
            if profile_detail:
                pair_cache_plan['host_write_seconds'] += (
                    time.perf_counter() - start)
            pair_cache_plan['host_write_bytes'] += int(contiguous.nbytes)

    execution = execute_gamma_reciprocal_blocks(
        len(vectors), memory_plan.initial_block_size, worker, accept_result,
        phase='zero_order_packed', is_oom=_is_gpu_out_of_memory,
        block_alignment=memory_plan.block_alignment,
        on_oom=_release_cached_gpu_blocks)
    return _Reciprocal(
        ft_opt, evaluator, vectors, coulomb, execution.final_block_size,
        memory_plan, execution, full_grid_count, gamma_symmetry,
        pair_cache=pair_cache, pair_cache_plan=pair_cache_plan,
        profile_detail=profile_detail)


def _make_reciprocal_pair_cache(ngrid, npair, options, profile_detail):
    '''Allocate one bounded host cache for selected real-Gamma pair Fourier data.'''
    requested = options['pair_cache']
    cache_bytes = int(ngrid) * 2 * int(npair) * np.dtype(np.float64).itemsize
    requested_storage = (
        'double_pass' if requested == 'recompute' else 'host_cache')
    plan = plan_adjoint_host_cache(
        cache_bytes,
        memory_fraction=options['pair_cache_host_memory_fraction'],
        minimum_reserve_bytes=(
            options['pair_cache_host_minimum_reserve_bytes']),
        maximum_bytes=options['pair_cache_host_max_bytes'],
        requested_storage=requested_storage,
        fallback=options['pair_cache_fallback'])
    plan.update({
        'requested_mode': requested,
        'resolved_mode': (
            'cache' if plan['resolved_storage'] == 'host_cache'
            else 'recompute'),
        'host_storage': options['pair_cache_host'],
        'grid_count': int(ngrid),
        'unique_pair_count': int(npair),
        'predicted_cache_bytes': cache_bytes,
        'actual_cache_bytes': 0,
        'host_write_bytes': 0,
        'host_read_bytes': 0,
        'host_write_seconds': 0.0,
        'host_read_seconds': 0.0,
        'hit_count': 0,
        'profile_detail': bool(profile_detail),
        'release_seconds': 0.0,
        'released': False,
    })
    if plan['resolved_storage'] != 'host_cache':
        return None, plan
    start = time.perf_counter()
    try:
        shape = (2 * int(ngrid), int(npair))
        if options['pair_cache_host'] == 'mapped':
            cache = empty_mapped(shape, dtype=np.float64)
        else:
            cache = np.empty(shape, dtype=np.float64)
    except (MemoryError, OSError, cp.cuda.memory.OutOfMemoryError,
            cp.cuda.runtime.CUDARuntimeError) as error:
        if not options['pair_cache_fallback']:
            raise
        plan['resolved_storage'] = 'double_pass'
        plan['resolved_mode'] = 'recompute'
        plan['fallback_reason'] = f'{type(error).__name__}: {error}'
        plan['allocation_seconds'] = time.perf_counter() - start
        return None, plan
    plan['allocation_seconds'] = time.perf_counter() - start
    plan['actual_cache_bytes'] = int(cache.nbytes)
    return cache, plan


def _metric_inverse(auxcell, metric):
    '''Return the primitive-basis inverse auxiliary metric on the GPU.'''
    transformed = auxcell.apply_CT_mat_C(metric)
    if auxcell.cell.cart:
        raise NotImplementedError('Cartesian auxiliary metrics are unsupported')
    coefficients = cp.asarray(auxcell.ctr_coeff)
    return coefficients @ cp.linalg.solve(transformed, coefficients.T)


def _metric_source(states, naux):
    '''Build the signed auxiliary-metric derivative source.'''
    source = cp.zeros((naux, naux))
    for state in states:
        source += state.coulomb_weight * (
            state.aux_vector[:, None] * state.aux_vector[None, :])
        contract(
            'rij,sji->rs', state.three_center_oo, state.three_center_oo,
            alpha=-.5 * state.exchange_weight, beta=1., out=source)
    return source


def _long_range_derivative(intopt, reciprocal, states, metric_source, nao,
                           naux, primitive_source=None,
                           mo_response_factors=None,
                           return_components=False, detail_profile=None):
    '''Contract signed or direct-adjoint LR sources in reciprocal blocks.

    ``primitive_source`` is the exact reverse source which contracts directly
    with ``dB``.  ``mo_response_factors`` instead holds the exact
    ``C,U,A,K,Gamma`` representation and materializes only a transaction-local
    AO source.  These two inputs are mutually exclusive.  The signed-state
    branch remains the quadratic reference implementation used by the public
    component helpers.  With ``return_components=True`` the direct-adjoint
    routes also return their separate ``dB`` and ``dM`` contractions.  The G-vector
    traversal remains single-pass; only its final per-atom reductions are
    kept apart for the public fused-term ledger.  ``detail_profile`` is an
    optional response-profile mapping used only by the validation path.  When
    omitted, no CUDA events or additional synchronization are introduced.
    '''
    if primitive_source is not None and mo_response_factors is not None:
        raise ValueError(
            'primitive_source and mo_response_factors are mutually exclusive')
    profiler = (
        _LongRangeDetailProfiler() if detail_profile is not None else None)
    cell = intopt.cell
    auxcell = intopt.auxcell
    ft_opt = reciprocal.ft_opt
    pair_addresses = ft_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    bas_ij_idx, bas_ij_img_idx, shell_offsets = aft_jk._generate_shl_pairs(
        ft_opt)
    aft_envs = ft_opt.aft_envs
    shared_memory = aft_jk._estimate_max_shm_size(cell, (1, 0))
    kernel = libpbc.PBC_ft_aopair_ek_ip1
    ao_total = cp.zeros((cell.natm, 3))
    aux_total = cp.zeros((3, naux))

    def worker(p0, p1):
        if mo_response_factors is not None:
            return _long_range_rhf_mo_block(
                cell, auxcell, mo_response_factors, metric_source,
                reciprocal.vectors[p0:p1], reciprocal.coulomb[p0:p1],
                reciprocal.evaluator, nao, naux, i_addr, j_addr, kernel,
                aft_envs, shell_offsets, shared_memory, bas_ij_idx,
                bas_ij_img_idx, ft_opt.permutation_symmetry,
                detail_profiler=profiler)
        if primitive_source is not None:
            return _long_range_adjoint_block(
                cell, auxcell, primitive_source, metric_source,
                reciprocal.vectors[p0:p1], reciprocal.coulomb[p0:p1],
                reciprocal.evaluator, nao, naux, i_addr, j_addr, kernel,
                aft_envs, shell_offsets, shared_memory, bas_ij_idx,
                bas_ij_img_idx, ft_opt.permutation_symmetry,
                selected_pair_g=reciprocal.selected_pair_block(p0, p1),
                detail_profiler=profiler)
        return _long_range_block(
            cell, auxcell, states, metric_source,
            reciprocal.vectors[p0:p1], reciprocal.coulomb[p0:p1],
            reciprocal.evaluator, nao, naux, i_addr, j_addr, kernel,
            aft_envs, shell_offsets, shared_memory, bas_ij_idx,
            bas_ij_img_idx, ft_opt.permutation_symmetry,
            detail_profiler=profiler)

    def accept_result(_p0, _p1, value):
        event = None if profiler is None else profiler.begin('atom_reduction')
        ao_total[...] += value[0]
        aux_total[...] += value[1]
        if profiler is not None:
            profiler.end('atom_reduction', event)

    execution = execute_gamma_reciprocal_blocks(
        len(reciprocal.vectors), reciprocal.block_size, worker, accept_result,
        phase='long_range_derivative', is_oom=_is_gpu_out_of_memory,
        block_alignment=reciprocal.memory_plan.block_alignment,
        on_oom=_release_cached_gpu_blocks)

    if profiler is not None:
        # Resolve all queued Fourier/contraction events before timing the
        # blocking D2H copies below; otherwise those copies would include and
        # double count the complete transaction queue.
        profiler.resolve_gpu()
        reduction_start = time.perf_counter()
    dimensions = auxcell.ao_loc[1:] - auxcell.ao_loc[:-1]
    atom_for_aux = np.repeat(auxcell._bas[:, ATOM_OF], dimensions)
    grouped = groupby(atom_for_aux, aux_total.T.get(), op='sum')
    source_result = ao_total.get() * 2
    metric_result = np.zeros_like(source_result)
    if len(grouped) < cell.natm:
        metric_result[np.unique(atom_for_aux)] += grouped
    else:
        metric_result += grouped
    result = source_result + metric_result
    if profiler is not None:
        profiler.add_atom_reduction_host_seconds(
            time.perf_counter() - reduction_start)
        detail_profile.update(profiler.finish(execution.block_count))
    if return_components:
        return result, execution, {
            'source': source_result,
            'metric': metric_result,
        }
    return result, execution


def _long_range_rhf_mo_block(
        cell, auxcell, factors, metric_source, vectors, coulomb, evaluator,
        nao, naux, i_addr, j_addr, kernel, aft_envs, shell_offsets,
        shared_memory, bas_ij_idx, bas_ij_img_idx, permutation_symmetry,
        detail_profiler=None):
    r'''Consume one reciprocal block through exact RHF MO source factors.

    ``F_G`` is generated by the same native AO-pair Fourier evaluator as the
    packed oracle.  The helper contracts ``H_G=C.T F_G C`` and
    ``X_G=C.T F_G U`` directly with ``A_P`` and ``K_P``; it also constructs
    the transaction-local AO source required by
    ``PBC_ft_aopair_ek_ip1``.  No persistent ``(naux,npair)`` source exists.
    '''
    ngrid = len(vectors)
    auxiliary_event = (
        None if detail_profiler is None
        else detail_profiler.begin('auxiliary_ft'))
    aux_g = ft_ao.ft_ao(auxcell, vectors).T
    aux_weighted = aux_g.conj() * coulomb
    if detail_profiler is not None:
        detail_profiler.end('auxiliary_ft', auxiliary_event)

    pair_event = (
        None if detail_profiler is None
        else detail_profiler.begin('ao_pair_ft'))
    pair_g = evaluator(vectors).view(np.float64).reshape(
        nao, nao, ngrid * 2)
    pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]
    if detail_profiler is not None:
        detail_profiler.end('ao_pair_ft', pair_event)

    source_event = (
        None if detail_profiler is None
        else detail_profiler.begin('source_contraction'))
    aux_source, ao_source = contract_rhf_mo_response_long_range_block(
        factors, pair_g, aux_g, aux_weighted,
        include_metric_source=False)
    if detail_profiler is not None:
        detail_profiler.end('source_contraction', source_event)
        metric_event = detail_profiler.begin('metric_contraction')
    contract('sr,sG->rG', metric_source, aux_g.view(np.float64),
             alpha=-1., beta=1., out=aux_source)
    if detail_profiler is not None:
        detail_profiler.end('metric_contraction', metric_event)
        auxiliary_event = detail_profiler.begin('auxiliary_ft')

    aux_source.view(np.complex128)[:] *= coulomb
    derivative_aux_g = cp.empty((naux, ngrid), dtype=np.complex128)
    auxiliary = cp.empty((3, naux))
    for axis in range(3):
        cp.multiply(aux_g, -1j * vectors[:, axis], out=derivative_aux_g)
        auxiliary[axis] = cp.einsum(
            'ag,ag->a', derivative_aux_g.view(np.float64), aux_source)
    if detail_profiler is not None:
        detail_profiler.end('auxiliary_ft', auxiliary_event)

    derivative = cp.zeros((cell.natm, 3))
    vector_transpose = cp.asarray(vectors.T.ravel())
    pair_event = (
        None if detail_profiler is None
        else detail_profiler.begin('ao_pair_ft'))
    error = kernel(
        ctypes.cast(derivative.data.ptr, ctypes.c_void_p),
        ctypes.cast(ao_source.data.ptr, ctypes.c_void_p),
        ctypes.cast(vector_transpose.data.ptr, ctypes.c_void_p),
        ctypes.byref(aft_envs), ctypes.c_int(len(shell_offsets) - 1),
        ctypes.c_int(ngrid), ctypes.c_int(shared_memory),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(shell_offsets.data.ptr, ctypes.c_void_p),
        ctypes.c_int(permutation_symmetry))
    if detail_profiler is not None:
        detail_profiler.end('ao_pair_ft', pair_event)
    if error != 0:
        raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')
    return derivative, auxiliary


def _long_range_adjoint_block(cell, auxcell, primitive_source, metric_source,
                              vectors, coulomb, evaluator, nao, naux,
                              i_addr, j_addr, kernel, aft_envs,
                              shell_offsets, shared_memory, bas_ij_idx,
                              bas_ij_img_idx, permutation_symmetry,
                              selected_pair_g=None, detail_profiler=None):
    '''Contract one LR transaction with direct primitive ``dB``/``dM`` sources.'''
    ngrid = len(vectors)
    event = (None if detail_profiler is None
             else detail_profiler.begin('auxiliary_ft'))
    aux_g = ft_ao.ft_ao(auxcell, vectors).T
    aux_weighted = aux_g.conj() * coulomb
    if detail_profiler is not None:
        detail_profiler.end('auxiliary_ft', event)
    aux_source = cp.empty((naux, ngrid * 2))
    ao_source = cp.empty((nao, nao, ngrid * 2))
    source_event = (None if detail_profiler is None
                    else detail_profiler.begin('source_contraction'))
    if isinstance(primitive_source, _PackedPrimitiveSource):
        layout = primitive_source.layout
        values = primitive_source.values
        address = cp.asarray(layout['address'], dtype=cp.int64)
        weights = cp.asarray(layout['weights'], dtype=values.dtype)
        if selected_pair_g is None:
            if detail_profiler is not None:
                # Pair generation is a Fourier operation, not source algebra.
                detail_profiler.end('source_contraction', source_event)
                pair_event = detail_profiler.begin('ao_pair_ft')
            pair_g = evaluator(vectors).view(np.float64).reshape(
                nao, nao, ngrid * 2)
            pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]
            selected = pair_g.reshape(nao * nao, ngrid * 2)[address]
            if detail_profiler is not None:
                detail_profiler.end('ao_pair_ft', pair_event)
                source_event = detail_profiler.begin('source_contraction')
        else:
            selected = cp.asarray(selected_pair_g)
            if selected.shape != (len(address), ngrid * 2):
                raise ValueError(
                    'cached reciprocal AO-pair block has wrong shape')
        contract('rk,kG->rG', values, selected, out=aux_source)
        packed_ao_source = cp.empty((len(address), ngrid * 2))
        contract('rk,rG->kG', values, aux_weighted.view(np.float64),
                 out=packed_ao_source)
        packed_ao_source /= weights[:, None]
        ao_source[...] = 0.
        row = cp.asarray(layout['row'], dtype=cp.int32)
        column = cp.asarray(layout['column'], dtype=cp.int32)
        ao_source[row, column] = packed_ao_source
        ao_source[column, row] = packed_ao_source
    else:
        if detail_profiler is not None:
            detail_profiler.end('source_contraction', source_event)
            pair_event = detail_profiler.begin('ao_pair_ft')
        pair_g = evaluator(vectors).view(np.float64).reshape(
            nao, nao, ngrid * 2)
        pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]
        if detail_profiler is not None:
            detail_profiler.end('ao_pair_ft', pair_event)
            source_event = detail_profiler.begin('source_contraction')
        contract('rpq,pqG->rG', primitive_source, pair_g, out=aux_source)
        contract(
            'rpq,rG->pqG', primitive_source, aux_weighted.view(np.float64),
            out=ao_source)
    if detail_profiler is not None:
        detail_profiler.end('source_contraction', source_event)
        metric_event = detail_profiler.begin('metric_contraction')
    contract('sr,sG->rG', metric_source, aux_g.view(np.float64),
             alpha=-1., beta=1., out=aux_source)
    if detail_profiler is not None:
        detail_profiler.end('metric_contraction', metric_event)
        auxiliary_event = detail_profiler.begin('auxiliary_ft')
    aux_source_complex = aux_source.view(np.complex128)
    aux_source_complex *= coulomb
    derivative_aux_g = cp.empty((naux, ngrid), dtype=np.complex128)
    auxiliary = cp.empty((3, naux))
    for axis in range(3):
        cp.multiply(aux_g, -1j * vectors[:, axis], out=derivative_aux_g)
        auxiliary[axis] = cp.einsum(
            'ag,ag->a', derivative_aux_g.view(np.float64), aux_source)
    if detail_profiler is not None:
        detail_profiler.end('auxiliary_ft', auxiliary_event)

    derivative = cp.zeros((cell.natm, 3))
    vector_transpose = cp.asarray(vectors.T.ravel())
    pair_event = (None if detail_profiler is None
                  else detail_profiler.begin('ao_pair_ft'))
    error = kernel(
        ctypes.cast(derivative.data.ptr, ctypes.c_void_p),
        ctypes.cast(ao_source.data.ptr, ctypes.c_void_p),
        ctypes.cast(vector_transpose.data.ptr, ctypes.c_void_p),
        ctypes.byref(aft_envs), ctypes.c_int(len(shell_offsets) - 1),
        ctypes.c_int(ngrid), ctypes.c_int(shared_memory),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(shell_offsets.data.ptr, ctypes.c_void_p),
        ctypes.c_int(permutation_symmetry))
    if detail_profiler is not None:
        detail_profiler.end('ao_pair_ft', pair_event)
    if error != 0:
        raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')
    return derivative, auxiliary


def _long_range_block(cell, auxcell, states, metric_source, vectors,
                      coulomb, evaluator, nao, naux, i_addr, j_addr,
                      kernel, aft_envs, shell_offsets, shared_memory,
                      bas_ij_idx, bas_ij_img_idx, permutation_symmetry,
                      detail_profiler=None):
    '''Build and consume one transaction-local reciprocal derivative block.'''
    ngrid = len(vectors)
    pair_event = (None if detail_profiler is None
                  else detail_profiler.begin('ao_pair_ft'))
    pair_g = evaluator(vectors).view(np.float64).reshape(
        nao, nao, ngrid * 2)
    pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]
    if detail_profiler is not None:
        detail_profiler.end('ao_pair_ft', pair_event)

    auxiliary_event = (None if detail_profiler is None
                       else detail_profiler.begin('auxiliary_ft'))
    aux_g = ft_ao.ft_ao(auxcell, vectors).T
    aux_weighted = aux_g.conj() * coulomb
    aux_weighted_real = aux_weighted.view(np.float64)
    if detail_profiler is not None:
        detail_profiler.end('auxiliary_ft', auxiliary_event)
    aux_source = cp.zeros((naux, ngrid * 2))
    ao_source = cp.zeros((nao, nao, ngrid * 2))
    rank = max(state.rank for state in states)
    temp = cp.empty((rank, nao, ngrid * 2))
    occ = cp.empty((rank, rank, ngrid * 2))
    rho = cp.empty(ngrid * 2)
    potential = cp.empty(ngrid * 2)

    source_event = (None if detail_profiler is None
                    else detail_profiler.begin('source_contraction'))
    for state in states:
        tmp = temp[:state.rank]
        occupied = occ[:state.rank, :state.rank]
        contract('pqG,qp->G', pair_g, state.density_sorted, out=rho)
        contract('r,G->rG', state.aux_vector, rho,
                 alpha=state.coulomb_weight, beta=1., out=aux_source)

        contract('pqG,pi->iqG', pair_g, state.factor_r, out=tmp)
        contract('iqG,qj->ijG', tmp, state.factor_l, out=occupied)
        contract('rji,ijG->rG', state.three_center_oo, occupied,
                 alpha=-.5 * state.exchange_weight, beta=1., out=aux_source)

        contract('rij,rG->ijG', state.three_center_oo, aux_weighted_real,
                 out=occupied)
        contract('jiG,qi->jqG', occupied, state.factor_r, out=tmp)
        contract('jqG,pj->pqG', tmp, state.factor_l,
                 alpha=-.5 * state.exchange_weight, beta=1., out=ao_source)
        contract('r,rG->G', state.aux_vector, aux_weighted_real,
                 out=potential)
        contract('pq,G->pqG', state.density_sorted, potential,
                 alpha=state.coulomb_weight, beta=1., out=ao_source)

    if detail_profiler is not None:
        detail_profiler.end('source_contraction', source_event)
        metric_event = detail_profiler.begin('metric_contraction')
    contract('sr,sG->rG', metric_source, aux_g.view(np.float64),
             alpha=-1., beta=1., out=aux_source)
    if detail_profiler is not None:
        detail_profiler.end('metric_contraction', metric_event)
        auxiliary_event = detail_profiler.begin('auxiliary_ft')
    aux_weighted_source = aux_source.view(np.complex128)
    aux_weighted_source *= coulomb
    derivative_aux_g = cp.empty((naux, ngrid), dtype=np.complex128)
    auxiliary = cp.empty((3, naux))
    for axis in range(3):
        cp.multiply(aux_g, -1j * vectors[:, axis], out=derivative_aux_g)
        auxiliary[axis] = cp.einsum(
            'ag,ag->a', derivative_aux_g.view(np.float64), aux_source)
    if detail_profiler is not None:
        detail_profiler.end('auxiliary_ft', auxiliary_event)

    derivative = cp.zeros((cell.natm, 3))
    vector_transpose = cp.asarray(vectors.T.ravel())
    pair_event = (None if detail_profiler is None
                  else detail_profiler.begin('ao_pair_ft'))
    error = kernel(
        ctypes.cast(derivative.data.ptr, ctypes.c_void_p),
        ctypes.cast(ao_source.data.ptr, ctypes.c_void_p),
        ctypes.cast(vector_transpose.data.ptr, ctypes.c_void_p),
        ctypes.byref(aft_envs), ctypes.c_int(len(shell_offsets) - 1),
        ctypes.c_int(ngrid), ctypes.c_int(shared_memory),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(shell_offsets.data.ptr, ctypes.c_void_p),
        ctypes.c_int(permutation_symmetry))
    if detail_profiler is not None:
        detail_profiler.end('ao_pair_ft', pair_event)
    if error != 0:
        raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')
    return derivative, auxiliary


def _unpack_packed_primitive_source(
        packed, selected, inverse, weights, p0, p1,
        pair_block_size=_PACKED_SOURCE_PAIR_BLOCK_SIZE):
    r'''Expand one unique-pair source slice into libpbc's raw pair order.

    For raw AO-pair index ``k`` and auxiliary-slice index ``r``, this helper
    evaluates

    .. math::

       X_{k r} =
       S_{r,\iota(k)} / w_{\iota(k)}

    directly into ``packed[k, p0 + r]``.  A full CuPy expression such as
    ``selected[:, inverse]`` first materialises the complete
    ``(width, raw_pair_count)`` gather and another division result.  At
    Water64/GTH-DZVP/ETB one such array is 12.8 GiB.  Blocking the raw-pair
    index bounds the principal temporary by
    ``pair_block_size * width * itemsize`` while preserving the exact mapping.

    Args:
        packed : cupy.ndarray
            Existing ``(raw_pair_count, naux_batch)`` libpbc source buffer.
        selected : cupy.ndarray
            ``(width, unique_pair_count)`` primitive-adjoint slice.
        inverse : cupy.ndarray
            Raw-to-unique AO-pair address ``iota(k)``.
        weights : cupy.ndarray
            Unique-pair multiplicities ``w`` used by the canonical Gamma
            layout.
        p0, p1 : int
            Destination auxiliary-column interval in ``packed``.

    Kwargs:
        pair_block_size : int
            Maximum number of raw AO pairs in one CuPy gather.  This affects
            workspace and kernel-launch count only; it does not change the
            GDF approximation or numerical result.

    Returns:
        None
            ``packed[:, p0:p1]`` is overwritten in place.
    '''
    pair_block_size = int(pair_block_size)
    if pair_block_size < 1:
        raise ValueError('pair_block_size must be positive')
    width = int(p1 - p0)
    if selected.ndim != 2 or int(selected.shape[0]) != width:
        raise ValueError(
            'selected auxiliary width does not match destination slice')
    if packed.ndim != 2 or int(packed.shape[0]) != int(inverse.size):
        raise ValueError(
            'packed and inverse raw AO-pair dimensions do not match')
    for q0, q1 in lib.prange(0, int(inverse.size), pair_block_size):
        address = inverse[q0:q1]
        gathered = cp.take(selected, address, axis=1)
        gathered /= weights[address][None]
        packed[q0:q1, p0:p1] = gathered.T
        del gathered, address


def _short_range_derivative(intopt, states, pair_addresses, batch_size,
                            aux_block, nao, naux, primitive_source=None,
                            mo_response_factors=None,
                            return_components=False):
    '''Contract signed or direct-adjoint SR derivative kernels without dCDERI.

    The direct-adjoint branch receives a packed or dense AO source and only
    reorders each bounded auxiliary slice for libpbc.  The MO-factorized
    branch instead evaluates the requested raw AO pairs directly from
    ``C,U,A,K`` into that same transaction buffer.  Both avoid re-forming the
    two full-rank ``P+DZ`` and ``P-DZ`` state sources.
    '''
    if primitive_source is not None and mo_response_factors is not None:
        raise ValueError(
            'primitive_source and mo_response_factors are mutually exclusive')
    cell = intopt.cell
    auxcell = intopt.auxcell
    auxiliary_shell_limit = max(1, min(int(auxcell.nbas), POOL_SIZE))
    shell_pair_limit = max(1, POOL_SIZE // auxiliary_shell_limit)
    _, gout_stride, shared_sizes = derivative_int3c2e_scheme()
    lmax = cell.uniq_l_ctr[:, 0].max()
    laux = auxcell.uniq_l_ctr[:, 0].max()
    shared_memory = shared_sizes[:laux + 1, :lmax + 1, :lmax + 1].max()
    shell_pairs, shell_pair_offsets = cell.aggregate_shl_pairs(
        intopt.bas_ij_cache, shell_pair_limit)
    ao_pair_loc = get_ao_pair_loc(
        cell.uniq_l_ctr[:, 0], intopt.bas_ij_cache, cart=True)
    aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
    aux_offsets, aux_patterns = _split_l_ctr_pattern(
        aux_offsets, auxcell.uniq_l_ctr,
        min(batch_size, auxiliary_shell_limit))
    aux_offsets_gpu = cp.asarray(aux_offsets, dtype=np.int32)

    max_aux_shells = max(1, int(np.diff(aux_offsets).max()))
    max_pairs = max(1, int(np.diff(cp.asnumpy(shell_pair_offsets)).max()))
    if max_aux_shells * max_pairs > POOL_SIZE:
        raise RuntimeError('Gamma GDF SR derivative task batch exceeds POOL_SIZE')
    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty(
        workers * POOL_SIZE * (MAX_IMGS_PER_TASK + 2) + 1,
        dtype=np.uint32)
    head = pool[-1:]
    task_pool = empty_aligned(
        (workers, POOL_SIZE * 16), np.int32, alignment=128)
    ao_derivative = cp.zeros((cell.natm, 3))
    aux_derivative = cp.zeros((cell.natm, 3))
    diffuse_exponents = cp.asarray(intopt.diffuse_exps)
    diffuse_coefficients = cp.asarray(intopt.diffuse_coefs)
    log_cutoff = math.log(intopt.cutoff)
    aux0 = 0
    mo_pair_indices = (
        prepare_rhf_mo_short_range_pair_indices(pair_addresses, nao)
        if mo_response_factors is not None else None)

    for batch_id, _ in enumerate(aux_patterns[:, 0]):
        aux_ao_offset = auxcell.ao_loc[aux_offsets[batch_id]]
        naux_batch = (auxcell.ao_loc[aux_offsets[batch_id + 1]]
                      - aux_ao_offset)
        packed = cp.empty((len(pair_addresses), naux_batch))
        for p0, p1 in lib.prange(0, naux_batch, aux_block):
            width = p1 - p0
            aux1 = aux0 + width
            if mo_response_factors is not None:
                build_rhf_mo_response_short_range_source_block(
                    mo_response_factors, pair_addresses,
                    auxiliary_start=aux0, auxiliary_stop=aux1,
                    out=packed[:, p0:p1], pair_indices=mo_pair_indices)
                aux0 = aux1
                continue
            if primitive_source is None:
                source = cp.zeros((nao, nao, width))
                for state in states:
                    source += state.coulomb_weight * (
                        state.density_sorted[:, :, None]
                        * state.aux_vector[None, None, aux0:aux1])
                    temp = cp.empty((state.rank, nao, width))
                    contract('rji,qj->iqr', state.three_center_oo[aux0:aux1],
                             state.factor_l, out=temp)
                    contract('iqr,pi->pqr', temp, state.factor_r,
                             alpha=-.5 * state.exchange_weight, beta=1.,
                             out=source)
            else:
                if isinstance(primitive_source, _PackedPrimitiveSource):
                    layout = primitive_source.layout
                    inverse = cp.asarray(layout['inverse'], dtype=cp.int64)
                    weights = cp.asarray(
                        layout['weights'], dtype=primitive_source.values.dtype)
                    selected = primitive_source.values[aux0:aux1]
                    _unpack_packed_primitive_source(
                        packed, selected, inverse, weights, p0, p1)
                    aux0 = aux1
                    continue
                source = primitive_source[aux0:aux1].transpose(1, 2, 0)
            cp.take(source.reshape(-1, width), pair_addresses, axis=0,
                    out=packed[:, p0:p1])
            aux0 = aux1

        error = libpbc.PBCsr_ejk_int3c2e_ip1(
            ctypes.cast(ao_derivative.data.ptr, ctypes.c_void_p),
            ctypes.cast(aux_derivative.data.ptr, ctypes.c_void_p),
            ctypes.cast(packed.data.ptr, ctypes.c_void_p), lib.c_null_ptr(),
            ctypes.byref(intopt.int3c2e_envs),
            ctypes.cast(pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(task_pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(head.data.ptr, ctypes.c_void_p), ctypes.c_int(shared_memory),
            ctypes.c_int(len(shell_pair_offsets) - 1), ctypes.c_int(1),
            ctypes.cast(shell_pairs.data.ptr, ctypes.c_void_p),
            ctypes.cast(shell_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(aux_offsets_gpu[batch_id:].data.ptr, ctypes.c_void_p),
            ctypes.cast(intopt.img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(intopt.img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(aux_ao_offset), ctypes.c_int(auxcell.nbas),
            ctypes.c_int(naux_batch),
            ctypes.cast(diffuse_exponents.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefficients.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if error != 0:
            raise RuntimeError('PBCsr_ejk_int3c2e_ip1 failed')
    source_result = ao_derivative.get() * 2
    metric_result = aux_derivative.get() * 2
    result = source_result + metric_result
    if return_components:
        return result, {
            'source': source_result,
            'metric': metric_result,
        }
    return result


def _short_range_derivative_channels(intopt, pair_addresses, batch_size,
                                     aux_block, nao, naux, target_source,
                                     fock_source, kernel):
    r'''Contract two primitive adjoints in one short-range CUDA traversal.

    Both ``target_source`` and ``fock_source`` contract the *same* derivative
    three-center integrals.  The patched CUDA ABI therefore shares the costly
    Rys/primitive evaluation while keeping separate AO- and auxiliary-center
    reductions.  No all-atom ``dB`` tensor is formed.

    Args:
        intopt : gpu4pyscf.pbc.df.int3c2e.SRInt3c2eOpt
            Built Gamma three-center optimizer.
        pair_addresses : cupy.ndarray
            Packed sorted-AO pair addresses.
        batch_size, aux_block : int
            Provider-owned bounded auxiliary batching controls.
        nao, naux : int
            Sorted AO and auxiliary dimensions.
        target_source, fock_source : cupy.ndarray
            Real primitive adjoints with shape ``(naux, nao, nao)``.
        kernel : ctypes._CFuncPtr
            Audited ``PBCsr_ejk_int3c2e_ip1_two_channel`` endpoint.

    Returns:
        tuple
            Target value, Fock value, and their respective source/metric
            component dictionaries, all host NumPy arrays in Ha/Bohr.
    '''
    target_source = cp.asarray(target_source, dtype=cp.float64)
    fock_source = cp.asarray(fock_source, dtype=cp.float64)
    expected_shape = (int(naux), int(nao), int(nao))
    if target_source.shape != expected_shape or fock_source.shape != expected_shape:
        raise ValueError(
            'two-channel primitive sources must both have shape '
            f'{expected_shape}, got {target_source.shape} and {fock_source.shape}')
    cell = intopt.cell
    auxcell = intopt.auxcell
    auxiliary_shell_limit = max(1, min(int(auxcell.nbas), POOL_SIZE))
    shell_pair_limit = max(1, POOL_SIZE // auxiliary_shell_limit)
    _, gout_stride, shared_sizes = derivative_int3c2e_scheme()
    lmax = cell.uniq_l_ctr[:, 0].max()
    laux = auxcell.uniq_l_ctr[:, 0].max()
    shared_memory = shared_sizes[:laux + 1, :lmax + 1, :lmax + 1].max()
    shell_pairs, shell_pair_offsets = cell.aggregate_shl_pairs(
        intopt.bas_ij_cache, shell_pair_limit)
    ao_pair_loc = get_ao_pair_loc(
        cell.uniq_l_ctr[:, 0], intopt.bas_ij_cache, cart=True)
    aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
    aux_offsets, aux_patterns = _split_l_ctr_pattern(
        aux_offsets, auxcell.uniq_l_ctr,
        min(batch_size, auxiliary_shell_limit))
    aux_offsets_gpu = cp.asarray(aux_offsets, dtype=np.int32)
    max_aux_shells = max(1, int(np.diff(aux_offsets).max()))
    max_pairs = max(1, int(np.diff(cp.asnumpy(shell_pair_offsets)).max()))
    if max_aux_shells * max_pairs > POOL_SIZE:
        raise RuntimeError('Gamma GDF SR derivative task batch exceeds POOL_SIZE')
    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty(
        workers * POOL_SIZE * (MAX_IMGS_PER_TASK + 2) + 1,
        dtype=np.uint32)
    head = pool[-1:]
    task_pool = empty_aligned(
        (workers, POOL_SIZE * 16), np.int32, alignment=128)
    target_ao = cp.zeros((cell.natm, 3))
    target_aux = cp.zeros((cell.natm, 3))
    fock_ao = cp.zeros((cell.natm, 3))
    fock_aux = cp.zeros((cell.natm, 3))
    diffuse_exponents = cp.asarray(intopt.diffuse_exps)
    diffuse_coefficients = cp.asarray(intopt.diffuse_coefs)
    log_cutoff = math.log(intopt.cutoff)
    aux0 = 0

    for batch_id, _ in enumerate(aux_patterns[:, 0]):
        aux_ao_offset = auxcell.ao_loc[aux_offsets[batch_id]]
        naux_batch = (auxcell.ao_loc[aux_offsets[batch_id + 1]]
                      - aux_ao_offset)
        target_packed = cp.empty((len(pair_addresses), naux_batch))
        fock_packed = cp.empty_like(target_packed)
        for p0, p1 in lib.prange(0, naux_batch, aux_block):
            width = p1 - p0
            aux1 = aux0 + width
            target_slice = target_source[aux0:aux1].transpose(1, 2, 0)
            fock_slice = fock_source[aux0:aux1].transpose(1, 2, 0)
            cp.take(target_slice.reshape(-1, width), pair_addresses, axis=0,
                    out=target_packed[:, p0:p1])
            cp.take(fock_slice.reshape(-1, width), pair_addresses, axis=0,
                    out=fock_packed[:, p0:p1])
            aux0 = aux1

        error = kernel(
            ctypes.cast(target_ao.data.ptr, ctypes.c_void_p),
            ctypes.cast(target_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(target_packed.data.ptr, ctypes.c_void_p),
            ctypes.cast(fock_ao.data.ptr, ctypes.c_void_p),
            ctypes.cast(fock_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(fock_packed.data.ptr, ctypes.c_void_p),
            ctypes.byref(intopt.int3c2e_envs),
            ctypes.cast(pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(task_pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(head.data.ptr, ctypes.c_void_p), ctypes.c_int(shared_memory),
            ctypes.c_int(len(shell_pair_offsets) - 1), ctypes.c_int(1),
            ctypes.cast(shell_pairs.data.ptr, ctypes.c_void_p),
            ctypes.cast(shell_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(aux_offsets_gpu[batch_id:].data.ptr, ctypes.c_void_p),
            ctypes.cast(intopt.img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(intopt.img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(aux_ao_offset), ctypes.c_int(auxcell.nbas),
            ctypes.c_int(naux_batch),
            ctypes.cast(diffuse_exponents.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefficients.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if error != 0:
            raise RuntimeError('PBCsr_ejk_int3c2e_ip1_two_channel failed')

    target_components = {
        'source': target_ao.get() * 2,
        'metric': target_aux.get() * 2,
    }
    fock_components = {
        'source': fock_ao.get() * 2,
        'metric': fock_aux.get() * 2,
    }
    target_value = target_components['source'] + target_components['metric']
    fock_value = fock_components['source'] + fock_components['metric']
    return target_value, fock_value, target_components, fock_components


def _atmlst(cell, atmlst):
    '''Validate requested atom indices for a per-atom derivative result.'''
    if atmlst is None:
        atmlst = range(cell.natm)
    atoms = np.asarray(tuple(int(ia) for ia in atmlst), dtype=int)
    if atoms.ndim != 1 or np.any(atoms < 0) or np.any(atoms >= cell.natm):
        raise ValueError('atom index is outside the periodic cell')
    return atoms
