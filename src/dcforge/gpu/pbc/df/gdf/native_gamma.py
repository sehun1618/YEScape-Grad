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
an explicit numerical oracle.
'''

import ctypes
import math
import time

import cupy as cp
import numpy as np
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.df.grad.rhf import factorize_dm
from gpu4pyscf.gto.mole import groupby
from gpu4pyscf.lib.cupy_helper import asarray, contract, empty_aligned, get_avail_mem
from gpu4pyscf.pbc.df import aft_jk, ft_ao
from gpu4pyscf.pbc.df.grad.rhf import (
    _split_l_ctr_pattern,
    get_ao_pair_loc,
    int3c2e_scheme,
)
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.df.int3c2e import MAX_IMGS_PER_TASK, POOL_SIZE, SRInt3c2eOpt, libpbc
from gpu4pyscf.pbc.df.rsdf_builder import _guess_omega, _weighted_coulG_kpts
from pyscf import lib
from pyscf.gto import ATOM_OF

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
    build_rhf_primitive_adjoint,
    build_rhf_primitive_adjoint_channels,
)
from dcforge.gpu.pbc.df.gdf.two_channel import get_short_range_kernel


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


def target_j_fock_derivative(mf_hf, dm0, dz, atmlst=None,
                             return_metadata=False):
    r'''Return the fused target-J and signed source-Fock two-electron term.

    The result is

    .. math::

       E_J[P]^\lambda-
       \{Q[P+D_Z]^\lambda-Q[P-D_Z]^\lambda\}.

    A single primitive traversal builds the metric-solved three-center tensor
    once, then contracts its derivative with the direct ``P,D_Z`` reverse
    adjoint.  This is algebraically identical to a target ``P`` state plus
    the signed ``P+D_Z`` and ``P-D_Z`` polarization states, but avoids their
    two full-rank signed three-center tensors.  It also avoids the mutable
    native derivative state that makes separately dispatched GDF observables
    unsafe.

    The returned target-J plus Z-Fock observable is intentionally coupled and
    must remain ``de_j_fock_native``.  With ``return_metadata=True``, the
    same traversal additionally exposes its exact primitive ``dB`` source,
    auxiliary ``dM`` metric, and Ewald components.  Those components audit
    the fused observable; they are not a claim that target J and source Fock
    were dispatched as separate native derivatives.
    '''
    provider = mf_hf.with_df
    _check_gamma_gdf(provider, dm0, dz)
    if provider.auxcell is None:
        provider.build(j_only=False)
    cell = provider.cell
    profile = {}
    stage_start = time.perf_counter()
    intopt = SRInt3c2eOpt(
        cell, provider.auxcell, float(_guess_omega(cell)), None).build()
    _finish_profile_stage(profile, 'optimizer_seconds', stage_start)
    value, metadata = _direct_adjoint_derivative(
        intopt, dm0, dz, exxdiv=mf_hf.exxdiv,
        reciprocal_options=_reciprocal_options(provider))
    metadata['profile']['optimizer_seconds'] = profile['optimizer_seconds']
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
    }
    if options['maximum_block_size'] is not None:
        options['maximum_block_size'] = int(options['maximum_block_size'])
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
    batch_size, aux_block = _work_blocks(
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
    for state in states:
        state.three_center_oo = cp.einsum(
            'uv,vij->uij', metric_inverse, state.three_center_oo,
            optimize=True)
        state.aux_vector = state.three_center_oo.trace(axis1=1, axis2=2)

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
        'profile': profile,
        'term_components': term_components,
        'term_component_residual_max_abs': float(np.max(np.abs(
            np.asarray(value, dtype=float) - component_sum))),
    }
    return np.asarray(value, dtype=float), metadata


def _direct_adjoint_derivative(intopt, reference_density, response_density,
                               exxdiv, reciprocal_options):
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

    Returns:
        tuple
            All-atom Ha/Bohr derivative and resolved scheduler/adjoint
            metadata.
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
    batch_size, aux_block = _work_blocks(
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
    solved = cp.einsum(
        'uv,vij->uij', metric_inverse, primitive_state.three_center_oo,
        optimize=True)
    primitive_state.three_center_oo = None
    del metric_inverse
    source, metric_source, adjoint_metadata = build_rhf_primitive_adjoint(
        solved, reference_state.density_sorted, response_state.density_sorted,
        include_target_j=True, response_sign=-1.)
    stage_start = _finish_profile_stage(
        profile, 'metric_solve_and_adjoint_seconds', stage_start)
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
    stage_start = _finish_profile_stage(
        profile, 'metric_derivative_seconds', stage_start)
    long_range, derivative_execution, long_range_components = _long_range_derivative(
        intopt, reciprocal, (), metric_source, nao, naux,
        primitive_source=source, return_components=True)
    stage_start = _finish_profile_stage(
        profile, 'long_range_derivative_seconds', stage_start)
    value += long_range
    short_range, short_range_components = _short_range_derivative(
        intopt, (), pair_addresses, batch_size, aux_block, nao, naux,
        primitive_source=source, return_components=True)
    value += short_range
    stage_start = _finish_profile_stage(
        profile, 'short_range_derivative_seconds', stage_start)
    ewald = ewald_cross_derivative(
        cell, reference_state.density_physical,
        response_state.density_physical, exxdiv)
    value += ewald
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
        'adjoint': adjoint_metadata,
        'profile': profile,
        'solved_three_center_bytes': solved_bytes,
        # Two full-rank polarization tensors of this shape are absent.
        'signed_state_full_rank_bytes_avoided': 2 * naux * nao ** 2 * 8,
        'term_components': term_components,
        'term_component_residual_max_abs': float(np.max(np.abs(
            np.asarray(value, dtype=float) - component_sum))),
    }
    return np.asarray(value, dtype=float), metadata


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
    batch_size, aux_block = _work_blocks(
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
    solved = cp.einsum(
        'uv,vij->uij', metric_inverse, primitive_state.three_center_oo,
        optimize=True)
    primitive_state.three_center_oo = None
    del metric_inverse
    target, fock, adjoint_metadata = (
        build_rhf_primitive_adjoint_channels(
            solved, reference_state.density_sorted,
            response_state.density_sorted, response_sign=-1.))
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
    '''Choose conservative auxiliary blocks from currently free device memory.'''
    free = int(get_avail_mem(exclude_memory_pool=True))
    occupied = sum(naux * state.rank ** 2 * 8 for state in states)
    free = max(1, free - occupied)
    batch = max(1, min(naux, int(.40 * free / (npair * nbvk * 8))))
    size = states[0].density_sorted.size
    block = max(1, min(naux, int(.25 * free / (len(states) * size * 8))))
    return int(batch), max(1, int(block // 8 * 8))


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
                temp = cp.empty((state.rank, nao, width))
                contract('pqr,pi->iqr', three_center, state.factor_r, out=temp)
                contract(
                    'iqr,qj->rij', temp, state.factor_l,
                    out=state.three_center_oo[offset:p2])
            offset = p2


class _Reciprocal:
    '''Zero-order reciprocal data reused by the metric and derivative passes.'''

    def __init__(self, ft_opt, evaluator, vectors, coulomb, block_size,
                 memory_plan, zero_order_execution, full_grid_count,
                 gamma_conjugate_symmetry):
        self.ft_opt = ft_opt
        self.evaluator = evaluator
        self.vectors = vectors
        self.coulomb = coulomb
        self.block_size = int(block_size)
        self.memory_plan = memory_plan
        self.zero_order_execution = zero_order_execution
        self.full_grid_count = int(full_grid_count)
        self.gamma_conjugate_symmetry = bool(gamma_conjugate_symmetry)

    def plan_metadata(self):
        '''Return the memory plan plus the exact reciprocal reduction scope.'''
        metadata = self.memory_plan.as_dict()
        metadata.update({
            'full_grid_count': self.full_grid_count,
            'evaluated_grid_count': len(self.vectors),
            'gamma_conjugate_symmetry': self.gamma_conjugate_symmetry,
        })
        return metadata


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
        **{
            name: value for name, value in reciprocal_options.items()
            if name != 'gamma_conjugate_symmetry'
        })

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
                           return_components=False):
    '''Contract signed or direct-adjoint LR sources in reciprocal blocks.

    ``primitive_source`` is the exact reverse source which contracts directly
    with ``dB``.  When it is supplied, ``states`` is empty and no signed
    occupied-factor workspace is allocated.  The signed-state branch remains
    the quadratic reference implementation used by the public component
    helpers.  With ``return_components=True`` the direct-adjoint route also
    returns its separate ``dB`` and ``dM`` contractions.  The G-vector
    traversal remains single-pass; only its final per-atom reductions are
    kept apart for the public fused-term ledger.
    '''
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
        if primitive_source is not None:
            return _long_range_adjoint_block(
                cell, auxcell, primitive_source, metric_source,
                reciprocal.vectors[p0:p1], reciprocal.coulomb[p0:p1],
                reciprocal.evaluator, nao, naux, i_addr, j_addr, kernel,
                aft_envs, shell_offsets, shared_memory, bas_ij_idx,
                bas_ij_img_idx, ft_opt.permutation_symmetry)
        return _long_range_block(
            cell, auxcell, states, metric_source,
            reciprocal.vectors[p0:p1], reciprocal.coulomb[p0:p1],
            reciprocal.evaluator, nao, naux, i_addr, j_addr, kernel,
            aft_envs, shell_offsets, shared_memory, bas_ij_idx,
            bas_ij_img_idx, ft_opt.permutation_symmetry)

    def accept_result(_p0, _p1, value):
        ao_total[...] += value[0]
        aux_total[...] += value[1]

    execution = execute_gamma_reciprocal_blocks(
        len(reciprocal.vectors), reciprocal.block_size, worker, accept_result,
        phase='long_range_derivative', is_oom=_is_gpu_out_of_memory,
        block_alignment=reciprocal.memory_plan.block_alignment,
        on_oom=_release_cached_gpu_blocks)

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
    if return_components:
        return result, execution, {
            'source': source_result,
            'metric': metric_result,
        }
    return result, execution


def _long_range_adjoint_block(cell, auxcell, primitive_source, metric_source,
                              vectors, coulomb, evaluator, nao, naux,
                              i_addr, j_addr, kernel, aft_envs,
                              shell_offsets, shared_memory, bas_ij_idx,
                              bas_ij_img_idx, permutation_symmetry):
    '''Contract one LR transaction with direct primitive ``dB``/``dM`` sources.'''
    ngrid = len(vectors)
    pair_g = evaluator(vectors).view(np.float64).reshape(
        nao, nao, ngrid * 2)
    pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]
    aux_g = ft_ao.ft_ao(auxcell, vectors).T
    aux_weighted = aux_g.conj() * coulomb
    aux_source = cp.empty((naux, ngrid * 2))
    contract('rpq,pqG->rG', primitive_source, pair_g, out=aux_source)
    ao_source = cp.empty((nao, nao, ngrid * 2))
    contract('rpq,rG->pqG', primitive_source, aux_weighted.view(np.float64),
             out=ao_source)
    contract('sr,sG->rG', metric_source, aux_g.view(np.float64),
             alpha=-1., beta=1., out=aux_source)
    aux_source_complex = aux_source.view(np.complex128)
    aux_source_complex *= coulomb
    derivative_aux_g = cp.empty((naux, ngrid), dtype=np.complex128)
    auxiliary = cp.empty((3, naux))
    for axis in range(3):
        cp.multiply(aux_g, -1j * vectors[:, axis], out=derivative_aux_g)
        auxiliary[axis] = cp.einsum(
            'ag,ag->a', derivative_aux_g.view(np.float64), aux_source)

    derivative = cp.zeros((cell.natm, 3))
    vector_transpose = cp.asarray(vectors.T.ravel())
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
    if error != 0:
        raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')
    return derivative, auxiliary


def _long_range_block(cell, auxcell, states, metric_source, vectors,
                      coulomb, evaluator, nao, naux, i_addr, j_addr,
                      kernel, aft_envs, shell_offsets, shared_memory,
                      bas_ij_idx, bas_ij_img_idx, permutation_symmetry):
    '''Build and consume one transaction-local reciprocal derivative block.'''
    ngrid = len(vectors)
    pair_g = evaluator(vectors).view(np.float64).reshape(
        nao, nao, ngrid * 2)
    pair_g[j_addr, i_addr] = pair_g[i_addr, j_addr]

    aux_g = ft_ao.ft_ao(auxcell, vectors).T
    aux_weighted = aux_g.conj() * coulomb
    aux_weighted_real = aux_weighted.view(np.float64)
    aux_source = cp.zeros((naux, ngrid * 2))
    ao_source = cp.zeros((nao, nao, ngrid * 2))
    rank = max(state.rank for state in states)
    temp = cp.empty((rank, nao, ngrid * 2))
    occ = cp.empty((rank, rank, ngrid * 2))
    rho = cp.empty(ngrid * 2)
    potential = cp.empty(ngrid * 2)

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

    contract('sr,sG->rG', metric_source, aux_g.view(np.float64),
             alpha=-1., beta=1., out=aux_source)
    aux_weighted_source = aux_source.view(np.complex128)
    aux_weighted_source *= coulomb
    derivative_aux_g = cp.empty((naux, ngrid), dtype=np.complex128)
    auxiliary = cp.empty((3, naux))
    for axis in range(3):
        cp.multiply(aux_g, -1j * vectors[:, axis], out=derivative_aux_g)
        auxiliary[axis] = cp.einsum(
            'ag,ag->a', derivative_aux_g.view(np.float64), aux_source)

    derivative = cp.zeros((cell.natm, 3))
    vector_transpose = cp.asarray(vectors.T.ravel())
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
    if error != 0:
        raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')
    return derivative, auxiliary


def _short_range_derivative(intopt, states, pair_addresses, batch_size,
                            aux_block, nao, naux, primitive_source=None,
                            return_components=False):
    '''Contract signed or direct-adjoint SR derivative kernels without dCDERI.

    The direct-adjoint branch receives a preassembled ``(naux, nao, nao)``
    source and only reorders each bounded auxiliary slice for libpbc.  It
    avoids re-forming the two full-rank ``P+DZ`` and ``P-DZ`` state sources.
    '''
    cell = intopt.cell
    auxcell = intopt.auxcell
    auxiliary_shell_limit = max(1, min(int(auxcell.nbas), POOL_SIZE))
    shell_pair_limit = max(1, POOL_SIZE // auxiliary_shell_limit)
    _, gout_stride, shared_sizes = int3c2e_scheme(-1, 54)
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

    for batch_id, _ in enumerate(aux_patterns[:, 0]):
        aux_ao_offset = auxcell.ao_loc[aux_offsets[batch_id]]
        naux_batch = (auxcell.ao_loc[aux_offsets[batch_id + 1]]
                      - aux_ao_offset)
        packed = cp.empty((len(pair_addresses), naux_batch))
        for p0, p1 in lib.prange(0, naux_batch, aux_block):
            width = p1 - p0
            aux1 = aux0 + width
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
    _, gout_stride, shared_sizes = int3c2e_scheme(-1, 54)
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
