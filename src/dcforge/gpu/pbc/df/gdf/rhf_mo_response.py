r'''Exact MO-projected zero-order data for real-Gamma RHF GDF response.

This module implements the production ``mo_direct`` zero-order representation
for the native Gaussian-density-fitting response.  It removes the global
AO-pair tensor and projects
each short-range and reciprocal three-center block directly into the occupied
MO space,

.. math::

   B^{oo}_{Pij} = C_{\mu i} B_{P\mu\nu} C_{\nu j},
   \qquad
   B^{ou}_{Pij} = C_{\mu i} B_{P\mu\nu} U_{\nu j},

and solves the same auxiliary metric used by the packed oracle route,

.. math::

   O = M^{-1} B^{oo}, \qquad R = M^{-1} B^{ou}.

Here ``C`` is the bare occupied RHF coefficient matrix and ``U=C_v Z`` is the
bare response-orbital matrix.  No square-root occupation factors and no AO
overlap matrices enter these primitive projections.  Both matrices must
already use GPU4PySCF's sorted/padded AO representation.

From ``O`` and the generally nonsymmetric ``R``, the module constructs the
exact factorized primitive adjoint

.. math::

   S_P = C A_P C^T + C K_P U^T + U K_P C^T

and its auxiliary metric source ``Gamma``.  LR and SR consumers live in
``rhf_factorized_consumers`` and are integrated by ``native_gamma``.
The packed route remains unchanged as the numerical oracle and exact
unsupported-representation fallback.
'''

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import asarray, contract, get_avail_mem
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.df.rsdf_builder import _weighted_coulG_kpts
from pyscf import lib

from dcforge.gpu.pbc.df.gaussian.reciprocal import (
    execute_gamma_reciprocal_blocks,
    gamma_conjugate_reciprocal_indices,
    plan_gamma_reciprocal_blocks,
)


class RHFMOResponseFactors:
    r'''Factorized real-Gamma RHF primitive-adjoint representation.

    Attributes:
        occupied_coefficients : cupy.ndarray, shape (nao, nocc)
            Sorted-AO bare occupied coefficients ``C``.
        response_orbitals : cupy.ndarray, shape (nao, nocc)
            Sorted-AO bare response orbitals ``U=C_v Z``.
        occupied_projection : cupy.ndarray or None
            Metric-solved projection ``O=C^T(M^{-1}B)C``.
            The high-level low-memory builder sets this to ``None`` after
            reusing its storage for ``K``; the algebra-only constructor keeps
            it by default for validation.
        mixed_projection : cupy.ndarray or None
            Metric-solved projection ``R=C^T(M^{-1}B)U``.  ``R`` is not
            symmetrized.  It is ``None`` when its storage was reused for
            ``A``.
        auxiliary_reference, auxiliary_response : cupy.ndarray, shape (naux,)
            ``a=2 trace(O)`` and ``b=2 trace(R)``.
        source_factor, response_factor : cupy.ndarray
            Occupied-space factors ``A`` and ``K`` in
            ``S=C A C^T+C K U^T+U K C^T``.
        metric_source : cupy.ndarray, shape (naux, naux)
            Exact auxiliary metric adjoint ``Gamma=S:W``.
    '''

    def __init__(
            self, occupied_coefficients, response_orbitals,
            occupied_projection, mixed_projection, auxiliary_reference,
            auxiliary_response, source_factor, response_factor,
            metric_source, include_target_j, response_sign):
        self.occupied_coefficients = occupied_coefficients
        self.response_orbitals = response_orbitals
        self.occupied_projection = occupied_projection
        self.mixed_projection = mixed_projection
        self.auxiliary_reference = auxiliary_reference
        self.auxiliary_response = auxiliary_response
        self.source_factor = source_factor
        self.response_factor = response_factor
        self.metric_source = metric_source
        self.include_target_j = bool(include_target_j)
        self.response_sign = float(response_sign)


class RHFMOReciprocal:
    '''Zero-order reciprocal objects retained for derivative consumers.'''

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
        '''Return the scheduler plan with the exact Gamma reduction scope.'''
        metadata = self.memory_plan.as_dict()
        metadata.update({
            'full_grid_count': self.full_grid_count,
            'evaluated_grid_count': len(self.vectors),
            'gamma_conjugate_symmetry': self.gamma_conjugate_symmetry,
        })
        return metadata


def build_short_range_rhf_mo_projections(
        intopt, occupied_coefficients, response_orbitals,
        auxiliary_batch_size=None, auxiliary_block_size=None):
    r'''Build short-range ``Boo`` and ``Bou`` without an AO-pair tensor.

    Args:
        intopt : gpu4pyscf.pbc.df.int3c2e.SRInt3c2eOpt
            Built real-Gamma short-range three-center optimizer.
        occupied_coefficients : cupy.ndarray, shape (nao, nocc)
            Bare occupied coefficients ``C`` in the optimizer's sorted and
            padded AO order.
        response_orbitals : cupy.ndarray, shape (nao, nocc)
            Bare response orbitals ``U=C_v Z`` in the same AO order.

    Kwargs:
        auxiliary_batch_size : int or None
            Native evaluator auxiliary batch.  ``None`` selects a bounded
            value from currently free device memory.
        auxiliary_block_size : int or None
            Maximum auxiliary rows projected together.  ``None`` uses the
            same joint memory plan as the evaluator batch.

    Returns:
        tuple
            ``(Boo, Bou, metadata)``.  Both tensors have shape
            ``(naux,nocc,nocc)`` and are real FP64 arrays.  No packed or dense
            persistent ``B[P,mu,nu]`` is allocated.

    Notes:
        The temporary AO matrix is symmetric within each auxiliary block.
        Consequently off-diagonal AO pairs are visited through both matrix
        positions.  No packed-pair factor of two belongs in ``Boo`` or
        ``Bou``; those weights are required only when a symmetric source is
        stored as one value per unique AO pair.
    '''
    occupied, response = _validate_sorted_orbitals(
        intopt, occupied_coefficients, response_orbitals)
    nao, nocc = map(int, occupied.shape)
    naux = int(intopt.auxcell.ao_loc[-1])
    pair_addresses = intopt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    if len(pair_addresses) < 1:
        raise ValueError('short-range optimizer has no AO pairs')
    i_addr, j_addr = divmod(pair_addresses, nao)
    batch_size, block_size, plan = _short_range_work_blocks(
        len(pair_addresses), naux, len(intopt.bvkmesh_Ls), nao, nocc,
        auxiliary_batch_size=auxiliary_batch_size,
        auxiliary_block_size=auxiliary_block_size)

    evaluator, _, _, auxiliary_offsets = intopt.int3c2e_evaluator(
        aux_batch_size=batch_size, cart=True)
    occupied_projection = cp.empty(
        (naux, nocc, nocc), dtype=cp.float64)
    mixed_projection = cp.empty_like(occupied_projection)
    ao_block = cp.zeros((nao, nao, block_size), dtype=cp.float64)
    compressed_buffer = cp.empty(
        (batch_size, len(pair_addresses)), dtype=cp.float64)
    transformed = cp.empty(
        (nocc, nao, block_size), dtype=cp.float64)

    offset = 0
    for batch_id in range(len(auxiliary_offsets) - 1):
        compressed = evaluator(
            aux_batch_id=batch_id, out=compressed_buffer)[:, 0, :]
        for p0, p1 in lib.prange(0, compressed.shape[1], block_size):
            width = p1 - p0
            p2 = offset + width
            if p2 > naux:
                raise RuntimeError(
                    'short-range projection produced more auxiliary rows '
                    f'than expected ({p2}>{naux})')
            three_center = ao_block[:, :, :width]
            # Screened shell-pair lists do not necessarily cover every dense
            # AO position.  Clear the reused tail block so an omitted pair can
            # never inherit a value from the preceding auxiliary batch.
            three_center[...] = 0.
            three_center[j_addr, i_addr] = three_center[i_addr, j_addr] = (
                compressed[:, p0:p1])
            work = transformed[:, :, :width]
            contract('pqr,pi->iqr', three_center, occupied, out=work)
            contract(
                'iqr,qj->rij', work, occupied,
                out=occupied_projection[offset:p2])
            contract(
                'iqr,qj->rij', work, response,
                out=mixed_projection[offset:p2])
            offset = p2
    if offset != naux:
        raise RuntimeError(
            f'short-range projection produced {offset} of {naux} rows')
    return occupied_projection, mixed_projection, plan


def add_reciprocal_rhf_mo_projections(
        intopt, occupied_projection, mixed_projection, metric,
        occupied_coefficients, response_orbitals, reciprocal_options):
    r'''Accumulate reciprocal ``Boo``, ``Bou`` and ``M`` in one traversal.

    The Fourier evaluator, exact real-Gamma ``G,-G`` representative set,
    integer multiplicities, Coulomb weights, transactional scheduler, and OOM
    rebatching policy are the same primitives used by the native packed
    route.  Only the accepted three-center result differs: each AO-pair
    Fourier block is projected immediately to ``oo`` and ``ou`` space.

    All three output arrays are modified in place.  The returned reciprocal
    object retains the Fourier evaluator and resolved schedule needed by a
    subsequent factorized derivative consumer.
    '''
    occupied, response = _validate_sorted_orbitals(
        intopt, occupied_coefficients, response_orbitals)
    occupied_projection, mixed_projection, metric = _validate_projections(
        intopt, occupied_projection, mixed_projection, metric,
        occupied.shape[1])
    options = _validate_reciprocal_options(reciprocal_options)
    cell = intopt.cell
    auxcell = intopt.auxcell
    mesh = intopt.mesh
    nao, nocc = map(int, occupied.shape)
    naux = int(auxcell.ao_loc[-1])
    pair_addresses = intopt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)

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
    gamma_symmetry = options['gamma_conjugate_symmetry']
    if gamma_symmetry:
        indices, multiplicities = gamma_conjugate_reciprocal_indices(mesh)
        vectors = vectors[indices]
        coulomb = coulomb[indices] * asarray(multiplicities)

    memory_plan = plan_gamma_reciprocal_blocks(
        len(vectors), nao, naux, nocc,
        free_device_bytes=int(get_avail_mem(exclude_memory_pool=True)),
        workspace_model='rhf_mo_direct',
        **{name: options[name] for name in (
            'memory_fraction', 'minimum_reserve_bytes', 'block_alignment',
            'maximum_block_size')})

    def worker(p0, p1):
        ngrid = p1 - p0
        auxiliary_fourier = ft_ao.ft_ao(auxcell, vectors[p0:p1]).T
        weighted_auxiliary = auxiliary_fourier * coulomb[p0:p1]
        pair_fourier = evaluator(vectors[p0:p1]).view(np.float64).reshape(
            nao, nao, ngrid * 2)
        pair_fourier[j_addr, i_addr] = pair_fourier[i_addr, j_addr]
        transformed = contract(
            'pqG,pi->iqG', pair_fourier, occupied)
        occupied_fourier = contract(
            'iqG,qj->ijG', transformed, occupied)
        mixed_fourier = contract(
            'iqG,qj->ijG', transformed, response)
        return (auxiliary_fourier, weighted_auxiliary,
                occupied_fourier, mixed_fourier)

    def accept_result(_p0, _p1, value):
        auxiliary_fourier, weighted_auxiliary, occupied_fourier, \
            mixed_fourier = value
        contract(
            'iG,jG->ij', auxiliary_fourier.view(np.float64),
            weighted_auxiliary.view(np.float64), beta=1., out=metric)
        contract(
            'rG,ijG->rij', weighted_auxiliary.view(np.float64),
            occupied_fourier, beta=1., out=occupied_projection)
        contract(
            'rG,ijG->rij', weighted_auxiliary.view(np.float64),
            mixed_fourier, beta=1., out=mixed_projection)

    execution = execute_gamma_reciprocal_blocks(
        len(vectors), memory_plan.initial_block_size, worker, accept_result,
        phase='zero_order_rhf_mo_projected', is_oom=_is_gpu_out_of_memory,
        block_alignment=memory_plan.block_alignment,
        on_oom=_release_cached_gpu_blocks)
    return RHFMOReciprocal(
        ft_opt, evaluator, vectors, coulomb, execution.final_block_size,
        memory_plan, execution, full_grid_count, gamma_symmetry)


def solve_rhf_mo_projections_inplace(
        metric_inverse, occupied_projection, mixed_projection):
    r'''Apply one auxiliary inverse to ``Boo`` and ``Bou`` in place.

    ``metric_inverse`` must be the primitive-basis inverse returned by the
    native ``_metric_inverse(auxcell, metric)`` operation.  Each solve uses
    the native bounded AO-column algorithm, so no second complete ``O`` or
    ``R`` tensor is retained.  ``R`` is intentionally left nonsymmetric.

    Returns:
        tuple of dict
            Native workspace plans for the ``oo`` and ``ou`` solves.
    '''
    inverse = _as_real_array(metric_inverse, 'metric_inverse')
    occupied = _as_projection_stack(
        occupied_projection, 'occupied_projection')
    mixed = _as_projection_stack(mixed_projection, 'mixed_projection')
    if occupied.shape != mixed.shape:
        raise ValueError('occupied and mixed projections must have one shape')
    naux = int(occupied.shape[0])
    if inverse.shape != (naux, naux):
        raise ValueError(
            'metric_inverse shape must match the auxiliary projection axis')

    # Import lazily to avoid the native_gamma/rhf_mo_response import cycle
    # while reusing the exact bounded solve implementation.
    from dcforge.gpu.pbc.df.gdf.native_gamma import (
        _solve_three_center_inplace,
    )

    occupied_plan = _solve_three_center_inplace(inverse, occupied)
    mixed_plan = _solve_three_center_inplace(inverse, mixed)
    # O is symmetric analytically because W is symmetric in its AO indices.
    # Enforce that representation against contraction-order roundoff.  R has
    # different left/right orbitals and must never be symmetrized.
    occupied[...] = .5 * (
        occupied + occupied.transpose(0, 2, 1))
    return occupied_plan, mixed_plan


def make_rhf_mo_response_factors(
        occupied_coefficients, response_orbitals, occupied_projection,
        mixed_projection, include_target_j=False, response_sign=1.,
        overwrite_projections=False):
    r'''Construct ``a,b,A,K,Gamma`` from metric-solved ``O`` and ``R``.

    ``R=C^T W U`` is generally nonsymmetric.  Both ``R`` and ``R.T`` appear
    explicitly in ``A``; replacing it by a symmetrized matrix would lose the
    orientation needed by the exact AO source.  ``response_sign`` is the
    signed-polarization coefficient used by the current native packed route.
    With ``overwrite_projections=True``, ``O`` storage becomes ``K`` and
    ``R`` storage becomes ``A`` only after ``a``, ``b``, and ``Gamma`` have
    been completed.  This retains two, rather than four, persistent
    ``(naux,nocc,nocc)`` stacks.
    '''
    occupied = _as_real_array(
        occupied_coefficients, 'occupied_coefficients')
    response = _as_real_array(response_orbitals, 'response_orbitals')
    if occupied.ndim != 2 or response.shape != occupied.shape:
        raise ValueError(
            'occupied_coefficients and response_orbitals must have one '
            '(nao,nocc) shape')
    occupied_projection = _as_projection_stack(
        occupied_projection, 'occupied_projection')
    mixed_projection = _as_projection_stack(
        mixed_projection, 'mixed_projection')
    if occupied_projection.shape != mixed_projection.shape:
        raise ValueError('occupied and mixed projections must have one shape')
    if occupied_projection.shape[1:] != (
            occupied.shape[1], occupied.shape[1]):
        raise ValueError(
            'projection occupied dimensions do not match the MO factors')
    response_sign = _response_sign(response_sign)
    overwrite_projections = bool(overwrite_projections)

    auxiliary_reference = 2. * cp.trace(
        occupied_projection, axis1=1, axis2=2)
    auxiliary_response = 2. * cp.trace(
        mixed_projection, axis1=1, axis2=2)
    nocc = int(occupied.shape[1])
    diagonal = cp.arange(nocc)

    # Gamma still needs the original O/R values.  Complete it before the
    # optional O->K and R->A storage reuse below.
    cross = contract(
        'pij,qij->pq', mixed_projection, occupied_projection)
    metric_source = -4. * response_sign * (cross + cross.T)
    metric_source += 2. * response_sign * (
        cp.outer(auxiliary_response, auxiliary_reference)
        + cp.outer(auxiliary_reference, auxiliary_response))
    if include_target_j:
        metric_source += cp.outer(
            auxiliary_reference, auxiliary_reference)

    if overwrite_projections:
        # The transpose makes R->A an overlapping transformation.  Form one
        # bounded-lifetime symmetric temporary before overwriting R; unlike a
        # persistent A allocation, it is released before derivative kernels.
        symmetric_mixed = (
            mixed_projection + mixed_projection.transpose(0, 2, 1))
        source_factor = mixed_projection
        source_factor[...] = -2. * response_sign * symmetric_mixed
        del symmetric_mixed
    else:
        source_factor = -2. * response_sign * (
            mixed_projection + mixed_projection.transpose(0, 2, 1))
    source_factor[:, diagonal, diagonal] += (
        4. * response_sign * auxiliary_response[:, None])
    if include_target_j:
        source_factor[:, diagonal, diagonal] += (
            2. * auxiliary_reference[:, None])

    if overwrite_projections:
        response_factor = occupied_projection
        response_factor *= -2. * response_sign
    else:
        response_factor = -2. * response_sign * occupied_projection
    response_factor[:, diagonal, diagonal] += (
        2. * response_sign * auxiliary_reference[:, None])

    return RHFMOResponseFactors(
        occupied, response,
        None if overwrite_projections else occupied_projection,
        None if overwrite_projections else mixed_projection,
        auxiliary_reference, auxiliary_response, source_factor,
        response_factor, metric_source, include_target_j, response_sign)


def build_rhf_mo_response_factors(
        intopt, occupied_coefficients, response_orbitals,
        reciprocal_options, include_target_j=False, response_sign=1.,
        auxiliary_batch_size=None, auxiliary_block_size=None,
        retain_projections=False):
    r'''Build the complete exact MO-projected zero-order response state.

    This orchestration evaluates short-range and reciprocal ``Boo/Bou``,
    forms the native auxiliary metric, applies the native primitive-basis
    inverse in place, and returns ``O,R,a,b,A,K,Gamma``.  It never constructs
    a persistent ``(naux,npair)`` or ``(naux,nao,nao)`` three-center tensor.
    By default, the completed ``O`` and ``R`` buffers are reused in place for
    ``K`` and ``A`` after ``Gamma`` has been formed.  Set
    ``retain_projections=True`` only for an algebra/debug audit that also
    needs the explicit solved projections.

    Returns:
        tuple
            ``(factors, int2opt, reciprocal, metadata)``.  ``int2opt`` and
            ``reciprocal`` are retained because factorized derivative
            consumers need the identical metric and Fourier conventions.
    '''
    occupied, response = _validate_sorted_orbitals(
        intopt, occupied_coefficients, response_orbitals)
    occupied_projection, mixed_projection, short_range_plan = (
        build_short_range_rhf_mo_projections(
            intopt, occupied, response,
            auxiliary_batch_size=auxiliary_batch_size,
            auxiliary_block_size=auxiliary_block_size))

    auxcell = intopt.auxcell
    auxcell.rcut = _estimate_sr_2c2e_rcut(
        auxcell, intopt.omega, auxcell.precision * 1.e-6)
    int2opt = Int2c2eOpt(auxcell)
    metric = int2opt.int2c2e(sort_output=False)
    reciprocal = add_reciprocal_rhf_mo_projections(
        intopt, occupied_projection, mixed_projection, metric,
        occupied, response, reciprocal_options)

    # This lazy import deliberately calls the current native transformation:
    # apply_CT_mat_C, Cartesian-domain gate, contraction coefficients, and the
    # primitive-basis linear solve therefore remain bit-for-bit the same
    # definition as the packed oracle.
    from dcforge.gpu.pbc.df.gdf.native_gamma import _metric_inverse

    metric_inverse = _metric_inverse(auxcell, metric)
    del metric
    occupied_solve, mixed_solve = solve_rhf_mo_projections_inplace(
        metric_inverse, occupied_projection, mixed_projection)
    del metric_inverse
    factors = make_rhf_mo_response_factors(
        occupied, response, occupied_projection, mixed_projection,
        include_target_j=include_target_j, response_sign=response_sign,
        overwrite_projections=not bool(retain_projections))
    metadata = {
        'route': 'experimental_real_gamma_rhf_mo_projected_zero_order',
        'domain': 'real_gamma_closed_shell_rhf',
        'source_storage': 'factorized_mo_no_packed_ao_pairs',
        'include_target_j': bool(include_target_j),
        'response_sign': float(_response_sign(response_sign)),
        'nao': int(occupied.shape[0]),
        'nocc': int(occupied.shape[1]),
        'naux': int(occupied_projection.shape[0]),
        'short_range_workspace_plan': short_range_plan,
        'occupied_metric_solve_workspace_plan': occupied_solve,
        'mixed_metric_solve_workspace_plan': mixed_solve,
        'reciprocal_plan': reciprocal.plan_metadata(),
        'reciprocal_zero_order': (
            reciprocal.zero_order_execution.as_dict()),
        'raw_screened_three_center_bytes_not_allocated': int(
            occupied_projection.shape[0]
            * short_range_plan['raw_pair_count'] * 8),
        'dense_three_center_bytes_not_allocated': int(
            occupied_projection.shape[0] * occupied.shape[0] ** 2 * 8),
        'projection_storage_reused_for_factors': not bool(
            retain_projections),
        'persistent_projection_bytes': (
            int(occupied_projection.nbytes + mixed_projection.nbytes)
            if retain_projections else 0),
        'persistent_factor_bytes': int(
            factors.source_factor.nbytes + factors.response_factor.nbytes
            + factors.metric_source.nbytes
            + factors.auxiliary_reference.nbytes
            + factors.auxiliary_response.nbytes),
        'production_integrated': False,
    }
    return factors, int2opt, reciprocal, metadata


def _short_range_work_blocks(
        raw_pair_count, naux, nbvk, nao, nocc,
        auxiliary_batch_size=None, auxiliary_block_size=None):
    '''Plan evaluator and direct MO-projection buffers jointly.'''
    available = int(get_avail_mem(exclude_memory_pool=True))
    itemsize = np.dtype(np.float64).itemsize
    persistent = 2 * int(naux) * int(nocc) ** 2 * itemsize
    reserve = min(512 * 1024 ** 2, max(0, available // 8))
    usable = max(1, available - persistent - reserve)
    evaluator_bytes_per_aux = max(
        1, int(raw_pair_count) * max(1, int(nbvk)) * itemsize)
    projection_bytes_per_aux = (
        int(nao) ** 2 + int(nocc) * int(nao)) * itemsize

    if auxiliary_batch_size is None:
        batch = max(1, min(
            int(naux), int(.35 * usable) // evaluator_bytes_per_aux))
    else:
        batch = _positive_block(
            auxiliary_batch_size, naux, 'auxiliary_batch_size')
    if auxiliary_block_size is None:
        block = max(1, min(
            int(naux), int(.35 * usable)
            // max(1, projection_bytes_per_aux)))
        if block >= 8:
            block = max(8, block // 8 * 8)
    else:
        block = _positive_block(
            auxiliary_block_size, naux, 'auxiliary_block_size')

    return int(batch), int(block), {
        'layout': 'direct_sorted_ao_to_oo_and_ou',
        'raw_pair_count': int(raw_pair_count),
        'available_bytes': available,
        'persistent_projection_bytes': persistent,
        'reserve_bytes': reserve,
        'usable_workspace_bytes': usable,
        'evaluator_aux_batch_size': int(batch),
        'projection_aux_block_size': int(block),
        'evaluator_estimated_bytes': int(
            batch * evaluator_bytes_per_aux),
        'projection_estimated_bytes': int(
            block * projection_bytes_per_aux),
        'persistent_packed_three_center_allocated': False,
    }


def _validate_sorted_orbitals(intopt, occupied_coefficients,
                              response_orbitals):
    '''Validate bare real ``C,U`` factors in the optimizer's sorted AO basis.'''
    occupied = _as_real_array(
        occupied_coefficients, 'occupied_coefficients')
    response = _as_real_array(response_orbitals, 'response_orbitals')
    if occupied.ndim != 2 or response.ndim != 2:
        raise ValueError(
            'occupied_coefficients and response_orbitals must be 2D')
    if occupied.shape != response.shape:
        raise ValueError(
            'occupied_coefficients and response_orbitals must have one shape')
    if occupied.shape[1] < 1:
        raise ValueError('occupied_coefficients must contain an occupied MO')
    required_nao = int(intopt.cell.ao_loc[-1])
    if occupied.shape[0] != required_nao:
        raise ValueError(
            'MO factors must use the optimizer sorted/padded AO dimension '
            f'{required_nao}, got {occupied.shape[0]}')
    return occupied, response


def _validate_projections(
        intopt, occupied_projection, mixed_projection, metric, nocc):
    '''Validate mutable zero-order outputs before reciprocal accumulation.'''
    occupied = _as_projection_stack(
        occupied_projection, 'occupied_projection')
    mixed = _as_projection_stack(mixed_projection, 'mixed_projection')
    metric = _as_real_array(metric, 'metric')
    naux = int(intopt.auxcell.ao_loc[-1])
    expected = (naux, int(nocc), int(nocc))
    if occupied.shape != expected or mixed.shape != expected:
        raise ValueError(
            f'projection shapes must both be {expected}, got '
            f'{occupied.shape} and {mixed.shape}')
    if metric.shape != (naux, naux):
        raise ValueError(
            f'metric shape must be {(naux, naux)}, got {metric.shape}')
    return occupied, mixed, metric


def _validate_reciprocal_options(options):
    '''Return the native scheduler controls required by this experiment.'''
    required = (
        'memory_fraction', 'minimum_reserve_bytes', 'block_alignment',
        'maximum_block_size', 'gamma_conjugate_symmetry')
    try:
        resolved = {
            'memory_fraction': float(options['memory_fraction']),
            'minimum_reserve_bytes': int(options['minimum_reserve_bytes']),
            'block_alignment': int(options['block_alignment']),
            'maximum_block_size': options['maximum_block_size'],
            'gamma_conjugate_symmetry': bool(
                options['gamma_conjugate_symmetry']),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            'reciprocal_options must define ' + ', '.join(required)) from error
    if resolved['maximum_block_size'] is not None:
        resolved['maximum_block_size'] = int(
            resolved['maximum_block_size'])
    return resolved


def _as_projection_stack(value, name):
    '''Return a real FP64 ``(naux,nocc,nocc)`` projection stack.'''
    array = _as_real_array(value, name)
    if (array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 1
            or array.shape[1] != array.shape[2]):
        raise ValueError(f'{name} must have shape (naux,nocc,nocc)')
    return array


def _as_real_array(value, name):
    '''Return one real FP64 CuPy array or reject complex Gamma data.'''
    array = cp.asarray(value)
    if cp.iscomplexobj(array):
        imaginary = float(cp.max(cp.abs(array.imag)).get())
        if imaginary > 1.e-10:
            raise NotImplementedError(
                f'{name} must be real for real-Gamma RHF response')
        array = array.real
    return cp.asarray(array, dtype=cp.float64)


def _response_sign(value):
    '''Return an accepted signed-polarization coefficient.'''
    value = float(value)
    if value not in (-1., 1.):
        raise ValueError('response_sign must be +1 or -1')
    return value


def _positive_block(value, maximum, name):
    '''Return one positive block capped at the complete auxiliary extent.'''
    value = int(value)
    if value < 1:
        raise ValueError(f'{name} must be positive')
    return min(value, int(maximum))


def _is_gpu_out_of_memory(error):
    '''Identify CUDA allocation failures eligible for reciprocal rebatching.'''
    if isinstance(error, cp.cuda.memory.OutOfMemoryError):
        return True
    message = str(error).lower()
    return (
        error.__class__.__name__ == 'CUDARuntimeError'
        and ('out of memory' in message
             or 'cudaerrormemoryallocation' in message)
    )


def _release_cached_gpu_blocks(_start, _stop, _error):
    '''Return a failed transaction's cached blocks before retrying it.'''
    cp.get_default_memory_pool().free_all_blocks()


__all__ = [
    'RHFMOReciprocal',
    'RHFMOResponseFactors',
    'add_reciprocal_rhf_mo_projections',
    'build_rhf_mo_response_factors',
    'build_short_range_rhf_mo_projections',
    'make_rhf_mo_response_factors',
    'solve_rhf_mo_projections_inplace',
]
