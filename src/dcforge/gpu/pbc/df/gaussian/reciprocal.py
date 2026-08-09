'''Bounded reciprocal-space work scheduling for native Gamma Gaussian DF.

Native GDF and RSDF response use AO-pair Fourier tensors in their
zero-order and derivative reciprocal passes.  Its storage is proportional to
``nAO**2 * nG_block`` and must therefore be decided before an allocation is
attempted.  This module owns only that execution policy: it neither changes a
reciprocal vector nor evaluates a density-fitting integral.

The scheduler is deliberately independent of CuPy.  It can be unit-tested
with a synthetic allocation failure and the native CUDA route supplies the
backend-specific out-of-memory predicate.
'''

import numpy as np

COMPLEX128_BYTES = 16
DEFAULT_MEMORY_FRACTION = .10
DEFAULT_MINIMUM_RESERVE_BYTES = 512 * 1024 ** 2
DEFAULT_BLOCK_ALIGNMENT = 32
DEFAULT_MAX_BLOCK_SIZE = 512
DEFAULT_WORKSPACE_MODEL = 'generic'
RHF_MO_DIRECT_WORKSPACE_MODEL = 'rhf_mo_direct'


def gamma_conjugate_reciprocal_indices(mesh):
    r'''Return one representative of every real-Gamma ``G,-G`` pair.

    For real Gamma-point AO products, auxiliary functions, and adjoint
    sources, the reciprocal integrand at ``-G`` is the complex conjugate of
    the integrand at ``G``.  The final real contraction can therefore
    evaluate one representative with multiplicity two.  Even FFT meshes
    contain negative Nyquist vectors whose positive partners are absent from
    ``numpy.fft.fftfreq``; those vectors are retained individually with
    multiplicity one instead of being treated as self-conjugate.

    Args:
        mesh : sequence of three int
            Resolved three-dimensional reciprocal mesh.

    Returns:
        tuple of numpy.ndarray
            Flat representative indices and their integer multiplicities.
            The multiplicities always sum to ``prod(mesh)``.
    '''
    mesh = np.asarray(mesh, dtype=np.int64)
    if mesh.shape != (3,) or np.any(mesh < 1):
        raise ValueError('Gamma reciprocal mesh must contain three positive values')
    axes = [
        np.rint(np.fft.fftfreq(int(size), d=1. / int(size))).astype(np.int64)
        for size in mesh
    ]
    vectors = np.stack(
        np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    addresses = {tuple(vector): index for index, vector in enumerate(vectors)}
    used = np.zeros(len(vectors), dtype=bool)
    representatives = []
    multiplicities = []
    for index, vector in enumerate(vectors):
        if used[index]:
            continue
        partner = addresses.get(tuple(-vector))
        representatives.append(index)
        if partner is not None and partner != index:
            used[index] = used[partner] = True
            multiplicities.append(2)
        else:
            used[index] = True
            multiplicities.append(1)
    representatives = np.asarray(representatives, dtype=np.int64)
    multiplicities = np.asarray(multiplicities, dtype=np.float64)
    if int(np.sum(multiplicities)) != int(np.prod(mesh)) or not np.all(used):
        raise RuntimeError('Gamma reciprocal conjugate partition is incomplete')
    return representatives, multiplicities


class ReciprocalBlockPlan:
    '''Static reciprocal-workspace decision for one Gamma Gaussian-DF pass.'''

    def __init__(self, ngrids, nao, naux, nocc_max, free_device_bytes,
                 memory_fraction, minimum_reserve_bytes, budget_bytes,
                 reserve_bytes, build_bytes_per_grid,
                 derivative_bytes_per_grid, block_alignment,
                 initial_block_size, maximum_block_size,
                 workspace_model=DEFAULT_WORKSPACE_MODEL,
                 build_values_per_grid=None,
                 derivative_values_per_grid=None):
        self.ngrids = int(ngrids)
        self.nao = int(nao)
        self.naux = int(naux)
        self.nocc_max = int(nocc_max)
        self.free_device_bytes = int(free_device_bytes)
        self.memory_fraction = float(memory_fraction)
        self.minimum_reserve_bytes = int(minimum_reserve_bytes)
        self.budget_bytes = int(budget_bytes)
        self.reserve_bytes = int(reserve_bytes)
        self.build_bytes_per_grid = int(build_bytes_per_grid)
        self.derivative_bytes_per_grid = int(derivative_bytes_per_grid)
        self.workspace_model = str(workspace_model)
        self.build_values_per_grid = (
            None if build_values_per_grid is None
            else dict(build_values_per_grid))
        self.derivative_values_per_grid = (
            None if derivative_values_per_grid is None
            else dict(derivative_values_per_grid))
        self.bytes_per_grid = max(
            self.build_bytes_per_grid, self.derivative_bytes_per_grid)
        self.block_alignment = int(block_alignment)
        self.initial_block_size = int(initial_block_size)
        self.maximum_block_size = int(maximum_block_size)
        self.estimated_block_working_bytes = (
            self.initial_block_size * self.bytes_per_grid)

    def as_dict(self):
        '''Return JSON-safe data retained by the gradient driver.'''
        return {
            'ngrids': self.ngrids,
            'nao': self.nao,
            'naux': self.naux,
            'nocc_max': self.nocc_max,
            'free_device_bytes_at_plan': self.free_device_bytes,
            'memory_fraction': self.memory_fraction,
            'minimum_reserve_bytes': self.minimum_reserve_bytes,
            'reserve_bytes': self.reserve_bytes,
            'budget_bytes': self.budget_bytes,
            'build_bytes_per_grid': self.build_bytes_per_grid,
            'derivative_bytes_per_grid': self.derivative_bytes_per_grid,
            'bytes_per_grid': self.bytes_per_grid,
            'workspace_model': self.workspace_model,
            'build_values_per_grid': self.build_values_per_grid,
            'derivative_values_per_grid': self.derivative_values_per_grid,
            'block_alignment': self.block_alignment,
            'initial_block_size': self.initial_block_size,
            'maximum_block_size': self.maximum_block_size,
            'estimated_block_working_bytes': self.estimated_block_working_bytes,
        }


class ReciprocalBlockExecution:
    '''Completed transactional traversal of a reciprocal grid.'''

    def __init__(self, phase, initial_block_size, final_block_size,
                 minimum_block_size, completed_count, block_count,
                 rebatch_count):
        self.phase = str(phase)
        self.initial_block_size = int(initial_block_size)
        self.final_block_size = int(final_block_size)
        self.minimum_block_size = int(minimum_block_size)
        self.completed_count = int(completed_count)
        self.block_count = int(block_count)
        self.rebatch_count = int(rebatch_count)

    def as_dict(self):
        '''Return JSON-safe completion information for one pass.'''
        return {
            'phase': self.phase,
            'initial_block_size': self.initial_block_size,
            'final_block_size': self.final_block_size,
            'minimum_block_size': self.minimum_block_size,
            'completed_count': self.completed_count,
            'block_count': self.block_count,
            'rebatch_count': self.rebatch_count,
        }


class _ReciprocalBlockScheduler:
    '''Advance only after a native reciprocal block was fully accumulated.'''

    def __init__(self, total_count, block_size):
        if int(total_count) < 1 or int(block_size) < 1:
            raise ValueError('total_count and block_size must be positive')
        self.total_count = int(total_count)
        self.block_size = min(int(block_size), self.total_count)
        self.completed_count = 0
        self.block_count = 0
        self.rebatch_count = 0
        self.minimum_block_size = self.block_size

    @property
    def done(self):
        '''Whether every reciprocal vector has been committed.'''
        return self.completed_count == self.total_count

    def next_range(self):
        '''Return the uncommitted interval without advancing it.'''
        if self.done:
            raise StopIteration('all reciprocal vectors are complete')
        start = self.completed_count
        return start, min(self.total_count, start + self.block_size)

    def accept(self, stop):
        '''Commit precisely the interval returned by :meth:`next_range`.'''
        _, expected_stop = self.next_range()
        if int(stop) != expected_stop:
            raise ValueError(
                f'block ending at {stop} does not match pending reciprocal '
                f'interval ending at {expected_stop}')
        self.completed_count = expected_stop
        self.block_count += 1

    def shrink_after_oom(self, alignment):
        '''Halve an uncommitted interval, retaining its starting vector.'''
        if self.block_size == 1:
            start, _ = self.next_range()
            raise MemoryError(
                'native Gamma Gaussian-DF reciprocal response ran out of GPU memory '
                f'at block size one (vector {start}/{self.total_count})')
        block_size = max(1, self.block_size // 2)
        if block_size >= int(alignment):
            block_size = max(
                int(alignment), block_size // int(alignment) * int(alignment))
        self.block_size = block_size
        self.minimum_block_size = min(self.minimum_block_size, block_size)
        self.rebatch_count += 1
        return block_size

    def result(self, phase, initial_block_size):
        '''Build the public completion record.'''
        return ReciprocalBlockExecution(
            phase, initial_block_size, self.block_size,
            self.minimum_block_size, self.completed_count,
            self.block_count, self.rebatch_count)


def plan_gamma_reciprocal_blocks(ngrids, nao, naux, nocc_max,
                                 free_device_bytes,
                                 memory_fraction=DEFAULT_MEMORY_FRACTION,
                                 minimum_reserve_bytes=
                                 DEFAULT_MINIMUM_RESERVE_BYTES,
                                 block_alignment=DEFAULT_BLOCK_ALIGNMENT,
                                 maximum_block_size=DEFAULT_MAX_BLOCK_SIZE,
                                 workspace_model=DEFAULT_WORKSPACE_MODEL):
    r'''Plan a bounded Gamma reciprocal block from its largest live tensors.

    Args:
        ngrids : int
            Number of reciprocal vectors in the derivative mesh.
        nao : int
            Sorted AO dimension.
        naux : int
            Auxiliary AO dimension.
        nocc_max : int
            Largest low-rank signed-density factor width.
        free_device_bytes : int
            Free CUDA memory immediately before reciprocal allocation.

    Kwargs:
        memory_fraction : float
            Maximum fraction of currently free device memory assigned to this
            transaction.  The default is deliberately conservative because
            CuPy/CUDA kernels can require opaque temporary workspaces.
        minimum_reserve_bytes : int
            Memory not assigned to reciprocal work even when the fraction
            would permit it.
        block_alignment : int
            Preferred SIMD-friendly G-vector alignment.
        maximum_block_size : int or None
            Absolute cap on one transaction.  ``None`` disables the cap.
        workspace_model : {'generic', 'rhf_mo_direct'}
            Live-tensor accounting model.  The default preserves the
            established signed/packed planner.  ``'rhf_mo_direct'`` includes
            both occupied projections in zero order and the complete
            ``C/U/A/K`` contraction workspace of the exact MO-factorized
            long-range derivative.

    Returns:
        ReciprocalBlockPlan
            A reproducible allocation decision.

    Notes:
        The zero-order pass retains an AO-pair Fourier tensor, an occupied
        transform, and two auxiliary Fourier arrays.  The derivative pass
        additionally retains a second AO-pair source and two auxiliary
        sources.  The larger per-vector value is used for both passes, so a
        block that succeeds in the first pass is also preflighted for the
        second pass.
    '''
    values = (ngrids, nao, naux, nocc_max, free_device_bytes)
    if any(int(value) < 1 for value in values):
        raise ValueError('grid, AO, auxiliary, and memory dimensions must be positive')
    if not 0. < float(memory_fraction) <= 1.:
        raise ValueError('memory_fraction must lie in (0, 1]')
    if int(minimum_reserve_bytes) < 0:
        raise ValueError('minimum_reserve_bytes cannot be negative')
    if int(block_alignment) < 1:
        raise ValueError('block_alignment must be positive')
    if maximum_block_size is not None and int(maximum_block_size) < 1:
        raise ValueError('maximum_block_size must be positive when set')

    workspace_model = str(workspace_model).lower()
    if workspace_model == DEFAULT_WORKSPACE_MODEL:
        build_components = {
            'ao_pair_fourier': int(nao) ** 2,
            'occupied_ao_transform': int(nocc_max) * int(nao),
            'occupied_pair_projection': int(nocc_max) ** 2,
            'auxiliary_fourier_channels': 2 * int(naux),
        }
        derivative_components = {
            'ao_pair_and_derivative_source': 2 * int(nao) ** 2,
            'occupied_ao_transform': int(nocc_max) * int(nao),
            'occupied_pair_projection': int(nocc_max) ** 2,
            'auxiliary_fourier_and_sources': 4 * int(naux),
            'reciprocal_vector_coordinates': 2,
        }
    elif workspace_model == RHF_MO_DIRECT_WORKSPACE_MODEL:
        build_components = {
            'ao_pair_fourier': int(nao) ** 2,
            'occupied_ao_transform': int(nocc_max) * int(nao),
            'occupied_and_mixed_pair_projections': 2 * int(nocc_max) ** 2,
            'auxiliary_fourier_channels': 2 * int(naux),
        }
        derivative_components = {
            'ao_pair_and_derivative_source': 2 * int(nao) ** 2,
            # C/U transforms and A/K back-transforms are distinct live
            # contraction channels in the transactional MO consumer.
            'occupied_ao_contraction_channels': (
                2 * int(nocc_max) * int(nao)),
            'occupied_pair_contraction_channels': 4 * int(nocc_max) ** 2,
            'auxiliary_fourier_and_sources': 4 * int(naux),
            'reciprocal_vector_coordinates': 2,
        }
    else:
        raise ValueError(
            'workspace_model must be generic or rhf_mo_direct')

    build_values = sum(build_components.values())
    derivative_values = sum(derivative_components.values())
    build_bytes = build_values * COMPLEX128_BYTES
    derivative_bytes = derivative_values * COMPLEX128_BYTES
    bytes_per_grid = max(build_bytes, derivative_bytes)
    fraction_budget = int(int(free_device_bytes) * float(memory_fraction))
    reserve_budget = max(0, int(free_device_bytes) - int(minimum_reserve_bytes))
    budget = min(fraction_budget, reserve_budget)
    reserve = int(free_device_bytes) - budget
    block_size = min(int(ngrids), budget // bytes_per_grid)
    if maximum_block_size is not None:
        block_size = min(block_size, int(maximum_block_size))
    if block_size < 1:
        raise MemoryError(
            'insufficient GPU memory for one native Gamma Gaussian-DF reciprocal '
            f'vector: budget={budget} bytes, required={bytes_per_grid} '
            f'bytes, free={free_device_bytes} bytes')
    if block_size >= int(block_alignment):
        block_size = max(
            int(block_alignment),
            block_size // int(block_alignment) * int(block_alignment))
    return ReciprocalBlockPlan(
        ngrids, nao, naux, nocc_max, free_device_bytes, memory_fraction,
        minimum_reserve_bytes, budget, reserve, build_bytes, derivative_bytes,
        block_alignment, block_size,
        int(ngrids) if maximum_block_size is None else maximum_block_size,
        workspace_model=workspace_model,
        build_values_per_grid=build_components,
        derivative_values_per_grid=derivative_components)


def execute_gamma_reciprocal_blocks(total_count, initial_block_size, worker,
                                    accept_result, phase, is_oom,
                                    block_alignment=DEFAULT_BLOCK_ALIGNMENT,
                                    on_oom=None):
    r'''Execute reciprocal blocks transactionally with OOM rebatching.

    ``worker(start, stop)`` may allocate and evaluate a native CUDA block.
    ``accept_result`` runs only when that complete block succeeds.  An OOM
    therefore retries the same starting vector at half the block size; no
    reciprocal contribution is silently omitted or accumulated twice.
    '''
    scheduler = _ReciprocalBlockScheduler(total_count, initial_block_size)
    while not scheduler.done:
        start, stop = scheduler.next_range()
        try:
            result = worker(start, stop)
        except Exception as error:
            if not is_oom(error):
                raise
            if on_oom is not None:
                on_oom(start, stop, error)
            try:
                scheduler.shrink_after_oom(block_alignment)
            except MemoryError as minimum_error:
                raise minimum_error from error
            continue
        accept_result(start, stop, result)
        scheduler.accept(stop)
    return scheduler.result(phase, initial_block_size)
