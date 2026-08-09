'''Native Gamma RSDF three-center source-response contractions.

The central RSDF factor is ``C = L^{-1}B``.  Once reverse-Cholesky algebra
has produced ``dE/dB``, this module contracts it with the exact nuclear
derivative of the pre-metric source ``B``.  Short-range three-center
integrals and long-range Fourier derivatives are evaluated by GPU4PySCF CUDA
primitives.  In particular, it never asks the CPU implementation to build
the all-atom ``dB`` tensor.
'''

import ctypes
import math
from time import perf_counter

import cupy as cp
import numpy as np
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib.cupy_helper import asarray, empty_aligned, get_avail_mem
from gpu4pyscf.pbc.df import aft_jk, ft_ao
from gpu4pyscf.pbc.df.grad.rhf import (
    _split_l_ctr_pattern,
    get_ao_pair_loc,
)
from gpu4pyscf.pbc.df.int3c2e import (
    MAX_IMGS_PER_TASK,
    POOL_SIZE,
    SRInt3c2eOpt,
    libpbc,
)
from gpu4pyscf.pbc.df.rsdf_builder import _weighted_coulG_kpts
from pyscf import lib
from pyscf.gto import ATOM_OF

from dcforge.gpu.pbc.df.compatibility import derivative_int3c2e_scheme
from dcforge.gpu.pbc.df.gaussian.reciprocal import (
    execute_gamma_reciprocal_blocks,
    gamma_conjugate_reciprocal_indices,
    plan_gamma_reciprocal_blocks,
)


def build_source_optimizer(with_df):
    r'''Build the Gamma CUDA optimizer for the provider's RSDF source split.

    Args:
        with_df : dcforge.gpu.pbc.df.rsdf.RSDF
            Built provider.  ``omega`` and ``mesh_compact`` define this
            source; the independently resolved metric settings are not used.

    Returns:
        gpu4pyscf.pbc.df.int3c2e.SRInt3c2eOpt
            Sorted, built device optimizer with its source mesh fixed.
    '''
    if getattr(with_df, 'backend_name', None) != 'rsdf':
        raise TypeError('source response requires a dcforge GPU RSDF provider')
    if with_df.auxcell is None:
        with_df.build(j_only=False)
    cell = with_df.cell.copy()
    auxcell = with_df.auxcell.copy()
    cell.precision = float(with_df.precision_R)
    auxcell.precision = float(with_df.precision_R)
    optimizer = SRInt3c2eOpt(cell, auxcell, float(with_df.omega), None)
    optimizer.mesh = np.asarray(with_df.mesh_compact, dtype=int)
    return optimizer.build()


def contract_source_derivative(with_df, source_adjoints):
    r'''Contract one or more Gamma ``dE/dB`` tensors on the GPU.

    Args:
        with_df : dcforge.gpu.pbc.df.rsdf.RSDF
            Built RSDF provider.
        source_adjoints : cupy.ndarray
            Physical-AO source adjoints of shape
            ``(nobservable, nao*nao, naux)``.  This is the output of
            :func:`dcforge.gpu.pbc.df.rsdf.adjoint.push_cderi_adjoints`.

    Returns:
        tuple
            ``(value, metadata)``.  ``value`` has shape
            ``(nobservable,natm,3)`` in Ha/Bohr; metadata records the CUDA
            short-range/reciprocal timings and reciprocal memory traversal.

    Notes:
        The physical-to-sorted-AO transformation occurs once before either
        derivative pass.  The short-range and reciprocal terms are then
        contracted directly into nuclear scalars.  No ``dB`` or ``dC``
        tensor survives the function.
    '''
    optimizer = build_source_optimizer(with_df)
    start = perf_counter()
    sources = _sorted_sources(optimizer, source_adjoints)
    setup_seconds = perf_counter() - start

    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()
    start_event.record()
    short_range = _contract_short_range(optimizer, sources)
    stop_event.record()
    stop_event.synchronize()
    short_range_seconds = cp.cuda.get_elapsed_time(start_event, stop_event) / 1000.

    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()
    start_event.record()
    reciprocal, reciprocal_metadata = _contract_reciprocal(
        optimizer, sources, with_df)
    stop_event.record()
    stop_event.synchronize()
    reciprocal_seconds = cp.cuda.get_elapsed_time(start_event, stop_event) / 1000.
    return short_range + reciprocal, {
        'source_owner': 'GPU4PySCF native short-range and reciprocal kernels',
        'setup_seconds': setup_seconds,
        'short_range_seconds': short_range_seconds,
        'reciprocal_seconds': reciprocal_seconds,
        'reciprocal_memory': reciprocal_metadata,
    }


def _sorted_sources(optimizer, source_adjoints):
    '''Map physical ``(nao**2,naux)`` source adjoints to sorted CUDA AOs.'''
    source_adjoints = cp.asarray(source_adjoints)
    if source_adjoints.ndim == 2:
        source_adjoints = source_adjoints[None]
    physical_nao = int(optimizer.cell.cell.nao_nr())
    physical_naux = int(optimizer.auxcell.cell.nao_nr())
    expected = (physical_nao ** 2, physical_naux)
    if source_adjoints.ndim != 3 or source_adjoints.shape[1:] != expected:
        raise ValueError(
            'source adjoints must have shape '
            f'(nobservable,{expected[0]},{expected[1]}), got '
            f'{source_adjoints.shape}')
    if source_adjoints.dtype.kind == 'c':
        imaginary = float(cp.max(cp.abs(source_adjoints.imag)).get())
        if imaginary > 1.0e-10:
            raise NotImplementedError(
                'Gamma RSDF source adjoints must be real; maximum imaginary '
                f'component is {imaginary:.3e}')
        source_adjoints = source_adjoints.real
    output = []
    for source in source_adjoints:
        source = source.reshape(physical_nao, physical_nao, physical_naux)
        source = source.transpose(2, 0, 1)
        source = optimizer.cell.apply_C_mat_CT(source)
        source = optimizer.auxcell.apply_C_dot(source, axis=0)
        output.append(source.transpose(1, 2, 0))
    return cp.stack(output)


class _ShortRangeContext:
    '''Shared indexing and CUDA workspaces for Gamma short-range sources.'''

    def __init__(self, **values):
        self.__dict__.update(values)

    @classmethod
    def build(cls, optimizer):
        cell = optimizer.cell
        auxcell = optimizer.auxcell
        auxiliary_limit = max(1, min(int(auxcell.nbas), POOL_SIZE))
        shell_pair_limit = max(1, POOL_SIZE // auxiliary_limit)
        _unused, gout_stride, shared_sizes = derivative_int3c2e_scheme()
        lmax = int(cell.uniq_l_ctr[:, 0].max())
        laux = int(auxcell.uniq_l_ctr[:, 0].max())
        shared_memory = int(
            shared_sizes[:laux + 1, :lmax + 1, :lmax + 1].max())
        bas_ij_idx, shell_pair_offsets = cell.aggregate_shl_pairs(
            optimizer.bas_ij_cache, shell_pair_limit)
        ao_pair_loc = get_ao_pair_loc(
            cell.uniq_l_ctr[:, 0], optimizer.bas_ij_cache, cart=True)
        auxiliary_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
        auxiliary_offsets, patterns = _split_l_ctr_pattern(
            auxiliary_offsets, auxcell.uniq_l_ctr, auxiliary_limit)
        _validate_task_pool(shell_pair_offsets, auxiliary_offsets)
        pool = cp.empty(
            gpu_specs['multiProcessorCount'] * POOL_SIZE
            * (MAX_IMGS_PER_TASK + 2) + 1,
            dtype=np.uint32)
        return cls(
            optimizer=optimizer,
            cell=cell,
            auxcell=auxcell,
            bas_ij_idx=bas_ij_idx,
            shell_pair_offsets=shell_pair_offsets,
            ao_pair_loc=ao_pair_loc,
            auxiliary_offsets=auxiliary_offsets,
            auxiliary_offsets_gpu=cp.asarray(auxiliary_offsets, dtype=cp.int32),
            batch_count=len(patterns),
            gout_stride=gout_stride,
            shared_memory=shared_memory,
            pool=pool,
            head=pool[-1:],
            task_pool=empty_aligned(
                (gpu_specs['multiProcessorCount'], POOL_SIZE * 16),
                np.int32, alignment=128),
            diffuse_exponents=cp.asarray(optimizer.diffuse_exps),
            diffuse_coefficients=cp.asarray(optimizer.diffuse_coefs),
            log_cutoff=math.log(optimizer.cutoff),
        )

    def auxiliary_ao_range(self, batch_id):
        start = int(self.auxcell.ao_loc[self.auxiliary_offsets[batch_id]])
        stop = int(self.auxcell.ao_loc[self.auxiliary_offsets[batch_id + 1]])
        return start, stop

    def run(self, batch_id, compressed, ao_derivative, aux_derivative):
        start, stop = self.auxiliary_ao_range(batch_id)
        error = libpbc.PBCsr_ejk_int3c2e_ip1(
            ctypes.cast(ao_derivative.data.ptr, ctypes.c_void_p),
            ctypes.cast(aux_derivative.data.ptr, ctypes.c_void_p),
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(), ctypes.byref(self.optimizer.int3c2e_envs),
            ctypes.cast(self.pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.task_pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.head.data.ptr, ctypes.c_void_p),
            ctypes.c_int(self.shared_memory),
            ctypes.c_int(len(self.shell_pair_offsets) - 1), ctypes.c_int(1),
            ctypes.cast(self.bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.shell_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(
                self.auxiliary_offsets_gpu[batch_id:].data.ptr,
                ctypes.c_void_p),
            ctypes.cast(self.optimizer.img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.optimizer.img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(start), ctypes.c_int(self.auxcell.nbas),
            ctypes.c_int(stop - start),
            ctypes.cast(self.diffuse_exponents.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.diffuse_coefficients.data.ptr, ctypes.c_void_p),
            ctypes.c_float(self.log_cutoff))
        if error != 0:
            raise RuntimeError('PBCsr_ejk_int3c2e_ip1 failed')


def _contract_short_range(optimizer, sources):
    '''Contract ``dB_SR`` with all sorted source adjoints in one kernel plan.'''
    sources = cp.asarray(sources)
    if sources.ndim != 4:
        raise ValueError(
            'sorted Gamma sources must have shape '
            '(nobservable,nao,nao,naux)')
    context = _ShortRangeContext.build(optimizer)
    pair_addresses = optimizer.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    ao_derivative = cp.zeros((len(sources), context.cell.natm, 3))
    aux_derivative = cp.zeros_like(ao_derivative)
    for batch_id in range(context.batch_count):
        start, stop = context.auxiliary_ao_range(batch_id)
        width = stop - start
        compressed = cp.take(
            sources[..., start:stop].reshape(len(sources), -1, width),
            pair_addresses, axis=1)
        for source_id in range(len(sources)):
            context.run(
                batch_id, compressed[source_id], ao_derivative[source_id],
                aux_derivative[source_id])
    return (ao_derivative + aux_derivative) * 2.0


def _validate_task_pool(shell_pair_offsets, auxiliary_offsets):
    shell_pairs = np.diff(cp.asnumpy(shell_pair_offsets))
    auxiliary_shells = np.diff(auxiliary_offsets)
    if int(shell_pairs.max()) * int(auxiliary_shells.max()) > POOL_SIZE:
        raise RuntimeError('RSDF source derivative batch exceeds CUDA POOL_SIZE')


def _contract_reciprocal(optimizer, sources, with_df):
    '''Contract Gamma ``dB_LR+dB_vbar`` through bounded G-vector blocks.'''
    cell = optimizer.cell
    auxcell = optimizer.auxcell
    mesh = np.asarray(optimizer.mesh, dtype=int)
    vectors = asarray(auxcell.get_Gv(mesh))
    coulomb = _gamma_coulomb_vector(
        _weighted_coulG_kpts(cell, mesh, 0.0, float(optimizer.omega)),
        len(vectors))
    full_grid_count = len(vectors)
    gamma_symmetry = bool(
        with_df.source_reciprocal_gamma_conjugate_symmetry)
    if gamma_symmetry:
        indices, multiplicities = gamma_conjugate_reciprocal_indices(mesh)
        vectors = vectors[indices]
        coulomb = coulomb[indices] * asarray(multiplicities)
    ft_optimizer = ft_ao.FTOpt.from_intopt(optimizer)
    evaluator = ft_optimizer.ft_evaluator(
        compressing=False, cart=True, original_ao_order=False)[0]
    pair_addresses = ft_optimizer.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    nao = int(sources.shape[1])
    naux = int(sources.shape[-1])
    i_addr, j_addr = divmod(pair_addresses, nao)
    kernel_inputs = _fourier_kernel_inputs(cell, ft_optimizer)
    free_bytes = int(get_avail_mem(exclude_memory_pool=True))
    plan = plan_gamma_reciprocal_blocks(
        len(vectors), nao, naux * len(sources), 1, free_bytes,
        memory_fraction=float(with_df.source_reciprocal_memory_fraction),
        minimum_reserve_bytes=int(with_df.source_reciprocal_minimum_reserve_bytes),
        block_alignment=int(with_df.source_reciprocal_block_alignment),
        maximum_block_size=with_df.source_reciprocal_max_block_size)
    ao_derivative = cp.zeros((len(sources), cell.natm, 3))
    aux_partial = cp.zeros((len(sources), 3, naux))

    def worker(start, stop):
        return _reciprocal_block(
            cell, auxcell, sources, vectors, coulomb, evaluator, start, stop,
            nao=nao, i_addr=i_addr, j_addr=j_addr,
            permutation_symmetry=ft_optimizer.permutation_symmetry,
            **kernel_inputs)

    def accept(_start, _stop, result):
        ao_derivative[:] += result[0]
        aux_partial[:] += result[1]

    execution = execute_gamma_reciprocal_blocks(
        len(vectors), plan.initial_block_size, worker, accept,
        phase='rsdf_source_reciprocal', is_oom=_is_gpu_out_of_memory,
        block_alignment=plan.block_alignment, on_oom=_release_cached_blocks)
    value = ao_derivative * 2.0 + _group_auxiliary_batch(auxcell, aux_partial)
    return value, {
        'policy': {
            'memory_fraction': float(with_df.source_reciprocal_memory_fraction),
            'minimum_reserve_bytes': int(
                with_df.source_reciprocal_minimum_reserve_bytes),
            'maximum_block_size': with_df.source_reciprocal_max_block_size,
            'lifetime_shortening': (
                'AO-pair and source-contracted grid tensors are released '
                'before the next reciprocal block'),
            'gamma_conjugate_symmetry': gamma_symmetry,
            'full_grid_count': int(full_grid_count),
            'evaluated_grid_count': len(vectors),
        },
        'plan': plan.as_dict(),
        'execution': execution.as_dict(),
    }


def _reciprocal_block(cell, auxcell, sources, vectors, coulomb, evaluator,
                      start, stop, *, nao, i_addr, j_addr,
                      permutation_symmetry, **kernel_inputs):
    '''Evaluate one uncommitted Gamma reciprocal source-derivative block.'''
    width = int(stop - start)
    block_vectors = vectors[start:stop]
    pair_real = evaluator(block_vectors).view(np.float64).reshape(
        nao, nao, width * 2)
    pair_real[j_addr, i_addr] = pair_real[i_addr, j_addr]
    pair_complex = pair_real.view(np.complex128)
    auxiliary_g = ft_ao.ft_ao(auxcell, block_vectors).T
    auxiliary_weighted = auxiliary_g.conj() * coulomb[start:stop]
    auxiliary_source = cp.einsum(
        'spqr,pqG->srG', sources, pair_complex, optimize=True)
    auxiliary_source *= coulomb[start:stop]
    auxiliary_source = cp.ascontiguousarray(auxiliary_source).view(np.float64)
    auxiliary_partial = _auxiliary_phase_derivative(
        auxiliary_g, block_vectors, auxiliary_source)
    del pair_complex, pair_real, auxiliary_source
    ao_source = cp.einsum(
        'spqr,rG->spqG', sources, auxiliary_weighted, optimize=True)
    ao_source = cp.ascontiguousarray(ao_source).view(np.float64)
    ao_derivative = cp.stack([
        _run_fourier_kernel(
            cell, source, block_vectors, width,
            permutation_symmetry=permutation_symmetry, **kernel_inputs)
        for source in ao_source])
    return ao_derivative, auxiliary_partial


def _auxiliary_phase_derivative(auxiliary_g, vectors, source_real):
    '''Return source contractions of the auxiliary Fourier center response.'''
    naux = auxiliary_g.shape[0]
    result = cp.empty((len(source_real), 3, naux))
    for axis in range(3):
        derivative = auxiliary_g * (-1j * vectors[:, axis])
        derivative = cp.ascontiguousarray(derivative).view(np.float64)
        result[:, axis] = cp.einsum(
            'rG,srG->sr', derivative, source_real, optimize=True)
    return result


def _fourier_kernel_inputs(cell, ft_optimizer):
    bas_ij_idx, bas_ij_img_idx, shell_offsets = (
        aft_jk._generate_shl_pairs(ft_optimizer))
    return {
        'kernel': libpbc.PBC_ft_aopair_ek_ip1,
        'aft_envs': ft_optimizer.aft_envs,
        'shell_offsets': shell_offsets,
        'shared_memory': aft_jk._estimate_max_shm_size(cell, (1, 0)),
        'bas_ij_idx': bas_ij_idx,
        'bas_ij_img_idx': bas_ij_img_idx,
    }


def _run_fourier_kernel(cell, source, vectors, width, *, kernel, aft_envs,
                        shell_offsets, shared_memory, bas_ij_idx,
                        bas_ij_img_idx, permutation_symmetry):
    '''Contract one Fourier AO-pair derivative CUDA kernel invocation.'''
    output = cp.zeros((cell.natm, 3))
    vector_transpose = cp.asarray(vectors.T.ravel())
    error = kernel(
        ctypes.cast(output.data.ptr, ctypes.c_void_p),
        ctypes.cast(source.data.ptr, ctypes.c_void_p),
        ctypes.cast(vector_transpose.data.ptr, ctypes.c_void_p),
        ctypes.byref(aft_envs), ctypes.c_int(len(shell_offsets) - 1),
        ctypes.c_int(width), ctypes.c_int(shared_memory),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(shell_offsets.data.ptr, ctypes.c_void_p),
        ctypes.c_int(permutation_symmetry))
    if error != 0:
        raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')
    return output


def _gamma_coulomb_vector(value, ngrids):
    '''Normalise GPU4PySCF's Gamma Coulomb row to one device vector.'''
    value = asarray(value)
    if value.ndim == 2:
        if value.shape[0] != 1:
            raise ValueError('Gamma RSDF expects one Coulomb-weight row')
        value = value[0]
    if value.ndim != 1 or value.size != int(ngrids):
        raise ValueError('Gamma source Coulomb weights do not match its mesh')
    return value


def _group_auxiliary_batch(auxcell, partials):
    '''Reduce auxiliary-function center derivatives to nuclear centers.'''
    dimensions = auxcell.ao_loc[1:] - auxcell.ao_loc[:-1]
    atom_ids = cp.asarray(
        np.repeat(auxcell._bas[:, ATOM_OF], dimensions), dtype=cp.int32)
    outputs = []
    for partial in partials:
        output = cp.zeros((auxcell.natm, 3))
        cp.add.at(output, atom_ids, partial.T)
        outputs.append(output)
    return cp.stack(outputs)


def _is_gpu_out_of_memory(error):
    '''Recognise CUDA allocation errors that can safely be rebached.'''
    if isinstance(error, cp.cuda.memory.OutOfMemoryError):
        return True
    return (error.__class__.__name__ == 'CUDARuntimeError'
            and ('out of memory' in str(error).lower()
                 or 'cudaerrormemoryallocation' in str(error).lower()))


def _release_cached_blocks(_start, _stop, _error):
    '''Release failed transactional buffers before retrying the same range.'''
    cp.get_default_memory_pool().free_all_blocks()
