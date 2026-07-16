"""Bounded direct contractions for RSDF short-range source derivatives."""

from __future__ import annotations

import ctypes
import math

import cupy as cp
import numpy as np
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib.cupy_helper import empty_aligned
from gpu4pyscf.pbc.df.grad.rhf import (
    _split_l_ctr_pattern,
    get_ao_pair_loc,
    int3c2e_scheme,
)
from gpu4pyscf.pbc.df.int3c2e import (
    MAX_IMGS_PER_TASK,
    POOL_SIZE,
    libpbc,
)
from pyscf import lib


def contract_gamma(optimizer, source, pair_addresses):
    """Contract one real Gamma source in the optimizer's sorted basis."""

    return contract_gamma_batch(
        optimizer,
        cp.asarray(source)[None],
        pair_addresses,
    )[0]


def contract_gamma_batch(optimizer, sources, pair_addresses):
    """Contract several Gamma sources while sharing the SR kernel context."""

    sources = cp.asarray(sources)
    if sources.ndim != 4:
        raise ValueError(
            "Gamma sources must have shape (nsource,nao,nao,naux), "
            f"got {sources.shape}"
        )
    context = _KernelContext.build(optimizer, bvk_ncells=1)
    ao_derivative, auxiliary_derivative = context.output_arrays(
        len(sources)
    )
    for batch_id in range(context.batch_count):
        start, stop = context.auxiliary_ao_range(batch_id)
        width = stop - start
        compressed = cp.take(
            sources[..., start:stop].reshape(len(sources), -1, width),
            pair_addresses,
            axis=1,
        )
        for source_id in range(len(sources)):
            context.run_kernel(
                batch_id,
                compressed[source_id],
                ao_derivative[source_id],
                auxiliary_derivative[source_id],
            )
    return (ao_derivative + auxiliary_derivative) * 2.0


def contract_kpoint(
    optimizer,
    ordered_source,
    momentum,
    pair_addresses,
):
    """Contract all sampled-k source pairs in one BvK traversal."""

    return contract_kpoint_batch(
        optimizer,
        cp.asarray(ordered_source)[None],
        momentum,
        pair_addresses,
    )[0]


def contract_kpoint_batch(
    optimizer,
    ordered_sources,
    momentum,
    pair_addresses,
):
    """Contract several sampled-k sources in one shared BvK traversal."""

    ordered_sources = cp.asarray(ordered_sources)
    if ordered_sources.ndim != 6:
        raise ValueError(
            "sampled-k sources must have shape "
            "(nsource,nkpts,nkpts,nao,nao,naux), "
            f"got {ordered_sources.shape}"
        )
    nsource = len(ordered_sources)
    nkpts = int(ordered_sources.shape[1])
    bvk_ncells = len(optimizer.bvkmesh_Ls)
    context = _KernelContext.build(
        optimizer,
        bvk_ncells=bvk_ncells,
    )
    ao_derivative, auxiliary_derivative = context.output_arrays(nsource)
    order_ki = (
        momentum.ijk_conserv.T * nkpts
        + cp.arange(nkpts)[:, None]
    ).ravel()
    for batch_id in range(context.batch_count):
        start, stop = context.auxiliary_ao_range(batch_id)
        crystal_sources = ordered_sources[..., start:stop]
        compressed = _sources_to_bvk(
            crystal_sources,
            order_ki,
            momentum,
            pair_addresses,
            bvk_ncells=bvk_ncells,
        )
        for source_id in range(nsource):
            context.run_kernel(
                batch_id,
                compressed[source_id],
                ao_derivative[source_id],
                auxiliary_derivative[source_id],
            )
    return (ao_derivative + auxiliary_derivative) * 2.0


def _sources_to_bvk(
    crystal_sources,
    order_ki,
    momentum,
    pair_addresses,
    *,
    bvk_ncells,
):
    nsource, nkpts, _unused, nao, _nao2, width = (
        crystal_sources.shape
    )
    swapped = cp.empty_like(crystal_sources).reshape(
        nsource,
        nkpts**2,
        nao,
        nao,
        width,
    )
    swapped[:, order_ki] = crystal_sources.reshape(
        nsource,
        nkpts**2,
        nao,
        nao,
        width,
    )
    swapped = swapped.reshape(
        nsource,
        nkpts,
        nkpts,
        nao,
        nao,
        width,
    )
    mixed = cp.einsum(
        "sKJpqr,LK->sJpqLr",
        swapped,
        momentum.exp_lk_conj,
        optimize=True,
    )
    real_space = cp.einsum(
        "sJpqLr,NJ->sqNpLr",
        mixed,
        momentum.exp_lk,
        optimize=True,
    )
    real_space = real_space.reshape(
        nsource,
        -1,
        bvk_ncells,
        width,
    ).real
    return cp.take(real_space, pair_addresses, axis=1)


class _KernelContext:
    """Immutable indexing and workspace shared by SR source batches."""

    def __init__(self, **values):
        self.__dict__.update(values)

    @classmethod
    def build(cls, optimizer, *, bvk_ncells):
        cell = optimizer.cell
        auxcell = optimizer.auxcell
        auxiliary_limit = max(
            1,
            min(int(auxcell.nbas), POOL_SIZE // int(bvk_ncells)),
        )
        shell_pair_limit = max(
            1,
            POOL_SIZE // (auxiliary_limit * int(bvk_ncells)),
        )
        _, gout_stride, shared_sizes = int3c2e_scheme(-1, 54)
        lmax = int(cell.uniq_l_ctr[:, 0].max())
        laux = int(auxcell.uniq_l_ctr[:, 0].max())
        shared_memory = int(
            shared_sizes[: laux + 1, : lmax + 1, : lmax + 1].max()
        )
        bas_ij_idx, shell_pair_offsets = cell.aggregate_shl_pairs(
            optimizer.bas_ij_cache,
            shell_pair_limit,
        )
        ao_pair_loc = get_ao_pair_loc(
            cell.uniq_l_ctr[:, 0],
            optimizer.bas_ij_cache,
            cart=True,
        )
        auxiliary_offsets = np.append(
            0,
            np.cumsum(auxcell.l_ctr_counts),
        )
        auxiliary_offsets, auxiliary_patterns = _split_l_ctr_pattern(
            auxiliary_offsets,
            auxcell.uniq_l_ctr,
            auxiliary_limit,
        )
        _validate_task_pool_size(
            shell_pair_offsets,
            auxiliary_offsets,
            multiplier=bvk_ncells,
        )
        pool = cp.empty(
            gpu_specs["multiProcessorCount"]
            * POOL_SIZE
            * (MAX_IMGS_PER_TASK + 2)
            + 1,
            dtype=np.uint32,
        )
        return cls(
            optimizer=optimizer,
            cell=cell,
            auxcell=auxcell,
            bas_ij_idx=bas_ij_idx,
            shell_pair_offsets=shell_pair_offsets,
            ao_pair_loc=ao_pair_loc,
            auxiliary_offsets=auxiliary_offsets,
            auxiliary_offsets_gpu=cp.asarray(
                auxiliary_offsets,
                dtype=np.int32,
            ),
            batch_count=len(auxiliary_patterns),
            gout_stride=gout_stride,
            shared_memory=shared_memory,
            pool=pool,
            head=pool[-1:],
            task_pool=empty_aligned(
                (gpu_specs["multiProcessorCount"], POOL_SIZE * 16),
                np.int32,
                alignment=128,
            ),
            diffuse_exponents=cp.asarray(optimizer.diffuse_exps),
            diffuse_coefficients=cp.asarray(optimizer.diffuse_coefs),
            log_cutoff=math.log(optimizer.cutoff),
        )

    def output_arrays(self, source_count=1):
        ao = cp.zeros((int(source_count), self.cell.natm, 3))
        return ao, cp.zeros_like(ao)

    def auxiliary_ao_range(self, batch_id):
        start = int(
            self.auxcell.ao_loc[self.auxiliary_offsets[batch_id]]
        )
        stop = int(
            self.auxcell.ao_loc[self.auxiliary_offsets[batch_id + 1]]
        )
        return start, stop

    def run_kernel(
        self,
        batch_id,
        compressed,
        ao_derivative,
        auxiliary_derivative,
    ):
        start, stop = self.auxiliary_ao_range(batch_id)
        error = libpbc.PBCsr_ejk_int3c2e_ip1(
            ctypes.cast(ao_derivative.data.ptr, ctypes.c_void_p),
            ctypes.cast(auxiliary_derivative.data.ptr, ctypes.c_void_p),
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.byref(self.optimizer.int3c2e_envs),
            ctypes.cast(self.pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.task_pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.head.data.ptr, ctypes.c_void_p),
            ctypes.c_int(self.shared_memory),
            ctypes.c_int(len(self.shell_pair_offsets) - 1),
            ctypes.c_int(1),
            ctypes.cast(self.bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.shell_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(
                self.auxiliary_offsets_gpu[batch_id:].data.ptr,
                ctypes.c_void_p,
            ),
            ctypes.cast(self.optimizer.img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.optimizer.img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(start),
            ctypes.c_int(self.auxcell.nbas),
            ctypes.c_int(stop - start),
            ctypes.cast(self.diffuse_exponents.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.diffuse_coefficients.data.ptr, ctypes.c_void_p),
            ctypes.c_float(self.log_cutoff),
        )
        if error != 0:
            raise RuntimeError("PBCsr_ejk_int3c2e_ip1 failed")


def _validate_task_pool_size(
    shell_pair_offsets,
    auxiliary_offsets,
    *,
    multiplier=1,
):
    shell_pairs = np.diff(cp.asnumpy(shell_pair_offsets))
    auxiliary_shells = np.diff(auxiliary_offsets)
    maximum_tasks = (
        int(shell_pairs.max())
        * int(auxiliary_shells.max())
        * int(multiplier)
    )
    if maximum_tasks > POOL_SIZE:
        raise RuntimeError(
            "RSDF source derivative batch exceeds CUDA POOL_SIZE: "
            f"{maximum_tasks} > {POOL_SIZE}"
        )
