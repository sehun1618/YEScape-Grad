"""Source-contracted CUDA derivative of native GDF local PP part 1.

GPU4PySCF evaluates this operator as a short-range three-center integral plus
a long-range Fourier contraction. Building every AO derivative matrix would
require the unavailable ``ft_ao_ip1`` tensor interface. Energy and adjoint
gradient assembly only need contractions with a Hermitian AO density, so this
module calls the existing source-contracted CUDA kernels instead.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math

import cupy as cp
import numpy as np
from pyscf import lib
from pyscf.pbc.df import aft as cpu_aft
from pyscf.pbc.lib.kpts_helper import is_zero

from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.df.int3c2e_bdiv import _split_l_ctr_pattern
from gpu4pyscf.lib.cupy_helper import contract, empty_aligned, get_avail_mem
from gpu4pyscf.pbc.df import aft_jk, ft_ao, rsdf_builder
from gpu4pyscf.pbc.df.grad.rhf import int3c2e_scheme
from gpu4pyscf.pbc.df.int3c2e import (
    MAX_IMGS_PER_TASK,
    POOL_SIZE,
    SRInt3c2eOpt,
)
from gpu4pyscf.pbc.lib.kpts_helper import fft_matrix
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh


_NATIVE_OMEGA = 0.2
DEFAULT_RECIPROCAL_MEMORY_FRACTION = 0.10
DEFAULT_RECIPROCAL_MAX_GRID_BLOCK_SIZE = 1024
DEFAULT_RECIPROCAL_BLOCK_ALIGNMENT = 32


@dataclass(frozen=True)
class LocalPart1DerivativeComponents:
    """Device-resident pieces of one contracted local-part-1 derivative."""

    short_range_ao: cp.ndarray
    short_range_center: cp.ndarray
    long_range_ao: cp.ndarray
    long_range_center: cp.ndarray

    @property
    def total(self) -> cp.ndarray:
        """Return ``d Tr[D Vloc1] / dR`` in ``Ha/Bohr``."""

        return 2.0 * (
            self.short_range_ao
            + self.short_range_center
            + self.long_range_ao
        ) + self.long_range_center


class NativeGDFLocalPart1Derivative:
    """Contract native GPU4PySCF local-part-1 derivatives with a density.

    The accepted density has shape ``(Nk,nao,nao)`` in the original AO order.
    The result already includes the Brillouin-zone factor ``1/Nk``.
    """

    def __init__(
        self,
        cell,
        kpts,
        *,
        omega: float = _NATIVE_OMEGA,
        reciprocal_memory_fraction=(
            DEFAULT_RECIPROCAL_MEMORY_FRACTION
        ),
        reciprocal_max_grid_block_size=(
            DEFAULT_RECIPROCAL_MAX_GRID_BLOCK_SIZE
        ),
    ):
        self.cell = cell
        self.kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
        self.nkpts = len(self.kpts)
        self.omega = float(omega)
        if abs(self.omega - _NATIVE_OMEGA) > 1.0e-14:
            raise NotImplementedError(
                "native GPU4PySCF local part 1 currently fixes omega=0.2"
            )
        self.reciprocal_memory_fraction = float(
            reciprocal_memory_fraction
        )
        if not 0.0 < self.reciprocal_memory_fraction <= 1.0:
            raise ValueError(
                "reciprocal_memory_fraction must be in (0, 1]"
            )
        self.reciprocal_max_grid_block_size = int(
            reciprocal_max_grid_block_size
        )
        if self.reciprocal_max_grid_block_size < 1:
            raise ValueError(
                "reciprocal_max_grid_block_size must be positive"
            )
        self.last_reciprocal_execution = None
        self.kmesh = np.asarray(
            kpts_to_kmesh(cell, self.kpts, bound_by_supmol=True),
            dtype=int,
        )
        if int(np.prod(self.kmesh)) != self.nkpts:
            raise NotImplementedError(
                "source-contracted local part 1 requires a complete MP mesh"
            )

        self.fakenuc = cpu_aft._fake_nuc(cell, with_pseudo=True)
        self.charges = -cp.asarray(cell.atom_charges(), dtype=cp.float64)
        self.short_range = SRInt3c2eOpt(
            cell,
            self.fakenuc,
            omega=-self.omega,
            bvk_kmesh=self.kmesh,
        ).build()

        cutoff = rsdf_builder.estimate_ke_cutoff_for_omega(cell, self.omega)
        mesh = cell.cutoff_to_mesh(cutoff)
        self.mesh = np.asarray(cell.symmetrize_mesh(mesh), dtype=int)
        self.gv, _base, self.weights = cell.get_Gv_weights(self.mesh)
        self.weighted_coulomb = rsdf_builder._weighted_coulG_LR(
            cell,
            self.gv,
            self.omega,
            self.weights,
        )
        self._g0_pseudo_correction = self._build_g0_correction()

        self.fourier = ft_ao.FTOpt(cell, self.kmesh).build()
        self.fourier_kernel = self.fourier.gen_ft_kernel(
            transform_ao=False,
            kpts=self.kpts,
        )
        self._fourier_shell_pairs = aft_jk._generate_shl_pairs(self.fourier)
        self._fourier_shared_memory = aft_jk._estimate_max_shm_size(
            self.fourier.cell,
            (1, 0),
        )

    def contract(self, density) -> cp.ndarray:
        """Return the analytic contraction for one Hermitian AO density."""

        return self.contract_components(density).total

    def contract_batch(self, densities) -> cp.ndarray:
        """Contract several Gamma densities while sharing Fourier geometry."""

        return self.contract_components_batch(densities).total

    def contract_components(self, density) -> LocalPart1DerivativeComponents:
        """Return the four terms separately for diagnostics and tests."""

        dm = self._normalize_density(density)
        components = self.contract_components_batch(dm[None])
        return LocalPart1DerivativeComponents(
            short_range_ao=components.short_range_ao[0],
            short_range_center=components.short_range_center[0],
            long_range_ao=components.long_range_ao[0],
            long_range_center=components.long_range_center[0],
        )

    def contract_components_batch(
        self,
        densities,
    ) -> LocalPart1DerivativeComponents:
        """Return component contractions for a leading density-set axis."""

        batch = cp.asarray(densities)
        if batch.ndim == 3:
            batch = batch[None]
        if batch.ndim != 4:
            raise ValueError(
                "density batch must have shape (nset,nkpts,nao,nao), "
                f"got {batch.shape}"
            )
        normalized = cp.stack(
            [self._normalize_density(density) for density in batch]
        )
        short_ao = []
        short_center = []
        for density in normalized:
            ao, center = self._contract_short_range(density)
            short_ao.append(ao)
            short_center.append(center)
        long_ao, long_center = self._contract_long_range_batch(normalized)
        return LocalPart1DerivativeComponents(
            short_range_ao=cp.stack(short_ao),
            short_range_center=cp.stack(short_center),
            long_range_ao=long_ao,
            long_range_center=long_center,
        )

    def _normalize_density(self, density) -> cp.ndarray:
        dm = cp.asarray(density, dtype=cp.complex128)
        if dm.ndim == 2:
            dm = dm[None]
        expected = (self.nkpts, self.cell.nao_nr(), self.cell.nao_nr())
        if dm.shape != expected:
            raise ValueError(f"density shape {dm.shape} does not match {expected}")
        hermiticity = cp.max(cp.abs(dm - dm.transpose(0, 2, 1).conj()))
        if float(hermiticity) > 1.0e-8:
            raise ValueError("local-part-1 contraction requires Hermitian density")
        return dm

    def _contract_short_range(self, density):
        sorted_density = self.short_range.cell.apply_C_mat_CT(density)
        if is_zero(self.kpts):
            bvk_density = cp.asarray(sorted_density.real, order="C")
        else:
            phases = fft_matrix(self.short_range.bvk_kmesh)
            bvk_complex = contract("Lk,kpq->Lpq", phases, sorted_density)
            imaginary = float(cp.max(cp.abs(bvk_complex.imag)))
            if imaginary > 1.0e-6:
                raise ValueError(
                    "k-point density does not map to a real BvK density; "
                    f"max imaginary component is {imaginary:.3e}"
                )
            bvk_density = cp.asarray(
                bvk_complex.real / self.nkpts,
                order="C",
            )

        orbital = cp.zeros((self.cell.natm, 3), dtype=cp.float64)
        center = cp.zeros_like(orbital)
        self._launch_short_range_kernel(bvk_density, orbital, center)
        return orbital, center

    def _launch_short_range_kernel(self, density, orbital, center):
        opt = self.short_range
        cell = opt.cell
        auxcell = opt.auxcell
        _nsp, gout_stride, shared_memory = int3c2e_scheme(-1, 54)
        lmax = int(cell.uniq_l_ctr[:, 0].max())
        laux = int(auxcell.uniq_l_ctr[:, 0].max())
        shared_memory_size = int(
            shared_memory[: laux + 1, : lmax + 1, : lmax + 1].max()
        )
        shell_pairs, shell_offsets, aux_offsets = (
            _bounded_short_range_batches(opt)
        )
        diffuse_exponents = cp.asarray(opt.diffuse_exps)
        diffuse_coefficients = cp.asarray(opt.diffuse_coefs)

        workers = int(gpu_specs["multiProcessorCount"])
        pool = cp.empty(
            workers * POOL_SIZE * (MAX_IMGS_PER_TASK + 2) + 1,
            dtype=cp.uint32,
        )
        head = pool[-1:]
        task_pool = empty_aligned(
            (workers, POOL_SIZE * 16),
            np.int32,
            alignment=128,
        )
        charges = cp.ascontiguousarray(self.charges)
        environments = opt.int3c2e_envs
        error = ft_ao.libpbc.PBCsr_ejk_int3c2e_ip1(
            _pointer(orbital),
            _pointer(center),
            _pointer(density),
            _pointer(charges),
            ctypes.byref(environments),
            _pointer(pool),
            _pointer(task_pool),
            _pointer(head),
            ctypes.c_int(shared_memory_size),
            ctypes.c_int(len(shell_offsets) - 1),
            ctypes.c_int(len(aux_offsets) - 1),
            _pointer(shell_pairs),
            _pointer(shell_offsets),
            _pointer(aux_offsets),
            _pointer(opt.img_idx),
            _pointer(opt.img_offsets),
            _pointer(gout_stride),
            lib.c_null_ptr(),
            ctypes.c_int(0),
            ctypes.c_int(auxcell.nbas),
            ctypes.c_int(auxcell.nao),
            _pointer(diffuse_exponents),
            _pointer(diffuse_coefficients),
            ctypes.c_float(math.log(opt.cutoff)),
        )
        if error != 0:
            raise RuntimeError("PBCsr_ejk_int3c2e_ip1 failed")

    def _contract_long_range(self, density):
        orbital, center = self._contract_long_range_batch(density[None])
        return orbital[0], center[0]

    def _contract_long_range_batch(self, densities):
        sorted_density = cp.stack(
            [self.fourier.cell.apply_C_mat_CT(density) for density in densities]
        )
        bvk_density = cp.stack(
            [self._fourier_bvk_density(density) for density in sorted_density]
        )
        orbital = cp.zeros(
            (len(densities), self.cell.natm, 3),
            dtype=cp.float64,
        )
        center = cp.zeros_like(orbital)

        block_size = self._fourier_block_size()
        block_count = int(math.ceil(len(self.gv) / block_size))
        self.last_reciprocal_execution = {
            "grid_count": int(len(self.gv)),
            "block_size": int(block_size),
            "block_count": block_count,
            "memory_fraction": self.reciprocal_memory_fraction,
            "maximum_grid_block_size": (
                self.reciprocal_max_grid_block_size
            ),
            "density_count": int(len(densities)),
            "estimated_pair_fourier_bytes_per_block": int(
                block_size
                * self.nkpts
                * int(self.fourier.cell.nao) ** 2
                * np.dtype(np.complex128).itemsize
            ),
        }
        shell_pairs, image_pairs, shell_offsets = self._fourier_shell_pairs
        environments = self.fourier.aft_envs
        for g0, g1 in lib.prange(0, len(self.gv), block_size):
            gv_block = self.gv[g0:g1]
            auxiliary = ft_ao.ft_ao(self.fakenuc, gv_block).conj()
            potential = auxiliary.dot(self.charges)
            potential *= self.weighted_coulomb[g0:g1]
            if g0 == 0:
                potential[0] -= self._g0_pseudo_correction

            pair_fourier = self.fourier_kernel(gv_block)
            pair_fourier = pair_fourier.transpose(0, 2, 3, 1)
            rho = contract("skji,kijg->sg", sorted_density, pair_fourier)
            phase_derivative = (
                1j
                * cp.asarray(gv_block)[:, None, :]
                * auxiliary[:, :, None]
                * self.charges[None, :, None]
                * self.weighted_coulomb[g0:g1, None, None]
            )
            center += contract("sg,gax->sax", rho, phase_derivative).real

            gv_transposed = cp.asarray(gv_block.T.ravel())
            potential = cp.ascontiguousarray(potential)
            for source_id in range(len(densities)):
                error = ft_ao.libpbc.PBC_ft_aopair_ej_ip1(
                    _pointer(orbital[source_id]),
                    _pointer(bvk_density[source_id]),
                    _pointer(potential),
                    _pointer(gv_transposed),
                    ctypes.byref(environments),
                    ctypes.c_int(len(shell_offsets) - 1),
                    ctypes.c_int(g1 - g0),
                    ctypes.c_int(self._fourier_shared_memory),
                    _pointer(shell_pairs),
                    _pointer(image_pairs),
                    _pointer(shell_offsets),
                    ctypes.c_int(int(self.fourier.permutation_symmetry)),
                )
                if error != 0:
                    raise RuntimeError("PBC_ft_aopair_ej_ip1 failed")

        if not self.fourier.permutation_symmetry:
            orbital *= 0.5
        orbital /= self.nkpts
        center /= self.nkpts
        return orbital, center

    def _fourier_bvk_density(self, sorted_density):
        if is_zero(self.kpts):
            return cp.asarray(sorted_density.real, order="C")
        phases = cp.exp(
            1j
            * cp.asarray(self.fourier.bvkmesh_Ls).dot(
                cp.asarray(self.kpts).T
            )
        )
        bvk_density = contract("Lk,kpq->Lpq", phases, sorted_density)
        imaginary = float(cp.max(cp.abs(bvk_density.imag)))
        if imaginary > 1.0e-6:
            raise ValueError(
                "k-point density does not map to a real Fourier BvK density; "
                f"max imaginary component is {imaginary:.3e}"
            )
        return cp.asarray(bvk_density.real, order="C")

    def _fourier_block_size(self) -> int:
        free_memory = (
            get_avail_mem(exclude_memory_pool=True)
            * self.reciprocal_memory_fraction
        )
        nao = int(self.fourier.cell.nao)
        values_per_grid = max(1, self.nkpts * nao * nao)
        size = int(free_memory / (values_per_grid * 16 * 2))
        alignment = DEFAULT_RECIPROCAL_BLOCK_ALIGNMENT
        size = max(1, size)
        if size >= alignment:
            size = max(alignment, size // alignment * alignment)
        return min(
            size,
            len(self.gv),
            self.reciprocal_max_grid_block_size,
        )

    def _build_g0_correction(self):
        if not (
            self.cell.dimension == 3
            or (
                self.cell.dimension == 2
                and self.cell.low_dim_ft_type != "inf_vacuum"
            )
        ):
            return cp.asarray(0.0)
        exponents = cp.asarray(np.hstack(self.fakenuc.bas_exps()))
        return self.charges.dot(np.pi / exponents) / self.cell.vol


def _pointer(array):
    return ctypes.cast(array.data.ptr, ctypes.c_void_p)


def _bounded_short_range_batches(opt):
    """Build SR task batches that fit the fixed CUDA task pool.

    One ``PBCsr_ejk_int3c2e_ip1`` work item is the product of an AO shell
    pair, a fake-nucleus auxiliary shell, and a BvK cell.  GPU4PySCF's CUDA
    kernel allocates ``POOL_SIZE`` task records per worker, so passing the
    complete water20 shell products in one batch writes beyond that buffer.
    The corruption is non-deterministic and appears as a size-dependent force
    error.  Bounding both shell axes preserves the exact contraction while
    keeping every work group within the kernel allocation.
    """

    cell = opt.cell
    auxcell = opt.auxcell
    bvk_ncells = max(1, len(opt.bvkmesh_Ls))
    auxiliary_shells = max(1, int(auxcell.nbas))
    auxiliary_limit = max(
        1,
        min(auxiliary_shells, POOL_SIZE // bvk_ncells),
    )
    shell_pair_limit = max(
        1,
        POOL_SIZE // (auxiliary_limit * bvk_ncells),
    )

    shell_pairs, shell_offsets = cell.aggregate_shl_pairs(
        opt.bas_ij_cache,
        shell_pair_limit,
    )
    host_aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
    host_aux_offsets, _ = _split_l_ctr_pattern(
        host_aux_offsets,
        auxcell.uniq_l_ctr,
        auxiliary_limit,
    )
    _validate_short_range_batches(
        shell_offsets,
        host_aux_offsets,
        bvk_ncells,
    )
    return (
        shell_pairs,
        shell_offsets,
        cp.asarray(host_aux_offsets, dtype=cp.int32),
    )


def _validate_short_range_batches(
    shell_pair_offsets,
    auxiliary_shell_offsets,
    bvk_ncells,
):
    """Reject a batch layout that could overflow the CUDA task pool."""

    shell_pair_counts = np.diff(cp.asnumpy(shell_pair_offsets))
    auxiliary_shell_counts = np.diff(np.asarray(auxiliary_shell_offsets))
    if shell_pair_counts.size == 0 or auxiliary_shell_counts.size == 0:
        return
    largest_batch = int(
        shell_pair_counts.max()
        * auxiliary_shell_counts.max()
        * int(bvk_ncells)
    )
    if largest_batch > POOL_SIZE:
        raise RuntimeError(
            "local-part-1 SR derivative batch exceeds CUDA POOL_SIZE: "
            f"{largest_batch} > {POOL_SIZE}"
        )
