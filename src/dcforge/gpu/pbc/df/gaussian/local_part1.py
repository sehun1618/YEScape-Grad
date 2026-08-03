'''Source-contracted real-Gamma Gaussian-provider local-GTH part 1.'''

import ctypes
import math

import cupy as cp
import numpy as np
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.df.int3c2e_bdiv import _split_l_ctr_pattern
from gpu4pyscf.lib.cupy_helper import contract, empty_aligned, get_avail_mem
from gpu4pyscf.pbc.df import aft_jk, ft_ao, rsdf_builder
from gpu4pyscf.pbc.df.grad.rhf import int3c2e_scheme
from gpu4pyscf.pbc.df.int3c2e import (
    MAX_IMGS_PER_TASK,
    POOL_SIZE,
    SRInt3c2eOpt,
    fill_triu_bvk,
)
from pyscf import lib
from pyscf.pbc.df import aft as cpu_aft
from pyscf.pbc.lib.kpts_helper import is_zero

_AUTO_OMEGA_INITIAL = .4
_AUTO_RECIPROCAL_GRID_TARGET = 3000.
_RECIPROCAL_MEMORY_FRACTION = .10
_RECIPROCAL_MAX_BLOCK_SIZE = 1024
_RECIPROCAL_BLOCK_ALIGNMENT = 32


def resolve_pp_loc_part1(cell, omega=None):
    r'''Resolve the Gaussian-provider local-part-1 SR/LR plan.

    GPU4PySCF chooses the pseudopotential part-1 range split from the most
    compact AO exponent, cell volume, lattice and ``cell.precision``.  DC-Forge
    retains that exact deterministic policy so the user controls accuracy
    through ``cell.precision`` instead of tuning an implementation parameter.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built three-dimensional periodic cell.

    Kwargs:
        omega : float or None
            Explicit diagnostic/legacy split.  ``None`` applies the installed
            GPU4PySCF 1.8.0 local-part-1 policy: start from ``0.4``, rescale
            the cutoff towards 3000 reciprocal vectors, and invert the error
            estimate for the final omega.

    Returns:
        dict
            Resolved ``omega``, ``ke_cutoff``, integer ``mesh`` and ``mode``.
            The dictionary contains numerical execution metadata, not an
            independently mutable physical model.
    '''
    if cell.dimension != 3:
        raise NotImplementedError(
            'Gaussian-provider local part 1 requires 3D PBC')
    if omega is None:
        initial_cutoff = rsdf_builder.estimate_ke_cutoff_for_omega(
            cell, _AUTO_OMEGA_INITIAL)
        initial_mesh = np.asarray(
            cell.cutoff_to_mesh(initial_cutoff), dtype=int)
        initial_grid_count = int(np.prod(initial_mesh, dtype=np.int64))
        if initial_grid_count < 1:
            raise RuntimeError('local-part-1 auto mesh contains no grid points')
        target_cutoff = initial_cutoff * (
            _AUTO_RECIPROCAL_GRID_TARGET / initial_grid_count) ** (2. / 3.)
        omega = rsdf_builder.estimate_omega_for_ke_cutoff(
            cell, target_cutoff)
        mode = 'gpu4pyscf_auto'
    else:
        omega = float(omega)
        if not np.isfinite(omega) or omega <= 0.:
            raise ValueError('pp_loc_part1_omega must be positive or None')
        mode = 'explicit'

    ke_cutoff = rsdf_builder.estimate_ke_cutoff_for_omega(cell, omega)
    mesh = np.asarray(cell.symmetrize_mesh(
        cell.cutoff_to_mesh(ke_cutoff)), dtype=int)
    return {
        'omega': float(omega),
        'ke_cutoff': float(ke_cutoff),
        'mesh': mesh,
        'mode': mode,
    }


def get_pp_loc_part1(cell, kpts=None, omega=None, mesh=None):
    r'''Return the Gaussian-provider local-part-1 AO matrix.

    This is GPU4PySCF's short-range three-center plus long-range Fourier
    algebra with the provider-resolved ``omega`` and ``mesh`` supplied
    explicitly.  The same values are passed to :class:`LocalPart1Derivative`,
    making the zero-order operator and its nuclear derivative one numerical
    representation rather than two independently guessed plans.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built periodic cell.

    Kwargs:
        kpts : ndarray or None
            One Gamma point or ``None``.  Sampled-k zero-order and analytic
            derivatives remain outside the current DC-Forge contract.
        omega : float or None
            Resolved positive range split.  ``None`` selects the automatic
            policy through :func:`resolve_pp_loc_part1`.
        mesh : sequence of 3 int or None
            Reciprocal mesh paired with ``omega``.  Supplying one without a
            positive omega is rejected.

    Returns:
        cupy.ndarray
            Local-part-1 AO matrix.  An explicit one-dimensional Gamma point
            returns ``(nao,nao)``; ``None`` retains the provider-compatible
            leading singleton k-point axis.
    '''
    if mesh is None:
        plan = resolve_pp_loc_part1(cell, omega)
        omega = plan['omega']
        mesh = plan['mesh']
    else:
        if omega is None:
            raise ValueError('an explicit local-part-1 mesh requires omega')
        omega = float(omega)
        if not np.isfinite(omega) or omega <= 0.:
            raise ValueError('local-part-1 omega must be positive')
        mesh = np.asarray(mesh, dtype=int)
        if mesh.shape != (3,) or np.any(mesh < 1):
            raise ValueError('local-part-1 mesh must contain three positive ints')

    is_single_kpt = kpts is not None and np.asarray(kpts).ndim == 1
    is_gamma = kpts is None or is_zero(kpts)
    if not is_gamma:
        raise NotImplementedError(
            'DC-Forge Gaussian-provider local-part-1 value currently '
            'requires Gamma point')
    if is_single_kpt:
        kpts = np.asarray(kpts).reshape(1, 3)
    bvk_kmesh = np.ones(3, dtype=int)

    fakenuc = cpu_aft._fake_nuc(cell, with_pseudo=True)
    intopt = SRInt3c2eOpt(
        cell, fakenuc, omega=-omega, bvk_kmesh=bvk_kmesh).build()
    charges = -cp.asarray(cell.atom_charges(), dtype=cp.float64)
    value = _contract_auxvec_gamma(intopt, charges)

    gv, _base, weights = cell.get_Gv_weights(mesh)
    potential = ft_ao.ft_ao(fakenuc, gv).conj().dot(charges)
    potential *= rsdf_builder._weighted_coulG_LR(
        cell, gv, omega, weights)
    exponents = cp.asarray(np.hstack(fakenuc.bas_exps()))
    potential[0] -= charges.dot(np.pi / exponents) / cell.vol
    fourier = ft_ao.FTOpt.from_intopt(intopt)
    value += fourier.contract_rhoG(potential, gv, kpts)
    if is_single_kpt and value.ndim == 3:
        value = value[0]
    return value


def _contract_auxvec_gamma(intopt, auxiliary):
    '''Contract short-range three-center values in bounded Gamma batches.

    GPU4PySCF's ``SRInt3c2eOpt.contract_auxvec`` submits all auxiliary shells
    with up to one million shell pairs to a CUDA task pool of fixed
    ``POOL_SIZE``.  Large cells can terminate the process before an error code
    is returned.  The upstream batched evaluator uses the same integral kernel
    while grouping both axes to the pool capacity; only its contracted lower
    triangle is retained here.
    '''
    if int(np.prod(intopt.bvk_kmesh)) != 1:
        raise NotImplementedError('bounded local part 1 requires Gamma point')
    auxiliary = cp.asarray(auxiliary, dtype=cp.float64)
    naux = int(intopt.auxcell.nao)
    if auxiliary.shape != (naux,):
        raise ValueError(
            f'local-part-1 auxiliary vector must have shape ({naux},)')

    aux_batch_size = max(1, min(naux, 64))
    evaluate, _sorting, pair_offsets, aux_offsets = (
        intopt.int3c2e_evaluator(
            aux_batch_size=aux_batch_size, cart=True))
    pair_addresses = intopt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    compressed = cp.zeros(len(pair_addresses), dtype=cp.float64)
    for pair_id in range(len(pair_offsets) - 1):
        pair0, pair1 = pair_offsets[pair_id:pair_id + 2]
        for aux_id in range(len(aux_offsets) - 1):
            aux0, aux1 = aux_offsets[aux_id:aux_id + 2]
            block = evaluate(
                shl_pair_batch_id=pair_id, aux_batch_id=aux_id)
            compressed[pair0:pair1] += cp.einsum(
                'pLa,a->p', block, auxiliary[aux0:aux1])

    nao = int(intopt.cell.nao)
    value = cp.zeros(nao * nao, dtype=cp.float64)
    value[pair_addresses] = compressed
    value = value.reshape(nao, 1, nao).transpose(1, 0, 2)
    value = fill_triu_bvk(
        cp.asarray(value, order='C'), nao, np.ones(3, dtype=int))
    return intopt.cell.apply_CT_mat_C(value)


class LocalPart1Derivative:
    r'''Contract Gaussian-provider local-pseudopotential part 1 at real Gamma.

    PySCF/GPU4PySCF split the Coulomb-like local GTH potential into a
    short-range three-center integral and a long-range Fourier term.  The
    nuclear gradient only needs their contraction with a Hermitian density,
    so this class never constructs ``(natm,3,nao,nao)`` derivative matrices.

    Args:
        cell : pyscf.pbc.gto.Cell
            Three-dimensional real-Gamma GTH cell.

    Kwargs:
        omega : float or None
            Provider-resolved positive SR/LR split.  ``None`` applies
            :func:`resolve_pp_loc_part1`.
        mesh : sequence of 3 int or None
            Provider-resolved reciprocal mesh paired with ``omega``.
        reciprocal_memory_fraction : float
            Fraction of currently free device memory available to a Fourier
            AO-pair block.
        reciprocal_max_block_size : int
            Maximum reciprocal vectors in one Fourier block.

    Notes:
        Short- and long-range contributions are summed exactly, so ``omega``
        is an internal representation choice rather than a physical
        range-separated approximation.  Production passes the provider-owned
        plan used by zero-order Hcore; an explicit ``0.2`` remains available
        only through the provider's diagnostic override.
    '''

    def __init__(self, cell, omega=None, mesh=None,
                 reciprocal_memory_fraction=_RECIPROCAL_MEMORY_FRACTION,
                 reciprocal_max_block_size=_RECIPROCAL_MAX_BLOCK_SIZE):
        if cell.dimension != 3:
            raise NotImplementedError(
                'Gaussian-provider local part 1 requires 3D PBC')
        if mesh is None:
            plan = resolve_pp_loc_part1(cell, omega)
            omega = plan['omega']
            mesh = plan['mesh']
        elif omega is None:
            raise ValueError('an explicit local-part-1 mesh requires omega')
        self.cell = cell
        self.kpts = np.zeros((1, 3))
        self.omega = float(omega)
        self.mesh = np.asarray(mesh, dtype=int)
        if not np.isfinite(self.omega) or self.omega <= 0.:
            raise ValueError('local-part-1 omega must be positive')
        if self.mesh.shape != (3,) or np.any(self.mesh < 1):
            raise ValueError('local-part-1 mesh must contain three positive ints')
        self.reciprocal_memory_fraction = float(
            reciprocal_memory_fraction)
        self.reciprocal_max_block_size = int(reciprocal_max_block_size)
        if not 0. < self.reciprocal_memory_fraction <= 1.:
            raise ValueError('reciprocal_memory_fraction must be in (0,1]')
        if self.reciprocal_max_block_size < 1:
            raise ValueError('reciprocal_max_block_size must be positive')

        self.fakenuc = cpu_aft._fake_nuc(cell, with_pseudo=True)
        self.charges = -cp.asarray(cell.atom_charges(), dtype=cp.float64)
        self.short_range = SRInt3c2eOpt(
            cell, self.fakenuc, omega=-self.omega,
            bvk_kmesh=np.ones(3, dtype=int)).build()

        self.gv, _base, self.weights = cell.get_Gv_weights(self.mesh)
        self.weighted_coulomb = rsdf_builder._weighted_coulG_LR(
            cell, self.gv, self.omega, self.weights)
        self.g0_correction = self._build_g0_correction()

        self.fourier = ft_ao.FTOpt(cell, np.ones(3, dtype=int)).build()
        self.fourier_kernel = self.fourier.gen_ft_kernel(
            transform_ao=False, kpts=self.kpts)
        self.fourier_shell_pairs = aft_jk._generate_shl_pairs(self.fourier)
        self.fourier_shared_memory = aft_jk._estimate_max_shm_size(
            self.fourier.cell, (1, 0))
        self.last_reciprocal_execution = None

    def contract_batch(self, densities):
        r'''Return ``d Tr[D V_loc,1] / dR`` for several Gamma densities.

        Args:
            densities : cupy.ndarray of shape (nset,nao,nao)
                Real symmetric AO density matrices.

        Returns:
            tuple
                Total derivatives ``(nset,natm,3)`` and a dictionary of the
                four SR/LR AO/center contractions before final summation.
        '''
        dms = self._normalize_densities(densities)
        short_ao = []
        short_center = []
        for dm in dms:
            ao, center = self._contract_short_range(dm)
            short_ao.append(ao)
            short_center.append(center)
        short_ao = cp.stack(short_ao)
        short_center = cp.stack(short_center)
        long_ao, long_center = self._contract_long_range_batch(dms)
        components = {
            'short_range_ao': short_ao,
            'short_range_center': short_center,
            'long_range_ao': long_ao,
            'long_range_center': long_center,
        }
        total = 2 * (short_ao + short_center + long_ao) + long_center
        return total, components

    def _normalize_densities(self, densities):
        dms = cp.asarray(densities)
        if dms.ndim == 2:
            dms = dms[None]
        nao = self.cell.nao_nr()
        if dms.ndim != 3 or tuple(dms.shape[1:]) != (nao, nao):
            raise ValueError(
                'local-part-1 density batch must have shape '
                f'(nset,{nao},{nao}), got {dms.shape}')
        imaginary = float(cp.max(cp.abs(dms.imag)).get())
        asymmetry = float(cp.max(cp.abs(
            dms - dms.transpose(0, 2, 1).conj())).get())
        if imaginary > 1e-10:
            raise NotImplementedError(
                'native local-part-1 contraction requires real Gamma')
        if asymmetry > 1e-9:
            raise ValueError('local-part-1 densities must be Hermitian')
        return dms.real

    def _contract_short_range(self, density):
        density = self.short_range.cell.apply_C_mat_CT(density[None])
        density = cp.asarray(density.real, order='C')
        orbital = cp.zeros((self.cell.natm, 3), dtype=cp.float64)
        center = cp.zeros_like(orbital)
        self._launch_short_range_kernel(density, orbital, center)
        return orbital, center

    def _launch_short_range_kernel(self, density, orbital, center):
        opt = self.short_range
        cell = opt.cell
        auxcell = opt.auxcell
        _nsp, gout_stride, shared_memory = int3c2e_scheme(-1, 54)
        lmax = int(cell.uniq_l_ctr[:, 0].max())
        laux = int(auxcell.uniq_l_ctr[:, 0].max())
        shared_memory_size = int(
            shared_memory[:laux + 1, :lmax + 1, :lmax + 1].max())
        shell_pairs, shell_offsets, aux_offsets = (
            _bounded_short_range_batches(opt))
        diffuse_exponents = cp.asarray(opt.diffuse_exps)
        diffuse_coefficients = cp.asarray(opt.diffuse_coefs)

        workers = int(gpu_specs['multiProcessorCount'])
        pool = cp.empty(
            workers * POOL_SIZE * (MAX_IMGS_PER_TASK + 2) + 1,
            dtype=cp.uint32)
        head = pool[-1:]
        task_pool = empty_aligned(
            (workers, POOL_SIZE * 16), np.int32, alignment=128)
        charges = cp.ascontiguousarray(self.charges)
        environments = opt.int3c2e_envs
        error = ft_ao.libpbc.PBCsr_ejk_int3c2e_ip1(
            _pointer(orbital), _pointer(center), _pointer(density),
            _pointer(charges), ctypes.byref(environments), _pointer(pool),
            _pointer(task_pool), _pointer(head),
            ctypes.c_int(shared_memory_size),
            ctypes.c_int(len(shell_offsets) - 1),
            ctypes.c_int(len(aux_offsets) - 1), _pointer(shell_pairs),
            _pointer(shell_offsets), _pointer(aux_offsets),
            _pointer(opt.img_idx), _pointer(opt.img_offsets),
            _pointer(gout_stride), lib.c_null_ptr(), ctypes.c_int(0),
            ctypes.c_int(auxcell.nbas), ctypes.c_int(auxcell.nao),
            _pointer(diffuse_exponents), _pointer(diffuse_coefficients),
            ctypes.c_float(math.log(opt.cutoff)))
        if error != 0:
            raise RuntimeError('PBCsr_ejk_int3c2e_ip1 failed')

    def _contract_long_range_batch(self, densities):
        sorted_dms = cp.stack([
            self.fourier.cell.apply_C_mat_CT(dm[None]) for dm in densities])
        bvk_dms = cp.asarray(sorted_dms[:, 0].real, order='C')
        orbital = cp.zeros(
            (len(densities), self.cell.natm, 3), dtype=cp.float64)
        center = cp.zeros_like(orbital)

        block_size = self._fourier_block_size()
        self.last_reciprocal_execution = {
            'grid_count': len(self.gv),
            'block_size': int(block_size),
            'block_count': math.ceil(len(self.gv) / block_size),
            'density_count': len(densities),
        }
        shell_pairs, image_pairs, shell_offsets = self.fourier_shell_pairs
        environments = self.fourier.aft_envs
        for g0, g1 in lib.prange(0, len(self.gv), block_size):
            gv = self.gv[g0:g1]
            auxiliary = ft_ao.ft_ao(self.fakenuc, gv).conj()
            potential = auxiliary.dot(self.charges)
            potential *= self.weighted_coulomb[g0:g1]
            if g0 == 0:
                potential[0] -= self.g0_correction

            pair_fourier = self.fourier_kernel(gv)
            pair_fourier = pair_fourier.transpose(0, 2, 3, 1)
            rho = contract('sji,ijg->sg', sorted_dms[:, 0], pair_fourier[0])
            phase = (
                1j * cp.asarray(gv)[:, None, :]
                * auxiliary[:, :, None] * self.charges[None, :, None]
                * self.weighted_coulomb[g0:g1, None, None])
            center += contract('sg,gax->sax', rho, phase).real

            gv_t = cp.asarray(gv.T.ravel())
            potential = cp.ascontiguousarray(potential)
            for iset in range(len(densities)):
                error = ft_ao.libpbc.PBC_ft_aopair_ej_ip1(
                    _pointer(orbital[iset]), _pointer(bvk_dms[iset]),
                    _pointer(potential), _pointer(gv_t),
                    ctypes.byref(environments),
                    ctypes.c_int(len(shell_offsets) - 1),
                    ctypes.c_int(g1 - g0),
                    ctypes.c_int(self.fourier_shared_memory),
                    _pointer(shell_pairs), _pointer(image_pairs),
                    _pointer(shell_offsets),
                    ctypes.c_int(int(self.fourier.permutation_symmetry)))
                if error != 0:
                    raise RuntimeError('PBC_ft_aopair_ej_ip1 failed')

        if not self.fourier.permutation_symmetry:
            orbital *= .5
        return orbital, center

    def _fourier_block_size(self):
        free_memory = (
            get_avail_mem(exclude_memory_pool=True)
            * self.reciprocal_memory_fraction)
        nao = int(self.fourier.cell.nao)
        size = int(free_memory / max(1, nao * nao * 32))
        alignment = _RECIPROCAL_BLOCK_ALIGNMENT
        if size >= alignment:
            size = max(alignment, size // alignment * alignment)
        return min(max(1, size), len(self.gv),
                   self.reciprocal_max_block_size)

    def _build_g0_correction(self):
        exponents = cp.asarray(np.hstack(self.fakenuc.bas_exps()))
        return self.charges.dot(np.pi / exponents) / self.cell.vol


def _pointer(array):
    '''Return a CUDA device pointer for one CuPy array.'''
    return ctypes.cast(array.data.ptr, ctypes.c_void_p)


def _bounded_short_range_batches(opt):
    '''Build SR task batches that fit GPU4PySCF's fixed CUDA task pool.'''
    cell = opt.cell
    auxcell = opt.auxcell
    ncells = max(1, len(opt.bvkmesh_Ls))
    naux_shell = max(1, int(auxcell.nbas))
    aux_limit = max(1, min(naux_shell, POOL_SIZE // ncells))
    pair_limit = max(1, POOL_SIZE // (aux_limit * ncells))
    shell_pairs, shell_offsets = cell.aggregate_shl_pairs(
        opt.bas_ij_cache, pair_limit)
    aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
    aux_offsets, _ = _split_l_ctr_pattern(
        aux_offsets, auxcell.uniq_l_ctr, aux_limit)
    largest = (
        int(np.diff(cp.asnumpy(shell_offsets)).max())
        * int(np.diff(aux_offsets).max()) * ncells)
    if largest > POOL_SIZE:
        raise RuntimeError(
            'local-part-1 SR derivative batch exceeds CUDA POOL_SIZE')
    return shell_pairs, shell_offsets, cp.asarray(
        aux_offsets, dtype=cp.int32)
