'''Density-contracted real-Gamma GDF GTH pseudopotential strain.'''

import ctypes
import time

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
from gpu4pyscf.pbc.df.aft_jk import (
    _estimate_max_shm_size,
    _generate_shl_pairs,
)
from gpu4pyscf.pbc.df.ft_ao import FTOpt, libpbc
from gpu4pyscf.pbc.gto import int1e
from pyscf import lib

from dcforge.cpu.pbc.df.gaussian.fake_nuclear import (
    make_fake_nuclear_cell,
)
from dcforge.cpu.pbc.df.gaussian.pseudo import (
    contract_vpploc_part2_strain,
    contract_vppnl_strain,
)
from dcforge.cpu.pbc.df.gaussian.strain import (
    auxiliary_fourier_strain,
    weighted_coulomb_strain,
)
from dcforge.cpu.pbc.df.gdf.pp_strain import (
    _local_part2_potential_fourier_strain,
    _reciprocal_mesh,
)
from dcforge.cpu.pbc.df.gdf.stress import (
    contract_short_range_source_strain,
    contract_vbar_source_strain,
)
from dcforge.gpu.pbc.df.gaussian.local_part1 import get_pp_loc_part1
from dcforge.gpu.pbc.df.gaussian.strain import contract_overlap_strain

_RECIPROCAL_MEMORY_FRACTION = .25
_RECIPROCAL_BLOCK_ALIGNMENT = 16
_RECIPROCAL_MAX_BLOCK_SIZE = 16384


def contract_gdf_pseudopotential_strain(
        with_df, densities, reciprocal_mesh=None, block_size=None,
        source_block_size=None, value_tol=1.e-7):
    r'''Contract the provider-matched GTH PP strain with several densities.

    The first production batch contains ``P`` and ``D_Z`` so the skeleton
    and nonvariational Hcore terms share one PP construction and one direct
    local-part1 contraction.  Local part 2 and the nonlocal projector use the
    exact Gaussian-provider integrals with analytic lattice-image moments;
    both are density-contracted before an AO derivative matrix is formed.
    The reciprocal local-part2 implementation remains an explicit oracle,
    but is not the production path because its cost scales with the PP mesh.
    The returned ledger exposes the remaining host/device boundaries rather
    than describing this mixed development path as fully device-native.

    Args:
        with_df : dcforge.gpu.pbc.df.gdf.GDF
            Active source/target provider.  Its resolved local-part1 plan and
            cached zero-order PP matrix define the representation.
        densities : cupy.ndarray or ndarray
            Hermitian real-Gamma density batch ``(nset,nao,nao)``.

    Kwargs:
        reciprocal_mesh, block_size
            Compatibility controls for the retained reciprocal local-part2
            oracle.  The exact provider-matched production contraction does
            not use a reciprocal mesh or G-vector block.
        source_block_size : int or None
            Bounded reciprocal block for transparent local part 1.
        value_tol : float
            Maximum error between the contracted analytic representation and
            the provider's zero-order pseudopotential matrix.

    Returns:
        dict
            Provider PP value matrix, per-density value and raw strain
            contractions, term ledger and explicit host/device boundaries.
    '''
    cell = with_df.cell
    density_device = cp.asarray(densities)
    if density_device.ndim == 2:
        density_device = density_device[None]
    density_host = cp.asnumpy(density_device)
    started = time.perf_counter()
    local_part1_value, local_part1_contraction, local_part1 = (
        contract_local_part1_strain(
            with_df, density_device, block_size=source_block_size,
            return_metadata=True))
    local_part1_seconds = time.perf_counter() - started

    local_part2_value, local_part2_contraction, local_part2 = (
        contract_vpploc_part2_strain(
            cell, density_host, return_metadata=True))
    local_part2['reciprocal_oracle_controls'] = {
        'mesh': (
            None if reciprocal_mesh is None
            else np.asarray(reciprocal_mesh, dtype=int)),
        'block_size': (
            None if block_size is None else int(block_size)),
        'used_by_production': False,
    }

    started = time.perf_counter()
    (nonlocal_value_contraction, nonlocal_contraction,
     nonlocal_part) = contract_vppnl_strain(
         cell, density_host, return_metadata=True)
    nonlocal_seconds = time.perf_counter() - started

    value_contraction = (
        local_part1_value + local_part2_value
        + nonlocal_value_contraction)
    raw_contraction = (
        local_part1_contraction + local_part2_contraction
        + nonlocal_contraction)
    provider_value = cp.asarray(with_df.get_pp(np.zeros(3)))
    if provider_value.ndim == 3:
        provider_value = provider_value[0]
    provider_value_host = cp.asnumpy(provider_value)
    provider_contraction = lib.einsum(
        'ij,sji->s', provider_value_host, density_host).real
    value_parity = float(np.max(np.abs(
        value_contraction - provider_contraction)))
    if value_parity > float(value_tol):
        raise RuntimeError(
            'contracted GDF PP representation does not reproduce the '
            f'provider value: {value_parity:.3e} > {float(value_tol):.3e}')

    return {
        'name': 'V_PP_contracted',
        'primitive_analytic': True,
        'production_analytic': True,
        'gpu_native_complete': False,
        'low_level_numerical_components': (),
        'value': provider_value_host,
        'value_contraction': value_contraction,
        'provider_value_contraction': provider_contraction,
        'value_contraction_parity_max_abs': value_parity,
        'raw_contraction': raw_contraction,
        'density_count': len(density_host),
        'local_part1': {
            **local_part1,
            'value_contraction': local_part1_value,
            'raw_contraction': local_part1_contraction,
            'wall_seconds': float(local_part1_seconds),
        },
        'local_part2': local_part2,
        'nonlocal': {
            **nonlocal_part,
            'value_contraction': nonlocal_value_contraction,
            'raw_contraction': nonlocal_contraction,
            'wall_seconds': float(nonlocal_seconds),
        },
        'device_contracted_primitives': ('local_part1_long_range',),
        'host_contracted_primitives': (
            'local_part1_short_range', 'local_part1_vbar',
            'local_part2_exact_integral', 'nonlocal_exact_integral'),
        'host_matrix_primitives': (),
        'silent_cpu_fallback': False,
    }


def contract_local_part1_strain(
        with_df, densities, block_size=None, return_metadata=False):
    r'''Contract the provider-matched GTH local-part1 strain directly.

    PySCF/GPU4PySCF split the error-function local pseudopotential into

    .. math::

        V_{loc,1}=B_{SR}[-Z]+B_{LR}[-Z]+B_{vbar}[-Z]
                  +V_{G=0}^{pseudo}.

    The old transparent stress oracle first formed every AO matrix in
    ``(3,3,nao,nao)`` form.  Its long-range part additionally retained the
    much larger ``(3,3,nG,nao,nao)`` AO-pair Fourier response.  This routine
    instead contracts the same four terms with every supplied density before
    the next intermediate is built:

    * ``B_SR`` uses the package C/OpenMP three-center direct contraction;
    * ``B_LR`` uses GPU4PySCF's analytic Fourier AO-pair strain kernel;
    * ``B_vbar`` uses the analytic overlap/volume source contraction; and
    * ``V_G0^pseudo`` uses the native GPU overlap strain contraction.

    No displaced Cell, numerical derivative, changed mesh, or altered range
    separation is introduced.  The provider's resolved ``omega`` and integer
    mesh are reused exactly.

    Args:
        with_df : dcforge.gpu.pbc.df.gdf.GDF
            Active provider owning the local-part1 plan.
        densities : cupy.ndarray or ndarray
            One Hermitian real-Gamma AO density or a leading density batch.

    Kwargs:
        block_size : int or None
            Bounded reciprocal block for the long-range Fourier response.
        return_metadata : bool
            Return the term and execution-boundary ledger.

    Returns:
        tuple
            Per-density central value and raw strain contractions.  With
            ``return_metadata=True`` a third metadata dictionary is added.
    '''
    cell = with_df.cell
    density_batch = _real_density_batch(
        cell, densities, 'local-part1')
    density_host = cp.asnumpy(density_batch)
    plan = with_df.get_pp_loc_part1_plan()
    omega = float(plan['omega'])
    mesh = np.asarray(plan['mesh'], dtype=int)
    fake_nuclear = make_fake_nuclear_cell(cell, with_pseudo=True)
    charges = -np.asarray(cell.atom_charges(), dtype=float)
    if fake_nuclear.nao_nr() != len(charges):
        raise NotImplementedError(
            'local-part1 stress requires one fake nuclear Gaussian for '
            'every pseudized atom')

    nao = int(cell.nao_nr())
    nfake = int(fake_nuclear.nao_nr())
    source_adjoint = lib.einsum(
        'sji,P->sijP', density_host, charges).reshape(
            len(density_batch), nao * nao, nfake)

    short_started = time.perf_counter()
    short_range = contract_short_range_source_strain(
        cell, fake_nuclear, omega, source_adjoint)
    short_seconds = time.perf_counter() - short_started

    long_started = time.perf_counter()
    long_range = _contract_local_part1_long_range_strain(
        cell, fake_nuclear, charges, density_batch, omega, mesh,
        block_size=block_size)
    long_seconds = time.perf_counter() - long_started

    vbar_started = time.perf_counter()
    vbar = contract_vbar_source_strain(
        cell, fake_nuclear, omega, source_adjoint)
    vbar_seconds = time.perf_counter() - vbar_started

    zero_started = time.perf_counter()
    exponents = np.hstack(fake_nuclear.bas_exps())
    coefficient = float(
        charges.dot(np.pi / exponents) / float(cell.vol))
    overlap_value = int1e.int1e_ovlp(cell)
    overlap_contraction = cp.asnumpy(contract(
        'ij,sji->s', overlap_value, density_batch).real)
    overlap_response = np.asarray([
        contract_overlap_strain(cell, density)
        for density in density_batch
    ])
    pseudo_g0_value = -coefficient * overlap_contraction
    pseudo_g0_overlap = -coefficient * overlap_response
    pseudo_g0_volume = (
        -pseudo_g0_value[:, None, None] * np.eye(3)[None])
    pseudo_g0_response = pseudo_g0_overlap + pseudo_g0_volume
    zero_seconds = time.perf_counter() - zero_started

    raw = (
        np.asarray(short_range['raw_contraction'])
        + np.asarray(long_range['raw_contraction'])
        + np.asarray(vbar['raw_contraction'])
        + pseudo_g0_response)
    provider_value = get_pp_loc_part1(
        cell, np.zeros(3), omega=omega, mesh=mesh)
    value = cp.asnumpy(contract(
        'ij,sji->s', provider_value, density_batch).real)
    metadata = {
        'name': 'V_PP_local_part1_contracted',
        'primitive_analytic': True,
        'production_analytic': True,
        'gpu_native_complete': False,
        'low_level_numerical_components': (),
        'omega': omega,
        'mesh': mesh,
        'short_range': short_range,
        'long_range': long_range,
        'vbar': vbar,
        'pseudo_g0': {
            'value_contraction': pseudo_g0_value,
            'raw_contraction': pseudo_g0_response,
            'overlap_response': pseudo_g0_overlap,
            'volume_response': pseudo_g0_volume,
            'wall_seconds': float(zero_seconds),
            'execution_owner': 'gpu_overlap_direct',
        },
        'component_wall_seconds': {
            'short_range': float(short_seconds),
            'long_range': float(long_seconds),
            'vbar': float(vbar_seconds),
            'pseudo_g0': float(zero_seconds),
        },
        'execution_owner': 'host_SR_direct_GPU_LR_direct',
        'device_contracted_primitives': ('B_LR', 'pseudo_G0'),
        'host_contracted_primitives': ('B_SR', 'B_vbar'),
        'host_matrix_primitives': (),
        'silent_cpu_fallback': False,
    }
    if return_metadata:
        return value, raw, metadata
    return value, raw


def _contract_local_part1_long_range_strain(
        cell, fake_nuclear, charges, densities, omega, mesh,
        block_size=None):
    r'''Contract ``B_LR[-Z]`` without a full Fourier strain tensor.'''
    grid_vectors = np.asarray(cell.get_Gv(mesh), dtype=float)
    ngrid = len(grid_vectors)
    started = time.perf_counter()

    ft_opt = FTOpt(cell, np.ones(3, dtype=int))
    ft_kernel = ft_opt.gen_ft_kernel(
        transform_ao=False, kpts=np.zeros((1, 3)))
    sorted_cell = ft_opt.cell
    sorted_density = sorted_cell.apply_C_mat_CT(densities)
    sorted_density = cp.asarray(sorted_density.real, order='C')
    block_size = _resolve_block_size(
        int(sorted_cell.nao), ngrid, block_size)

    pair_indices, image_indices, shell_pair_offsets = (
        _generate_shl_pairs(ft_opt))
    shell_pair_batches = int(len(shell_pair_offsets) - 1)
    shared_memory = int(_estimate_max_shm_size(sorted_cell, (1, 0)))
    aft_envs = ft_opt.aft_envs
    kernel = libpbc.PBC_ft_aopair_ej_strain_deriv

    nset = len(sorted_density)
    value = cp.zeros(nset, dtype=cp.float64)
    pair_response = cp.zeros((nset, 3, 3), dtype=cp.float64)
    potential_response = cp.zeros_like(pair_response)
    unused_atom_gradient = cp.zeros((sorted_cell.natm, 3))
    block_count = 0
    peak_explicit_block_bytes = 0

    for start, stop in lib.prange(0, ngrid, block_size):
        block_count += 1
        vectors = grid_vectors[start:stop]
        pair_fourier = ft_kernel(vectors).transpose(0, 2, 3, 1)[0]
        density_fourier = contract(
            'sji,ijG->sG', sorted_density, pair_fourier)
        auxiliary = auxiliary_fourier_strain(fake_nuclear, vectors)
        coulomb = weighted_coulomb_strain(
            cell, vectors, omega=omega)
        factors = np.asarray(auxiliary['value'])
        factors1 = np.asarray(auxiliary['raw_derivative'])
        weights = np.asarray(coulomb['value'])
        weights1 = np.asarray(coulomb['raw_derivative'])
        potential_host = lib.einsum(
            'PG,P,G->G', factors, charges, weights)
        potential1_host = lib.einsum(
            'xyPG,P,G->xyG', factors1, charges, weights)
        potential1_host += lib.einsum(
            'PG,P,xyG->xyG', factors, charges, weights1)
        potential = cp.asarray(potential_host)
        potential1 = cp.asarray(potential1_host)
        value += contract(
            'sG,G->s', density_fourier, potential.conj()).real
        potential_response += contract(
            'sG,xyG->sxy', density_fourier,
            potential1.conj()).real

        reciprocal_weight = cp.asarray(potential.conj(), order='C')
        vectors_device = cp.asarray(vectors.T.ravel())
        pair_fourier = None
        density_fourier = None
        for density_id in range(nset):
            error = kernel(
                ctypes.cast(
                    unused_atom_gradient.data.ptr, ctypes.c_void_p),
                ctypes.cast(
                    pair_response[density_id].data.ptr, ctypes.c_void_p),
                ctypes.cast(
                    sorted_density[density_id:density_id + 1].data.ptr,
                    ctypes.c_void_p),
                ctypes.cast(
                    reciprocal_weight.data.ptr, ctypes.c_void_p),
                ctypes.cast(vectors_device.data.ptr, ctypes.c_void_p),
                ctypes.byref(aft_envs),
                ctypes.c_int(shell_pair_batches),
                ctypes.c_int(stop - start),
                ctypes.c_int(shared_memory),
                ctypes.cast(pair_indices.data.ptr, ctypes.c_void_p),
                ctypes.cast(image_indices.data.ptr, ctypes.c_void_p),
                ctypes.cast(shell_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.c_int(int(ft_opt.permutation_symmetry)),
            )
            if error != 0:
                raise RuntimeError(
                    'PBC_ft_aopair_ej_strain_deriv failed for GTH '
                    'local-part1 long-range stress')
        peak_explicit_block_bytes = max(
            peak_explicit_block_bytes,
            int((stop - start) * (
                16 * sorted_cell.nao ** 2
                + 16 * nset + 16 + 16 * 9)))

    pair_response *= 2.
    raw = pair_response + potential_response
    value_host = cp.asnumpy(value)
    raw_host = cp.asnumpy(raw)
    return {
        'name': 'B_LR_contracted',
        'primitive_analytic': True,
        'production_analytic': True,
        'low_level_numerical_components': (),
        'value_contraction': value_host,
        'raw_contraction': raw_host,
        'ao_pair_response': cp.asnumpy(pair_response),
        'potential_response': cp.asnumpy(potential_response),
        'mesh': np.asarray(mesh, dtype=int),
        'grid_points': int(ngrid),
        'block_size': int(block_size),
        'block_count': int(block_count),
        'peak_explicit_block_bytes': int(peak_explicit_block_bytes),
        'wall_seconds': float(time.perf_counter() - started),
        'execution_owner': 'gpu_fourier_pair_direct',
    }


def contract_local_part2_strain(
        cell, densities, mesh=None, block_size=None,
        return_metadata=False):
    r'''Contract the analytic GTH local-part2 strain with Gamma densities.

    PySCF's Gaussian providers evaluate the second local GTH term with exact
    three-center integrals.  The converged reciprocal representation used by
    the transparent stress oracle is

    .. math::

        E_{loc,2}[D]
        =\Omega^{-1}\sum_G \rho_D(G)v_2(G)^*,
        \qquad
        \rho_D(G)=\sum_{\mu\nu}D_{\nu\mu}F_{\mu\nu}(G).

    This routine keeps exactly that mesh and radial GTH potential.  It avoids
    the host ``(3,3,nG,nao,nao)`` AO-pair derivative by sending each density
    directly to GPU4PySCF's analytic Fourier AO-pair strain kernel.  The
    potential-shape and inverse-volume responses are accumulated from the
    same zero-order ``rho_D(G)`` block.  No displaced Cell or finite
    difference is used.

    Args:
        cell : pyscf.pbc.gto.Cell
            Three-dimensional real-Gamma Cell with GTH pseudopotentials.
        densities : cupy.ndarray or ndarray
            One ``(nao,nao)`` Hermitian density or a leading batch of such
            densities.  The intended production batch is source ``P`` and
            the Hermitian Z-transition density ``D_Z``.

    Kwargs:
        mesh : array_like or None
            Fixed converged local-part2 reciprocal mesh.  ``None`` resolves
            the same mesh as the CPU analytic matrix oracle.
        block_size : int or None
            Number of reciprocal vectors processed at once.  ``None`` uses
            a device-memory-aware bounded block without changing the sum.
        return_metadata : bool
            Return execution and storage diagnostics together with the
            contracted value and strain.

    Returns:
        tuple
            ``(value, raw_derivative)`` with shapes ``(nset,)`` and
            ``(nset,3,3)``.  If ``return_metadata`` is true, a third metadata
            dictionary is returned.

    Notes:
        ``PBC_ft_aopair_ej_strain_deriv`` traverses one triangular Gamma
        shell-pair representative.  Its raw strain accumulator is therefore
        multiplied by two, following GPU4PySCF's own
        :func:`get_ej_strain_deriv` normalization.  This factor belongs only
        to the AO-pair response; the potential and volume terms already use
        the complete density Fourier factor once.
    '''
    if int(cell.dimension) != 3:
        raise NotImplementedError(
            'GDF local-part2 strain requires a three-dimensional Cell')
    if not cell._pseudo:
        raise NotImplementedError(
            'GDF local-part2 strain requires GTH pseudopotentials')

    density_batch = cp.asarray(densities)
    if density_batch.ndim == 2:
        density_batch = density_batch[None]
    nao = int(cell.nao_nr())
    if density_batch.shape[1:] != (nao, nao):
        raise ValueError(
            'local-part2 density batch must have shape '
            f'(nset,{nao},{nao})')
    if cp.iscomplexobj(density_batch):
        imaginary = float(cp.max(cp.abs(density_batch.imag)).get())
        if imaginary > 1.e-10:
            raise NotImplementedError(
                'local-part2 stress requires real Gamma densities')
        density_batch = density_batch.real
    asymmetry = float(cp.max(cp.abs(
        density_batch - density_batch.transpose(0, 2, 1))).get())
    if asymmetry > 1.e-9:
        raise ValueError('local-part2 stress densities must be symmetric')

    mesh = _reciprocal_mesh(cell, mesh)
    grid_vectors = np.asarray(cell.get_Gv(mesh), dtype=float)
    ngrid = len(grid_vectors)
    started = time.perf_counter()

    ft_opt = FTOpt(cell, np.ones(3, dtype=int))
    ft_kernel = ft_opt.gen_ft_kernel(
        transform_ao=False, kpts=np.zeros((1, 3)))
    sorted_cell = ft_opt.cell
    sorted_density = sorted_cell.apply_C_mat_CT(density_batch)
    sorted_density = cp.asarray(sorted_density.real, order='C')
    block_size = _resolve_block_size(
        int(sorted_cell.nao), ngrid, block_size)

    pair_indices, image_indices, shell_pair_offsets = (
        _generate_shl_pairs(ft_opt))
    shell_pair_batches = int(len(shell_pair_offsets) - 1)
    shared_memory = int(_estimate_max_shm_size(sorted_cell, (1, 0)))
    aft_envs = ft_opt.aft_envs
    kernel = libpbc.PBC_ft_aopair_ej_strain_deriv

    nset = len(density_batch)
    value = cp.zeros(nset, dtype=cp.float64)
    pair_response = cp.zeros((nset, 3, 3), dtype=cp.float64)
    potential_response = cp.zeros_like(pair_response)
    inverse_volume = 1. / float(cell.vol)
    unused_atom_gradient = cp.zeros((sorted_cell.natm, 3))
    block_count = 0
    peak_explicit_block_bytes = 0

    for start, stop in lib.prange(0, ngrid, block_size):
        block_count += 1
        vectors = grid_vectors[start:stop]
        pair_fourier = ft_kernel(vectors).transpose(0, 2, 3, 1)[0]
        density_fourier = contract(
            'sji,ijG->sG', sorted_density, pair_fourier)
        potential_host, potential1_host = (
            _local_part2_potential_fourier_strain(cell, vectors))
        potential = cp.asarray(potential_host)
        potential1 = cp.asarray(potential1_host)
        value += inverse_volume * contract(
            'sG,G->s', density_fourier, potential.conj()).real
        potential_response += inverse_volume * contract(
            'sG,xyG->sxy', density_fourier,
            potential1.conj()).real

        reciprocal_weight = cp.asarray(
            potential.conj() * inverse_volume, order='C')
        vectors_device = cp.asarray(vectors.T.ravel())
        pair_fourier = None
        density_fourier = None
        for density_id in range(nset):
            error = kernel(
                ctypes.cast(
                    unused_atom_gradient.data.ptr, ctypes.c_void_p),
                ctypes.cast(
                    pair_response[density_id].data.ptr, ctypes.c_void_p),
                ctypes.cast(
                    sorted_density[density_id:density_id + 1].data.ptr,
                    ctypes.c_void_p),
                ctypes.cast(reciprocal_weight.data.ptr, ctypes.c_void_p),
                ctypes.cast(vectors_device.data.ptr, ctypes.c_void_p),
                ctypes.byref(aft_envs),
                ctypes.c_int(shell_pair_batches),
                ctypes.c_int(stop - start),
                ctypes.c_int(shared_memory),
                ctypes.cast(pair_indices.data.ptr, ctypes.c_void_p),
                ctypes.cast(image_indices.data.ptr, ctypes.c_void_p),
                ctypes.cast(shell_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.c_int(int(ft_opt.permutation_symmetry)),
            )
            if error != 0:
                raise RuntimeError(
                    'PBC_ft_aopair_ej_strain_deriv failed for GTH '
                    'local-part2 stress')
        peak_explicit_block_bytes = max(
            peak_explicit_block_bytes,
            int((stop - start) * (
                16 * sorted_cell.nao ** 2
                + 16 * nset + 16 + 16 * 9)))

    pair_response *= 2.
    volume_response = (
        -cp.eye(3)[None] * value[:, None, None])
    derivative = pair_response + potential_response + volume_response
    value_host = cp.asnumpy(value)
    derivative_host = cp.asnumpy(derivative)
    metadata = {
        'name': 'V_PP_local_part2_contracted',
        'primitive_analytic': True,
        'production_analytic': True,
        'low_level_numerical_components': (),
        'owner': 'gpu4pyscf_fourier_ao_pair_strain',
        'mesh': np.asarray(mesh, dtype=int),
        'grid_points': int(ngrid),
        'block_size': int(block_size),
        'block_count': int(block_count),
        'density_count': int(nset),
        'pair_response': cp.asnumpy(pair_response),
        'potential_response': cp.asnumpy(potential_response),
        'volume_response': cp.asnumpy(volume_response),
        'peak_explicit_block_bytes': int(peak_explicit_block_bytes),
        'wall_seconds': float(time.perf_counter() - started),
        'host_formula_boundary': (
            'GTH_local_part2_radial_potential_and_strain',),
    }
    if return_metadata:
        return value_host, derivative_host, metadata
    return value_host, derivative_host


def _resolve_block_size(sorted_nao, ngrid, requested):
    '''Return a bounded G-vector block for value plus strain contractions.'''
    if requested is not None:
        requested = int(requested)
        if requested <= 0:
            raise ValueError('local-part2 block_size must be positive')
        return min(int(ngrid), requested)
    available = int(get_avail_mem(exclude_memory_pool=True))
    budget = int(_RECIPROCAL_MEMORY_FRACTION * available)
    bytes_per_grid = max(1, 16 * (int(sorted_nao) ** 2 + 12))
    raw = max(
        _RECIPROCAL_BLOCK_ALIGNMENT,
        budget // bytes_per_grid)
    aligned = max(
        _RECIPROCAL_BLOCK_ALIGNMENT,
        raw // _RECIPROCAL_BLOCK_ALIGNMENT
        * _RECIPROCAL_BLOCK_ALIGNMENT)
    return min(
        int(ngrid), int(_RECIPROCAL_MAX_BLOCK_SIZE), int(aligned))


def _real_density_batch(cell, densities, name):
    '''Return a validated symmetric real-Gamma density batch on the GPU.'''
    density_batch = cp.asarray(densities)
    if density_batch.ndim == 2:
        density_batch = density_batch[None]
    nao = int(cell.nao_nr())
    if density_batch.ndim != 3 or density_batch.shape[1:] != (nao, nao):
        raise ValueError(
            f'{name} density batch must have shape (nset,{nao},{nao})')
    if cp.iscomplexobj(density_batch):
        imaginary = float(cp.max(cp.abs(density_batch.imag)).get())
        if imaginary > 1.e-10:
            raise NotImplementedError(
                f'{name} stress requires real Gamma densities')
        density_batch = density_batch.real
    asymmetry = float(cp.max(cp.abs(
        density_batch - density_batch.transpose(0, 2, 1))).get())
    if asymmetry > 1.e-9:
        raise ValueError(f'{name} stress densities must be symmetric')
    return density_batch


__all__ = [
    'contract_gdf_pseudopotential_strain',
    'contract_local_part1_strain',
    'contract_local_part2_strain',
]
