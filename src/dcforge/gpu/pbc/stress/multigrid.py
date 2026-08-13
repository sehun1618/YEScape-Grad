'''XC-only multigrid strain derivatives for GPU4PySCF 1.8.1.'''

import ctypes
import inspect

import cupy as cp
import gpu4pyscf
import numpy as np
from gpu4pyscf.pbc.dft import multigrid_v2
from gpu4pyscf.pbc.dft.multigrid_v2 import _rks_exc_strain_deriv
from pyscf.gto.mole import ATOM_OF

_SUPPORTED_GPU4PYSCF_VERSION = '1.8.1'
_EXPECTED_PARAMETERS = (
    'ni', 'xc_code', 'dm_kpts', 'kpts', 'with_j', 'with_nuc')


def get_vxc_strain(mf_grad, dm=None, with_j=False, with_nuc=False,
                   return_metadata=False):
    r'''Return GPU multigrid XC-only energy-like strain derivative.

    This adapter deliberately calls only the private XC/MG primitive, never
    GPU4PySCF's complete FFTDF-RKS stress kernel.  Hartree J remains owned by
    DC-Forge GDF and pseudopotential strain remains owned by the future GDF
    Hcore ledger.

    GPU4PySCF 1.8.1 evaluates the multigrid density response using nine
    internal ``1e-4`` Cell displacements.  The result is consequently labelled
    semi-analytic and cannot set ``production_analytic=True``.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.rhf_dft.Gradients
            Active GPU fixed-density gradient object.

    Kwargs:
        dm : cupy.ndarray or None
            Fixed spin-summed source-RHF density.  The converged source density
            is used when omitted.
        with_j, with_nuc : bool
            Must both be false.  They are visible only to audit ownership at
            the call boundary.
        return_metadata : bool
            Return ``(work_xc, metadata)`` when true.

    Returns:
        ndarray of shape (3, 3) or tuple
            Host NumPy energy-like XC strain derivative in Hartree and
            optional provenance metadata.
    '''
    if with_j or with_nuc:
        raise NotImplementedError(
            'DC-Forge XC strain requires with_j=False and with_nuc=False')
    _check_private_api()
    mf = mf_grad.base
    if not isinstance(mf._numint, multigrid_v2.MultiGridNumInt):
        raise TypeError('GPU XC strain requires a multigrid-v2 target')
    if mf.multigrid_with_j:
        raise NotImplementedError('GPU XC strain requires with_j=False')
    if dm is None:
        if not mf.mf_hf.converged:
            raise RuntimeError('run source RHF before requesting XC strain')
        dm = mf.make_hf_rdm1()
    dm = cp.asarray(dm)
    if dm.ndim != 2 or dm.shape != (mf.cell.nao_nr(),) * 2:
        raise ValueError('GPU XC strain density must have shape (nao, nao)')

    work_xc = np.asarray(_rks_exc_strain_deriv(
        mf._numint, mf.xc, dm[None], None,
        with_j=False, with_nuc=False))
    if work_xc.shape != (3, 3) or not np.all(np.isfinite(work_xc)):
        raise RuntimeError('GPU multigrid XC strain returned an invalid tensor')
    metadata = {
        'target_xc_stress_owner': 'gpu4pyscf_multigrid_v2',
        'gpu4pyscf_version': str(gpu4pyscf.__version__),
        'with_j': False,
        'with_nuc': False,
        'mesh': np.asarray(mf._numint.mesh, dtype=int).tolist(),
        'production_analytic': False,
        'matched_target_integrator': True,
        'low_level_numerical_components': [
            'multigrid_density_strain_cell_displacement'],
    }
    if return_metadata:
        return work_xc, metadata
    return work_xc


def get_vxc_strain_uniform_grid_oracle(
        mf_grad, dm=None, with_j=False, with_nuc=False,
        return_metadata=False):
    r'''Return a fully analytic uniform-grid XC-only strain oracle.

    GPU4PySCF's ordinary real-Gamma uniform-grid stress path evaluates AO
    homogeneous-strain derivatives directly.  Its GGA path differentiates
    ``rho`` and ``grad rho``; its meta-GGA path additionally differentiates
    the kinetic-energy density ``tau``.  This helper evaluates only that XC
    contraction on the exact integer mesh used by the active multigrid
    target:

    .. math::

        W_{xc}^{(xy)}
        = \int w^{(xy)} e_{xc}
        + \int w\sum_n v_{xc,n}\rho_n^{(xy)}.

    No displaced Cell, target SCF, Hartree J or pseudopotential term is
    evaluated.  The result is an independent mathematical oracle, not yet a
    production multigrid derivative: it uses the same global uniform mesh and
    LibXC functional but not multigrid-v2's screened pair/task traversal.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.rhf_dft.Gradients
            Active fixed-density GPU gradient object.

    Kwargs:
        dm : cupy.ndarray or None
            Fixed spin-summed source-RHF density.
        with_j, with_nuc : bool
            Must both be false to preserve the GDF + XC-only ownership
            contract.
        return_metadata : bool
            Return ``(work_xc, metadata)`` when true.

    Returns:
        ndarray of shape (3, 3) or tuple
            Analytic energy-like XC strain in Hartree and optional
            provenance metadata.
    '''
    from gpu4pyscf.pbc.dft import rks as gpu_rks
    from gpu4pyscf.pbc.dft.gen_grid import UniformGrids
    from gpu4pyscf.pbc.grad import rks as gpu_rks_grad
    from gpu4pyscf.pbc.grad import rks_stress

    if with_j or with_nuc:
        raise NotImplementedError(
            'DC-Forge XC strain requires with_j=False and with_nuc=False')
    mf = mf_grad.base
    if not isinstance(mf._numint, multigrid_v2.MultiGridNumInt):
        raise TypeError('GPU XC strain oracle requires a multigrid-v2 target')
    if mf.multigrid_with_j:
        raise NotImplementedError('GPU XC strain oracle requires with_j=False')
    if dm is None:
        if not mf.mf_hf.converged:
            raise RuntimeError('run source RHF before requesting XC strain')
        dm = mf.make_hf_rdm1()
    dm = cp.asarray(dm)
    if dm.ndim != 2 or dm.shape != (mf.cell.nao_nr(),) * 2:
        raise ValueError('GPU XC strain density must have shape (nao, nao)')

    oracle = gpu_rks.RKS(mf.cell, xc=mf.xc)
    oracle.grids = UniformGrids(mf.cell)
    oracle.grids.mesh = np.asarray(mf._numint.mesh, dtype=int).copy()
    oracle_grad = gpu_rks_grad.Gradients(oracle)
    work_xc = np.asarray(rks_stress.get_vxc(
        oracle_grad, mf.cell, dm,
        with_j=False, with_nuc=False))
    if work_xc.shape != (3, 3) or not np.all(np.isfinite(work_xc)):
        raise RuntimeError(
            'GPU uniform-grid XC strain returned an invalid tensor')
    xctype = oracle._numint._xc_type(mf.xc)
    density_variables = {
        'LDA': ('rho',),
        'GGA': ('rho', 'grad_rho'),
        'MGGA': ('rho', 'grad_rho', 'tau'),
    }[xctype]
    metadata = {
        'target_xc_stress_owner': (
            'gpu4pyscf_uniform_grid_analytic_oracle'),
        'gpu4pyscf_version': str(gpu4pyscf.__version__),
        'with_j': False,
        'with_nuc': False,
        'mesh': np.asarray(mf._numint.mesh, dtype=int).tolist(),
        'production_analytic': False,
        'primitive_analytic': True,
        'matched_target_integrator': False,
        'matched_global_uniform_mesh': True,
        'density_variables': density_variables,
        'low_level_numerical_components': [],
        'remaining_owner_difference': (
            'uniform AO traversal instead of multigrid-v2 screened tasks'),
    }
    if return_metadata:
        return work_xc, metadata
    return work_xc


def get_vxc_strain_analytic(
        mf_grad, dm=None, with_j=False, with_nuc=False,
        image_block_size=None, return_metadata=False):
    r'''Return the analytic XC-only multigrid strain derivative.

    This function differentiates the *same* screened Gaussian-pair tasks and
    pair meshes that produced the active multigrid energy.  It reuses
    GPU4PySCF's analytic XC nuclear-gradient CUDA kernels and converts their
    center derivatives to a homogeneous-strain derivative,

    .. math::

        \frac{\partial E_{xc}}{\partial\varepsilon_{xy}}
        = \sum_{I\mathbf L}
          (R_{Iy}+L_y)\frac{\partial E_{xc}}
                                {\partial R_{I\mathbf L,x}}
          + \int r_y\frac{\delta E_{xc}}{\delta r_x}
          + \delta_{xy}E_{xc}.

    The first term is separated into atom-center and lattice-image moments.
    The grid-motion term is obtained by multiplying the pair-mesh XC weight
    by the Cartesian grid coordinate before invoking the same analytic CUDA
    derivative kernel.  For GGA and meta-GGA functionals, an additional
    Cartesian-gradient metric term is required after integration by parts,

    .. math::

        W_{xy}^{\nabla\rho}
        = -\int v_{\nabla\rho,y}\,\partial_x\rho\,d\mathbf r.

    The meta-GGA ``tau`` response needs no numerical Cell displacement.  The
    existing ``evaluate_xc_with_tau_gradient_driver`` already contracts the
    Gaussian first and second derivatives; the same center/grid moments turn
    that contraction into the analytic ``tau`` strain response.

    Only real Gamma-point RHF is accepted.  Target Hartree and
    pseudopotential terms remain disabled because they belong to the GDF
    stress ledger.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.rhf_dft.Gradients
            Active GPU fixed-density gradient object.

    Kwargs:
        dm : cupy.ndarray or None
            Fixed spin-summed source-RHF density.  The converged source
            density is used when omitted.
        with_j, with_nuc : bool
            Must both be false.
        image_block_size : int or None
            Maximum number of absolute lattice-image channels contracted in
            one batch.  ``None`` selects a density-size-aware block bounded
            to approximately 128 MiB.
        return_metadata : bool
            Return ``(work_xc, metadata)`` when true.

    Returns:
        ndarray of shape (3, 3) or tuple
            Analytic energy-like XC strain derivative in Hartree and optional
            provenance metadata.
    '''
    if with_j or with_nuc:
        raise NotImplementedError(
            'DC-Forge XC strain requires with_j=False and with_nuc=False')
    _check_private_api()
    mf = mf_grad.base
    ni = mf._numint
    if not isinstance(ni, multigrid_v2.MultiGridNumInt):
        raise TypeError('GPU analytic XC strain requires a multigrid-v2 target')
    if mf.multigrid_with_j:
        raise NotImplementedError('GPU analytic XC strain requires with_j=False')
    if dm is None:
        if not mf.mf_hf.converged:
            raise RuntimeError('run source RHF before requesting XC strain')
        dm = mf.make_hf_rdm1()
    dm = cp.asarray(dm, dtype=np.float64, order='C')
    if dm.ndim != 2 or dm.shape != (mf.cell.nao_nr(),) * 2:
        raise ValueError('GPU XC strain density must have shape (nao, nao)')

    xctype = ni._xc_type(mf.xc)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        raise NotImplementedError(
            f'analytic multigrid XC strain does not support {xctype}')
    if ni.sorted_gaussian_pairs is None:
        ni.build(xctype)

    cell = mf.cell
    mesh = np.asarray(ni.mesh, dtype=int)
    ngrids = int(np.prod(mesh))
    weight = cell.vol / ngrids
    dm_kpts = dm[None]

    density_g = multigrid_v2.evaluate_density_on_g_mesh(
        ni, dm_kpts, None, xctype)
    rho = multigrid_v2.ifft_in_place(
        density_g.reshape(1, -1, *mesh)).real.reshape(1, -1, ngrids)
    rho /= weight
    rho = cp.asarray(rho, dtype=np.float64, order='C')
    exc, vxc = ni.eval_xc_eff(
        mf.xc, rho[0], deriv=1, xctype=xctype, spin=0)[:2]

    exc_energy = cp.einsum(
        'g,g->', rho[0, 0], exc.ravel()).real * weight
    work_xc = cp.eye(3, dtype=np.float64) * exc_energy

    xc_on_g_mesh = _make_effective_xc_weights(
        cell, mesh, weight, vxc, xctype)
    dms = multigrid_v2._format_dms(dm_kpts, None)
    center_work = cp.zeros((3, 3), dtype=np.float64)
    grid_work = cp.zeros((3, 3), dtype=np.float64)
    image_work = cp.zeros((3, 3), dtype=np.float64)
    image_moment_metadata = {
        'pair_groups': 0,
        'unique_translations_max': 0,
        'translation_block_size_max': 0,
        'translation_blocks': 0,
        'driver_calls': 0,
    }

    for pairs in ni.sorted_gaussian_pairs:
        xc_weights = _interpolate_xc_weights(
            xc_on_g_mesh, mesh, pairs)
        pair_dm = _make_pair_density_matrix(dms, pairs)
        multigrid_v2.libgpbc.update_dxyz_dabc(
            pairs['dxyz_dabc'].ctypes)
        img_phase = multigrid_v2.image_phase_for_kpts(
            cell, pairs['neighboring_images'], None)

        center_gradient = cp.zeros((cell.natm, 3), dtype=np.float64)
        multigrid_v2.evaluate_xc_gradient_wrapper(
            center_gradient, pairs, xc_weights, pair_dm, img_phase,
            ignore_imag=True, with_tau=(xctype == 'MGGA'))
        center_work += cp.einsum(
            'ax,ay->xy', center_gradient,
            cp.asarray(cell.atom_coords(), dtype=np.float64))

        for y in range(3):
            grid_coordinate = _pair_mesh_coordinate(pairs, y)
            grid_gradient = cp.zeros((cell.natm, 3), dtype=np.float64)
            multigrid_v2.evaluate_xc_gradient_wrapper(
                grid_gradient, pairs,
                cp.ascontiguousarray(
                    xc_weights * grid_coordinate[None, None]),
                pair_dm, img_phase, ignore_imag=True,
                with_tau=(xctype == 'MGGA'))
            grid_work[:, y] -= cp.sum(grid_gradient, axis=0)

        image_part, image_metadata = _contract_image_center_moments(
            cell, pairs, xc_weights, pair_dm, img_phase,
            with_tau=(xctype == 'MGGA'),
            image_block_size=image_block_size)
        image_work += image_part
        for name in ('pair_groups', 'translation_blocks', 'driver_calls'):
            image_moment_metadata[name] += image_metadata[name]
        for name in (
                'unique_translations_max', 'translation_block_size_max'):
            image_moment_metadata[name] = max(
                image_moment_metadata[name], image_metadata[name])

    work_xc += center_work + image_work + grid_work
    gradient_metric_work = cp.zeros((3, 3), dtype=np.float64)
    if xctype in ('GGA', 'MGGA'):
        gradient_metric_work -= cp.einsum(
            'yg,xg->xy', vxc[1:4], rho[0, 1:4]).real * weight
        work_xc += gradient_metric_work

    work_xc = cp.asnumpy(work_xc)
    if work_xc.shape != (3, 3) or not np.all(np.isfinite(work_xc)):
        raise RuntimeError(
            'GPU analytic multigrid XC strain returned an invalid tensor')
    metadata = {
        'target_xc_stress_owner': (
            'dcforge_gpu4pyscf_multigrid_v2_analytic_pair_tasks'),
        'gpu4pyscf_version': str(gpu4pyscf.__version__),
        'with_j': False,
        'with_nuc': False,
        'mesh': mesh.tolist(),
        'production_analytic': True,
        'primitive_analytic': True,
        'matched_target_integrator': True,
        'matched_global_uniform_mesh': True,
        'density_variables': {
            'LDA': ('rho',),
            'GGA': ('rho', 'grad_rho'),
            'MGGA': ('rho', 'grad_rho', 'tau'),
        }[xctype],
        'strain_terms': {
            'volume': float(exc_energy.get()),
            'atom_center': cp.asnumpy(center_work).tolist(),
            'lattice_image': cp.asnumpy(image_work).tolist(),
            'grid_motion': cp.asnumpy(grid_work).tolist(),
            'gradient_metric': cp.asnumpy(gradient_metric_work).tolist(),
        },
        'image_moment': image_moment_metadata,
        'image_block_size_requested': image_block_size,
        'low_level_numerical_components': [],
    }
    if return_metadata:
        return work_xc, metadata
    return work_xc


def _make_effective_xc_weights(cell, mesh, weight, vxc, xctype):
    '''Transform LibXC variables to the weights used by MG pair kernels.'''
    ngrids = int(np.prod(mesh))
    xc_on_g_mesh = vxc.reshape(1, -1, *mesh) * weight
    xc_on_g_mesh = multigrid_v2.fft_in_place(
        xc_on_g_mesh).reshape(1, -1, ngrids)
    if xctype == 'LDA':
        return xc_on_g_mesh

    vectors = multigrid_v2.get_Gv(cell, mesh)
    xc_on_g_mesh[:, 0] -= multigrid_v2.contract(
        'ngp,pg->np', xc_on_g_mesh[:, 1:4], vectors) * 1j
    if xctype == 'GGA':
        return xc_on_g_mesh[:, :1]
    return cp.concatenate((
        xc_on_g_mesh[:, :1], xc_on_g_mesh[:, 4:5]), axis=1)


def _interpolate_xc_weights(xc_on_g_mesh, mesh, pairs):
    '''Interpolate global reciprocal XC weights to one active pair mesh.'''
    global_weights = xc_on_g_mesh.reshape(
        xc_on_g_mesh.shape[:2] + tuple(mesh))
    interpolated = global_weights[
        :, :, pairs['fft_grid'][0][:, None, None],
        pairs['fft_grid'][1][:, None], pairs['fft_grid'][2]]
    return cp.asarray(
        multigrid_v2.ifft_in_place(interpolated).real, order='C')


def _make_pair_density_matrix(dms, pairs):
    '''Apply the active MG primitive contractions to the AO density.'''
    density = dms[
        :, :, pairs['ao_indices_in_localized'][:, None],
        pairs['concatenated_ao_indices']]
    diffuse = dms[
        :, :, pairs['ao_indices_in_diffused'][:, None],
        pairs['ao_indices_in_localized']]
    nlocal = diffuse.shape[3]
    density[:, :, :, nlocal:] += diffuse.transpose(0, 1, 3, 2).conj()
    density = cp.einsum(
        'nkij,pi->nkpj', density, pairs['coeff_in_localized'])
    return cp.einsum(
        'nkpj,qj->nkpq', density, pairs['concatenated_coeff'])


def _pair_mesh_coordinate(pairs, component):
    '''Return one Cartesian coordinate on an active MG pair mesh.'''
    mesh = tuple(int(value) for value in pairs['mesh'])
    dxyz = cp.asarray(pairs['dxyz_dabc'], dtype=np.float64)
    return (
        cp.arange(mesh[0], dtype=np.float64)[:, None, None]
        * dxyz[0, component]
        + cp.arange(mesh[1], dtype=np.float64)[None, :, None]
        * dxyz[1, component]
        + cp.arange(mesh[2], dtype=np.float64)[None, None, :]
        * dxyz[2, component])


def _contract_image_center_moments(
        cell, pairs, xc_weights, pair_dm, img_phase, with_tau,
        image_block_size=None):
    '''Contract lattice-image center motion with analytic AO derivatives.

    The upstream gradient ABI accumulates both AO roles into the same atom
    slot.  Stress needs their *different* lattice-image vectors.  We therefore
    duplicate only the atom labels in the integral environment: the bra role
    writes to ``0:natm`` and the ket role to ``natm:2*natm``.  The Gaussian
    parameters, coordinates, pair screening, blocks and mesh are unchanged.

    Absolute image-pair indices are remapped to compact translation channels.
    Weighting the density channel by ``L_i`` or ``L_j`` then obtains all image
    moments in six CUDA calls per pair level, rather than one call per image.
    '''
    phase, original_image_map = img_phase
    if phase.shape != (1, 1):
        raise NotImplementedError(
            'analytic MG strain currently supports real Gamma only')
    del original_image_map

    natm = cell.natm
    nbas = pairs['bas'].shape[0]
    atm = cp.ascontiguousarray(cp.concatenate((pairs['atm'], pairs['atm'])))
    bas = cp.ascontiguousarray(cp.concatenate((pairs['bas'], pairs['bas'])))
    bas[nbas:, ATOM_OF] += natm
    images = cp.asarray(pairs['neighboring_images'], dtype=np.float64)
    nimages = len(images)
    out = cp.zeros((3, 3), dtype=np.float64)
    diagnostics = {
        'pair_groups': 0,
        'unique_translations_max': 0,
        'translation_block_size_max': 0,
        'translation_blocks': 0,
        'driver_calls': 0,
    }

    for angular_pairs in pairs['per_angular_pairs']:
        diagnostics['pair_groups'] += 1
        absolute_indices = angular_pairs['image_indices']
        unique_indices = cp.unique(absolute_indices)
        if len(unique_indices) == 0:
            continue
        diagnostics['unique_translations_max'] = max(
            diagnostics['unique_translations_max'], len(unique_indices))

        shell_to_ao = angular_pairs['shell_to_ao_indices']
        shell_to_ao = cp.ascontiguousarray(cp.concatenate((
            shell_to_ao[:-1], shell_to_ao)))
        j_shells = cp.ascontiguousarray(
            angular_pairs['j_shells'] + nbas)

        base_dm = cp.ascontiguousarray(pair_dm[:, 0])
        if image_block_size is None:
            block_size = _image_translation_block_size(
                base_dm, len(unique_indices))
        else:
            if int(image_block_size) < 1:
                raise ValueError('image_block_size must be positive')
            block_size = min(int(image_block_size), len(unique_indices))
        diagnostics['translation_block_size_max'] = max(
            diagnostics['translation_block_size_max'], block_size)
        for p0 in range(0, len(unique_indices), block_size):
            diagnostics['translation_blocks'] += 1
            block_indices = unique_indices[p0:p0 + block_size]
            nblock = len(block_indices)
            # Translation channel zero is deliberately empty.  Absolute
            # image pairs outside this block map to it and therefore make no
            # contribution, while selected pairs map to 1:nblock.
            image_map = cp.zeros(nimages * nimages, dtype=cp.int32)
            image_map[block_indices] = cp.arange(
                1, nblock + 1, dtype=cp.int32)
            image_i = block_indices // nimages
            image_j = block_indices % nimages
            zero = cp.zeros(1, dtype=np.float64)

            for y in range(3):
                bra_weights = cp.concatenate((zero, images[image_i, y]))
                bra_dm = cp.ascontiguousarray(
                    base_dm[:, None]
                    * bra_weights[None, :, None, None])
                bra_gradient = _call_role_separated_gradient(
                    pairs, angular_pairs, xc_weights, bra_dm, image_map,
                    nblock + 1, atm, bas, j_shells, shell_to_ao,
                    with_tau)
                diagnostics['driver_calls'] += 1
                out[:, y] += cp.sum(bra_gradient[:natm], axis=0)
                del bra_dm, bra_gradient

                ket_weights = cp.concatenate((zero, images[image_j, y]))
                ket_dm = cp.ascontiguousarray(
                    base_dm[:, None]
                    * ket_weights[None, :, None, None])
                ket_gradient = _call_role_separated_gradient(
                    pairs, angular_pairs, xc_weights, ket_dm, image_map,
                    nblock + 1, atm, bas, j_shells, shell_to_ao,
                    with_tau)
                diagnostics['driver_calls'] += 1
                out[:, y] += cp.sum(ket_gradient[natm:], axis=0)
                del ket_dm, ket_gradient
    return out, diagnostics


def _image_translation_block_size(
        base_dm, ntranslations, target_bytes=128 << 20):
    '''Return a bounded absolute-image channel count.

    One weighted density channel has the same storage as ``base_dm``.  The
    empty guard channel and one active weighted batch are kept below 128 MiB;
    the bra and ket arrays are allocated and released sequentially.
    '''
    if ntranslations < 1:
        return 1
    bytes_per_translation = max(int(base_dm.nbytes), 1)
    available = max(int(target_bytes) // bytes_per_translation - 1, 1)
    return min(int(ntranslations), available)


def _call_role_separated_gradient(
        pairs, angular_pairs, xc_weights, density, image_map,
        ntranslations, atm, bas, j_shells, shell_to_ao, with_tau):
    '''Invoke the upstream analytic MG gradient ABI with split AO roles.'''
    if with_tau:
        driver = multigrid_v2.libgpbc.evaluate_xc_with_tau_gradient_driver
    else:
        driver = multigrid_v2.libgpbc.evaluate_xc_gradient_driver
    gradient = cp.zeros((len(atm), 3), dtype=np.float64)
    i_angular, j_angular = angular_pairs['angular']
    nimages = len(pairs['neighboring_images'])
    err = driver(
        multigrid_v2.cast_to_pointer(gradient),
        multigrid_v2.cast_to_pointer(xc_weights),
        multigrid_v2.cast_to_pointer(density),
        ctypes.c_int(int(i_angular)), ctypes.c_int(int(j_angular)),
        multigrid_v2.cast_to_pointer(
            angular_pairs['screened_shell_pairs']),
        multigrid_v2.cast_to_pointer(angular_pairs['i_shells']),
        multigrid_v2.cast_to_pointer(j_shells),
        ctypes.c_int(len(angular_pairs['j_shells'])),
        multigrid_v2.cast_to_pointer(shell_to_ao),
        ctypes.c_int(density.shape[-2]),
        ctypes.c_int(density.shape[-1]),
        multigrid_v2.cast_to_pointer(
            angular_pairs['pair_indices_per_block']),
        multigrid_v2.cast_to_pointer(
            angular_pairs['accumulated_counts_per_block']),
        multigrid_v2.cast_to_pointer(
            angular_pairs['sorted_block_index']),
        ctypes.c_int(len(angular_pairs['sorted_block_index'])),
        multigrid_v2.cast_to_pointer(angular_pairs['image_indices']),
        multigrid_v2.cast_to_pointer(pairs['neighboring_images']),
        ctypes.c_int(nimages),
        multigrid_v2.cast_to_pointer(image_map),
        ctypes.c_int(ntranslations),
        multigrid_v2.cast_to_pointer(pairs['mesh']),
        multigrid_v2.cast_to_pointer(atm),
        multigrid_v2.cast_to_pointer(bas),
        multigrid_v2.cast_to_pointer(pairs['env']),
        ctypes.c_int(density.shape[0]),
        ctypes.c_int(pairs['is_non_orthogonal']),
        ctypes.c_int(0),
    )
    if err != 0:
        raise RuntimeError(
            'GPU analytic multigrid image-moment contraction failed for '
            f'li={i_angular} lj={j_angular}')
    return gradient


def _check_private_api():
    version = str(gpu4pyscf.__version__)
    if version != _SUPPORTED_GPU4PYSCF_VERSION:
        raise RuntimeError(
            'GPU multigrid strain bridge requires GPU4PySCF '
            f'{_SUPPORTED_GPU4PYSCF_VERSION}, found {version}')
    parameters = tuple(inspect.signature(
        _rks_exc_strain_deriv).parameters)
    if parameters != _EXPECTED_PARAMETERS:
        raise RuntimeError(
            'GPU4PySCF _rks_exc_strain_deriv signature changed: '
            f'{parameters!r}')


__all__ = [
    'get_vxc_strain',
    'get_vxc_strain_analytic',
    'get_vxc_strain_uniform_grid_oracle',
]
