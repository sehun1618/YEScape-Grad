'''Exact-integral local and nonlocal GTH strain/gradient primitives.'''

import ctypes
import importlib
import time
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib
from pyscf.gto import moleintor
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.scf import _vhf


def contract_vpploc_part2_strain(
        cell, densities, return_metadata=False):
    r'''Contract the exact GDF GTH local-part2 strain analytically.

    PySCF evaluates the polynomial-Gaussian part of the local GTH
    pseudopotential as periodic three-center one-electron integrals.  For a
    symmetric real-Gamma density, exchange of the two primary AO centers
    makes their strain contributions equal.  With libcint's ``ip1`` sign
    convention the density-contracted response is therefore

    .. math::

        E_{loc,2}^{(xy)}[D]
        =2\sum_{ijA,L_i,L_j}D_{ij}
          I_{ijA,x}^{ip1}(L_i,L_j)
          (R_A-R_i-L_i)_y.

    The package callback is executed inside PySCF's screened periodic
    three-center driver.  It retains ``L_i`` only long enough to apply the
    last factor and contracts every supplied density before leaving the
    shell/image loop.  Consequently neither a reciprocal mesh nor a
    ``(3,3,nao,nao)`` derivative matrix is formed.

    Args:
        cell : pyscf.pbc.gto.Cell
            Three-dimensional real-Gamma Cell with GTH pseudopotentials.
        densities : ndarray
            One Hermitian AO density or a leading batch of such densities.

    Kwargs:
        return_metadata : bool
            Return the exact-integral execution ledger.

    Returns:
        tuple
            Per-density central values and raw ``(3,3)`` strain
            contractions.  With ``return_metadata=True`` a third metadata
            dictionary is returned.
    '''
    if int(cell.dimension) != 3:
        raise NotImplementedError(
            'exact local-part2 GTH strain requires a 3D Cell')
    if not cell._pseudo:
        raise NotImplementedError(
            'exact local-part2 GTH strain requires pseudopotentials')
    batch = _validate_real_gamma_densities(
        cell, densities, 'local-part2 strain')
    started = time.perf_counter()
    raw = np.zeros((len(batch), 3, 3), dtype=float)
    channel_seconds = {}
    channel_count = 0
    intors = (
        'int3c2e_ip1', 'int3c1e_ip1',
        'int3c1e_ip1_r2_origk', 'int3c1e_ip1_r4_origk',
        'int3c1e_ip1_r6_origk')
    lattice_vectors = np.asarray(
        cell.get_lattice_Ls(), dtype=float, order='C')
    extension = _load_vpploc2_strain_extension()
    fill = extension.DCFORGE_vpploc2_strain_fill_gs1

    for polynomial_order in range(1, 5):
        fakecell = pp_int.fake_cell_vloc(cell, polynomial_order)
        fakecell.precision = pp_int.EPS_PPL
        if fakecell.nbas == 0:
            continue
        channel_started = time.perf_counter()
        neighbor_list = pbcgto.build_neighbor_list_for_shlpairs(
            fakecell, cell, lattice_vectors)
        try:
            raw += _contract_vpploc2_channel(
                cell, fakecell, batch, intors[polynomial_order],
                lattice_vectors, neighbor_list, fill)
        finally:
            pbcgto.free_neighbor_list(neighbor_list)
        channel_seconds[str(polynomial_order)] = float(
            time.perf_counter() - channel_started)
        channel_count += 1

    provider = np.asarray(pp_int.get_pp_loc_part2(cell))
    value = lib.einsum('ij,sji->s', provider, batch).real
    metadata = {
        'name': 'V_PP_local_part2_exact_contracted',
        'primitive_analytic': True,
        'production_analytic': True,
        'low_level_numerical_components': (),
        'polynomial_channels': int(channel_count),
        'lattice_vectors': len(lattice_vectors),
        'channel_wall_seconds': channel_seconds,
        'wall_seconds': float(time.perf_counter() - started),
        'storage': 'screened_image_integral_direct_contraction',
        'execution_owner': 'pyscf_exact_integral_host_direct',
        'reciprocal_mesh': None,
        'silent_fallback': False,
    }
    if return_metadata:
        return value, raw, metadata
    return value, raw


def _contract_vpploc2_channel(
        cell, fakecell, batch, intor, lattice_vectors, neighbor_list,
        fill):
    '''Apply one polynomial local-GTH channel in the screened C driver.'''
    intor = cell._add_suffix(intor)
    intor, _integral_components = moleintor._get_intor_and_comp(
        intor, 3)
    primary_basis_count = int(cell.nbas)
    atm, bas, env = gto.conc_env(
        cell._atm, cell._bas, cell._env,
        cell._atm, cell._bas, cell._env)
    ao_locations = moleintor.make_loc(bas, intor)
    auxiliary_locations = fakecell.ao_loc_nr()
    ao_locations = np.asarray(np.hstack((
        ao_locations,
        ao_locations[-1] + auxiliary_locations[1:])), dtype=np.int32)
    atm, bas, env = gto.conc_env(
        atm, bas, env, fakecell._atm, fakecell._bas, fakecell._env)
    atm = np.asarray(atm, dtype=np.int32)
    bas = np.asarray(bas, dtype=np.int32)
    env = np.asarray(env, dtype=np.double)
    env[gto.PTR_EXPCUTOFF] = abs(np.log(cell.precision))
    optimizer = _vhf.make_cintopt(atm, bas, env, intor)
    pp_int.libpbc.CINTdel_pairdata_optimizer(optimizer)

    nobs = len(batch)
    output = np.zeros((cell.natm, nobs * 9), dtype=np.double)
    phase = np.ones(1, dtype=np.complex128)
    kpoint_pair = np.zeros(1, dtype=np.int32)
    shell_slice = (
        0, primary_basis_count,
        primary_basis_count, 2 * primary_basis_count,
        2 * primary_basis_count,
        2 * primary_basis_count + fakecell.nbas)
    driver = pp_int.libpbc.PBCnr3c1e_screened_nuc_grad_drv
    driver(
        getattr(moleintor.libcgto, intor), fill,
        output.ctypes.data_as(ctypes.c_void_p),
        batch.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(nobs * 9),
        ctypes.c_int(len(lattice_vectors)),
        lattice_vectors.ctypes.data_as(ctypes.c_void_p),
        phase.ctypes.data_as(ctypes.c_void_p),
        kpoint_pair.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int * 6)(*shell_slice),
        ao_locations.ctypes.data_as(ctypes.c_void_p), optimizer,
        lib.c_null_ptr(),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
        bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(primary_basis_count),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size),
        ctypes.c_int(cell.nao_nr()), ctypes.byref(neighbor_list))
    return output[0].reshape(nobs, 3, 3)


def get_vppnl_deriv(cell, kpts):
    r'''Return the Gaussian-DF nonlocal GTH matrix derivative.

    PySCF GDF and RSGDF evaluate each nonlocal channel from analytic projector
    half-integrals ``I`` as ``V_nl = I^dagger h I``.  This routine applies the
    product rule to the same half-integrals, including projector-center and
    atom-owned AO-center translations.

    Returns:
        ndarray of shape (natm, nkpts, 3, nao, nao)
            Exact-integral nonlocal pseudopotential response in Ha/Bohr.
    '''
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    fakecell, hl_blocks = pp_int.fake_cell_vnl(cell)
    values = pp_int._int_vnl(cell, fakecell, hl_blocks, kpts)
    derivative_intors = (
        'int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2')
    derivatives = pp_int._int_vnl(
        cell, fakecell, hl_blocks, kpts, derivative_intors, comp=3)

    # PySCF's first historical intor is ip1 rather than ip2.  Its own
    # vppnl_nuc_grad performs this sign conversion before contraction.
    if len(derivatives[0]) > 0:
        for k in range(len(kpts)):
            derivatives[0][k] *= -1

    nao = cell.nao_nr()
    v1_projector = np.zeros(
        (cell.natm, len(kpts), 3, nao, nao), dtype=np.complex128)
    v1_ao = np.zeros_like(v1_projector)
    aoslices = cell.aoslice_by_atom()
    for k in range(len(kpts)):
        offsets = [0, 0, 0]
        for shell, hl in enumerate(hl_blocks):
            l = int(fakecell.bas_angular(shell))
            nm = 2 * l + 1
            nl = np.asarray(hl).shape[0]
            proj, proj1 = _projector_shell(
                values, derivatives, k, offsets, nl, nm, nao)
            half1 = lib.einsum(
                'ixmp,ij,jmq->xpq', proj1.conj(), hl, proj)
            ia = int(fakecell._bas[shell, 0])
            v1_projector[ia, k] += half1 + half1.transpose(
                0, 2, 1).conj()
            for ja in range(cell.natm):
                p0, p1 = aoslices[ja, 2:]
                v1_ao[ja, k, :, p0:p1] -= half1[:, p0:p1]
                v1_ao[ja, k, :, :, p0:p1] -= half1[:, p0:p1].transpose(
                    0, 2, 1).conj()
    return v1_projector + v1_ao


def contract_vppnl_deriv(cell, densities, kpts=None, atmlst=None):
    r'''Contract nonlocal GTH derivatives with several densities at once.

    The projector half-integrals and their derivatives are generated once.
    A leading density-set contraction then produces only
    ``(nset,natm,3)`` output, avoiding the full atom-by-AO-square derivative
    retained by :func:`get_vppnl_deriv`.
    '''
    kpts = np.zeros((1, 3)) if kpts is None else np.asarray(
        kpts, dtype=float).reshape(-1, 3)
    nao = cell.nao_nr()
    batch = np.asarray(densities)
    if batch.ndim == 2:
        batch = batch[None, None]
    elif batch.ndim == 3:
        batch = batch[:, None] if len(kpts) == 1 else batch[None]
    if batch.shape[1:] != (len(kpts), nao, nao):
        raise ValueError(
            'nonlocal density batch must have shape '
            f'(nset,{len(kpts)},{nao},{nao}), got {batch.shape}')
    if atmlst is None:
        atmlst = range(cell.natm)
    atmlst = [int(atom_id) for atom_id in atmlst]

    fakecell, hl_blocks = pp_int.fake_cell_vnl(cell)
    values = pp_int._int_vnl(cell, fakecell, hl_blocks, kpts)
    derivative_intors = (
        'int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2')
    derivatives = pp_int._int_vnl(
        cell, fakecell, hl_blocks, kpts, derivative_intors, comp=3)
    if len(derivatives[0]) > 0:
        for k in range(len(kpts)):
            derivatives[0][k] *= -1

    gradient = np.zeros(
        (len(batch), cell.natm, 3), dtype=np.complex128)
    aoslices = cell.aoslice_by_atom()
    for k in range(len(kpts)):
        dm_hermitian = batch[:, k] + batch[:, k].transpose(0, 2, 1).conj()
        derivative_matrix = np.zeros((3, nao, nao), dtype=np.complex128)
        offsets = [0, 0, 0]
        for shell, hl in enumerate(hl_blocks):
            angular = int(fakecell.bas_angular(shell))
            projector_count = 2 * angular + 1
            radial_count = np.asarray(hl).shape[0]
            proj, proj1 = _projector_shell(
                values, derivatives, k, offsets,
                radial_count, projector_count, nao)
            half1 = lib.einsum(
                'ixmp,ij,jmq->xpq', proj1.conj(), hl, proj)
            derivative_matrix += half1
            atom_id = int(fakecell._bas[shell, 0])
            gradient[:, atom_id] += lib.einsum(
                'xpq,sqp->sx', half1, dm_hermitian)
        for atom_id, (_sh0, _sh1, p0, p1) in enumerate(aoslices):
            gradient[:, atom_id] -= lib.einsum(
                'xpq,sqp->sx', derivative_matrix[:, p0:p1],
                dm_hermitian[:, :, p0:p1])
    return gradient[:, atmlst].real


def contract_vppnl_strain(cell, densities, return_metadata=False):
    r'''Contract the exact-integral nonlocal GTH strain analytically.

    For one projector half integral at real Gamma,

    .. math::

        I_{A,p\mu}=\sum_L I_{A,p\mu}(L),

    translational invariance gives the fixed-fraction homogeneous-strain
    response

    .. math::

        I_{A,p\mu}^{(xy)}=
        \sum_L \frac{\partial I_{A,p\mu}(L)}{\partial R_{A,x}}
        (R_A-R_\mu-L)_y.

    PySCF already evaluates both ``I`` and its projector-center derivative
    for the exact Gaussian-provider nonlocal pseudopotential.  This routine
    calls the same periodic two-center driver once more with each lattice
    vector component as the image weight, obtaining the ``L_y`` moment
    analytically.  It then applies the product rule to

    .. math::

        V_{nl}=I^\dagger h I

    and immediately contracts with every density.  No reciprocal mesh,
    displaced Cell, finite difference, or ``(3,3,nao,nao)`` derivative
    matrix is used.

    Args:
        cell : pyscf.pbc.gto.Cell
            Three-dimensional real-Gamma GTH Cell.
        densities : ndarray
            One Hermitian AO density or a leading density batch.

    Kwargs:
        return_metadata : bool
            Return the exact-integral execution ledger.

    Returns:
        tuple
            Per-density central values and raw ``(3,3)`` strain
            contractions.  With ``return_metadata=True`` a third metadata
            dictionary is returned.
    '''
    if int(cell.dimension) != 3:
        raise NotImplementedError(
            'exact nonlocal GTH strain requires a three-dimensional Cell')
    batch = np.asarray(densities)
    if batch.ndim == 2:
        batch = batch[None]
    nao = int(cell.nao_nr())
    if batch.ndim != 3 or batch.shape[1:] != (nao, nao):
        raise ValueError(
            'nonlocal strain densities must have shape '
            f'(nset,{nao},{nao})')
    if np.iscomplexobj(batch):
        imaginary = float(np.max(np.abs(batch.imag)))
        if imaginary > 1.e-10:
            raise NotImplementedError(
                'exact nonlocal GTH strain requires real Gamma densities')
        batch = batch.real
    asymmetry = float(np.max(np.abs(batch - batch.transpose(0, 2, 1))))
    if asymmetry > 1.e-9:
        raise ValueError('nonlocal strain densities must be symmetric')
    batch = np.asarray(batch, dtype=float, order='C')

    kpts = np.zeros((1, 3))
    fakecell, hl_blocks = pp_int.fake_cell_vnl(cell)
    if not hl_blocks:
        value = np.zeros(len(batch))
        raw = np.zeros((len(batch), 3, 3))
        metadata = {
            'name': 'V_PP_nonlocal_exact_contracted',
            'primitive_analytic': True,
            'production_analytic': True,
            'low_level_numerical_components': (),
            'projector_shells': 0,
            'lattice_vectors': 0,
            'storage': 'half_integral_direct_contraction',
        }
        if return_metadata:
            return value, raw, metadata
        return value, raw

    values = pp_int._int_vnl(
        cell, fakecell, hl_blocks, kpts)
    derivative_intors = (
        'int1e_ipovlp', 'int1e_r2_origi_ip2',
        'int1e_r4_origi_ip2')
    derivatives = pp_int._int_vnl(
        cell, fakecell, hl_blocks, kpts,
        derivative_intors, comp=3)
    lattice_vectors, image_moments = _int_vnl_image_moments(
        cell, fakecell, hl_blocks, derivative_intors)

    # The first historical intor differentiates the bra electronic
    # coordinate.  PySCF's nonlocal nuclear gradient uses this same sign
    # conversion to obtain the projector-center derivative.
    if len(derivatives[0]) > 0:
        derivatives[0][0] *= -1
        image_moments[0] *= -1

    ao_centers = _ao_centers(cell)
    value = np.zeros(len(batch), dtype=np.complex128)
    raw = np.zeros((len(batch), 3, 3), dtype=np.complex128)
    offsets = [0, 0, 0]
    for shell, coupling in enumerate(hl_blocks):
        angular = int(fakecell.bas_angular(shell))
        magnetic = 2 * angular + 1
        radial = int(np.asarray(coupling).shape[0])
        projector = np.empty(
            (radial, magnetic, nao), dtype=np.complex128)
        projector_gradient = np.empty(
            (radial, 3, magnetic, nao), dtype=np.complex128)
        projector_image_moment = np.empty(
            (radial, 3, 3, magnetic, nao), dtype=np.complex128)
        for radial_id in range(radial):
            start = offsets[radial_id]
            stop = start + magnetic
            projector[radial_id] = np.asarray(
                values[radial_id][0])[start:stop]
            projector_gradient[radial_id] = np.asarray(
                derivatives[radial_id][0])[:, start:stop]
            projector_image_moment[radial_id] = np.asarray(
                image_moments[radial_id])[:, :, start:stop]
            offsets[radial_id] = stop

        atom_id = int(fakecell._bas[shell, 0])
        separation = cell.atom_coord(atom_id)[None] - ao_centers
        center_response = lib.einsum(
            'ixmp,py->xyimp', projector_gradient, separation)
        # image_moments is ordered as (lattice y, derivative x, ...).
        strain_projector = (
            center_response
            - projector_image_moment.transpose(2, 1, 0, 3, 4))
        value += lib.einsum(
            'imp,ij,jmq,sqp->s', projector.conj(), coupling,
            projector, batch)
        half_response = lib.einsum(
            'xyimp,ij,jmq,sqp->sxy', strain_projector.conj(),
            coupling, projector, batch)
        raw += half_response + half_response.conj()

    value = _real_contraction(value, 'nonlocal GTH value')
    raw = _real_contraction(raw, 'nonlocal GTH strain')
    metadata = {
        'name': 'V_PP_nonlocal_exact_contracted',
        'primitive_analytic': True,
        'production_analytic': True,
        'low_level_numerical_components': (),
        'projector_shells': len(hl_blocks),
        'lattice_vectors': len(lattice_vectors),
        'storage': 'half_integral_direct_contraction',
        'execution_owner': 'pyscf_exact_integral_host_direct',
        'reciprocal_mesh': None,
        'silent_fallback': False,
    }
    if return_metadata:
        return value, raw, metadata
    return value, raw


def _int_vnl_image_moments(cell, fakecell, hl_blocks, intors):
    '''Return ``sum_L L_y dI_x(L)`` from PySCF's periodic driver.'''
    rcut = max(float(cell.rcut), float(fakecell.rcut))
    lattice_vectors = np.asarray(
        cell.get_lattice_Ls(rcut=rcut), dtype=float)
    weights = np.asarray(
        lattice_vectors.T, dtype=np.complex128, order='C')
    nweights = len(weights)

    def int_ket(basis, intor):
        if len(basis) == 0:
            return []
        intor = cell._add_suffix(intor)
        atm, bas, env = gto.conc_env(
            cell._atm, cell._bas, cell._env,
            fakecell._atm, basis, fakecell._env)
        atm = np.asarray(atm, dtype=np.int32)
        bas = np.asarray(bas, dtype=np.int32)
        env = np.asarray(env, dtype=np.double)
        shell_slice = (
            cell.nbas, len(bas), 0, cell.nbas)
        ao_locations = gto.moleintor.make_loc(bas, intor)
        ni = int(
            ao_locations[shell_slice[1]]
            - ao_locations[shell_slice[0]])
        nj = int(
            ao_locations[shell_slice[3]]
            - ao_locations[shell_slice[2]])
        output = np.empty(
            (nweights, 3, ni, nj), dtype=np.complex128)
        pp_int.libpbc.PBCnr2c_drv(
            getattr(gto.moleintor.libcgto, intor),
            pp_int.libpbc.PBCnr2c_fill_ks1,
            output.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nweights), ctypes.c_int(3),
            ctypes.c_int(len(lattice_vectors)),
            lattice_vectors.ctypes.data_as(ctypes.c_void_p),
            weights.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int * 4)(*shell_slice),
            ao_locations.ctypes.data_as(ctypes.c_void_p),
            lib.c_null_ptr(),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))
        return output

    dimensions = np.asarray([len(coupling) for coupling in hl_blocks])
    moments = (
        int_ket(fakecell._bas[dimensions > 0], intors[0]),
        int_ket(fakecell._bas[dimensions > 1], intors[1]),
        int_ket(fakecell._bas[dimensions > 2], intors[2]),
    )
    return lattice_vectors, list(moments)


def _validate_real_gamma_densities(cell, densities, name):
    '''Return a C-contiguous symmetric real-Gamma density batch.'''
    batch = np.asarray(densities)
    if batch.ndim == 2:
        batch = batch[None]
    nao = int(cell.nao_nr())
    if batch.ndim != 3 or batch.shape[1:] != (nao, nao):
        raise ValueError(
            f'{name} densities must have shape (nset,{nao},{nao})')
    if np.iscomplexobj(batch):
        imaginary = float(np.max(np.abs(batch.imag)))
        if imaginary > 1.e-10:
            raise NotImplementedError(
                f'{name} requires real Gamma densities')
        batch = batch.real
    asymmetry = float(np.max(np.abs(batch - batch.transpose(0, 2, 1))))
    if asymmetry > 1.e-9:
        raise ValueError(f'{name} densities must be symmetric')
    return np.asarray(batch, dtype=np.double, order='C')


def _load_vpploc2_strain_extension():
    '''Load the package callback after exporting PySCF C symbols globally.'''
    pyscf_lib = Path(pyscf.__file__).resolve().parent / 'lib'
    libraries = (
        pyscf_lib / 'deps' / 'lib' / 'libcint.so',
        pyscf_lib / 'libcgto.so',
        pyscf_lib / 'libpbc.so',
    )
    for library in libraries:
        if not library.is_file():
            raise ImportError(
                f'required PySCF library was not found: {library}')
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    module = importlib.import_module(
        'dcforge.cpu.pbc.df.gaussian._vpploc2_strain')
    return ctypes.CDLL(
        str(Path(module.__file__).resolve()), mode=ctypes.RTLD_GLOBAL)


def _ao_centers(cell):
    '''Return the Cartesian center of every primary AO.'''
    centers = np.empty((cell.nao_nr(), 3), dtype=float)
    for atom_id, aoslice in enumerate(cell.aoslice_by_atom()):
        start, stop = aoslice[2:]
        centers[start:stop] = cell.atom_coord(atom_id)
    return centers


def _real_contraction(value, name, tolerance=1.e-8):
    '''Return a real-Gamma contraction or reject a complex residual.'''
    value = np.asarray(value)
    if np.iscomplexobj(value):
        imaginary = float(np.max(np.abs(value.imag)))
        if imaginary > tolerance:
            raise RuntimeError(
                f'{name} has imaginary residual {imaginary:.3e}')
        value = value.real
    return np.asarray(value, dtype=float)


def _projector_shell(values, derivatives, k, offsets, nl, nm, nao):
    proj = np.empty((nl, nm, nao), dtype=np.complex128)
    proj1 = np.empty((nl, 3, nm, nao), dtype=np.complex128)
    for i in range(nl):
        p0 = offsets[i]
        p1 = p0 + nm
        proj[i] = np.asarray(values[i][k])[p0:p1]
        proj1[i] = np.asarray(derivatives[i][k])[:, p0:p1]
        offsets[i] = p1
    return proj, proj1
