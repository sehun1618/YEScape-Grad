'''Analytic real-Gamma GDF GTH pseudopotential strain primitives.'''

import numpy as np
from pyscf import gto, lib
from pyscf.pbc.df import aft, ft_ao, rsdf_builder
from pyscf.pbc.gto import pseudo

from dcforge.cpu.pbc.df.gaussian.fake_nuclear import (
    make_fake_nuclear_cell,
)
from dcforge.cpu.pbc.df.gaussian.source import three_center_source
from dcforge.cpu.pbc.df.gaussian.strain import (
    ao_pair_fourier_strain,
    auxiliary_fourier_strain,
    matrix_strain_to_voigt,
    overlap_strain,
)
from dcforge.cpu.pbc.df.gdf.stress import three_center_source_strain


def gdf_pp_local_part1_strain(
        cell, omega=None, mesh=None, reciprocal_block_size=None):
    r'''Return GDF GTH local-part1 and its analytic strain response.

    PySCF's :class:`_RSNucBuilder` represents the error-function part of the
    local GTH potential as a range-separated three-center source with a fake
    nuclear Gaussian auxiliary Cell.  In the real-Gamma case,

    .. math::

        V_{loc,1}=\sum_A(-Z_A)B_A+V_{G=0}^{pseudo}.

    The first term reuses the explicit ``B_SR + B_LR + B_vbar`` strain
    ledger.  The finite pseudo ``G=0`` correction contains the AO-overlap and
    inverse-volume responses and is reported separately.

    Args:
        cell : pyscf.pbc.gto.Cell
            Three-dimensional real-Gamma Cell with GTH pseudopotentials.

    Kwargs:
        omega : float or None
            Fixed numerical range-separation parameter.  ``None`` resolves
            the same central setting as PySCF ``GDF.get_pp``.
        mesh : array_like or None
            Fixed three-center source mesh.  It must be supplied together
            with explicit ``omega``.  ``None`` uses the central PySCF
            resolution.
        reciprocal_block_size : int or None
            G-vector block used by the transparent three-center reference.

    Returns:
        dict
            Central matrix, raw/Voigt strain derivative, the complete
            three-center ledger and the separate pseudo-zero-mode response.
    '''
    _check_cell(cell)
    omega, mesh = _nuclear_builder_settings(cell, omega, mesh)
    fake_nuclear = make_fake_nuclear_cell(cell, with_pseudo=True)
    charges = -np.asarray(cell.atom_charges(), dtype=float)
    if fake_nuclear.nao_nr() != len(charges):
        raise NotImplementedError(
            'the first GTH stress release requires one fake nuclear '
            'Gaussian for every pseudized atom')
    source = three_center_source_strain(
        cell, fake_nuclear, omega, mesh,
        reciprocal_block_size=reciprocal_block_size)
    nao = int(cell.nao_nr())
    nfake = int(fake_nuclear.nao_nr())
    source_value = lib.einsum(
        'ijP,P->ij', source['value'].reshape(nao, nao, nfake), charges)
    source_response = lib.einsum(
        'xyijP,P->xyij',
        source['raw_derivative'].reshape(3, 3, nao, nao, nfake),
        charges)

    overlap = overlap_strain(cell)
    exponents = np.hstack(fake_nuclear.bas_exps())
    coefficient = float(
        charges.dot(np.pi / exponents) / float(cell.vol))
    zero_mode_value = -coefficient * np.asarray(overlap['value'])
    overlap_response = (
        -coefficient * np.asarray(overlap['raw_derivative']))
    volume_response = (
        -np.eye(3)[:, :, None, None]
        * zero_mode_value[None, None])
    zero_mode_response = overlap_response + volume_response
    value = _real_if_close(source_value + zero_mode_value)
    derivative = _real_if_close(source_response + zero_mode_response)
    return {
        'name': 'V_PP_local_part1',
        'primitive_analytic': True,
        'production_analytic': True,
        'low_level_numerical_components': [],
        'value': value,
        'raw_derivative': derivative,
        'voigt_derivative': matrix_strain_to_voigt(derivative),
        'three_center_response': _real_if_close(source_response),
        'pseudo_g0_value': _real_if_close(zero_mode_value),
        'pseudo_g0_response': _real_if_close(zero_mode_response),
        'pseudo_g0_overlap_response': _real_if_close(overlap_response),
        'pseudo_g0_volume_response': _real_if_close(volume_response),
        'source': source,
        'omega': float(omega),
        'mesh': np.asarray(mesh, dtype=int),
    }


def gdf_pp_local_part2_strain(cell, mesh=None, block_size=None):
    r'''Return the analytic Gaussian-polynomial local GTH matrix strain.

    For the electron potential :math:`v_2(G)` and the periodic AO-pair
    Fourier factor :math:`F_{\mu\nu}(G)`,

    .. math::

        V_{loc,2;\mu\nu}
          =\Omega^{-1}\sum_G F_{\mu\nu}(G)v_2(G).

    The derivative is kept as AO-pair shape/image response, analytic GTH
    radial-polynomial response, and inverse-volume response.  Reciprocal
    vectors are processed in bounded blocks and no displaced Cell is used.

    Kwargs:
        mesh : array_like or None
            Fixed reciprocal integer mesh.  ``None`` uses PySCF's converged
            GTH/AO cutoff estimate for the central Cell.
        block_size : int or None
            G-vectors per block.  ``None`` uses a PySCF-memory-aware planner.
    '''
    _check_cell(cell)
    mesh = _reciprocal_mesh(cell, mesh)
    vectors = np.asarray(cell.get_Gv(mesh), dtype=float)
    block_size = _pp_block_size(
        cell, len(vectors), block_size, pair_response=True)
    nao = int(cell.nao_nr())
    value = np.zeros((nao, nao), dtype=np.complex128)
    pair_response = np.zeros((3, 3, nao, nao), dtype=np.complex128)
    potential_response = np.zeros_like(pair_response)
    peak_block_bytes = 0
    inverse_volume = 1 / float(cell.vol)

    for start in range(0, len(vectors), block_size):
        block = vectors[start:start + block_size]
        pair = ao_pair_fourier_strain(cell, block)
        potential, potential1 = _local_part2_potential_fourier_strain(
            cell, block)
        pair_value = np.asarray(pair['value'])
        pair1 = np.asarray(pair['raw_derivative'])
        value += inverse_volume * lib.einsum(
            'Gij,G->ij', pair_value, potential.conj())
        pair_response += inverse_volume * lib.einsum(
            'xyGij,G->xyij', pair1, potential.conj())
        potential_response += inverse_volume * lib.einsum(
            'Gij,xyG->xyij', pair_value, potential1.conj())
        peak_block_bytes = max(peak_block_bytes, int(
            pair_value.nbytes + pair1.nbytes
            + potential.nbytes + potential1.nbytes))

    volume_response = -np.eye(3)[:, :, None, None] * value[None, None]
    derivative = pair_response + potential_response + volume_response
    value = _real_if_close(value)
    derivative = _real_if_close(derivative)
    return {
        'name': 'V_PP_local_part2',
        'primitive_analytic': True,
        'production_analytic': True,
        'low_level_numerical_components': [],
        'value': value,
        'raw_derivative': derivative,
        'voigt_derivative': matrix_strain_to_voigt(derivative),
        'ao_pair_response': _real_if_close(pair_response),
        'potential_response': _real_if_close(potential_response),
        'volume_response': _real_if_close(volume_response),
        'mesh': mesh,
        'grid_points': len(vectors),
        'block_size': int(block_size),
        'peak_block_bytes': int(peak_block_bytes),
    }


def gdf_pp_nonlocal_strain(cell, mesh=None, block_size=None):
    r'''Return the analytic reciprocal GTH nonlocal-projector strain.

    Each projector overlap is accumulated as

    .. math::

        Q_{A,lmi,\mu}
          =\sum_G p_{A,lmi}(G)^*F_\mu(G)/\sqrt{\Omega}.

    Fixed fractional atomic coordinates make the structure-factor phase
    invariant.  The response therefore consists of the projector radial and
    angular shape response, AO Fourier shape response, the
    ``1/sqrt(Omega)`` normalization, and the final ``1/Omega`` contraction.

    The channel ledger is retained, whereas reciprocal-grid intermediates
    are discarded block by block.
    '''
    _check_cell(cell)
    mesh = _reciprocal_mesh(cell, mesh)
    vectors = np.asarray(cell.get_Gv(mesh), dtype=float)
    block_size = _pp_block_size(
        cell, len(vectors), block_size, pair_response=False)
    nao = int(cell.nao_nr())
    channels = _nonlocal_channels(cell, nao)
    if not channels:
        zeros = np.zeros((nao, nao))
        return {
            'name': 'V_PP_nonlocal',
            'primitive_analytic': True,
            'production_analytic': True,
            'low_level_numerical_components': [],
            'value': zeros,
            'raw_derivative': np.zeros((3, 3, nao, nao)),
            'voigt_derivative': np.zeros((6, nao, nao)),
            'channels': [],
            'mesh': mesh,
            'grid_points': 0,
            'block_size': int(block_size),
            'peak_block_bytes': 0,
        }

    fake = _projector_angular_molecule()
    inverse_sqrt_volume = float(cell.vol) ** -.5
    peak_block_bytes = 0
    for start in range(0, len(vectors), block_size):
        block = vectors[start:start + block_size]
        structure_factors = cell.get_SI(Gv=block)
        ao_fourier = auxiliary_fourier_strain(cell, block)
        ao_value = np.asarray(ao_fourier['value']).T * inverse_sqrt_volume
        ao_response = (
            np.asarray(ao_fourier['raw_derivative']).transpose(0, 1, 3, 2)
            * inverse_sqrt_volume)
        ao_response -= (
            .5 * np.eye(3)[:, :, None, None] * ao_value[None, None])

        for channel in channels:
            phase = structure_factors[channel['atom_id']].conj()
            projector, projector_gradient = _projector_value_gradient(
                fake, block, channel['angular_momentum'],
                channel['count'], channel['radius'])
            projector_response = -lib.einsum(
                'Gx,ynmG->xynmG', block, projector_gradient)
            phased_projector = projector * phase[None, None]
            phased_response = (
                projector_response * phase[None, None, None, None])
            channel['overlap'] += lib.einsum(
                'nmG,Gp->nmp', phased_projector, ao_value)
            channel['response'] += lib.einsum(
                'xynmG,Gp->xynmp', phased_response, ao_value)
            channel['response'] += lib.einsum(
                'nmG,xyGp->xynmp', phased_projector, ao_response)
            peak_block_bytes = max(peak_block_bytes, int(
                ao_value.nbytes + ao_response.nbytes
                + projector.nbytes + projector_gradient.nbytes
                + projector_response.nbytes))

    value_accumulator = np.zeros((nao, nao), dtype=np.complex128)
    derivative_accumulator = np.zeros(
        (3, 3, nao, nao), dtype=np.complex128)
    channel_ledger = []
    for channel in channels:
        overlap = channel['overlap']
        response = channel['response']
        coupling = channel['coupling']
        channel_value = lib.einsum(
            'amp,ab,bmq->pq', overlap.conj(), coupling, overlap)
        channel_response = lib.einsum(
            'xyamp,ab,bmq->xypq',
            response.conj(), coupling, overlap)
        channel_response += lib.einsum(
            'amp,ab,xybmq->xypq',
            overlap.conj(), coupling, response)
        value_accumulator += channel_value
        derivative_accumulator += channel_response
        channel_ledger.append({
            'atom_id': int(channel['atom_id']),
            'angular_momentum': int(channel['angular_momentum']),
            'radial_projectors': int(channel['count']),
            'value_before_volume': _real_if_close(channel_value),
            'response_before_volume': _real_if_close(channel_response),
        })

    value = value_accumulator / float(cell.vol)
    projector_response = derivative_accumulator / float(cell.vol)
    volume_response = -np.eye(3)[:, :, None, None] * value[None, None]
    derivative = projector_response + volume_response
    value = _real_if_close(value)
    derivative = _real_if_close(derivative)
    return {
        'name': 'V_PP_nonlocal',
        'primitive_analytic': True,
        'production_analytic': True,
        'low_level_numerical_components': [],
        'value': value,
        'raw_derivative': derivative,
        'voigt_derivative': matrix_strain_to_voigt(derivative),
        'projector_response': _real_if_close(projector_response),
        'volume_response': _real_if_close(volume_response),
        'channels': channel_ledger,
        'mesh': mesh,
        'grid_points': len(vectors),
        'block_size': int(block_size),
        'peak_block_bytes': int(peak_block_bytes),
    }


def gdf_pseudopotential_strain(
        cell, local_part1_omega=None, local_part1_mesh=None,
        reciprocal_mesh=None, block_size=None,
        source_block_size=None):
    r'''Assemble the complete analytic real-Gamma GDF GTH PP strain.

    The returned matrix follows PySCF ``GDF.get_pp`` exactly at the central
    Cell when the converged reciprocal mesh is used:

    ``V_PP = V_local_part1 + V_local_part2 + V_nonlocal``.
    '''
    local_part1 = gdf_pp_local_part1_strain(
        cell, omega=local_part1_omega, mesh=local_part1_mesh,
        reciprocal_block_size=source_block_size)
    local_part2 = gdf_pp_local_part2_strain(
        cell, mesh=reciprocal_mesh, block_size=block_size)
    nonlocal_part = gdf_pp_nonlocal_strain(
        cell, mesh=reciprocal_mesh, block_size=block_size)
    value = _real_if_close(
        local_part1['value'] + local_part2['value']
        + nonlocal_part['value'])
    derivative = _real_if_close(
        local_part1['raw_derivative']
        + local_part2['raw_derivative']
        + nonlocal_part['raw_derivative'])
    return {
        'name': 'V_PP',
        'primitive_analytic': True,
        'production_analytic': True,
        'low_level_numerical_components': [],
        'value': value,
        'raw_derivative': derivative,
        'voigt_derivative': matrix_strain_to_voigt(derivative),
        'local_part1': local_part1,
        'local_part2': local_part2,
        'nonlocal': nonlocal_part,
    }


def _local_part2_potential_fourier_strain(cell, grid_vectors):
    '''Return electron local-part2 ``v(G)`` and reciprocal-shape response.'''
    structure_factors = cell.get_SI(Gv=grid_vectors)
    atomic_value, atomic_response = _local_part2_atomic_fourier_strain(
        cell, grid_vectors)
    value = -lib.einsum('AG,AG->G', structure_factors, atomic_value)
    response = -lib.einsum(
        'AG,AxyG->xyG', structure_factors, atomic_response)
    return value, response


def _local_part2_atomic_fourier_strain(cell, grid_vectors):
    r'''Return atom-resolved Gaussian-polynomial factors and ``G`` response.'''
    vectors = np.asarray(grid_vectors, dtype=float)
    norm2 = lib.einsum('Gx,Gx->G', vectors, vectors)
    dyadic = lib.einsum('Gx,Gy->xyG', vectors, vectors)
    values = np.zeros((cell.natm, len(vectors)))
    derivatives = np.zeros((cell.natm, 3, 3, len(vectors)))
    for atom_id in range(cell.natm):
        symbol = cell.atom_symbol(atom_id)
        if symbol not in cell._pseudo:
            raise NotImplementedError(
                'mixed all-electron/GTH PBC stress is not supported')
        radius, count, coefficients = cell._pseudo[symbol][1:4]
        reduced = norm2 * radius ** 2
        polynomial = np.zeros_like(reduced)
        polynomial_derivative = np.zeros_like(reduced)
        if count >= 1:
            polynomial += coefficients[0]
        if count >= 2:
            polynomial += coefficients[1] * (3 - reduced)
            polynomial_derivative -= coefficients[1]
        if count >= 3:
            polynomial += coefficients[2] * (
                15 - 10 * reduced + reduced ** 2)
            polynomial_derivative += coefficients[2] * (
                -10 + 2 * reduced)
        if count >= 4:
            polynomial += coefficients[3] * (
                105 - 105 * reduced + 21 * reduced ** 2 - reduced ** 3)
            polynomial_derivative += coefficients[3] * (
                -105 + 42 * reduced - 3 * reduced ** 2)
        amplitude = (2 * np.pi) ** 1.5 * radius ** 3
        damping = np.exp(-.5 * reduced)
        value = -amplitude * damping * polynomial
        derivative_norm2 = (
            -amplitude * radius ** 2 * damping
            * (polynomial_derivative - .5 * polynomial))
        values[atom_id] = value
        derivatives[atom_id] = (
            -2 * dyadic * derivative_norm2[None, None])
    return values, derivatives


def _nonlocal_channels(cell, nao):
    '''Allocate GTH projector channel accumulators.'''
    channels = []
    for atom_id in range(cell.natm):
        symbol = cell.atom_symbol(atom_id)
        if symbol not in cell._pseudo:
            raise NotImplementedError(
                'mixed all-electron/GTH PBC stress is not supported')
        for angular_momentum, projector_data in enumerate(
                cell._pseudo[symbol][5:]):
            radius, count, coupling = projector_data
            count = int(count)
            if count <= 0:
                continue
            magnetic = 2 * angular_momentum + 1
            channels.append({
                'atom_id': atom_id,
                'angular_momentum': angular_momentum,
                'radius': float(radius),
                'count': count,
                'coupling': np.asarray(coupling, dtype=float),
                'overlap': np.zeros(
                    (count, magnetic, nao), dtype=np.complex128),
                'response': np.zeros(
                    (3, 3, count, magnetic, nao), dtype=np.complex128),
            })
    return channels


def _projector_angular_molecule():
    '''Return the one-shell molecule used for reciprocal solid harmonics.'''
    fake = gto.Mole()
    fake._atm = np.zeros((1, gto.ATM_SLOTS), dtype=np.int32)
    fake._bas = np.zeros((1, gto.BAS_SLOTS), dtype=np.int32)
    pointer = gto.PTR_ENV_START
    fake._env = np.zeros(pointer + 10)
    fake._bas[0, gto.NPRIM_OF] = 1
    fake._bas[0, gto.NCTR_OF] = 1
    fake._bas[0, gto.PTR_EXP] = pointer + 3
    fake._bas[0, gto.PTR_COEFF] = pointer + 4
    return fake


def _projector_value_gradient(
        fake, vectors, angular_momentum, count, radius):
    '''Return GTH reciprocal projectors and Cartesian ``G`` gradients.'''
    pointer = gto.PTR_ENV_START
    fake._bas[0, gto.ANG_OF] = int(angular_momentum)
    fake._env[pointer + 3] = .5 * radius ** 2
    fake._env[pointer + 4] = (
        radius ** (angular_momentum + 1.5) * np.pi ** 1.25)
    base = np.asarray(fake.eval_gto('GTOval', vectors)).T
    base_gradient = np.asarray(
        fake.eval_gto('GTOval_ip', vectors)).transpose(0, 2, 1)
    reduced_norm2 = lib.einsum(
        'Gx,Gx->G', vectors, vectors) * radius ** 2
    projectors = []
    gradients = []
    for radial_index in range(int(count)):
        polynomial, derivative = _qli_polynomial_and_derivative(
            reduced_norm2, angular_momentum, radial_index)
        polynomial_gradient = (
            2 * radius ** 2 * vectors.T[:, None]
            * derivative[None, None])
        projectors.append(base * polynomial[None])
        gradients.append(
            base_gradient * polynomial[None, None]
            + base[None] * polynomial_gradient)
    return (
        np.asarray(projectors),
        np.asarray(gradients).transpose(1, 0, 2, 3),
    )


def _qli_polynomial_and_derivative(reduced_norm2, angular_momentum, index):
    '''Return PySCF GTH ``q_li`` polynomial and derivative in ``(G r_l)^2``.'''
    sqrt = np.sqrt
    constants = {
        (0, 0): (4 * sqrt(2.), 1., 0., 0.),
        (0, 1): (8 * sqrt(2 / 15), 3., 1., 0.),
        (0, 2): (16 / 3 * sqrt(2 / 105), 15., 10., 1.),
        (1, 0): (8 * sqrt(1 / 3), 1., 0., 0.),
        (1, 1): (16 * sqrt(1 / 105), 5., 1., 0.),
        (1, 2): (32 / 3 * sqrt(1 / 1155), 35., 14., 1.),
        (2, 0): (8 * sqrt(2 / 15), 1., 0., 0.),
        (2, 1): (16 / 3 * sqrt(2 / 105), 7., 1., 0.),
        (2, 2): (32 / 3 * sqrt(2 / 15015), 63., 18., 1.),
        (3, 0): (16 * sqrt(1 / 105), 1., 0., 0.),
        (3, 1): (32 / 3 * sqrt(1 / 1155), 9., 1., 0.),
        (3, 2): (64 / 45 * sqrt(1 / 1001), 99., 22., 1.),
        (4, 0): (16 / 3 * sqrt(2 / 105), 1., 0., 0.),
        (4, 1): (32 / 3 * sqrt(2 / 15015), 11., 1., 0.),
        (4, 2): (64 / 45 * sqrt(2 / 17017), 143., 26., 1.),
    }
    try:
        scale, constant, linear, quadratic = constants[
            (int(angular_momentum), int(index))]
    except KeyError as error:
        raise NotImplementedError(
            'unsupported GTH projector '
            f'l={angular_momentum}, i={index}') from error
    value = scale * (
        constant - linear * reduced_norm2
        + quadratic * reduced_norm2 ** 2)
    derivative = scale * (
        -linear + 2 * quadratic * reduced_norm2)
    return value, derivative


def _nuclear_builder_settings(cell, omega, mesh):
    '''Resolve or validate the local-part1 PySCF numerical split.'''
    if omega is None and mesh is None:
        builder = rsdf_builder._RSNucBuilder(
            cell, np.zeros((1, 3))).build()
        return float(builder.omega), np.asarray(builder.mesh, dtype=int)
    if omega is None or mesh is None:
        raise ValueError(
            'fixed local-part1 strain requires both omega and mesh')
    omega = abs(float(omega))
    mesh = _mesh(mesh)
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('local-part1 omega must be positive')
    return omega, mesh


def _reciprocal_mesh(cell, mesh):
    '''Return the fixed converged local-part2/nonlocal reciprocal mesh.'''
    if mesh is None:
        cutoff = aft.estimate_ke_cutoff(cell, cell.precision)
        mesh = cell.symmetrize_mesh(cell.cutoff_to_mesh(cutoff))
    return _mesh(mesh)


def _mesh(mesh):
    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError('PP mesh must contain three positive integers')
    return mesh


def _pp_block_size(cell, ngrid, requested, pair_response):
    '''Resolve a bounded reciprocal PP block from PySCF ``max_memory``.'''
    ngrid = int(ngrid)
    if requested is not None:
        requested = int(requested)
        if requested <= 0:
            raise ValueError('PP reciprocal block_size must be positive')
        return min(ngrid, requested)
    nao = int(cell.nao_nr())
    if pair_response:
        bytes_per_grid = 16 * 10 * nao ** 2
    else:
        bytes_per_grid = 16 * 24 * nao
    remaining = max(
        1., float(cell.max_memory) - float(lib.current_memory()[0]))
    budget = remaining * 1.e6 / 4
    return max(1, min(8192, ngrid, int(budget // bytes_per_grid)))


def _check_cell(cell):
    '''Enforce the first GDF stress release boundary.'''
    if int(cell.dimension) != 3:
        raise NotImplementedError('GTH PP strain requires a 3D Cell')
    if not cell._pseudo:
        raise NotImplementedError('GTH PP strain requires pseudopotentials')


def _real_if_close(value, tolerance=1.e-8):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tolerance:
        return value.real
    return value


def provider_pseudopotential_matrix(cell):
    '''Return PySCF's integral GTH PP matrix for central-value validation.'''
    builder = rsdf_builder._RSNucBuilder(
        cell, np.zeros((1, 3))).build()
    local_part1 = np.asarray(builder.get_pp_loc_part1()[0])
    local_part2 = np.asarray(pseudo.pp_int.get_pp_loc_part2(cell))
    nonlocal_part = np.asarray(pseudo.pp_int.get_pp_nl(cell))
    return {
        'value': _real_if_close(
            local_part1 + local_part2 + nonlocal_part),
        'local_part1': _real_if_close(local_part1),
        'local_part2': _real_if_close(local_part2),
        'nonlocal': _real_if_close(nonlocal_part),
        'omega': float(builder.omega),
        'mesh': np.asarray(builder.mesh, dtype=int),
    }


def reciprocal_local_part2_matrix(cell, mesh, block_size=None):
    '''Return the fixed-mesh local-part2 matrix without strain intermediates.'''
    mesh = _mesh(mesh)
    vectors = np.asarray(cell.get_Gv(mesh), dtype=float)
    block_size = _pp_block_size(
        cell, len(vectors), block_size, pair_response=True)
    value = np.zeros(
        (cell.nao_nr(), cell.nao_nr()), dtype=np.complex128)
    inverse_volume = 1 / float(cell.vol)
    for start in range(0, len(vectors), block_size):
        block = vectors[start:start + block_size]
        pair = ft_ao.ft_aopair(
            cell, block, aosym='s1',
            kpti_kptj=np.zeros((2, 3)), intor='GTO_ft_ovlp')
        potential, _potential1 = _local_part2_potential_fourier_strain(
            cell, block)
        value += inverse_volume * lib.einsum(
            'Gij,G->ij', pair, potential.conj())
    return _real_if_close(value)


def gdf_pp_local_part1_matrix(cell, omega, mesh):
    '''Return fixed-split GDF local-part1 without strain intermediates.'''
    _check_cell(cell)
    omega, mesh = _nuclear_builder_settings(cell, omega, mesh)
    fake_nuclear = make_fake_nuclear_cell(cell, with_pseudo=True)
    charges = -np.asarray(cell.atom_charges(), dtype=float)
    source = three_center_source(
        cell, fake_nuclear, omega, mesh,
        max_memory=cell.max_memory)
    nao = int(cell.nao_nr())
    value = lib.einsum(
        'ijP,P->ij', source.reshape(nao, nao, -1), charges)
    exponents = np.hstack(fake_nuclear.bas_exps())
    coefficient = float(
        charges.dot(np.pi / exponents) / float(cell.vol))
    overlap = np.asarray(cell.pbc_intor('int1e_ovlp', hermi=0))
    return _real_if_close(value - coefficient * overlap)
