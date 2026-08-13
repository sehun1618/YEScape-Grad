'''Analytic 3D ion-ion Ewald homogeneous-strain derivative.'''

import numpy as np
from pyscf import lib
from pyscf.pbc.tools import pbc as pbc_tools
from scipy.special import erfc

from dcforge.cpu.pbc.df.gaussian.strain import overlap_strain
from dcforge.cpu.pbc.stress.strain import (
    stress_tensor_to_voigt,
)


def ewald_strain(cell, eta=None, cutoff=None, mesh=None,
                 reciprocal_block_size=262144):
    r'''Return the Ewald energy ledger and raw ``dE/dstrain`` in Hartree.

    Fractional nuclear coordinates are fixed.  The central Ewald splitting
    parameter, real-space cutoff and integer reciprocal mesh are numerical
    partitions and are held fixed while differentiating the physical Cell.

    Kwargs:
        eta, cutoff : float or None
            Central Ewald parameters.  ``cell.get_ewald_params()`` supplies
            omitted values.
        mesh : array_like of length 3 or None
            Fixed reciprocal Ewald mesh derived from the central Cell when
            omitted.
        reciprocal_block_size : int
            Maximum number of reciprocal vectors contracted together.

    Returns:
        dict
            Total/component energies and raw strain derivatives.  All energy
            values and derivatives are in Hartree.
    '''
    if int(cell.dimension) != 3:
        raise NotImplementedError('analytic Ewald strain requires a 3D Cell')
    if bool(getattr(cell, 'use_particle_mesh_ewald', False)):
        raise NotImplementedError('particle-mesh Ewald strain is not implemented')
    if reciprocal_block_size <= 0:
        raise ValueError('reciprocal_block_size must be positive')
    if cell.natm == 0:
        zeros = np.zeros((3, 3))
        return _ewald_result(0., zeros, 0., zeros, 0., zeros, 0., zeros,
                             0., 0., np.zeros(3, dtype=int), 0)
    if eta is None or cutoff is None:
        eta0, cutoff0 = cell.get_ewald_params()
        if eta is None:
            eta = eta0
        if cutoff is None:
            cutoff = cutoff0
    eta = float(eta)
    cutoff = float(cutoff)
    if eta <= 0 or cutoff <= 0:
        raise ValueError('Ewald eta and cutoff must be positive')

    charges = np.asarray(cell.atom_charges(), dtype=float)
    coordinates = np.asarray(cell.atom_coords(), dtype=float)
    if mesh is None:
        log_precision = np.log(
            float(cell.precision) / (float(charges.sum()) * 16 * np.pi**2))
        ke_cutoff = -2 * eta**2 * log_precision
        mesh = np.asarray(cell.cutoff_to_mesh(ke_cutoff), dtype=int)
    else:
        mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError('Ewald mesh must contain three positive integers')

    lattice_images = np.asarray(
        cell.get_lattice_Ls(rcut=cutoff), dtype=float)
    real_value, real_derivative = _real_space_response(
        coordinates, charges, lattice_images, eta)
    self_value, self_derivative = _self_response(
        charges, eta, float(cell.vol))
    reciprocal_value, reciprocal_derivative = _reciprocal_response(
        cell, coordinates, charges, mesh, eta, reciprocal_block_size)
    value = real_value + self_value + reciprocal_value
    derivative = real_derivative + self_derivative + reciprocal_derivative
    return _ewald_result(
        value, derivative,
        real_value, real_derivative,
        self_value, self_derivative,
        reciprocal_value, reciprocal_derivative,
        eta, cutoff, mesh, len(lattice_images))


def gamma_madelung_strain(cell):
    r'''Return PySCF's Gamma Ewald-exchange Madelung strain.

    PySCF defines the full-range Gamma exchange correction through
    ``m = pbc.tools.madelung(cell, Gamma)``.  The same scalar is minus twice
    the Ewald energy of a single unit-charge probe Cell with a neutralizing
    background.  Reusing :func:`ewald_strain` therefore gives an analytic
    derivative without differentiating PySCF's parameter-selection code.

    Returns:
        dict
            Central ``value`` and raw/Voigt dimensionless-strain derivative.
    '''
    if int(cell.dimension) != 3:
        raise NotImplementedError(
            'Gamma Madelung strain currently requires a 3D Cell')
    coordinate_pointer = int(cell._env.size)
    probe = cell.copy(deep=False)
    probe._atm = np.asarray((
        (1, coordinate_pointer, 0, 0, 0, 0),
    ), dtype=np.int32)
    probe._env = np.append(cell._env, (0., 0., 0.))
    probe.unit = 'Bohr'
    probe.a = np.asarray(cell.lattice_vectors(), dtype=float)
    probe_response = ewald_strain(probe)
    value = float(pbc_tools.madelung(cell, np.zeros((1, 3))))
    derivative = -2 * np.asarray(probe_response['raw_derivative'])
    return {
        'name': 'madelung',
        'primitive_analytic': True,
        'production_analytic': True,
        'low_level_numerical_components': [],
        'value': value,
        'raw_derivative': derivative,
        'voigt_derivative': stress_tensor_to_voigt(derivative),
        'probe_ewald': probe_response,
    }


def gamma_ewald_exchange_strain(cell, density):
    r'''Differentiate PySCF ``K_G0 = m S P S`` at real Gamma.

    This matrix is added to the GDF exchange matrix when
    ``exxdiv='ewald'``.  The density is held fixed: the response contains the
    Madelung scalar, bra-overlap and ket-overlap derivatives only.  Orbital
    relaxation belongs to the central CPHF/Z-vector equation and must not be
    included here.

    Args:
        cell : pyscf.pbc.gto.Cell
            Three-dimensional real-Gamma Cell.
        density : ndarray of shape (nao,nao)
            Spin-summed source density.

    Returns:
        dict
            Central exchange-correction matrix, separate Madelung/overlap
            derivative terms and their analytic sum.
    '''
    density = np.asarray(density)
    nao = int(cell.nao_nr())
    if density.shape != (nao, nao):
        raise ValueError(
            'Gamma Ewald exchange density has incompatible AO dimensions')
    if np.iscomplexobj(density):
        maximum = float(np.max(np.abs(density.imag))) if density.size else 0.
        if maximum > 1.e-10:
            raise NotImplementedError(
                'complex Gamma Ewald exchange strain is not supported')
        density = density.real

    overlap = overlap_strain(cell)
    overlap_value = np.asarray(overlap['value'])
    overlap1 = np.asarray(overlap['raw_derivative'])
    madelung = gamma_madelung_strain(cell)
    projected_density = overlap_value @ density @ overlap_value
    value = madelung['value'] * projected_density
    madelung_response = (
        madelung['raw_derivative'][:, :, None, None]
        * projected_density[None, None])
    overlap_response = madelung['value'] * (
        lib.einsum(
            'xyik,kl,lj->xyij', overlap1, density, overlap_value)
        + lib.einsum(
            'ik,kl,xylj->xyij', overlap_value, density, overlap1))
    derivative = madelung_response + overlap_response
    return {
        'name': 'K_G0_ewald',
        'primitive_analytic': True,
        'production_analytic': True,
        'low_level_numerical_components': [],
        'value': np.asarray(value).real,
        'raw_derivative': np.asarray(derivative).real,
        'voigt_derivative': np.asarray((
            derivative[0, 0], derivative[1, 1], derivative[2, 2],
            .5 * (derivative[1, 2] + derivative[2, 1]),
            .5 * (derivative[0, 2] + derivative[2, 0]),
            .5 * (derivative[0, 1] + derivative[1, 0]),
        )).real,
        'madelung_response': np.asarray(madelung_response).real,
        'overlap_response': np.asarray(overlap_response).real,
        'madelung': madelung,
    }


def _real_space_response(coordinates, charges, lattice_images, eta):
    charge_products = charges[:, None] * charges[None, :]
    value = 0.
    derivative = np.zeros((3, 3))
    eta_factor = 2 * eta / np.sqrt(np.pi)
    for image in lattice_images:
        vectors = (
            coordinates[:, None] - coordinates[None, :] + image)
        radii2 = lib.einsum('ijx,ijx->ij', vectors, vectors)
        mask = radii2 >= 1.e-32
        radii = np.ones_like(radii2)
        radii[mask] = np.sqrt(radii2[mask])
        screened = np.zeros_like(radii)
        screened[mask] = erfc(eta * radii[mask]) / radii[mask]
        value += .5 * lib.einsum(
            'ij,ij->', charge_products, screened)

        radial_factor = np.zeros_like(radii)
        radial_factor[mask] = (
            -erfc(eta * radii[mask]) / radii[mask]**3
            - eta_factor * np.exp(-(eta * radii[mask])**2)
            / radii[mask]**2)
        derivative += .5 * lib.einsum(
            'ij,ij,ijx,ijy->xy', charge_products, radial_factor,
            vectors, vectors)
    return float(value), derivative


def _self_response(charges, eta, volume):
    point_self = -float(np.dot(charges, charges)) * eta / np.sqrt(np.pi)
    background = -.5 * float(charges.sum())**2 * np.pi / (eta**2 * volume)
    return point_self + background, -background * np.eye(3)


def _reciprocal_response(cell, coordinates, charges, mesh, eta, block_size):
    reciprocal_vectors, _base, weights = cell.get_Gv_weights(mesh)
    if not np.isscalar(weights):
        raise NotImplementedError(
            'analytic Ewald strain requires uniform 3D reciprocal weights')
    weights = float(weights)
    value = 0.
    derivative = np.zeros((3, 3))
    identity = np.eye(3)
    for start in range(0, len(reciprocal_vectors), block_size):
        vectors = np.asarray(
            reciprocal_vectors[start:start + block_size])
        norms2 = lib.einsum('gx,gx->g', vectors, vectors)
        mask = norms2 > 1.e-28
        if not np.any(mask):
            continue
        vectors = vectors[mask]
        norms2 = norms2[mask]
        phases = np.exp(-1j * coordinates @ vectors.T)
        structure = lib.einsum('a,ag->g', charges, phases)
        amplitudes = (structure.conj() * structure).real
        kernel = (
            4 * np.pi * weights / norms2
            * np.exp(-norms2 / (4 * eta**2)))
        weighted = amplitudes * kernel
        value += .5 * float(np.sum(weighted))
        logarithmic_response = (
            -identity[None]
            + (2 / norms2 + 1 / (2 * eta**2))[:, None, None]
            * lib.einsum('gx,gy->gxy', vectors, vectors))
        derivative += .5 * lib.einsum(
            'g,gxy->xy', weighted, logarithmic_response)
    return value, derivative


def _ewald_result(value, derivative, real_value, real_derivative,
                  self_value, self_derivative,
                  reciprocal_value, reciprocal_derivative,
                  eta, cutoff, mesh, lattice_image_count):
    return {
        'value': float(value),
        'raw_derivative': np.asarray(derivative),
        'voigt_derivative': stress_tensor_to_voigt(derivative),
        'real_value': float(real_value),
        'real_derivative': np.asarray(real_derivative),
        'self_value': float(self_value),
        'self_derivative': np.asarray(self_derivative),
        'reciprocal_value': float(reciprocal_value),
        'reciprocal_derivative': np.asarray(reciprocal_derivative),
        'eta': float(eta),
        'cutoff': float(cutoff),
        'mesh': np.asarray(mesh, dtype=int),
        'lattice_image_count': int(lattice_image_count),
    }
