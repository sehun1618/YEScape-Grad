'''Exact-exchange Ewald response terms shared by Gamma Gaussian DF.'''

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.grad.rhf import contract_h1e_dm
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.tools.pbc import madelung


def cross_derivative(cell, reference_density, response_density, exxdiv):
    r'''Return the exact-exchange Ewald cross derivative.

    Args:
        cell : pyscf.pbc.gto.Cell or gpu4pyscf sorted Cell
            Physical provider Cell or the native GDF sorted-cell wrapper.
        reference_density, response_density : cupy.ndarray
            Physical-AO reference density ``P`` and Hermitian transition
            density ``D_Z``.
        exxdiv : {'ewald', None}
            Source-HF exchange-divergence convention.

    Returns:
        numpy.ndarray
            All-atom cross derivative in Ha/Bohr.
    '''
    if exxdiv not in ('ewald', None):
        raise NotImplementedError(
            "Gamma Gaussian-DF response supports exxdiv='ewald' or None, "
            f'got {exxdiv!r}')
    if exxdiv is None or not (
            cell.dimension == 3 or
            (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum')):
        return np.zeros((cell.natm, 3))
    overlap = int1e.int1e_ovlp(cell)
    overlap_derivative = int1e.int1e_ipovlp(cell)
    cross = -(
        reference_density @ overlap @ response_density
        + response_density @ overlap @ reference_density)
    physical_cell = getattr(cell, 'cell', cell)
    value = contract_h1e_dm(
        physical_cell, overlap_derivative, cross, hermi=1)
    return np.asarray(value * madelung(
        cell, np.zeros((1, 3)), omega=0.), dtype=float)


def signed_derivative(cell, states, exxdiv):
    r'''Return the weighted-state exact-exchange Ewald derivative.

    Args:
        cell : pyscf.pbc.gto.Cell or gpu4pyscf sorted Cell
            Physical provider Cell or the native GDF sorted-cell wrapper.
        states : sequence
            Signed density states exposing ``exchange_weight`` and
            ``density_physical``.
        exxdiv : {'ewald', None}
            Source-HF exchange-divergence convention.

    Returns:
        numpy.ndarray
            All-atom signed Ewald derivative in Ha/Bohr.
    '''
    if exxdiv not in ('ewald', None):
        raise NotImplementedError(
            "Gamma Gaussian-DF response supports exxdiv='ewald' or None, "
            f'got {exxdiv!r}')
    if exxdiv is None or not (
            cell.dimension == 3 or
            (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum')):
        return np.zeros((cell.natm, 3))
    overlap = int1e.int1e_ovlp(cell)
    overlap_derivative = int1e.int1e_ipovlp(cell)
    product = cp.zeros_like(states[0].density_physical)
    for state in states:
        product += state.exchange_weight * (
            state.density_physical @ overlap @ state.density_physical)
    physical_cell = getattr(cell, 'cell', cell)
    value = contract_h1e_dm(
        physical_cell, overlap_derivative, product, hermi=1)
    return np.asarray(value * (.5 * madelung(
        cell, np.zeros((1, 3)), omega=0.)), dtype=float)
