'''Small-system forward GDF strain reference for stress development.'''

import ctypes
import importlib
import time
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib
from pyscf.gto import moleintor
from pyscf.pbc.df import rsdf, rsdf_helper
from pyscf.pbc.df.rsdf_builder import (
    estimate_rs_2c2e_rcut,
)
from pyscf.scf import _vhf

from dcforge.cpu.pbc.df.gaussian.adjoint import (
    make_cderi_observable_adjoints,
    push_cderi_adjoints,
)
from dcforge.cpu.pbc.df.gaussian.cderi import (
    cholesky_cderi,
    factor_metric,
    grad_cderi_with_factor,
    solve_cderi,
    solve_cderi_with_factor,
)
from dcforge.cpu.pbc.df.gaussian.fake_nuclear import (
    fake_nuclear_refuniq,
)
from dcforge.cpu.pbc.df.gaussian.geometry import copy_with_coordinates
from dcforge.cpu.pbc.df.gaussian.source import three_center_source
from dcforge.cpu.pbc.df.gaussian.source_short_range import short_range_source
from dcforge.cpu.pbc.df.gaussian.storage import load_cderi
from dcforge.cpu.pbc.df.gaussian.strain import (
    ao_pair_fourier_strain,
    auxiliary_fourier_strain,
    matrix_strain_to_voigt,
    overlap_strain,
    weighted_coulomb_strain,
)
from dcforge.cpu.pbc.df.gdf.metric import auxiliary_metric
from dcforge.cpu.pbc.df.gdf.response import response_settings
from dcforge.cpu.pbc.stress.strain import (
    deform_cell,
    fractional_coordinates,
    raw_strain_basis,
    voigt_strain_basis,
)


def g0_metric_strain(cell, auxcell, omega):
    r'''Return the analytic Gamma zero-mode metric and volume response.

    The GDF reciprocal zero mode is

    .. math::

        M^{G=0}_{PQ}
        =-\frac{\pi}{\omega^2\Omega}q_Pq_Q.

    The charge-normalized auxiliary coefficients and charges ``q_P`` are
    fixed under Cell strain.  Consequently only the inverse-volume factor
    responds, giving ``dM_G0/d epsilon_xy = -delta_xy M_G0``.

    Returns:
        dict
            ``value`` and raw ``(3,3,naux,naux)`` derivative in the exact
            auxiliary gauge supplied by ``auxcell``.
    '''
    omega = abs(float(omega))
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('GDF G=0 metric strain requires positive omega')
    if int(cell.dimension) != 3:
        raise NotImplementedError(
            'GDF metric strain currently requires a 3D Cell')
    charges = np.asarray(rsdf.get_aux_chg(auxcell), dtype=float)
    value = (
        -np.pi / (omega ** 2 * float(cell.vol))
        * np.outer(charges, charges))
    derivative = -np.eye(3)[:, :, None, None] * value[None, None]
    return {
        'name': 'M_G0',
        'primitive_analytic': True,
        'value': value,
        'raw_derivative': derivative,
        'voigt_derivative': matrix_strain_to_voigt(derivative),
    }


def long_range_metric_strain(cell, auxcell, omega, mesh,
                             reciprocal_block_size=None):
    r'''Return analytic :math:`M^{LR}` and its homogeneous-strain response.

    For

    .. math::

        M^{LR}_{PQ}=\sum_G F_P(G)^*w_G F_Q(G),

    the derivative is separated into the two auxiliary Fourier-factor
    responses and the weighted Coulomb-kernel response.  Reciprocal vectors,
    Cell volume and Gaussian Fourier shapes all move analytically.  The
    integer mesh and ``omega`` remain fixed at their central resolved values.

    Kwargs:
        reciprocal_block_size : int or None
            Number of reciprocal vectors contracted at once.  ``None`` uses
            the complete small-system reference grid.  Changing the block
            size changes storage only, not the mathematical sum.

    Returns:
        dict
            Total value/derivative and separately accumulated auxiliary and
            kernel derivative terms.
    '''
    mesh = _mesh(mesh)
    grid_vectors = np.asarray(cell.get_Gv(mesh), dtype=float)
    ngrid = len(grid_vectors)
    if reciprocal_block_size is None:
        block_size = ngrid
    else:
        block_size = int(reciprocal_block_size)
        if block_size <= 0:
            raise ValueError('reciprocal_block_size must be positive')
        block_size = min(block_size, ngrid)
    naux = int(auxcell.nao_nr())
    value = np.zeros((naux, naux), dtype=np.complex128)
    auxiliary_response = np.zeros(
        (3, 3, naux, naux), dtype=np.complex128)
    kernel_response = np.zeros_like(auxiliary_response)

    for start in range(0, ngrid, block_size):
        vectors = grid_vectors[start:start + block_size]
        auxiliary = auxiliary_fourier_strain(auxcell, vectors)
        kernel = weighted_coulomb_strain(cell, vectors, omega=omega)
        factors = np.asarray(auxiliary['value'])
        factors1 = np.asarray(auxiliary['raw_derivative'])
        weights = np.asarray(kernel['value'])
        weights1 = np.asarray(kernel['raw_derivative'])
        value += lib.einsum(
            'PG,G,QG->PQ', factors.conj(), weights, factors)
        auxiliary_response += lib.einsum(
            'xyPG,G,QG->xyPQ', factors1.conj(), weights, factors)
        auxiliary_response += lib.einsum(
            'PG,G,xyQG->xyPQ', factors.conj(), weights, factors1)
        kernel_response += lib.einsum(
            'PG,xyG,QG->xyPQ', factors.conj(), weights1, factors)

    derivative = auxiliary_response + kernel_response
    return {
        'name': 'M_LR',
        'primitive_analytic': True,
        'value': _real_if_close(value),
        'raw_derivative': _real_if_close(derivative),
        'voigt_derivative': matrix_strain_to_voigt(
            _real_if_close(derivative)),
        'auxiliary_response': _real_if_close(auxiliary_response),
        'kernel_response': _real_if_close(kernel_response),
        'reciprocal_block_size': int(block_size),
        'grid_points': int(ngrid),
    }


def short_range_metric_strain(auxcell, omega, precision=None):
    r'''Return analytic image-resolved :math:`M^{SR}` strain response.

    The transparent CPU reference retains every lattice image ``L`` and uses
    PySCF ``int2c2e_ip1/ip2`` derivatives.  Under affine strain, the two
    auxiliary centers move with velocities ``R_P`` and ``R_Q+L``.  Nuclear-
    center derivatives are the negative electron-coordinate derivatives, so

    .. math::

        (M^{SR}_{PQ,L})^{(xy)}
        =-(\partial_x^1 M_{PQ,L})R_{P,y}
         -(\partial_x^2 M_{PQ,L})(R_{Q,y}+L_y).

    This routine is analytic but intentionally materializes the full metric
    tensor; production stress will contract the same image terms directly
    with metric adjoints.
    '''
    omega = abs(float(omega))
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('short-range metric strain requires positive omega')
    if precision is None:
        precision = float(auxcell.precision) ** 1.5
    precision = float(precision)
    rcut = float(estimate_rs_2c2e_rcut(auxcell, omega, precision))
    lattice_vectors = np.asarray(
        auxcell.get_lattice_Ls(rcut=rcut), dtype=float)
    centers = _ao_centers(auxcell)
    naux = int(auxcell.nao_nr())
    value = np.zeros((naux, naux), dtype=np.complex128)
    derivative = np.zeros(
        (3, 3, naux, naux), dtype=np.complex128)

    with auxcell.with_short_range_coulomb(omega):
        for lattice in lattice_vectors:
            shifted = copy_with_coordinates(
                auxcell, auxcell.atom_coords() + lattice)
            with shifted.with_short_range_coulomb(omega):
                block = np.asarray(gto.intor_cross(
                    'int2c2e', auxcell, shifted))
                ip1 = np.asarray(gto.intor_cross(
                    'int2c2e_ip1', auxcell, shifted, comp=3))
                ip2 = np.asarray(gto.intor_cross(
                    'int2c2e_ip2', auxcell, shifted, comp=3))
            value += block
            derivative -= lib.einsum(
                'xPQ,Py->xyPQ', ip1, centers)
            derivative -= lib.einsum(
                'xPQ,Qy->xyPQ', ip2, centers + lattice)

    return {
        'name': 'M_SR',
        'primitive_analytic': True,
        'value': _real_if_close(value),
        'raw_derivative': _real_if_close(derivative),
        'voigt_derivative': matrix_strain_to_voigt(
            _real_if_close(derivative)),
        'lattice_vectors': len(lattice_vectors),
        'rcut_bohr': rcut,
    }


def auxiliary_metric_strain(cell, auxcell, omega, mesh,
                            precision=None, reciprocal_block_size=None):
    r'''Return the analytic Gamma GDF metric strain term ledger.

    The metric is assembled explicitly as

    .. math::

        M=M_{SR}+M_{LR}+M_{G=0},\qquad
        M^{(xy)}=M_{SR}^{(xy)}+M_{LR}^{(xy)}+M_{G=0}^{(xy)}.

    This is the P4 small-system forward reference.  It is analytic at the
    primitive level but not the production memory path because it retains
    ``(3,3,naux,naux)`` tensors.
    '''
    short_range = short_range_metric_strain(
        auxcell, omega, precision=precision)
    long_range = long_range_metric_strain(
        cell, auxcell, omega, mesh,
        reciprocal_block_size=reciprocal_block_size)
    zero_mode = g0_metric_strain(cell, auxcell, omega)
    value = (
        np.asarray(short_range['value'])
        + np.asarray(long_range['value'])
        + np.asarray(zero_mode['value']))
    derivative = (
        np.asarray(short_range['raw_derivative'])
        + np.asarray(long_range['raw_derivative'])
        + np.asarray(zero_mode['raw_derivative']))
    return {
        'name': 'M',
        'primitive_analytic': True,
        'production_analytic': False,
        'low_level_numerical_components': [],
        'storage': 'full_reference',
        'value': _real_if_close(value),
        'raw_derivative': _real_if_close(derivative),
        'voigt_derivative': matrix_strain_to_voigt(
            _real_if_close(derivative)),
        'short_range': short_range,
        'long_range': long_range,
        'g0': zero_mode,
    }


def vbar_source_strain(cell, auxcell, omega):
    r'''Return analytic :math:`B^{vbar}` overlap and volume responses.

    At Gamma,

    .. math::

        B^{vbar}_{\mu\nu,P}
        =-S_{\mu\nu}\frac{\pi q_P}{\omega^2\Omega}.

    The auxiliary charge is fixed in the central charge-normalized gauge.
    Homogeneous strain therefore acts on the AO overlap and on the inverse
    Cell volume.  These terms are returned separately so that neither the
    overlap Pulay response nor the zero-mode volume response is hidden in a
    reciprocal-source total.

    Returns:
        dict
            Source in ``(nao*nao,naux)`` ordering and raw
            ``(3,3,nao*nao,naux)`` derivative.
    '''
    omega = abs(float(omega))
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('GDF vbar source strain requires positive omega')
    overlap = overlap_strain(cell)
    overlap_value = np.asarray(overlap['value'])
    overlap1 = np.asarray(overlap['raw_derivative'])
    charges = np.asarray(rsdf.get_aux_chg(auxcell), dtype=float)
    coefficient = np.pi * charges / (omega ** 2 * float(cell.vol))
    value = -overlap_value.reshape(-1, 1) * coefficient[None]
    overlap_response = (
        -overlap1.reshape(3, 3, -1, 1)
        * coefficient[None, None, None])
    volume_response = (
        -np.eye(3)[:, :, None, None] * value[None, None])
    derivative = overlap_response + volume_response
    return {
        'name': 'B_vbar',
        'primitive_analytic': True,
        'value': _real_if_close(value),
        'raw_derivative': _real_if_close(derivative),
        'voigt_derivative': matrix_strain_to_voigt(
            _real_if_close(derivative)),
        'overlap_response': _real_if_close(overlap_response),
        'volume_response': _real_if_close(volume_response),
    }


def long_range_source_strain(cell, auxcell, omega, mesh,
                             rcut=None, reciprocal_block_size=None):
    r'''Return analytic :math:`B^{LR}` and its three response owners.

    The Gamma long-range source is

    .. math::

        B^{LR}_{\mu\nu,P}
        =\sum_G F_{\mu\nu}(G)w_G F_P(G)^*.

    Its strain derivative is retained as the explicit sum of AO-pair,
    auxiliary-Fourier, and weighted-kernel responses.  Every primitive is
    analytic and the central integer reciprocal mesh is fixed.  This forward
    routine retains the complete ``(3,3,nao**2,naux)`` tensor and is therefore
    a small-system reference, not the final reverse-adjoint path.

    Kwargs:
        rcut : float or None
            Primary AO-pair image cutoff in Bohr.
        reciprocal_block_size : int or None
            Grid points per block.  ``None`` uses the complete grid and
            traverses the AO-pair lattice images only once.
    '''
    mesh = _mesh(mesh)
    grid_vectors = np.asarray(cell.get_Gv(mesh), dtype=float)
    ngrid = len(grid_vectors)
    if reciprocal_block_size is None:
        block_size = ngrid
    else:
        block_size = int(reciprocal_block_size)
        if block_size <= 0:
            raise ValueError('reciprocal_block_size must be positive')
        block_size = min(block_size, ngrid)
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    shape = (3, 3, nao, nao, naux)
    value = np.zeros((nao, nao, naux), dtype=np.complex128)
    pair_response = np.zeros(shape, dtype=np.complex128)
    auxiliary_response = np.zeros(shape, dtype=np.complex128)
    kernel_response = np.zeros(shape, dtype=np.complex128)

    for start in range(0, ngrid, block_size):
        vectors = grid_vectors[start:start + block_size]
        pair = ao_pair_fourier_strain(cell, vectors, rcut=rcut)
        auxiliary = auxiliary_fourier_strain(auxcell, vectors)
        kernel = weighted_coulomb_strain(cell, vectors, omega=omega)
        pair_value = np.asarray(pair['value'])
        pair1 = np.asarray(pair['raw_derivative'])
        factors = np.asarray(auxiliary['value'])
        factors1 = np.asarray(auxiliary['raw_derivative'])
        weights = np.asarray(kernel['value'])
        weights1 = np.asarray(kernel['raw_derivative'])
        weighted_auxiliary = (factors * weights).conj()
        value += lib.einsum(
            'Gij,PG->ijP', pair_value, weighted_auxiliary)
        pair_response += lib.einsum(
            'xyGij,PG->xyijP', pair1, weighted_auxiliary)
        auxiliary_response += lib.einsum(
            'Gij,xyPG->xyijP', pair_value,
            (factors1 * weights[None, None]).conj())
        kernel_response += lib.einsum(
            'Gij,xyPG->xyijP', pair_value,
            (factors[:, None, None] * weights1).transpose(
                1, 2, 0, 3).conj())

    derivative = pair_response + auxiliary_response + kernel_response
    value = value.reshape(nao * nao, naux)
    derivative = derivative.reshape(3, 3, nao * nao, naux)
    pair_response = pair_response.reshape(3, 3, nao * nao, naux)
    auxiliary_response = auxiliary_response.reshape(
        3, 3, nao * nao, naux)
    kernel_response = kernel_response.reshape(3, 3, nao * nao, naux)
    return {
        'name': 'B_LR',
        'primitive_analytic': True,
        'production_analytic': False,
        'storage': 'full_reference',
        'value': _real_if_close(value),
        'raw_derivative': _real_if_close(derivative),
        'voigt_derivative': matrix_strain_to_voigt(
            _real_if_close(derivative)),
        'ao_pair_response': _real_if_close(pair_response),
        'auxiliary_response': _real_if_close(auxiliary_response),
        'kernel_response': _real_if_close(kernel_response),
        'reciprocal_block_size': block_size,
        'grid_points': ngrid,
    }


def short_range_source_strain(cell, auxcell, omega, precision=None):
    r'''Return analytic image-resolved :math:`B^{SR}` strain response.

    For each screened Gamma image triplet, translational invariance reduces
    the three center motions to

    .. math::

        (B^{SR})^{(xy)}
        =I_x^{ip1}(R_j-R_i)_y
         +I_x^{ip2}(R_j-R_P)_y.

    PySCF's existing ``PBCsr3c_g_drv`` continues to own shell screening and
    lattice traversal.  The DC-Forge callback only applies these two image
    weights while the image indices are still available.  No displaced Cell
    or finite difference is used.

    Returns:
        dict
            Central short-range source and raw analytic derivative in
            ``(3,3,nao**2,naux)`` ordering.
    '''
    omega = abs(float(omega))
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('GDF short-range source strain requires positive omega')
    if precision is None:
        precision = float(cell.precision)
    precision = float(precision)
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError('short-range source precision must be positive')
    context = _short_range_source_context(
        cell, auxcell, omega, precision)
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    packed = np.zeros(
        (9, nao * (nao + 1) // 2, naux), dtype=np.double)
    extension = _load_short_range_source_extension()
    fill_name = (
        'DCFORGE_sr3c_strain_cart' if cell.cart
        else 'DCFORGE_sr3c_strain_sph')
    driver = rsdf_helper.libpbc.PBCsr3c_g_drv
    ip1_name = cell._add_suffix('int3c2e_ip1')
    driver(
        getattr(rsdf_helper.libpbc, ip1_name),
        getattr(extension, fill_name),
        packed.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(9),
        ctypes.c_int(len(context['lattice_vectors'])),
        context['lattice_vectors'].ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int * 6)(
            0, cell.nbas, cell.nbas, 2 * cell.nbas,
            2 * cell.nbas, 2 * cell.nbas + auxcell.nbas),
        context['ao_locations'].ctypes.data_as(ctypes.c_void_p),
        context['optimizer'],
        context['shell_pair_mask'].ctypes.data_as(ctypes.c_void_p),
        context['reference_unique_map'].ctypes.data_as(ctypes.c_void_p),
        context['auxiliary_unique_map'].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(context['naux_unique']),
        context['unique_exponents'].ctypes.data_as(ctypes.c_void_p),
        context['distance_cutoffs'].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(context['distance_step']),
        context['three_center_cutoffs'].ctypes.data_as(ctypes.c_void_p),
        context['distance_locations'].ctypes.data_as(ctypes.c_void_p),
        context['atm'].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(cell.natm),
        context['bas'].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(cell.nbas),
        context['env'].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(context['env'].size),
    )
    derivative = np.empty((3, 3, nao * nao, naux))
    for component in range(9):
        full = lib.unpack_tril(packed[component].T).transpose(1, 2, 0)
        derivative.reshape(9, nao * nao, naux)[component] = full.reshape(
            nao * nao, naux)
    value = short_range_source(
        cell, auxcell, omega, precision=precision)
    return {
        'name': 'B_SR',
        'primitive_analytic': True,
        'production_analytic': True,
        'storage': 'direct_voigt_candidate',
        'value': _real_if_close(value),
        'raw_derivative': _real_if_close(derivative),
        'voigt_derivative': matrix_strain_to_voigt(
            _real_if_close(derivative)),
        'lattice_vectors': len(context['lattice_vectors']),
        'precision': precision,
    }


def three_center_source_strain(cell, auxcell, omega, mesh,
                               precision=None, rcut=None,
                               reciprocal_block_size=None):
    r'''Return the complete analytic Gamma pre-Cholesky source strain.

    The P4 source ledger is

    .. math::

        B=B_{SR}+B_{LR}+B_{vbar},\qquad
        B^{(xy)}=B_{SR}^{(xy)}+B_{LR}^{(xy)}+B_{vbar}^{(xy)}.

    Each component owns a distinct mathematical origin and remains available
    in the return dictionary.  The complete derivative is analytic, but this
    forward reference retains nine full source tensors.  Production stress
    will contract these primitives directly with reverse-CDERI adjoints.
    '''
    short_range = short_range_source_strain(
        cell, auxcell, omega, precision=precision)
    long_range = long_range_source_strain(
        cell, auxcell, omega, mesh, rcut=rcut,
        reciprocal_block_size=reciprocal_block_size)
    vbar = vbar_source_strain(cell, auxcell, omega)
    value = (
        np.asarray(short_range['value'])
        + np.asarray(long_range['value'])
        + np.asarray(vbar['value']))
    derivative = (
        np.asarray(short_range['raw_derivative'])
        + np.asarray(long_range['raw_derivative'])
        + np.asarray(vbar['raw_derivative']))
    return {
        'name': 'B',
        'primitive_analytic': True,
        'production_analytic': False,
        'low_level_numerical_components': [],
        'storage': 'full_reference',
        'value': _real_if_close(value),
        'raw_derivative': _real_if_close(derivative),
        'voigt_derivative': matrix_strain_to_voigt(
            _real_if_close(derivative)),
        'short_range': short_range,
        'long_range': long_range,
        'vbar': vbar,
    }


def make_gdf_strain_reference(with_df, value_tol=1.e-7,
                              reciprocal_block_size=None):
    r'''Return analytic forward ``B/M/C`` strain in the provider gauge.

    This P4 reference evaluates the complete analytic source and metric
    ledgers, factors the central metric once, and forms

    .. math::

        C_B^{(xy)}=L^{-1}B^{(xy)},\qquad
        C_M^{(xy)}=-L^{-1}L^{(xy)}C.

    The independently loaded provider CDERI is authoritative.  A mismatch in
    the auxiliary gauge, ``omega``, source mesh, metric mesh, sign, or
    Cholesky convention fails before any stress contraction is accepted.

    Args:
        with_df : dcforge.cpu.pbc.df.gdf.GDF
            Built real-Gamma GDF provider.

    Kwargs:
        value_tol : float
            Maximum central reconstructed/provider CDERI difference.
        reciprocal_block_size : int or None
            Reference G-vector block size for analytic LR terms.

    Returns:
        dict
            Central ``B/M/C``, analytic raw derivatives, resolved component
            ledgers and central provider parity.  Full tensors are retained,
            so ``production_analytic`` remains false despite analytic
            primitives.
    '''
    if getattr(with_df, 'auxcell', None) is None:
        with_df.build(j_only=False)
    cell = with_df.cell
    auxcell = with_df.auxcell
    omega, source_mesh, metric_mesh = response_settings(with_df)
    source = three_center_source_strain(
        cell, auxcell, omega, source_mesh,
        reciprocal_block_size=reciprocal_block_size)
    metric = auxiliary_metric_strain(
        cell, auxcell, omega, metric_mesh,
        reciprocal_block_size=reciprocal_block_size)
    nao = int(cell.nao_nr())
    factor, cderi = cholesky_cderi(
        source['value'], metric['value'], nao)
    provider_cderi = load_cderi(with_df)
    parity = float(np.max(np.abs(cderi - provider_cderi)))
    if parity > float(value_tol):
        raise RuntimeError(
            'analytic GDF strain reference does not reproduce provider '
            f'CDERI: {parity:.3e} > {float(value_tol):.3e}')
    cderi1_source, cderi1_metric = grad_cderi_with_factor(
        factor, cderi, source['raw_derivative'],
        metric['raw_derivative'], nao)
    cderi1 = cderi1_source + cderi1_metric
    return {
        'primitive_analytic': True,
        'production_analytic': False,
        'low_level_numerical_components': [],
        'storage': 'full_reference',
        'omega': float(omega),
        'source_mesh': np.asarray(source_mesh, dtype=int),
        'metric_mesh': np.asarray(metric_mesh, dtype=int),
        'source': source,
        'metric': metric,
        'factor': _real_if_close(factor),
        'cderi': _real_if_close(cderi),
        'provider_cderi': _real_if_close(provider_cderi),
        'cderi_value_parity_max_abs': parity,
        'cderi_source_raw_derivative': _real_if_close(cderi1_source),
        'cderi_metric_raw_derivative': _real_if_close(cderi1_metric),
        'cderi_raw_derivative': _real_if_close(cderi1),
    }


def contract_vbar_source_strain(cell, auxcell, omega, source_adjoint):
    r'''Contract :math:`B^{vbar,(m)}` directly with source adjoints.

    For one observable adjoint :math:`A^B_{\mu\nu,P}`, this routine evaluates

    .. math::

        w^{vbar}_m
        =\sum_{\mu\nu P}A^B_{\mu\nu,P}
          B^{vbar,(m)}_{\mu\nu,P}

    from the overlap response and inverse-volume response.  No
    ``(6,nao**2,naux)`` source tensor is formed.

    Args:
        cell : pyscf.pbc.gto.Cell
            Central primary Cell.
        auxcell : pyscf.pbc.gto.Cell
            Charge-normalized auxiliary Cell in the provider gauge.
        omega : float
            Fixed GDF numerical range-separation parameter.
        source_adjoint : ndarray of shape (nobs,nao**2,naux)
            Reverse-CDERI source adjoints ``dE_o/dB``.

    Returns:
        dict
            Raw and engineering-Voigt contractions, with overlap and volume
            owners kept separate.  The leading dimension enumerates
            observables.
    '''
    adjoint = _source_adjoint(source_adjoint, cell, auxcell)
    omega = abs(float(omega))
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('GDF vbar source strain requires positive omega')
    overlap = overlap_strain(cell)
    overlap1 = np.asarray(overlap['raw_derivative']).reshape(3, 3, -1)
    overlap_value = np.asarray(overlap['value']).reshape(-1)
    charges = np.asarray(rsdf.get_aux_chg(auxcell), dtype=float)
    coefficient = np.pi * charges / (omega ** 2 * float(cell.vol))

    overlap_response = -lib.einsum(
        'omP,xym,P->oxy', adjoint, overlap1, coefficient)
    central_contraction = -lib.einsum(
        'omP,m,P->o', adjoint, overlap_value, coefficient)
    volume_response = (
        -central_contraction[:, None, None] * np.eye(3)[None])
    raw = _real_gamma_contraction(
        overlap_response + volume_response, 'B_vbar contraction')
    return _contracted_ledger(
        'B_vbar', raw,
        overlap_response=_real_gamma_contraction(
            overlap_response, 'B_vbar overlap contraction'),
        volume_response=_real_gamma_contraction(
            volume_response, 'B_vbar volume contraction'))


def contract_long_range_source_strain(
        cell, auxcell, omega, mesh, source_adjoint, rcut=None,
        reciprocal_block_size=None):
    r'''Contract :math:`B^{LR,(m)}` in bounded reciprocal blocks.

    The AO-pair Fourier, auxiliary Fourier and Coulomb-kernel responses are
    each contracted with ``dE/dB`` before the next reciprocal block is
    generated.  Storage therefore scales with one ``G`` block and the central
    adjoint, not with six complete three-center source derivatives.

    Kwargs:
        rcut : float or None
            Primary AO-pair image cutoff in Bohr.
        reciprocal_block_size : int or None
            Number of reciprocal vectors in one analytic block.
    '''
    adjoint = _source_adjoint(source_adjoint, cell, auxcell)
    adjoint = adjoint.reshape(
        len(adjoint), cell.nao_nr(), cell.nao_nr(), auxcell.nao_nr())
    mesh = _mesh(mesh)
    grid_vectors = np.asarray(cell.get_Gv(mesh), dtype=float)
    ngrid = len(grid_vectors)
    block_size = _reciprocal_block_size(ngrid, reciprocal_block_size)
    shape = (len(adjoint), 3, 3)
    pair_response = np.zeros(shape, dtype=np.complex128)
    auxiliary_response = np.zeros(shape, dtype=np.complex128)
    kernel_response = np.zeros(shape, dtype=np.complex128)
    peak_block_bytes = 0

    for start in range(0, ngrid, block_size):
        vectors = grid_vectors[start:start + block_size]
        pair = ao_pair_fourier_strain(cell, vectors, rcut=rcut)
        auxiliary = auxiliary_fourier_strain(auxcell, vectors)
        kernel = weighted_coulomb_strain(cell, vectors, omega=omega)
        pair_value = np.asarray(pair['value'])
        pair1 = np.asarray(pair['raw_derivative'])
        factors = np.asarray(auxiliary['value'])
        factors1 = np.asarray(auxiliary['raw_derivative'])
        weights = np.asarray(kernel['value'])
        weights1 = np.asarray(kernel['raw_derivative'])
        weighted_auxiliary = (factors * weights).conj()
        pair_response += lib.einsum(
            'oijP,xyGij,PG->oxy',
            adjoint, pair1, weighted_auxiliary)
        auxiliary_response += lib.einsum(
            'oijP,Gij,xyPG->oxy', adjoint, pair_value,
            (factors1 * weights[None, None]).conj())
        weighted_kernel1 = (
            factors[:, None, None] * weights1).transpose(1, 2, 0, 3)
        kernel_response += lib.einsum(
            'oijP,Gij,xyPG->oxy', adjoint, pair_value,
            weighted_kernel1.conj())
        peak_block_bytes = max(peak_block_bytes, sum(
            array.nbytes for array in (
                pair_value, pair1, factors, factors1, weights, weights1)))

    raw = _real_gamma_contraction(
        pair_response + auxiliary_response + kernel_response,
        'B_LR contraction')
    return _contracted_ledger(
        'B_LR', raw,
        ao_pair_response=_real_gamma_contraction(
            pair_response, 'B_LR AO-pair contraction'),
        auxiliary_response=_real_gamma_contraction(
            auxiliary_response, 'B_LR auxiliary contraction'),
        kernel_response=_real_gamma_contraction(
            kernel_response, 'B_LR kernel contraction'),
        reciprocal_block_size=int(block_size),
        grid_points=int(ngrid),
        peak_block_bytes=int(peak_block_bytes))


def make_short_range_source_workspace(cell, auxcell, nobs):
    r'''Allocate the exact packed workspace consumed by the SR C callback.

    The callback ABI places its ``(nobs,9,npair)`` result immediately before
    the packed symmetric source ``(nobs,npair,naux)`` in one contiguous FP64
    buffer.  Exposing that layout lets a GPU caller transfer the final packed
    source directly, instead of first copying a dense
    ``(nobs,nao**2,naux)`` array to host and allocating a second packed copy.

    Args:
        cell, auxcell : pyscf.pbc.gto.Cell
            Primary and auxiliary Cells defining ``nao`` and ``naux``.
        nobs : int
            Number of independently contracted observables.

    Returns:
        dict
            Contiguous ``array`` plus writable ``packed_adjoint`` and
            ``result`` views, unique-pair indices, and byte metadata.
    '''
    nobs = int(nobs)
    if nobs < 1:
        raise ValueError('nobs must be positive')
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    row, column = np.tril_indices(nao)
    npair = len(row)
    result_size = nobs * 9 * npair
    packed_size = nobs * npair * naux
    array = np.zeros(result_size + packed_size, dtype=np.double)
    return {
        'array': array,
        'result': array[:result_size].reshape(nobs, 9, npair),
        'packed_adjoint': array[result_size:].reshape(nobs, npair, naux),
        'row': row,
        'column': column,
        'off_diagonal': row != column,
        'nobs': nobs,
        'nao': nao,
        'naux': naux,
        'npair': npair,
        'result_size': result_size,
        'workspace_bytes': int(array.nbytes),
        'packed_adjoint_bytes': int(packed_size * array.dtype.itemsize),
    }


def contract_short_range_source_strain(
        cell, auxcell, omega, source_adjoint=None, precision=None,
        workspace=None):
    r'''Contract :math:`B^{SR,(m)}` inside the screened C traversal.

    The package-owned callback evaluates the two image moments

    .. math::

        I_x^{ip1}(R_j-R_i)_y+I_x^{ip2}(R_j-R_P)_y

    and immediately contracts them with packed ``dE/dB``.  Its output is
    only ``(nobs,9,npair)``; it never creates the full
    ``(9,nao**2,naux)`` source derivative used by the forward oracle.
    Keeping ``npair`` until the OpenMP traversal ends gives every shell pair
    a disjoint output region and avoids atomic scalar accumulation.
    '''
    omega = abs(float(omega))
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('GDF short-range source strain requires positive omega')
    if precision is None:
        precision = float(cell.precision)
    precision = float(precision)
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError('short-range source precision must be positive')
    context = _short_range_source_context(
        cell, auxcell, omega, precision)
    if workspace is None:
        if source_adjoint is None:
            raise ValueError(
                'source_adjoint is required when workspace is not supplied')
        adjoint = _source_adjoint(source_adjoint, cell, auxcell)
        workspace = make_short_range_source_workspace(
            cell, auxcell, len(adjoint))
        dense = adjoint.reshape(
            workspace['nobs'], workspace['nao'], workspace['nao'],
            workspace['naux'])
        packed = workspace['packed_adjoint']
        row = workspace['row']
        column = workspace['column']
        packed[:] = dense[:, row, column]
        off_diagonal = workspace['off_diagonal']
        packed[:, off_diagonal] += dense[
            :, column[off_diagonal], row[off_diagonal]]
        input_storage = 'dense_host_packed_in_python'
    else:
        if source_adjoint is not None:
            raise ValueError(
                'pass either source_adjoint or a prefilled workspace, not both')
        if not isinstance(workspace, dict):
            raise TypeError('short-range source workspace must be a dict')
        nobs_required = int(workspace.get('nobs', 0))
        nao_required = int(cell.nao_nr())
        naux_required = int(auxcell.nao_nr())
        row_required, column_required = np.tril_indices(nao_required)
        npair_required = len(row_required)
        result_size_required = nobs_required * 9 * npair_required
        packed_size_required = nobs_required * npair_required * naux_required
        required = {
            'nobs': nobs_required,
            'nao': nao_required,
            'naux': naux_required,
            'npair': npair_required,
            'result_size': result_size_required,
            'row': row_required,
            'column': column_required,
            'off_diagonal': row_required != column_required,
            'packed_adjoint_bytes': int(
                packed_size_required * np.dtype(np.double).itemsize),
            'workspace_bytes': int(
                (result_size_required + packed_size_required)
                * np.dtype(np.double).itemsize),
        }
        for name in ('nobs', 'nao', 'naux', 'npair', 'result_size'):
            if int(workspace.get(name, -1)) != int(required[name]):
                raise ValueError(
                    f'short-range source workspace {name} is incompatible')
        array = np.asarray(workspace.get('array'))
        if (array.dtype != np.double or not array.flags.c_contiguous
                or array.size != (
                    result_size_required + packed_size_required)):
            raise ValueError(
                'short-range source workspace array must be one contiguous '
                'FP64 buffer with the expected size')
        # Recreate views from the caller's array instead of trusting mutable
        # dictionary views.  This also proves the packed source immediately
        # follows the result region required by the C ABI.
        required['array'] = array
        required['result'] = array[:required['result_size']].reshape(
            required['nobs'], 9, required['npair'])
        required['packed_adjoint'] = array[
            required['result_size']:].reshape(
                required['nobs'], required['npair'], required['naux'])
        workspace = required
        input_storage = 'prefilled_unique_pair_workspace'

    nobs = workspace['nobs']
    npair = workspace['npair']
    result_size = workspace['result_size']
    array = workspace['array']

    extension = _load_short_range_source_extension()
    fill_name = (
        'DCFORGE_sr3c_strain_contract_cart' if cell.cart
        else 'DCFORGE_sr3c_strain_contract_sph')
    rsdf_helper.libpbc.PBCsr3c_g_drv(
        getattr(rsdf_helper.libpbc, cell._add_suffix('int3c2e_ip1')),
        getattr(extension, fill_name),
        array.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nobs),
        ctypes.c_int(len(context['lattice_vectors'])),
        context['lattice_vectors'].ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int * 6)(
            0, cell.nbas, cell.nbas, 2 * cell.nbas,
            2 * cell.nbas, 2 * cell.nbas + auxcell.nbas),
        context['ao_locations'].ctypes.data_as(ctypes.c_void_p),
        context['optimizer'],
        context['shell_pair_mask'].ctypes.data_as(ctypes.c_void_p),
        context['reference_unique_map'].ctypes.data_as(ctypes.c_void_p),
        context['auxiliary_unique_map'].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(context['naux_unique']),
        context['unique_exponents'].ctypes.data_as(ctypes.c_void_p),
        context['distance_cutoffs'].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(context['distance_step']),
        context['three_center_cutoffs'].ctypes.data_as(ctypes.c_void_p),
        context['distance_locations'].ctypes.data_as(ctypes.c_void_p),
        context['atm'].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(cell.natm),
        context['bas'].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(cell.nbas),
        context['env'].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(context['env'].size),
    )
    raw = array[:result_size].reshape(nobs, 9, npair)
    raw = raw.sum(axis=-1).reshape(nobs, 3, 3)
    return _contracted_ledger(
        'B_SR', raw,
        lattice_vectors=len(context['lattice_vectors']),
        precision=precision,
        input_storage=input_storage,
        packed_adjoint_bytes=workspace['packed_adjoint_bytes'],
        workspace_bytes=workspace['workspace_bytes'])


def contract_three_center_source_strain(
        cell, auxcell, omega, mesh, source_adjoint, precision=None,
        rcut=None, reciprocal_block_size=None):
    r'''Contract the complete :math:`B_{SR}+B_{LR}+B_{vbar}` ledger.'''
    started = time.perf_counter()
    short_range = contract_short_range_source_strain(
        cell, auxcell, omega, source_adjoint, precision=precision)
    short_range_seconds = time.perf_counter() - started
    started = time.perf_counter()
    long_range = contract_long_range_source_strain(
        cell, auxcell, omega, mesh, source_adjoint, rcut=rcut,
        reciprocal_block_size=reciprocal_block_size)
    long_range_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vbar = contract_vbar_source_strain(
        cell, auxcell, omega, source_adjoint)
    vbar_seconds = time.perf_counter() - started
    raw = (
        short_range['raw_contraction']
        + long_range['raw_contraction']
        + vbar['raw_contraction'])
    return _contracted_ledger(
        'B', raw, short_range=short_range, long_range=long_range,
        vbar=vbar, component_wall_seconds={
            'short_range': float(short_range_seconds),
            'long_range': float(long_range_seconds),
            'vbar': float(vbar_seconds),
        })


def contract_g0_metric_strain(cell, auxcell, omega, metric_adjoint):
    r'''Contract the GDF zero-mode metric response with ``dE/dM``.'''
    adjoint = _metric_adjoint(metric_adjoint, auxcell)
    zero_mode = g0_metric_strain(cell, auxcell, omega)
    central = lib.einsum('oPQ,PQ->o', adjoint, zero_mode['value'])
    raw = -central[:, None, None] * np.eye(3)[None]
    return _contracted_ledger('M_G0', raw)


def contract_long_range_metric_strain(
        cell, auxcell, omega, mesh, metric_adjoint,
        reciprocal_block_size=None):
    r'''Contract :math:`M^{LR,(m)}` in bounded reciprocal blocks.

    The direct expression contains three contractions with the full
    auxiliary pair ``(P,Q)``:

    .. math::

        A_{PQ}(F_P^{(m)*}wF_Q
        +F_P^*wF_Q^{(m)}+F_P^*w^{(m)}F_Q).

    Since every metric derivative is symmetric in ``P,Q``, only
    ``sym(A)`` contributes.  This routine first forms the shared projection

    .. math::

        (AF)_{oPG}=\sum_Q A_{oPQ}F_{QG},

    once per reciprocal block.  The two auxiliary-factor legs are then
    complex conjugates and are accumulated as twice the real part, while the
    kernel leg reuses ``F^dagger AF``.  This is an exact contraction-order
    change: it replaces three expensive ``P,Q,G`` reductions by one matrix
    product plus linear-size reductions without changing the GDF owner,
    mesh, ``omega``, or strain algebra.
    '''
    adjoint = _metric_adjoint(metric_adjoint, auxcell)
    antisymmetry = adjoint - adjoint.transpose(0, 2, 1)
    antisymmetry_max = float(np.max(np.abs(antisymmetry)))
    adjoint = .5 * (adjoint + adjoint.transpose(0, 2, 1))
    mesh = _mesh(mesh)
    grid_vectors = np.asarray(cell.get_Gv(mesh), dtype=float)
    ngrid = len(grid_vectors)
    block_size = _reciprocal_block_size(ngrid, reciprocal_block_size)
    shape = (len(adjoint), 3, 3)
    auxiliary_response = np.zeros(shape, dtype=np.complex128)
    kernel_response = np.zeros(shape, dtype=np.complex128)
    peak_block_bytes = 0

    for start in range(0, ngrid, block_size):
        vectors = grid_vectors[start:start + block_size]
        auxiliary = auxiliary_fourier_strain(auxcell, vectors)
        kernel = weighted_coulomb_strain(cell, vectors, omega=omega)
        factors = np.asarray(auxiliary['value'])
        factors1 = np.asarray(auxiliary['raw_derivative'])
        weights = np.asarray(kernel['value'])
        weights1 = np.asarray(kernel['raw_derivative'])
        projected = lib.einsum(
            'oPQ,QG->oPG', adjoint, factors)
        auxiliary_response += 2 * lib.einsum(
            'xyPG,oPG,G->oxy',
            factors1.conj(), projected, weights).real
        bilinear = lib.einsum(
            'PG,oPG->oG', factors.conj(), projected)
        kernel_response += lib.einsum(
            'oG,xyG->oxy', bilinear, weights1).real
        peak_block_bytes = max(peak_block_bytes, sum(
            array.nbytes for array in (
                factors, factors1, weights, weights1,
                projected, bilinear)))

    raw = _real_gamma_contraction(
        auxiliary_response + kernel_response, 'M_LR contraction')
    return _contracted_ledger(
        'M_LR', raw,
        auxiliary_response=_real_gamma_contraction(
            auxiliary_response, 'M_LR auxiliary contraction'),
        kernel_response=_real_gamma_contraction(
            kernel_response, 'M_LR kernel contraction'),
        contraction_order='shared_metric_adjoint_projection',
        adjoint_antisymmetry_max_abs=antisymmetry_max,
        reciprocal_block_size=int(block_size),
        grid_points=int(ngrid),
        peak_block_bytes=int(peak_block_bytes))


def contract_short_range_metric_strain(
        auxcell, omega, metric_adjoint, precision=None,
        screening='gdf'):
    r'''Contract :math:`M^{SR,(m)}` directly in a screened C/OpenMP loop.

    For a symmetric real-Gamma metric adjoint :math:`A`, exchange of the two
    auxiliary centers converts the separate ``ip1`` and ``ip2`` terms into

    .. math::

        A_{PQ}M_{PQ}^{SR,(xy)}
        =\sum_{PQL} A_{PQ}
          (PQ_L|)^{ip1}_x (R_Q+L-R_P)_y.

    The package callback evaluates the screened PySCF/libcint shell block,
    applies its lattice-image moment, and consumes it immediately.  Only one
    ``(nobs,3,3)`` accumulator leaves C; neither the image stack nor a full
    ``(3,3,naux,naux)`` derivative metric is materialized.  The transparent
    two-integral Python implementation remains available as
    :func:`contract_short_range_metric_strain_oracle`.

    Args:
        auxcell : pyscf.pbc.gto.Cell
            Central charge-normalized auxiliary Cell.
        omega : float
            Positive fixed GDF range-separation parameter.
        metric_adjoint : ndarray
            One or more real-Gamma metric adjoints with shape
            ``(nobs,naux,naux)``.

    Kwargs:
        precision : float or None
            Integral-screening precision.  ``None`` preserves the validated
            stress convention ``auxcell.precision**1.5``.
        screening : {'gdf', 'rsdf'}
            Definition of the retained real-space lattice sum.  ``'gdf'``
            reproduces the transparent GDF stress oracle: one global image
            set and no shell-pair cutoff in the callback.  ``'rsdf'`` uses
            the exact reference-shell map, shell-pair cutoffs and minimal
            image list of :func:`pyscf.pbc.df.rsdf_helper.intor_j2c`.
            This is a representation choice, not an execution tolerance.

    Returns:
        dict
            Contracted ``M_SR`` ledger in Hartree with explicit host-C
            execution metadata.
    '''
    adjoint = _metric_adjoint(metric_adjoint, auxcell)
    antisymmetry = adjoint - adjoint.transpose(0, 2, 1)
    antisymmetry_max = float(np.max(np.abs(antisymmetry)))
    adjoint = np.asarray(
        .5 * (adjoint + adjoint.transpose(0, 2, 1)),
        dtype=np.double, order='C')
    omega = abs(float(omega))
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('short-range metric strain requires positive omega')
    screening = str(screening).lower()
    if screening not in ('gdf', 'rsdf'):
        raise ValueError("metric screening must be 'gdf' or 'rsdf'")
    if precision is None:
        precision = (
            float(auxcell.precision)
            if screening == 'rsdf'
            else float(auxcell.precision) ** 1.5)
    precision = float(precision)
    if screening == 'gdf':
        rcut = float(estimate_rs_2c2e_rcut(
            auxcell, omega, precision))
        lattice_vectors = np.asarray(
            auxcell.get_lattice_Ls(rcut=rcut),
            dtype=np.double, order='C')
        # The transparent GDF stress oracle evaluates every auxiliary shell
        # pair on one global image set.  Keep the callback cutoff infinite so
        # the direct C contraction is representation-identical to that
        # forward reference.
        refmap = np.zeros(auxcell.nbas, dtype=np.int32, order='C')
        cutoff_squared = np.asarray([np.inf], dtype=np.double, order='C')
        pair_cutoff_count = 1
    else:
        # Match rsdf_helper.intor_j2c exactly.  RSDF's finite Hamiltonian is
        # defined by a unique-reference-shell cutoff for every auxiliary
        # shell pair and by the corresponding minimal lattice-image list.
        # Replacing this with GDF's global rcut changes M_SR and therefore
        # leaves a finite strain error even when the central matrices agree.
        refmap, unique_atoms, unique_basis, unique_basis_locations = (
            rsdf_helper._get_refuniq_map(auxcell))
        cutoffs = rsdf_helper._get_2c2e_Rcut(
            unique_basis, omega, precision, lmp=True, lasympt=True,
            eta_correct=True, R_correct=False)
        atom_cutoffs = rsdf_helper._get_atom_Rcuts_2c(
            cutoffs, unique_basis_locations)
        lattice_vectors = np.asarray(
            rsdf_helper._get_Lsmin(
                auxcell, atom_cutoffs, unique_atoms),
            dtype=np.double, order='C')
        refmap = np.asarray(refmap, dtype=np.int32, order='C')
        cutoff_squared = np.asarray(
            cutoffs ** 2, dtype=np.double, order='C')
        rcut = float(atom_cutoffs.max())
        pair_cutoff_count = len(cutoffs)

    intor, _components = moleintor._get_intor_and_comp(
        auxcell._add_suffix('int2c2e_ip1'), 3)
    atm, bas, env = gto.conc_env(
        auxcell._atm, auxcell._bas, auxcell._env,
        auxcell._atm, auxcell._bas, auxcell._env)
    atm = np.asarray(atm, dtype=np.int32, order='C')
    bas = np.asarray(bas, dtype=np.int32, order='C')
    env = np.asarray(env, dtype=np.double, order='C')
    env[gto.PTR_RANGE_OMEGA] = -omega
    ao_locations = np.asarray(
        moleintor.make_loc(bas, intor), dtype=np.int32, order='C')
    shell_slice = (ctypes.c_int * 4)(
        0, auxcell.nbas, auxcell.nbas, 2 * auxcell.nbas)
    raw = np.zeros((len(adjoint), 3, 3), dtype=np.double, order='C')
    extension = _load_short_range_metric_extension()
    extension.DCFORGE_sr2c_adjoint_strain(
        getattr(moleintor.libcgto, intor),
        raw.ctypes.data_as(ctypes.c_void_p),
        adjoint.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(len(adjoint)),
        ctypes.c_int(len(lattice_vectors)),
        lattice_vectors.ctypes.data_as(ctypes.c_void_p),
        shell_slice,
        ao_locations.ctypes.data_as(ctypes.c_void_p),
        np.asarray(refmap, dtype=np.int32, order='C').ctypes.data_as(
            ctypes.c_void_p),
        cutoff_squared.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
    return _contracted_ledger(
        'M_SR', _real_gamma_contraction(raw, 'M_SR contraction'),
        lattice_vectors=len(lattice_vectors),
        rcut_bohr=rcut,
        precision=precision,
        screening=screening,
        pair_cutoff_count=pair_cutoff_count,
        storage='screened_image_integral_direct_adjoint',
        execution_owner='pyscf_libcint_host_c_openmp_direct',
        adjoint_antisymmetry_max_abs=antisymmetry_max,
        peak_block_bytes=0)


def contract_short_range_metric_strain_oracle(
        auxcell, omega, metric_adjoint, precision=None):
    r'''Contract image-resolved :math:`M^{SR,(m)}` with ``dE/dM``.

    Each lattice image produces only two derivative integral blocks.  They
    are consumed immediately, so neither the six metric derivatives nor a
    lattice-image stack is retained.
    '''
    adjoint = _metric_adjoint(metric_adjoint, auxcell)
    omega = abs(float(omega))
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('short-range metric strain requires positive omega')
    if precision is None:
        precision = float(auxcell.precision) ** 1.5
    precision = float(precision)
    rcut = float(estimate_rs_2c2e_rcut(auxcell, omega, precision))
    lattice_vectors = np.asarray(
        auxcell.get_lattice_Ls(rcut=rcut), dtype=float)
    centers = _ao_centers(auxcell)
    raw = np.zeros((len(adjoint), 3, 3), dtype=np.complex128)
    peak_block_bytes = 0

    with auxcell.with_short_range_coulomb(omega):
        for lattice in lattice_vectors:
            shifted = copy_with_coordinates(
                auxcell, auxcell.atom_coords() + lattice)
            with shifted.with_short_range_coulomb(omega):
                ip1 = np.asarray(gto.intor_cross(
                    'int2c2e_ip1', auxcell, shifted, comp=3))
                ip2 = np.asarray(gto.intor_cross(
                    'int2c2e_ip2', auxcell, shifted, comp=3))
            raw -= lib.einsum(
                'oPQ,xPQ,Py->oxy', adjoint, ip1, centers)
            raw -= lib.einsum(
                'oPQ,xPQ,Qy->oxy',
                adjoint, ip2, centers + lattice)
            peak_block_bytes = max(
                peak_block_bytes, int(ip1.nbytes + ip2.nbytes))

    return _contracted_ledger(
        'M_SR', _real_gamma_contraction(raw, 'M_SR contraction'),
        lattice_vectors=len(lattice_vectors), rcut_bohr=rcut,
        peak_block_bytes=peak_block_bytes,
        storage='python_image_integral_oracle',
        execution_owner='pyscf_intor_cross_host_python_oracle')


def contract_auxiliary_metric_strain(
        cell, auxcell, omega, mesh, metric_adjoint, precision=None,
        reciprocal_block_size=None, screening='gdf'):
    r'''Contract the complete :math:`M_{SR}+M_{LR}+M_{G0}` ledger.

    ``screening`` selects the provider's finite short-range Hamiltonian.  It
    does not alter the reciprocal mesh or numerical precision.
    '''
    started = time.perf_counter()
    short_range = contract_short_range_metric_strain(
        auxcell, omega, metric_adjoint, precision=precision,
        screening=screening)
    short_range_seconds = time.perf_counter() - started
    started = time.perf_counter()
    long_range = contract_long_range_metric_strain(
        cell, auxcell, omega, mesh, metric_adjoint,
        reciprocal_block_size=reciprocal_block_size)
    long_range_seconds = time.perf_counter() - started
    started = time.perf_counter()
    zero_mode = contract_g0_metric_strain(
        cell, auxcell, omega, metric_adjoint)
    zero_mode_seconds = time.perf_counter() - started
    raw = (
        short_range['raw_contraction']
        + long_range['raw_contraction']
        + zero_mode['raw_contraction'])
    return _contracted_ledger(
        'M', raw, short_range=short_range, long_range=long_range,
        g0=zero_mode, component_wall_seconds={
            'short_range': float(short_range_seconds),
            'long_range': float(long_range_seconds),
            'g0': float(zero_mode_seconds),
        }, short_range_screening=str(screening).lower())


def make_gdf_strain_response(
        with_df, dm0, dz, reciprocal_block_size=None,
        validate_central=False, value_tol=1.e-7):
    r'''Return production reverse-adjoint GDF strain observables.

    The two geometry-independent CDERI adjoints correspond to target Hartree
    and source-HF Fock response.  They are pushed once through

    .. math::

        C=L^{-1}B,\qquad M=LL^T,

    after which every analytic ``B`` and ``M`` strain primitive is contracted
    immediately.  The output contains only scalar raw/Voigt ledgers; no
    ``C^(m)``, ``B^(m)`` or ``M^(m)`` stack survives the call.

    Args:
        with_df : dcforge.cpu.pbc.df.gdf.GDF
            Built real-Gamma GDF provider.
        dm0 : ndarray of shape (nao,nao)
            Spin-summed source RHF density.
        dz : ndarray of shape (nao,nao)
            Hermitian Z-transition density.

    Kwargs:
        reciprocal_block_size : int or None
            Bounded G-vector block shared by source and metric LR consumers.
        validate_central : bool
            Rebuild zero-order ``B`` and require reconstructed/provider CDERI
            parity.  Production may disable this diagnostic reconstruction;
            scientific validation enables it.
        value_tol : float
            Central CDERI parity threshold when ``validate_central`` is true.

    Returns:
        dict
            Term-by-term source/metric contractions.  Observable order is
            ``target_j`` then ``source_fock`` and every Voigt array has shape
            ``(2,6)``.
    '''
    _check_real_gamma_gdf(with_df)
    if getattr(with_df, '_cderi', None) is None or with_df.auxcell is None:
        with_df.build(j_only=False)
    cell = with_df.cell
    auxcell = with_df.auxcell
    omega, source_mesh, metric_mesh1 = response_settings(with_df)
    metric = auxiliary_metric(cell, auxcell, omega, metric_mesh1)
    factor = factor_metric(metric)
    central = load_cderi(with_df)
    resolved_block_size = _automatic_strain_reciprocal_block_size(
        cell, auxcell, reciprocal_block_size, with_df.max_memory,
        max(int(np.prod(source_mesh)), int(np.prod(metric_mesh1))))
    parity = None
    if validate_central:
        source = three_center_source(
            cell, auxcell, omega, source_mesh,
            max_memory=with_df.max_memory)
        reconstructed = solve_cderi_with_factor(
            factor, source, cell.nao_nr())
        parity = float(np.max(np.abs(reconstructed - central)))
        if parity > float(value_tol):
            raise RuntimeError(
                'analytic GDF strain response does not reproduce provider '
                f'CDERI: {parity:.3e} > {float(value_tol):.3e}')

    observable_adjoints = np.asarray(
        make_cderi_observable_adjoints(central, dm0, dz))
    source_adjoint, metric_adjoint = push_cderi_adjoints(
        factor, central, observable_adjoints)
    source = contract_three_center_source_strain(
        cell, auxcell, omega, source_mesh, source_adjoint,
        reciprocal_block_size=resolved_block_size)
    metric_response = contract_auxiliary_metric_strain(
        cell, auxcell, omega, metric_mesh1, metric_adjoint,
        reciprocal_block_size=resolved_block_size)
    raw = source['raw_contraction'] + metric_response['raw_contraction']
    result = _contracted_ledger(
        'C', raw, source=source, metric=metric_response)
    result.update({
        'primitive_analytic': True,
        'production_analytic': True,
        'low_level_numerical_components': [],
        'storage': 'reverse_adjoint_direct_voigt',
        'observable_names': ('target_j', 'source_fock'),
        'omega': float(omega),
        'source_mesh': np.asarray(source_mesh, dtype=int),
        'metric_mesh': np.asarray(metric_mesh1, dtype=int),
        'reciprocal_block_size_requested': (
            None if reciprocal_block_size is None
            else int(reciprocal_block_size)),
        'reciprocal_block_size': int(resolved_block_size),
        'central_validation_enabled': bool(validate_central),
        'cderi_value_parity_max_abs': parity,
        'persistent_adjoint_bytes': int(
            observable_adjoints.nbytes + source_adjoint.nbytes
            + metric_adjoint.nbytes + factor.nbytes + central.nbytes),
    })
    with_df.cderi_strain_storage = result['storage']
    with_df.cderi_strain_reciprocal_block_size = (
        source['long_range']['reciprocal_block_size'])
    with_df.cderi_strain_persistent_bytes = result[
        'persistent_adjoint_bytes']
    return result


def finite_difference_gdf_strain(with_df, step=1.e-4):
    r'''Return fixed-representation raw-9 ``B/M/C`` strain references.

    This routine is intentionally a small-system oracle.  It holds the
    central GDF range-separation ``omega``, source mesh and metric mesh fixed,
    deforms both primary and auxiliary Cells, and materializes all nine
    source/metric/CDERI derivatives.  Production stress must instead contract
    analytic ``B^(strain)`` and ``M^(strain)`` directly with reverse adjoints.

    Args:
        with_df : dcforge.cpu.pbc.df.gdf.GDF
            Built Gamma GDF provider.

    Kwargs:
        step : float
            Dimensionless raw Cartesian strain displacement.

    Returns:
        dict
            Central ``source``, ``metric``, ``cderi`` and raw-9 derivative
            tensors, including independent direct-CDERI and differentiated-
            Cholesky parity.
    '''
    if getattr(with_df, 'auxcell', None) is None:
        with_df.build(j_only=False)
    cell = with_df.cell
    auxcell = with_df.auxcell
    if int(cell.dimension) != 3:
        raise NotImplementedError('GDF strain reference requires a 3D Cell')
    step = float(step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError('GDF strain reference step must be positive')
    omega, source_mesh, metric_mesh = response_settings(with_df)
    nao = int(cell.nao_nr())
    source = three_center_source(
        cell, auxcell, omega, source_mesh,
        max_memory=with_df.max_memory)
    metric = auxiliary_metric(cell, auxcell, omega, metric_mesh)
    factor, cderi = cholesky_cderi(source, metric, nao)

    source1 = np.empty((9,) + source.shape, dtype=source.dtype)
    metric1 = np.empty((9,) + metric.shape, dtype=metric.dtype)
    cderi1_direct = np.empty((9,) + cderi.shape, dtype=cderi.dtype)
    primary_mesh = np.asarray(cell.mesh, dtype=int)
    auxiliary_mesh = np.asarray(auxcell.mesh, dtype=int)
    for index, direction in enumerate(raw_strain_basis()):
        cell_plus = deform_cell(
            cell, direction, scale=step, mesh=primary_mesh)
        cell_minus = deform_cell(
            cell, direction, scale=-step, mesh=primary_mesh)
        aux_plus = _deform_auxiliary_cell(
            auxcell, direction, scale=step, mesh=auxiliary_mesh)
        aux_minus = _deform_auxiliary_cell(
            auxcell, direction, scale=-step, mesh=auxiliary_mesh)
        source_plus = three_center_source(
            cell_plus, aux_plus, omega, source_mesh,
            max_memory=with_df.max_memory)
        source_minus = three_center_source(
            cell_minus, aux_minus, omega, source_mesh,
            max_memory=with_df.max_memory)
        metric_plus = auxiliary_metric(
            cell_plus, aux_plus, omega, metric_mesh)
        metric_minus = auxiliary_metric(
            cell_minus, aux_minus, omega, metric_mesh)
        source1[index] = (source_plus - source_minus) / (2 * step)
        metric1[index] = (metric_plus - metric_minus) / (2 * step)
        cderi1_direct[index] = (
            solve_cderi(source_plus, metric_plus, nao)
            - solve_cderi(source_minus, metric_minus, nao)) / (2 * step)

    cderi1_source, cderi1_metric = grad_cderi_with_factor(
        factor, cderi, source1, metric1, nao)
    cderi1 = cderi1_source + cderi1_metric
    parity = float(np.max(np.abs(cderi1 - cderi1_direct)))
    return {
        'production_analytic': False,
        'low_level_numerical_components': [
            'gdf_source_metric_cell_finite_difference'],
        'step': step,
        'omega': float(omega),
        'source_mesh': np.asarray(source_mesh, dtype=int),
        'metric_mesh': np.asarray(metric_mesh, dtype=int),
        'source': _real_if_close(source),
        'metric': _real_if_close(metric),
        'cderi': _real_if_close(cderi),
        'source_raw_derivative': _real_if_close(
            source1.reshape(3, 3, *source.shape)),
        'metric_raw_derivative': _real_if_close(
            metric1.reshape(3, 3, *metric.shape)),
        'cderi_source_raw_derivative': _real_if_close(
            cderi1_source.reshape(3, 3, *cderi.shape)),
        'cderi_metric_raw_derivative': _real_if_close(
            cderi1_metric.reshape(3, 3, *cderi.shape)),
        'cderi_raw_derivative': _real_if_close(
            cderi1.reshape(3, 3, *cderi.shape)),
        'cderi_direct_raw_derivative': _real_if_close(
            cderi1_direct.reshape(3, 3, *cderi.shape)),
        'cderi_direct_parity_max_abs': parity,
    }


def gdf_jk_strain(cderi, cderi1, density):
    r'''Return fixed-density GDF ``J^(strain)`` and ``K^(strain)`` matrices.'''
    cderi = np.asarray(cderi)
    cderi1 = np.asarray(cderi1)
    density = np.asarray(density)
    if cderi1.shape[:2] != (3, 3) or cderi1.shape[2:] != cderi.shape:
        raise ValueError('GDF CDERI strain must have shape (3,3,naux,nao,nao)')
    if density.shape != cderi.shape[-2:]:
        raise ValueError('GDF strain density has incompatible AO dimensions')
    rho = lib.einsum('Pij,ji->P', cderi, density)
    rho1 = lib.einsum('xyPij,ji->xyP', cderi1, density)
    vj1 = lib.einsum('P,xyPij->xyij', rho, cderi1)
    vj1 += lib.einsum('xyP,Pij->xyij', rho1, cderi)
    vk1 = lib.einsum(
        'xyPij,jk,Pkl->xyil', cderi1, density, cderi)
    vk1 += lib.einsum(
        'Pij,jk,xyPkl->xyil', cderi, density, cderi1)
    return _real_if_close(vj1), _real_if_close(vk1)


def gdf_j_work(density, vj1):
    r'''Return target ``1/2 Tr[P J^(strain)[P]]`` in Hartree.'''
    return .5 * lib.einsum('xyij,ji->xy', vj1, density).real


def _contracted_ledger(name, raw, **components):
    '''Return one scalar raw/Voigt reverse-adjoint term ledger.'''
    raw = _real_gamma_contraction(raw, f'{name} contraction')
    if raw.ndim != 3 or raw.shape[-2:] != (3, 3):
        raise ValueError(
            f'{name} contracted strain must have shape (nobs,3,3)')
    result = {
        'name': name,
        'primitive_analytic': True,
        'production_analytic': True,
        'storage': 'contracted',
        'raw_contraction': raw,
        'voigt_contraction': lib.einsum(
            'vxy,oxy->ov', voigt_strain_basis(), raw),
    }
    result.update(components)
    return result


def _source_adjoint(adjoint, cell, auxcell):
    '''Return validated real ``(nobs,nao**2,naux)`` source adjoints.'''
    adjoint = _real_gamma_contraction(adjoint, 'source adjoint')
    expected = (cell.nao_nr() ** 2, auxcell.nao_nr())
    if adjoint.ndim == 2:
        adjoint = adjoint[None]
    if adjoint.ndim != 3 or adjoint.shape[1:] != expected:
        raise ValueError(
            'source adjoint must have shape '
            f'(nobs,{expected[0]},{expected[1]})')
    return np.asarray(adjoint, dtype=float, order='C')


def _metric_adjoint(adjoint, auxcell):
    '''Return validated real ``(nobs,naux,naux)`` metric adjoints.'''
    adjoint = _real_gamma_contraction(adjoint, 'metric adjoint')
    naux = auxcell.nao_nr()
    if adjoint.ndim == 2:
        adjoint = adjoint[None]
    if adjoint.ndim != 3 or adjoint.shape[1:] != (naux, naux):
        raise ValueError(
            'metric adjoint must have shape '
            f'(nobs,{naux},{naux})')
    return np.asarray(adjoint, dtype=float, order='C')


def _real_gamma_contraction(value, name, tolerance=1.e-8):
    '''Return a real contraction or reject an unsupported complex Gamma term.'''
    value = np.asarray(value)
    if np.iscomplexobj(value):
        maximum = float(np.max(np.abs(value.imag))) if value.size else 0.
        if maximum > tolerance:
            raise NotImplementedError(
                f'{name} has non-Gamma imaginary component {maximum:.3e}')
        value = value.real
    return value


def _reciprocal_block_size(ngrid, requested):
    '''Return a positive G-vector block size for direct contractions.'''
    ngrid = int(ngrid)
    if ngrid < 1:
        raise ValueError('reciprocal mesh contains no grid points')
    if requested is None:
        return ngrid
    requested = int(requested)
    if requested <= 0:
        raise ValueError('reciprocal_block_size must be positive')
    return min(requested, ngrid)


def _automatic_strain_reciprocal_block_size(
        cell, auxcell, requested, max_memory, ngrid):
    r'''Resolve a bounded analytic GDF strain reciprocal block.

    The dominant source workspace per reciprocal vector is the AO-pair value
    plus its nine raw strain responses.  Auxiliary Fourier values/responses
    are included in the estimate.  Automatic selection uses at most one
    quarter of currently unused PySCF ``max_memory`` and caps the block at
    4096 vectors so production never silently becomes a full-grid tensor for
    small-AO but very dense meshes.
    '''
    ngrid = int(ngrid)
    if requested is not None:
        return _reciprocal_block_size(ngrid, requested)
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    bytes_per_grid = (
        np.dtype(np.complex128).itemsize
        * (10 * nao ** 2 + 10 * naux)
        + np.dtype(np.float64).itemsize * 10)
    remaining_mb = max(
        1., float(max_memory) - float(lib.current_memory()[0]))
    budget = remaining_mb * 1.e6 / 4
    block_size = max(1, int(budget // bytes_per_grid))
    return min(4096, ngrid, block_size)


def _check_real_gamma_gdf(with_df):
    '''Reject provider representations outside the first stress release.'''
    kpts = np.asarray(with_df.kpts).reshape(-1, 3)
    if len(kpts) != 1 or np.max(np.abs(kpts)) > 1.e-10:
        raise NotImplementedError(
            'GDF strain response currently supports only real Gamma')
    if int(with_df.cell.dimension) != 3:
        raise NotImplementedError(
            'GDF strain response currently requires a 3D Cell')
    if getattr(with_df, 'j2c_eig_always', False):
        raise NotImplementedError(
            'GDF strain response requires Cholesky metric factorization')


def _real_if_close(value, tolerance=1.e-10):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tolerance:
        return value.real
    return value


def _short_range_source_context(cell, auxcell, omega, precision):
    '''Build PySCF-identical ME screening data for the strain callback.'''
    refmap, unique_atoms, unique_basis, unique_basis_locations = (
        rsdf_helper._get_refuniq_map(cell))
    if getattr(auxcell, '_dcforge_fake_nuclear', False):
        auxmap, _aux_atoms, unique_auxiliary, unique_aux_locations = (
            fake_nuclear_refuniq(auxcell))
        if _aux_atoms != unique_atoms:
            raise RuntimeError(
                'fake nuclear and orbital unique-atom order must agree')
        auxiliary_schwarz = _fake_nuclear_schwarz(
            unique_auxiliary, omega)
        naux_unique = int(unique_auxiliary.nbas)
    else:
        auxmap, _aux_atoms, unique_auxiliary, unique_aux_locations = (
            rsdf_helper._get_refuniq_map(auxcell))
        auxiliary_schwarz = rsdf_helper._get_schwartz_data(
            unique_auxiliary, omega, keep1ctr=False, safe=True)
        naux_unique = len(unique_auxiliary)
    distance_step = 1 / rsdf_helper.BOHR
    distance_cutoffs = rsdf_helper._get_schwartz_dcut(
        unique_basis, omega, precision / auxiliary_schwarz.max(),
        r0=cell.rcut)
    distance_lists = rsdf_helper._make_dijs_lst(
        distance_cutoffs, distance_step)
    distance_locations = np.cumsum(
        [0] + [len(values) for values in distance_lists]).astype(np.int32)
    zero_estimates = [np.zeros_like(values) for values in distance_lists]
    three_center_cutoffs = rsdf_helper._get_3c2e_Rcuts(
        unique_basis, unique_auxiliary, distance_lists, omega, precision,
        'ME', zero_estimates)
    basis_exponents = np.asarray([
        np.asarray(shell[1:])[:, 0].min() for shell in unique_basis])
    atom_cutoffs = rsdf_helper._get_atom_Rcuts_3c(
        three_center_cutoffs, distance_lists, basis_exponents,
        unique_basis_locations, unique_aux_locations)
    lattice_vectors = np.asarray(
        rsdf_helper._get_Lsmin(cell, atom_cutoffs, unique_atoms),
        dtype=float, order='C')

    intor = cell._add_suffix('int3c2e_ip1')
    atm, bas, env = gto.conc_env(
        cell._atm, cell._bas, cell._env,
        cell._atm, cell._bas, cell._env)
    ao_locations = moleintor.make_loc(bas, intor)
    auxiliary_locations = auxcell.ao_loc_nr(
        auxcell.cart or 'ssc' in intor)
    ao_locations = np.asarray(np.hstack((
        ao_locations,
        ao_locations[-1] + auxiliary_locations[1:])), dtype=np.int32)
    atm, bas, env = gto.conc_env(
        atm, bas, env, auxcell._atm, auxcell._bas, auxcell._env)
    env[gto.PTR_RANGE_OMEGA] = -omega
    optimizer = _vhf.make_cintopt(atm, bas, env, intor)
    rsdf_helper.libpbc.CINTdel_pairdata_optimizer(optimizer)
    return {
        'reference_unique_map': np.asarray(refmap, dtype=np.int32),
        'auxiliary_unique_map': np.asarray(auxmap, dtype=np.int32),
        'naux_unique': naux_unique,
        'unique_exponents': np.asarray(basis_exponents, dtype=float),
        'distance_cutoffs': np.asarray(distance_cutoffs, dtype=float) ** 2,
        'distance_step': float(distance_step),
        'three_center_cutoffs': np.asarray(
            three_center_cutoffs, dtype=float) ** 2,
        'distance_locations': distance_locations,
        'lattice_vectors': lattice_vectors,
        'shell_pair_mask': np.ones(
            (cell.nbas, cell.nbas), dtype=np.int8, order='C'),
        'ao_locations': ao_locations,
        'optimizer': optimizer,
        'atm': np.asarray(atm, dtype=np.int32),
        'bas': np.asarray(bas, dtype=np.int32),
        'env': np.asarray(env, dtype=float),
    }


def _fake_nuclear_schwarz(unique_fake_nuclear, omega):
    r'''Return exact SR two-center bounds for charge-normalized fake shells.

    ``rsdf_helper._get_schwartz_data`` rebuilds normalized orbital functions
    from a Python basis list.  GTH fake nuclear Gaussians instead carry the
    charge-normalized libcint coefficient installed by
    :func:`pyscf.pbc.df.aft._fake_nuc`.  Evaluating the representative shells
    in their original Cell retains that coefficient and therefore gives a
    conservative, basis-independent screening bound.
    '''
    values = np.zeros(unique_fake_nuclear.nbas, dtype=float)
    for shell_id in range(unique_fake_nuclear.nbas):
        integral = rsdf_helper._fintor_sreri(
            unique_fake_nuclear, 'int2c2e',
            (shell_id, shell_id + 1, shell_id, shell_id + 1),
            omega, True)
        values[shell_id] = np.linalg.norm(integral) ** .5
    return values


def _load_short_range_source_extension():
    '''Load the package-owned image-resolved short-range strain callback.'''
    pyscf_lib = Path(pyscf.__file__).resolve().parent / 'lib'
    libraries = (
        pyscf_lib / 'deps' / 'lib' / 'libcint.so',
        pyscf_lib / 'libcgto.so',
        pyscf_lib / 'libpbc.so',
    )
    for library in libraries:
        if not library.is_file():
            raise ImportError(f'required PySCF library was not found: {library}')
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    module = importlib.import_module(
        'dcforge.cpu.pbc.df.gaussian._sr3c_strain')
    return ctypes.CDLL(
        str(Path(module.__file__).resolve()), mode=ctypes.RTLD_GLOBAL)


def _load_short_range_metric_extension():
    '''Load the package-owned direct two-center strain contraction.'''
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
        'dcforge.cpu.pbc.df.gaussian._sr2c_adjoint_strain')
    return ctypes.CDLL(
        str(Path(module.__file__).resolve()), mode=ctypes.RTLD_GLOBAL)


def _deform_auxiliary_cell(auxcell, strain, scale, mesh):
    r'''Affinely move a charge-normalized ``make_modrho_basis`` Cell.

    Rebuilding this Cell, including through ``set_geom_``, can normalize an
    already modified auxiliary basis a second time when the input-unit label
    changes.  :func:`copy_with_coordinates` updates only nuclear-coordinate
    slots and therefore preserves the exact central auxiliary gauge.
    '''
    strain = np.asarray(strain, dtype=float)
    lattice = np.asarray(auxcell.lattice_vectors(), dtype=float)
    lattice_new = lattice @ (np.eye(3) + float(scale) * strain).T
    coordinates = fractional_coordinates(auxcell) @ lattice_new
    return copy_with_coordinates(
        auxcell, coordinates, lattice=lattice_new, mesh=mesh)


def _mesh(mesh):
    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError('mesh must contain three positive integers')
    return mesh


def _ao_centers(cell):
    centers = np.empty((cell.nao_nr(), 3), dtype=float)
    for atom_id, aoslice in enumerate(cell.aoslice_by_atom()):
        start, stop = aoslice[2:]
        centers[start:stop] = cell.atom_coord(atom_id)
    return centers
