"""Analytic Gamma metric strain for the exact PySCF RSDF representation."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from pyscf import gto
from pyscf.gto import moleintor
from pyscf.pbc.df import rsdf_helper

from .gdf_metric_strain import (
    kpoint_reciprocal_metric_strain,
    matrix_tensor_to_voigt,
)


@dataclass(frozen=True)
class GammaRSDFMetricStrain:
    """RSDF two-center metric and homogeneous-strain decomposition."""

    value: np.ndarray
    raw_derivative: np.ndarray
    short_range_value: np.ndarray
    short_range_derivative: np.ndarray
    reciprocal: object
    lattice_image_count: int

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class KPointRSDFMetricStrain:
    """Exact screened RSDF metric response at one momentum transfer."""

    value: np.ndarray
    raw_derivative: np.ndarray
    short_range_value: np.ndarray
    short_range_derivative: np.ndarray
    reciprocal: object
    lattice_image_count: int
    momentum_transfer: np.ndarray

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


def gamma_rsdf_metric_strain(
    cell, auxcell, parameters, *, reciprocal_block_size=None,
):
    """Differentiate the screened RSDF metric actually factorized by PySCF."""

    general = kpoint_rsdf_metric_strain(
        cell, auxcell, parameters, np.zeros(3),
        reciprocal_block_size=reciprocal_block_size,
    )
    return GammaRSDFMetricStrain(
        value=general.value,
        raw_derivative=general.raw_derivative,
        short_range_value=general.short_range_value,
        short_range_derivative=general.short_range_derivative,
        reciprocal=general.reciprocal,
        lattice_image_count=general.lattice_image_count,
    )


def kpoint_rsdf_metric_strain(
    cell, auxcell, parameters, q, *, reciprocal_block_size=None,
):
    """Differentiate the exact RSDF metric at fixed scaled transfer q."""

    transfer = _kpoint(q)
    short_value, short_derivative, image_count = _short_range_metric_strain(
        auxcell, parameters.metric_omega,
        q=transfer, precision=parameters.precision_j2c,
    )
    reciprocal = kpoint_reciprocal_metric_strain(
        cell, auxcell, parameters.metric_omega, parameters.metric_mesh,
        transfer, reciprocal_block_size=reciprocal_block_size,
    )
    return KPointRSDFMetricStrain(
        value=_real_if_close(short_value + reciprocal.value),
        raw_derivative=_real_if_close(
            short_derivative + reciprocal.raw_derivative
        ),
        short_range_value=_real_if_close(short_value),
        short_range_derivative=_real_if_close(short_derivative),
        reciprocal=reciprocal,
        lattice_image_count=image_count,
        momentum_transfer=transfer,
    )


def _short_range_metric_strain(auxcell, omega, *, q, precision):
    reference_map, unique_atoms, unique_basis, unique_basis_locations = (
        rsdf_helper._get_refuniq_map(auxcell)
    )
    cutoffs = rsdf_helper._get_2c2e_Rcut(
        unique_basis, float(omega), float(precision), lmp=True,
        lasympt=True, eta_correct=True, R_correct=False,
    )
    atom_cutoffs = rsdf_helper._get_atom_Rcuts_2c(
        cutoffs, unique_basis_locations
    )
    lattice = np.asarray(
        rsdf_helper._get_Lsmin(auxcell, atom_cutoffs, unique_atoms),
        dtype=float, order="C",
    )

    pair_cell = auxcell.copy(deep=False)
    pair_cell._atm, pair_cell._bas, pair_cell._env = atm, basis, env = (
        gto.conc_env(
            auxcell._atm, auxcell._bas, auxcell._env,
            auxcell._atm, auxcell._bas, auxcell._env,
        )
    )
    env[gto.PTR_RANGE_OMEGA] = -abs(float(omega))
    value_name, _ = moleintor._get_intor_and_comp(
        auxcell._add_suffix("int2c2e"), None
    )
    ip1_name, _ = moleintor._get_intor_and_comp(
        auxcell._add_suffix("int2c2e_ip1"), None
    )
    ao_locations = np.asarray(
        moleintor.make_loc(basis, value_name), dtype=np.int32
    )
    nbas = int(auxcell.nbas)
    shell_slice = np.asarray((0, nbas, nbas, 2 * nbas), dtype=np.int32)
    naux = int(auxcell.nao_nr())
    transfer = _kpoint(q)
    phase = np.asarray(
        np.exp(1j * np.dot(transfer, lattice.T)),
        dtype=np.complex128, order="C",
    )
    output = np.zeros((10, naux, naux), dtype=np.complex128)
    extension = _load_extension()
    extension.PBCsr2c_strain_kpt_drv(
        getattr(moleintor.libcgto, value_name),
        getattr(moleintor.libcgto, ip1_name),
        output.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(len(lattice)),
        lattice.ctypes.data_as(ctypes.c_void_p),
        phase.ctypes.data_as(ctypes.c_void_p),
        shell_slice.ctypes.data_as(ctypes.c_void_p),
        ao_locations.ctypes.data_as(ctypes.c_void_p),
        np.asarray(reference_map, dtype=np.int32).ctypes.data_as(ctypes.c_void_p),
        np.asarray(cutoffs**2, dtype=float).ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(pair_cell.natm),
        basis.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(pair_cell.nbas),
        env.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(env.size),
    )
    return output[0], output[1:].reshape(3, 3, naux, naux), int(len(lattice))


def _load_extension():
    from .gdf_source_strain import _load_short_range_strain_extension

    return _load_short_range_strain_extension()


def _real_if_close(value, tolerance=1e-10):
    array = np.asarray(value)
    if array.size and np.max(np.abs(array.imag)) <= tolerance:
        return array.real
    return array


def _kpoint(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"momentum transfer must have shape (3,), got {point.shape}")
    return point
