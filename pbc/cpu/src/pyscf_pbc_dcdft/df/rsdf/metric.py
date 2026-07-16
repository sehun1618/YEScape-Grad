r"""PySCF-RSDF two-center metric and its analytic center derivative.

For momentum transfer ``q`` the public PySCF RSDF backend constructs

.. math::

   M(q) = M_{\mathrm{SR}}(q;\omega_2)
        + M_{\mathrm{LR}}(q;\omega_2,\mathcal G_2)
        + \delta_{q0} M_{G=0},

where ``omega_j2c`` and ``mesh_j2c`` need not equal the parameters used by
the three-center source.  The short-range value uses PySCF's screened
``PBCsr2c`` lattice-sum driver.  Its ``ip1``/``ip2`` variants are called with
the identical image list and screening bounds, so this module differentiates
the metric actually factorized by RSDF rather than a continuum surrogate.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from pyscf import gto as mol_gto
from pyscf import lib
from pyscf.pbc.df import rsdf_helper
from pyscf.pbc.df.incore import libpbc

from ..gdf.metric import (
    g0_metric,
    g0_metric_derivative,
    long_range_metric,
    long_range_metric_derivative,
)


@dataclass(frozen=True)
class RSDFMetricDerivativeParts:
    """Short-range, long-range, and zero-mode metric derivatives."""

    short_range: np.ndarray
    long_range: np.ndarray
    g0: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """Return ``dM_SR + dM_LR + dM_G0`` in Hartree/Bohr."""

        return self.short_range + self.long_range + self.g0


def rsdf_metric(cell, auxcell, parameters, *, q: np.ndarray) -> np.ndarray:
    """Return the exact public-RSDF two-center metric for one ``q``.

    Parameters
    ----------
    cell, auxcell
        Primary and auxiliary periodic cells.
    parameters
        Resolved :class:`RSDFParameters` from the built provider.
    q
        Momentum transfer in reciprocal Bohr.
    """

    momentum = _qpoint(q)
    short_range = np.asarray(
        rsdf_helper.intor_j2c(
            auxcell,
            parameters.metric_omega,
            kpts=momentum.reshape(1, 3),
        )
    )[0]
    return _real_if_close(
        short_range
        + long_range_metric(
            cell,
            auxcell,
            parameters.metric_omega,
            mesh=parameters.metric_mesh,
            q=momentum,
        )
        + g0_metric(
            cell,
            auxcell,
            parameters.metric_omega,
            q=momentum,
        )
    )


def rsdf_metric_derivative(
    cell,
    auxcell,
    atom_id: int,
    parameters,
    *,
    q: np.ndarray,
) -> RSDFMetricDerivativeParts:
    """Return all Cartesian derivatives of the public-RSDF metric.

    The returned arrays have shape ``(3, naux, naux)`` and units
    Hartree/Bohr in the auxiliary Coulomb-metric convention.
    """

    momentum = _qpoint(q)
    return RSDFMetricDerivativeParts(
        short_range=_short_range_metric_derivative(
            auxcell,
            atom_id,
            parameters.metric_omega,
            q=momentum,
        ),
        long_range=long_range_metric_derivative(
            cell,
            auxcell,
            atom_id,
            parameters.metric_omega,
            mesh=parameters.metric_mesh,
            q=momentum,
        ),
        g0=g0_metric_derivative(auxcell, q=momentum),
    )


def _short_range_metric_derivative(auxcell, atom_id, omega, *, q):
    atom = _atom_id(auxcell, atom_id)
    ip1 = _short_range_integral(
        auxcell, omega, q=q, intor="int2c2e_ip1"
    )
    ip2 = _short_range_integral(
        auxcell, omega, q=q, intor="int2c2e_ip2"
    )
    p0, p1 = np.asarray(auxcell.aoslice_nr_by_atom(), dtype=int)[atom, 2:4]
    derivative = np.zeros_like(ip1)
    # PySCF ip integrals differentiate electronic coordinates.  Moving the
    # atom-centered Gaussian therefore contributes the opposite sign on its
    # row (ip1) and column (ip2) auxiliary blocks.
    derivative[:, p0:p1, :] -= ip1[:, p0:p1, :]
    derivative[:, :, p0:p1] -= ip2[:, :, p0:p1]
    return _real_if_close(derivative)


def _short_range_integral(auxcell, omega, *, q, intor):
    """Call PySCF's screened SR two-center driver for a derivative intor."""

    precision = float(auxcell.precision)
    ref_map, unique_atoms, unique_basis, unique_basis_loc = (
        rsdf_helper._get_refuniq_map(auxcell)
    )
    rcuts = rsdf_helper._get_2c2e_Rcut(
        unique_basis,
        float(omega),
        precision,
        lmp=True,
        lasympt=True,
        eta_correct=True,
        R_correct=False,
    )
    atom_rcuts = rsdf_helper._get_atom_Rcuts_2c(
        rcuts, unique_basis_loc
    )
    lattice_vectors = rsdf_helper._get_Lsmin(
        auxcell, atom_rcuts, unique_atoms
    )

    integral_name, components = mol_gto.moleintor._get_intor_and_comp(
        auxcell._add_suffix(intor), None
    )
    pair_cell = auxcell.copy(deep=False)
    pair_cell._atm, pair_cell._bas, pair_cell._env = atm, basis, env = (
        mol_gto.conc_env(
            auxcell._atm,
            auxcell._bas,
            auxcell._env,
            auxcell._atm,
            auxcell._bas,
            auxcell._env,
        )
    )
    env[mol_gto.PTR_RANGE_OMEGA] = -abs(float(omega))

    i0, i1 = 0, int(auxcell.nbas)
    j0, j1 = int(auxcell.nbas), 2 * int(auxcell.nbas)
    ao_loc = mol_gto.moleintor.make_loc(basis, integral_name)
    ni = int(ao_loc[i1] - ao_loc[i0])
    nj = int(ao_loc[j1] - ao_loc[j0])
    momentum = _qpoint(q).reshape(1, 3)
    output = np.empty(
        (1, components, ni, nj), dtype=np.complex128
    )
    phase = np.asarray(
        np.exp(1j * np.dot(momentum, lattice_vectors.T)), order="C"
    )

    driver = libpbc.PBCsr2c_k_drv
    fill = libpbc.PBCsr2c_fill_ks1
    integral = getattr(mol_gto.moleintor.libcgto, integral_name)
    driver(
        integral,
        fill,
        output.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(1),
        ctypes.c_int(components),
        ctypes.c_int(len(lattice_vectors)),
        lattice_vectors.ctypes.data_as(ctypes.c_void_p),
        phase.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int * 4)(i0, i1, j0, j1),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        lib.c_null_ptr(),
        ref_map.ctypes.data_as(ctypes.c_void_p),
        (rcuts**2).ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(pair_cell.natm),
        basis.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(pair_cell.nbas),
        env.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(env.size),
    )
    return output[0]


def _qpoint(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"q must have shape (3,), got {point.shape}")
    return point


def _atom_id(cell, atom_id):
    atom = int(atom_id)
    if not 0 <= atom < int(cell.natm):
        raise ValueError(f"atom_id out of range: {atom_id}")
    return atom


def _real_if_close(array, tolerance=1.0e-10):
    value = np.asarray(array)
    if value.size and np.max(np.abs(value.imag)) <= tolerance:
        return value.real
    return value
