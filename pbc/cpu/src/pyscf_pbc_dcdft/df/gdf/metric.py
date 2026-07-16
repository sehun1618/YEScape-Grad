r"""Auxiliary Coulomb metric and its fixed-cell nuclear derivative.

For momentum transfer ``q = k_j - k_i``, periodic Gaussian density fitting
uses the auxiliary metric

.. math::

   M_{PQ}(q) = (P_q | Q_q)
              = M^{\mathrm{SR}}_{PQ}(q)
              + M^{\mathrm{LR}}_{PQ}(q)
              + \delta_{q0} M^{G=0}_{PQ}.

The metric is geometry dependent because every auxiliary Gaussian is centered
on an atom.  Its derivative is the auxiliary-basis response which enters
``dC`` for the orthonormalized three-index tensor ``C = chol(M)^{-1} B``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import lib
from pyscf.pbc.df import ft_ao, rsdf
from pyscf.pbc.df.rsdf_builder import (
    _estimate_meshz,
    estimate_ke_cutoff_for_omega,
    estimate_rs_2c2e_rcut,
)
from pyscf.pbc.lib.kpts_helper import is_zero


@dataclass(frozen=True)
class MetricDerivativeParts:
    """Resolved fixed-cell derivative of the auxiliary metric."""

    short_range: np.ndarray
    long_range: np.ndarray
    g0: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """Return ``dM_SR + dM_LR + dM_G0``."""

        return self.short_range + self.long_range + self.g0


def recommended_metric_mesh(cell, auxcell, omega: float) -> np.ndarray:
    """Return the mesh used by PySCF's RSGDF two-center metric.

    The three-center source mesh is not automatically sufficient for the
    metric.  PySCF deliberately estimates a tighter cutoff from the auxiliary
    basis because a small error in ``M`` is amplified by its Cholesky solve.
    """

    precision = float(auxcell.precision) ** 1.5
    cutoff = estimate_ke_cutoff_for_omega(auxcell, float(omega), precision)
    mesh = np.asarray(auxcell.cutoff_to_mesh(cutoff), dtype=int)
    if cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum":
        mesh[2] = _estimate_meshz(auxcell)
    elif cell.dimension < 2:
        mesh[cell.dimension :] = cell.mesh[cell.dimension :]
    return np.asarray(cell.symmetrize_mesh(mesh), dtype=int)


def short_range_metric(auxcell, omega: float, *, q: np.ndarray) -> np.ndarray:
    """Return the analytic real-space short-range metric ``M_SR(q)``.

    This intentionally uses PySCF's public periodic integral path.  A previous
    low-level screened-driver prototype used a different cutoff convention and
    missed diagonal metric contributions.  ``estimate_rs_2c2e_rcut`` plus
    ``pbc_intor`` reproduces the production RSGDF convention directly.
    """

    momentum_transfer = _qpoint(q)
    with _short_range_context(auxcell, omega):
        value = np.asarray(
            auxcell.pbc_intor(
                "int2c2e",
                hermi=1,
                kpts=momentum_transfer.reshape(1, 3),
            )
        )[0]
    return _real_at_gamma(value, momentum_transfer)


def short_range_metric_derivative(
    auxcell,
    atom_id: int,
    omega: float,
    *,
    q: np.ndarray,
) -> np.ndarray:
    """Return ``dM_SR(q)/dR_A`` with shape ``(3, naux, naux)``.

    PySCF's ``ip1`` and ``ip2`` integrals are electron-coordinate
    derivatives.  Moving an auxiliary center therefore contributes their
    negative on the selected atom's row and column blocks.
    """

    atom = _atom_id(auxcell, atom_id)
    momentum_transfer = _qpoint(q)
    with _short_range_context(auxcell, omega):
        ip1 = np.asarray(
            auxcell.pbc_intor(
                "int2c2e_ip1",
                comp=3,
                hermi=0,
                kpts=momentum_transfer.reshape(1, 3),
                pbcopt=None,
            )
        )[0]
        ip2 = np.asarray(
            auxcell.pbc_intor(
                "int2c2e_ip2",
                comp=3,
                hermi=0,
                kpts=momentum_transfer.reshape(1, 3),
                pbcopt=None,
            )
        )[0]
    p0, p1 = _ao_slice(auxcell, atom)
    derivative = np.zeros_like(ip1)
    derivative[:, p0:p1, :] -= ip1[:, p0:p1, :]
    derivative[:, :, p0:p1] -= ip2[:, :, p0:p1]
    return _real_at_gamma(derivative, momentum_transfer)


def long_range_metric(
    cell,
    auxcell,
    omega: float,
    *,
    mesh: tuple[int, int, int] | np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Return the reciprocal-space long-range metric ``M_LR(q)``."""

    aux_fourier, coulomb, momentum_transfer, _grid_vectors = (
        _long_range_context(cell, auxcell, omega, mesh, q)
    )
    metric = (aux_fourier.conj() * coulomb) @ aux_fourier.T
    return _real_at_gamma(metric, momentum_transfer)


def long_range_metric_derivative(
    cell,
    auxcell,
    atom_id: int,
    omega: float,
    *,
    mesh: tuple[int, int, int] | np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Return the auxiliary-center derivative of ``M_LR(q)``."""

    atom = _atom_id(auxcell, atom_id)
    aux_fourier, coulomb, momentum_transfer, grid_vectors = (
        _long_range_context(cell, auxcell, omega, mesh, q)
    )
    p0, p1 = _ao_slice(auxcell, atom)
    momentum = grid_vectors + momentum_transfer
    derivatives = []
    for axis in range(3):
        d_aux = np.zeros_like(aux_fourier)
        d_aux[p0:p1] = (
            -1j
            * momentum[:, axis][None, :]
            * aux_fourier[p0:p1]
        )
        value = (d_aux.conj() * coulomb) @ aux_fourier.T
        value += (aux_fourier.conj() * coulomb) @ d_aux.T
        derivatives.append(value)
    return _real_at_gamma(np.asarray(derivatives), momentum_transfer)


def g0_metric(cell, auxcell, omega: float, *, q: np.ndarray) -> np.ndarray:
    """Return the zero-transfer charge correction, or zero for ``q != 0``."""

    momentum_transfer = _qpoint(q)
    naux = int(auxcell.nao_nr())
    if not is_zero(momentum_transfer):
        return np.zeros((naux, naux), dtype=np.complex128)
    charges = np.asarray(rsdf.get_aux_chg(auxcell), dtype=float)
    coefficient = np.pi / (float(omega) ** 2 * float(cell.vol))
    return -coefficient * np.outer(charges, charges)


def g0_metric_derivative(auxcell, *, q: np.ndarray) -> np.ndarray:
    """Return the fixed-cell center derivative of the G=0 correction.

    Auxiliary Gaussian charges and cell volume are translation invariant, so
    this derivative is exactly zero for nuclear displacements at fixed lattice.
    """

    _qpoint(q)
    naux = int(auxcell.nao_nr())
    return np.zeros((3, naux, naux), dtype=float)


def auxiliary_metric(
    cell,
    auxcell,
    omega: float,
    *,
    mesh: tuple[int, int, int] | np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Return ``M_SR + M_LR + M_G0`` in the RSGDF convention."""

    return (
        short_range_metric(auxcell, omega, q=q)
        + long_range_metric(cell, auxcell, omega, mesh=mesh, q=q)
        + g0_metric(cell, auxcell, omega, q=q)
    )


def auxiliary_metric_derivative(
    cell,
    auxcell,
    atom_id: int,
    omega: float,
    *,
    mesh: tuple[int, int, int] | np.ndarray,
    q: np.ndarray,
) -> MetricDerivativeParts:
    """Return the resolved fixed-cell derivative of the RSGDF metric."""

    return MetricDerivativeParts(
        short_range=short_range_metric_derivative(
            auxcell, atom_id, omega, q=q
        ),
        long_range=long_range_metric_derivative(
            cell, auxcell, atom_id, omega, mesh=mesh, q=q
        ),
        g0=g0_metric_derivative(auxcell, q=q),
    )


class _short_range_context:
    """Temporarily apply PySCF's converged SR cutoff and range operator."""

    def __init__(self, auxcell, omega):
        self.auxcell = auxcell
        self.omega = abs(float(omega))
        self.original_rcut = float(auxcell.rcut)

    def __enter__(self):
        precision = float(self.auxcell.precision) ** 1.5
        self.auxcell.rcut = estimate_rs_2c2e_rcut(
            self.auxcell, self.omega, precision
        )
        self.range_context = self.auxcell.with_short_range_coulomb(self.omega)
        self.range_context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self.range_context.__exit__(exc_type, exc_value, traceback)
        finally:
            self.auxcell.rcut = self.original_rcut


def _long_range_context(cell, auxcell, omega, mesh, q):
    momentum_transfer = _qpoint(q)
    mesh_array = _mesh(mesh)
    grid_vectors, grid_bases, _weights = auxcell.get_Gv_weights(mesh_array)
    reciprocal = auxcell.reciprocal_vectors()
    grid_indices = lib.cartesian_prod(
        [np.arange(len(values)) for values in grid_bases]
    )
    provider = rsdf.RSGDF(cell, kpts=momentum_transfer.reshape(1, 3))
    provider.auxcell = auxcell
    # Positive omega is the long-range erf kernel.  It equals the full kernel
    # minus PySCF's negative-omega short-range Fourier kernel.
    coulomb = provider.weighted_coulG(
        momentum_transfer, mesh=mesh_array, omega=float(omega)
    )
    aux_fourier = ft_ao.ft_ao(
        auxcell,
        grid_vectors,
        None,
        reciprocal,
        grid_indices,
        grid_bases,
        momentum_transfer,
    ).T
    return aux_fourier, coulomb, momentum_transfer, grid_vectors


def _real_at_gamma(array, momentum_transfer):
    value = np.asarray(array)
    return value.real if is_zero(momentum_transfer) else value


def _ao_slice(cell, atom_id):
    slices = np.asarray(cell.aoslice_nr_by_atom(), dtype=int)
    return int(slices[atom_id, 2]), int(slices[atom_id, 3])


def _atom_id(cell, atom_id):
    atom = int(atom_id)
    if atom < 0 or atom >= int(cell.natm):
        raise ValueError(f"atom_id out of range: {atom_id}")
    return atom


def _qpoint(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"q must have shape (3,), got {point.shape}")
    return point


def _mesh(value):
    mesh = np.asarray(value, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError(f"mesh must contain three positive integers: {mesh}")
    return mesh
