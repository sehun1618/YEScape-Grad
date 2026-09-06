r'''GDF auxiliary Coulomb metric and fixed-cell nuclear derivative.

For the Gamma-point GDF/RSDF factorization used here,

.. math::

    M_{PQ} = M^{\mathrm{SR}}_{PQ}
           + M^{\mathrm{LR}}_{PQ}
           + M^{G=0}_{PQ}.

The AO centers of the auxiliary basis move with their atoms, so both the
short- and long-range terms contribute to :math:`M^\lambda`.  The cell and
reciprocal mesh are fixed; stress derivatives are outside this module.
'''

from contextlib import contextmanager

import numpy as np
from pyscf.pbc.df.rsdf_builder import (
    _estimate_meshz,
    estimate_ke_cutoff_for_omega,
    estimate_rs_2c2e_rcut,
)
from pyscf.pbc.lib.kpts_helper import is_zero

from dcforge.cpu.pbc.df.gaussian.metric import (
    g0_metric,
    gen_grad_long_range_metric,
    long_range_metric,
)


def metric_mesh(cell, auxcell, omega):
    '''Return PySCF's converged reciprocal mesh for the two-center metric.

    Args:
        cell : pyscf.pbc.gto.Cell
            Primary periodic cell.  Its fixed lattice defines mesh symmetry.
        auxcell : pyscf.pbc.gto.Cell
            Auxiliary Gaussian cell.
        omega : float
            Positive range-separation parameter in Bohr**-1.

    Returns:
        ndarray of shape (3,)
            Symmetrized integer mesh.  This is normally tighter than the
            three-center source mesh because metric errors are amplified by
            the Cholesky solve.
    '''
    precision = float(auxcell.precision) ** 1.5
    cutoff = estimate_ke_cutoff_for_omega(auxcell, float(omega), precision)
    mesh = np.asarray(auxcell.cutoff_to_mesh(cutoff), dtype=int)
    if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
        mesh[2] = _estimate_meshz(auxcell)
    elif cell.dimension < 2:
        mesh[cell.dimension:] = cell.mesh[cell.dimension:]
    return np.asarray(cell.symmetrize_mesh(mesh), dtype=int)


def short_range_metric(auxcell, omega, q=None):
    r'''Return :math:`M^{\mathrm{SR}}` from analytic periodic integrals.

    Args:
        auxcell : pyscf.pbc.gto.Cell
            Auxiliary Gaussian cell.
        omega : float
            Positive range-separation parameter in Bohr**-1.
        q : ndarray of shape (3,)
            Momentum transfer.  The current public DC-Forge route uses zero.

    Returns:
        ndarray of shape (naux, naux)
            Short-range Coulomb metric.
    '''
    q = _qpoint(q)
    with _short_range_context(auxcell, omega):
        metric = np.asarray(auxcell.pbc_intor(
            'int2c2e', hermi=1, kpts=q.reshape(1, 3)))[0]
    return _real_at_gamma(metric, q)


def grad_short_range_metric(auxcell, omega, q=None, atmlst=None):
    r'''Return all atomic derivatives of :math:`M^{\mathrm{SR}}`.

    PySCF ``int2c2e_ip1`` and ``int2c2e_ip2`` differentiate the auxiliary AO
    with respect to the electron coordinate.  Moving its nuclear center gives
    the opposite sign.  The row and column blocks are therefore accumulated
    separately below.

    Returns:
        ndarray of shape (len(atmlst), 3, naux, naux)
            Fixed-cell derivative in Ha/Bohr.
    '''
    metric1 = gen_grad_short_range_metric(auxcell, omega, q)
    return metric1(atmlst)


def gen_grad_short_range_metric(auxcell, omega, q=None):
    r'''Generate atom-block assembly of :math:`(M^{SR})^\lambda`.'''
    q = _qpoint(q)
    with _short_range_context(auxcell, omega):
        ip1 = np.asarray(auxcell.pbc_intor(
            'int2c2e_ip1', comp=3, hermi=0,
            kpts=q.reshape(1, 3), pbcopt=None))[0]
        ip2 = np.asarray(auxcell.pbc_intor(
            'int2c2e_ip2', comp=3, hermi=0,
            kpts=q.reshape(1, 3), pbcopt=None))[0]

    naux = auxcell.nao_nr()
    aoslices = auxcell.aoslice_nr_by_atom()

    def metric1(atmlst=None):
        if atmlst is None:
            atmlst = range(auxcell.natm)
        atmlst = [int(ia) for ia in atmlst]
        value = np.zeros(
            (len(atmlst), 3, naux, naux),
            dtype=np.result_type(ip1, ip2))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            value[k, :, p0:p1] -= ip1[:, p0:p1]
            value[k, :, :, p0:p1] -= ip2[:, :, p0:p1]
        return _real_at_gamma(value, q)

    return metric1


def auxiliary_metric(cell, auxcell, omega, mesh, q=None):
    r'''Return the full PySCF RSGDF metric :math:`M`.

    The three terms remain public functions so their numerical conventions can
    be checked independently; this function is the full metric consumed by
    ``solve_cderi`` and ``grad_cderi``.
    '''
    return (
        short_range_metric(auxcell, omega, q)
        + long_range_metric(cell, auxcell, omega, mesh, q)
        + g0_metric(cell, auxcell, omega, q)
    )


def grad_auxiliary_metric(cell, auxcell, omega, mesh, q=None, atmlst=None):
    r'''Return the full fixed-cell metric derivative :math:`M^\lambda`.

    The :math:`G=0` term depends on auxiliary charges and the fixed cell volume,
    neither of which changes when an AO center moves.  Its nuclear derivative
    is therefore exactly zero and does not need a numerical allocation here.
    '''
    metric1 = gen_grad_auxiliary_metric(
        cell, auxcell, omega, mesh, q)
    return metric1(atmlst)


def gen_grad_auxiliary_metric(cell, auxcell, omega, mesh, q=None):
    r'''Generate atom-block assembly of the complete :math:`M^\lambda`.'''
    short_range1 = gen_grad_short_range_metric(auxcell, omega, q)
    long_range1 = gen_grad_long_range_metric(
        cell, auxcell, omega, mesh, q)

    def metric1(atmlst=None):
        return short_range1(atmlst) + long_range1(atmlst)

    return metric1


@contextmanager
def _short_range_context(auxcell, omega):
    '''Temporarily apply PySCF's short-range operator and converged cutoff.'''
    omega = abs(float(omega))
    precision = float(auxcell.precision) ** 1.5
    rcut = estimate_rs_2c2e_rcut(auxcell, omega, precision)
    rcut0 = float(auxcell.rcut)
    try:
        auxcell.rcut = rcut
        with auxcell.with_short_range_coulomb(omega):
            yield
    finally:
        auxcell.rcut = rcut0


def _real_at_gamma(value, q):
    value = np.asarray(value)
    return value.real if is_zero(q) else value


def _qpoint(q):
    q = np.zeros(3) if q is None else np.asarray(q, dtype=float)
    if q.shape != (3,):
        raise ValueError(f'q must have shape (3,), got {q.shape}')
    return q
