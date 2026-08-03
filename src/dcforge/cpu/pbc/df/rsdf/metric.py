r'''Public RSDF auxiliary metric and exact screened derivative.'''

import ctypes

import numpy as np
from pyscf import gto as mol_gto
from pyscf import lib
from pyscf.pbc.df import rsdf_helper
from pyscf.pbc.df.incore import libpbc

from dcforge.cpu.pbc.df.gaussian.metric import (
    g0_metric,
    gen_grad_long_range_metric,
    long_range_metric,
)


def auxiliary_metric(cell, auxcell, omega, mesh):
    r'''Return the exact public-RSDF metric :math:`M` at Gamma.

    RSDF's two-center short-range value is produced by
    ``rsdf_helper.intor_j2c``.  It is combined with the reciprocal long-range
    term and the zero-mode auxiliary-charge correction using the independently
    resolved ``omega_j2c`` and ``mesh_j2c`` settings.
    '''
    short_range = np.asarray(rsdf_helper.intor_j2c(
        auxcell, float(omega), kpts=np.zeros((1, 3))))[0]
    return (
        short_range
        + long_range_metric(cell, auxcell, omega, mesh)
        + g0_metric(cell, auxcell, omega)
    )


def grad_auxiliary_metric(cell, auxcell, omega, mesh, atmlst=None):
    r'''Return the exact fixed-cell RSDF metric derivative :math:`M^\lambda`.

    The short-range derivative uses the same screened ``PBCsr2c`` image list,
    cutoff matrix, range operator, and auxiliary precision as
    ``rsdf_helper.intor_j2c``.  Using the ordinary ``pbc_intor`` derivative is
    close but not representation-identical and leaves a measurable CDERI
    response error.
    '''
    metric1 = gen_grad_auxiliary_metric(cell, auxcell, omega, mesh)
    return metric1(atmlst)


def gen_grad_auxiliary_metric(cell, auxcell, omega, mesh):
    r'''Generate atom-block assembly of exact RSDF :math:`M^\lambda`.'''
    short_range1 = gen_grad_short_range_metric(auxcell, omega)
    long_range1 = gen_grad_long_range_metric(
        cell, auxcell, omega, mesh)

    def metric1(atmlst=None):
        return short_range1(atmlst) + long_range1(atmlst)

    return metric1


def grad_short_range_metric(auxcell, omega, atmlst=None):
    '''Return all atom-resolved derivatives of the screened RSDF metric.'''
    metric1 = gen_grad_short_range_metric(auxcell, omega)
    return metric1(atmlst)


def short_range_metric_coordinate_derivatives(auxcell, omega):
    r'''Return exact screened RSDF primitive ``ip1`` and ``ip2`` arrays.

    Args:
        auxcell : pyscf.pbc.gto.Cell
            Built auxiliary cell in the public RSDF AO ordering.
        omega : float
            Positive RSDF metric range-separation parameter.

    Returns:
        tuple
            ``(ip1, ip2)``.  ``ip1[x,p,q]`` differentiates the first
            auxiliary center and ``ip2[x,p,q]`` differentiates the second.

    Notes:
        This is intentionally lower-level than :func:`grad_short_range_metric`.
        GPU response code transfers these two exact screened primitives, then
        performs the observable contraction and atom reduction on the device.
        It does not transfer an assembled all-atom ``dM`` tensor.
    '''
    return (
        _short_range_integral(auxcell, omega, 'int2c2e_ip1'),
        _short_range_integral(auxcell, omega, 'int2c2e_ip2'),
    )


def gen_grad_short_range_metric(auxcell, omega):
    r'''Generate atom blocks from one exact screened ``ip1/ip2`` traversal.'''
    ip1, ip2 = short_range_metric_coordinate_derivatives(auxcell, omega)
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
        return _real_if_close(value)

    return metric1


def _short_range_integral(auxcell, omega, intor):
    precision = float(auxcell.precision)
    ref_map, unique_atoms, unique_basis, unique_basis_loc = (
        rsdf_helper._get_refuniq_map(auxcell))
    rcuts = rsdf_helper._get_2c2e_Rcut(
        unique_basis, float(omega), precision,
        lmp=True, lasympt=True, eta_correct=True, R_correct=False)
    atom_rcuts = rsdf_helper._get_atom_Rcuts_2c(
        rcuts, unique_basis_loc)
    lattice_vectors = rsdf_helper._get_Lsmin(
        auxcell, atom_rcuts, unique_atoms)

    intor, comp = mol_gto.moleintor._get_intor_and_comp(
        auxcell._add_suffix(intor), None)
    pair_cell = auxcell.copy(deep=False)
    pair_cell._atm, pair_cell._bas, pair_cell._env = atm, basis, env = (
        mol_gto.conc_env(
            auxcell._atm, auxcell._bas, auxcell._env,
            auxcell._atm, auxcell._bas, auxcell._env))
    env[mol_gto.PTR_RANGE_OMEGA] = -abs(float(omega))

    i0, i1 = 0, auxcell.nbas
    j0, j1 = auxcell.nbas, 2 * auxcell.nbas
    ao_loc = mol_gto.moleintor.make_loc(basis, intor)
    ni = int(ao_loc[i1] - ao_loc[i0])
    nj = int(ao_loc[j1] - ao_loc[j0])
    output = np.empty((1, comp, ni, nj), dtype=np.complex128)
    phase = np.asarray(
        np.exp(1j * np.dot(np.zeros((1, 3)), lattice_vectors.T)), order='C')

    libpbc.PBCsr2c_k_drv(
        getattr(mol_gto.moleintor.libcgto, intor),
        libpbc.PBCsr2c_fill_ks1,
        output.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(1), ctypes.c_int(comp),
        ctypes.c_int(len(lattice_vectors)),
        lattice_vectors.ctypes.data_as(ctypes.c_void_p),
        phase.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int * 4)(i0, i1, j0, j1),
        ao_loc.ctypes.data_as(ctypes.c_void_p), lib.c_null_ptr(),
        ref_map.ctypes.data_as(ctypes.c_void_p),
        (rcuts ** 2).ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pair_cell.natm),
        basis.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pair_cell.nbas),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size),
    )
    return output[0]


def _real_if_close(value, tol=1e-10):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tol:
        return value.real
    return value
