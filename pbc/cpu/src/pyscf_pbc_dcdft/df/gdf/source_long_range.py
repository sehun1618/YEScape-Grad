"""Long-range reciprocal-space three-center source for periodic GDF."""

from __future__ import annotations

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.df import ft_ao, rsdf
from pyscf.pbc.tools import k2gamma

from .ft_derivative import gen_ft_center_derivative_kernel


def long_range_source(
    cell,
    auxcell,
    omega: float,
    *,
    mesh: tuple[int, int, int] | np.ndarray,
    kpti: np.ndarray,
    kptj: np.ndarray,
) -> np.ndarray:
    """Return ``B_LR(ki,kj)`` with shape ``(nao**2, naux)``."""

    context = _context(cell, auxcell, omega, mesh, kpti, kptj)
    ao_pair = ft_ao.ft_aopair(
        cell,
        context["Gv"],
        aosym="s1",
        b=context["b"],
        gxyz=context["gxyz"],
        Gvbase=context["Gvbase"],
        kpti_kptj=np.asarray([context["ki"], context["kj"]]),
        q=context["q"],
        intor="GTO_ft_ovlp",
        comp=1,
    )
    weighted_aux = (context["auxG"] * context["coulG"]).T
    return ao_pair.reshape(len(context["Gv"]), -1).T @ weighted_aux.conj()


def long_range_source_derivative(
    cell,
    auxcell,
    atom_id: int,
    omega: float,
    *,
    mesh: tuple[int, int, int] | np.ndarray,
    kpti: np.ndarray,
    kptj: np.ndarray,
) -> np.ndarray:
    """Return AO-center plus auxiliary-center ``dB_LR/dR_A``."""

    atom = _validate_atom(cell, auxcell, atom_id)
    context = _context(cell, auxcell, omega, mesh, kpti, kptj)
    ao_derivative = _ao_pair_derivative(cell, atom, context)
    aux_derivative = _auxiliary_derivative(cell, auxcell, atom, context)
    return ao_derivative + aux_derivative


def long_range_source_derivatives(
    cell,
    auxcell,
    omega: float,
    *,
    mesh: tuple[int, int, int] | np.ndarray,
    kpti: np.ndarray,
    kptj: np.ndarray,
) -> np.ndarray:
    """Return all-atom ``dB_LR/dR`` while reusing one Fourier context."""

    context = _context(cell, auxcell, omega, mesh, kpti, kptj)
    ao_derivatives = _all_ao_pair_derivatives(cell, context)
    auxiliary_derivatives = _all_auxiliary_derivatives(
        cell, auxcell, context
    )
    return _real_if_close(ao_derivatives + auxiliary_derivatives)


def _all_ao_pair_derivatives(cell, context):
    supmol = _fourier_supermolecule(cell, context["kj"])
    kernel_i = gen_ft_center_derivative_kernel(supmol, center="i")
    kernel_j = gen_ft_center_derivative_kernel(supmol, center="j")
    common = (
        context["Gv"],
        context["gxyz"],
        context["Gvbase"],
        context["q"],
    )
    raw_i = kernel_i(*common, kptjs=np.asarray([context["kj"]]))
    raw_j = kernel_j(*common, kptjs=np.asarray([context["kj"]]))
    ip_i = raw_i[0][0] + 1j * raw_i[1][0]
    ip_j = raw_j[0][0] + 1j * raw_j[1][0]
    nao = int(cell.nao_nr())
    naux = int(context["auxG"].shape[0])
    ngrid = len(context["Gv"])
    weighted_aux = (context["auxG"] * context["coulG"]).T.conj()
    result = np.zeros(
        (int(cell.natm), 3, nao, nao, naux), dtype=np.complex128
    )
    for atom_id in range(cell.natm):
        p0, p1 = _ao_slice(cell, atom_id)
        count = p1 - p0
        for axis in range(3):
            rows = ip_i[axis, :, p0:p1, :].reshape(
                ngrid, count * nao
            )
            result[atom_id, axis, p0:p1, :, :] -= (
                rows.T @ weighted_aux
            ).reshape(count, nao, naux)
            columns = ip_j[axis, :, :, p0:p1].reshape(
                ngrid, nao * count
            )
            result[atom_id, axis, :, p0:p1, :] -= (
                columns.T @ weighted_aux
            ).reshape(nao, count, naux)
    return result.reshape(int(cell.natm), 3, nao * nao, naux)


def _all_auxiliary_derivatives(cell, auxcell, context):
    ao_pair = ft_ao.ft_aopair(
        cell,
        context["Gv"],
        aosym="s1",
        b=context["b"],
        gxyz=context["gxyz"],
        Gvbase=context["Gvbase"],
        kpti_kptj=np.asarray([context["ki"], context["kj"]]),
        q=context["q"],
        intor="GTO_ft_ovlp",
        comp=1,
    ).reshape(len(context["Gv"]), cell.nao_nr() ** 2)
    nao_pair = int(cell.nao_nr()) ** 2
    naux = int(auxcell.nao_nr())
    momentum = context["Gv"] + context["q"]
    result = np.zeros(
        (int(cell.natm), 3, nao_pair, naux), dtype=np.complex128
    )
    for atom_id in range(cell.natm):
        p0, p1 = _ao_slice(auxcell, atom_id)
        for axis in range(3):
            weighted = (
                -1j
                * momentum[:, axis][None, :]
                * context["auxG"][p0:p1]
                * context["coulG"][None, :]
            ).T
            result[atom_id, axis, :, p0:p1] = (
                ao_pair.T @ weighted.conj()
            )
    return result


def _ao_pair_derivative(cell, atom_id, context):
    supmol = _fourier_supermolecule(cell, context["kj"])
    kernel_i = gen_ft_center_derivative_kernel(supmol, center="i")
    kernel_j = gen_ft_center_derivative_kernel(supmol, center="j")
    common = (
        context["Gv"],
        context["gxyz"],
        context["Gvbase"],
        context["q"],
    )
    raw_i = kernel_i(*common, kptjs=np.asarray([context["kj"]]))
    raw_j = kernel_j(*common, kptjs=np.asarray([context["kj"]]))
    ip_i = raw_i[0][0] + 1j * raw_i[1][0]
    ip_j = raw_j[0][0] + 1j * raw_j[1][0]
    p0, p1 = _ao_slice(cell, atom_id)
    nao = int(cell.nao_nr())
    weighted_aux = (context["auxG"] * context["coulG"]).T
    result = []
    for axis in range(3):
        derivative = np.zeros(
            (len(context["Gv"]), nao, nao), dtype=np.complex128
        )
        derivative[:, p0:p1, :] -= ip_i[axis, :, p0:p1, :]
        derivative[:, :, p0:p1] -= ip_j[axis, :, :, p0:p1]
        result.append(
            derivative.reshape(len(context["Gv"]), nao * nao).T
            @ weighted_aux.conj()
        )
    return np.asarray(result)


def _auxiliary_derivative(cell, auxcell, atom_id, context):
    ao_pair = ft_ao.ft_aopair(
        cell,
        context["Gv"],
        aosym="s1",
        b=context["b"],
        gxyz=context["gxyz"],
        Gvbase=context["Gvbase"],
        kpti_kptj=np.asarray([context["ki"], context["kj"]]),
        q=context["q"],
        intor="GTO_ft_ovlp",
        comp=1,
    ).reshape(len(context["Gv"]), cell.nao_nr() ** 2)
    p0, p1 = _ao_slice(auxcell, atom_id)
    momentum = context["Gv"] + context["q"]
    result = []
    for axis in range(3):
        derivative = np.zeros_like(context["auxG"])
        derivative[p0:p1] = (
            -1j
            * momentum[:, axis][None, :]
            * context["auxG"][p0:p1]
        )
        weighted = (derivative * context["coulG"]).T
        result.append(ao_pair.T @ weighted.conj())
    return np.asarray(result)


def _context(cell, auxcell, omega, mesh, kpti, kptj):
    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    q = kj - ki
    mesh_array = _mesh(mesh)
    Gv, Gvbase, _weights = cell.get_Gv_weights(mesh_array)
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(values)) for values in Gvbase])
    provider = rsdf.RSGDF(cell, kpts=np.asarray([ki, kj]))
    provider.auxcell = auxcell
    provider.omega = float(omega)
    provider.mesh_compact = list(mesh_array)
    coulomb = provider.weighted_coulG(q, mesh=mesh_array)
    coulomb -= provider.weighted_coulG(q, mesh=mesh_array, omega=-float(omega))
    auxG = ft_ao.ft_ao(auxcell, Gv, None, b, gxyz, Gvbase, q).T
    return {
        "ki": ki,
        "kj": kj,
        "q": q,
        "Gv": Gv,
        "Gvbase": Gvbase,
        "b": b,
        "gxyz": gxyz,
        "coulG": coulomb,
        "auxG": auxG,
    }


def _fourier_supermolecule(cell, kptj):
    log = logger.new_logger(cell, 0)
    rs_cell = ft_ao._RangeSeparatedCell.from_cell(
        cell, ft_ao.KECUT_THRESHOLD, ft_ao.RCUT_THRESHOLD, log
    )
    bvk_mesh = k2gamma.kpts_to_kmesh(cell, np.asarray([kptj]))
    cutoff = ft_ao.estimate_rcut(rs_cell)
    supmol = ft_ao.ExtendedMole.from_cell(
        rs_cell, bvk_mesh, cutoff.max(), log
    )
    return supmol.strip_basis(cutoff)


def _validate_atom(cell, auxcell, atom_id):
    atom = int(atom_id)
    if atom < 0 or atom >= cell.natm or atom >= auxcell.natm:
        raise ValueError(f"atom_id out of range: {atom_id}")
    return atom


def _ao_slice(cell, atom_id):
    slices = np.asarray(cell.aoslice_by_atom(), dtype=int)
    return int(slices[atom_id, 2]), int(slices[atom_id, 3])


def _mesh(value):
    array = np.asarray(value, dtype=int)
    if array.shape != (3,) or np.any(array <= 0):
        raise ValueError(f"mesh must contain three positive integers: {array}")
    return [int(item) for item in array]


def _kpoint(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"k-point must have shape (3,), got {point.shape}")
    return point


def _real_if_close(array, tolerance=1e-10):
    value = np.asarray(array)
    if value.size and np.max(np.abs(value.imag)) <= tolerance:
        return value.real
    return value
