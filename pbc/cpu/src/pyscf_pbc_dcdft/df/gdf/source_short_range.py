"""Short-range three-center source and center derivative for periodic GDF."""

from __future__ import annotations

import numpy as np
from pyscf import lib
from pyscf.pbc.df import rsdf_helper


def short_range_source(
    cell,
    auxcell,
    omega: float,
    *,
    kpti: np.ndarray,
    kptj: np.ndarray,
) -> np.ndarray:
    """Return ``B_SR(ki,kj)`` with shape ``(nao**2, naux)``."""

    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    helper_symmetry = "s2" if _absolute_gamma_pair(ki, kj) else "s1"
    tensor = _helper_tensor(
        cell,
        auxcell,
        omega,
        kpti=ki,
        kptj=kj,
        intor="int3c2e",
        aosym=helper_symmetry,
    )
    if helper_symmetry == "s2":
        tensor = _unpack_packed_pair(tensor, cell.nao_nr())
    if tensor.shape[0] == 1:
        tensor = tensor[0]
    return np.asarray(tensor).reshape(cell.nao_nr() ** 2, auxcell.nao_nr())


def short_range_source_derivative(
    cell,
    auxcell,
    atom_id: int,
    omega: float,
    *,
    kpti: np.ndarray,
    kptj: np.ndarray,
    precision: float | None = None,
) -> np.ndarray:
    """Return ``dB_SR/dR_A`` as ``(3, nao**2, naux)``.

    ``int3c2e_ip1`` differentiates the first primary AO and
    ``int3c2e_ip2`` differentiates the auxiliary AO.  The second primary AO
    derivative is reconstructed from the swapped k-point pair and Hermitian
    AO-pair symmetry.  Integral derivatives are electron-coordinate
    derivatives, so nuclear-center derivatives carry a minus sign.
    """

    atom = _atom_id(cell, auxcell, atom_id)
    ki = _kpoint(kpti)
    kj = _kpoint(kptj)
    if _absolute_gamma_pair(ki, kj):
        return _gamma_derivative(
            cell, auxcell, atom, omega, precision=precision
        )

    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    ip1 = _helper_tensor(
        cell,
        auxcell,
        omega,
        kpti=ki,
        kptj=kj,
        intor="int3c2e_ip1",
        aosym="s1",
        precision=precision,
    ).reshape(3, nao, nao, naux)
    ip2 = _helper_tensor(
        cell,
        auxcell,
        omega,
        kpti=ki,
        kptj=kj,
        intor="int3c2e_ip2",
        aosym="s1",
        precision=precision,
    ).reshape(3, nao, nao, naux)
    swapped_ip1 = _helper_tensor(
        cell,
        auxcell,
        omega,
        kpti=kj,
        kptj=ki,
        intor="int3c2e_ip1",
        aosym="s1",
        precision=precision,
    ).reshape(3, nao, nao, naux)

    derivative = np.zeros_like(ip1)
    p0, p1 = _ao_slice(cell, atom)
    q0, q1 = _ao_slice(auxcell, atom)
    derivative[:, p0:p1, :, :] -= ip1[:, p0:p1, :, :]
    derivative[:, :, p0:p1, :] -= np.conj(
        np.swapaxes(swapped_ip1[:, p0:p1, :, :], 1, 2)
    )
    derivative[:, :, :, q0:q1] -= ip2[:, :, :, q0:q1]
    return _real_if_close(derivative.reshape(3, nao * nao, naux))


def _gamma_derivative(cell, auxcell, atom_id, omega, *, precision=None):
    """Return exact Gamma derivatives from PySCF's unpacked k driver.

    ``rsdf_helper`` forces packed ``s2`` storage when *all* supplied pairs are
    Gamma.  Packed ``int3c2e_ip1`` does not retain which primary AO center was
    differentiated when a shell pair spans two atoms.  Supplying a second,
    unused diagonal non-Gamma pair selects the general ``k`` driver and yields
    an exact, unpacked ``s1`` block for the first (still exactly Gamma) pair.
    This is a storage-layout choice, not a small-k approximation.
    """

    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    ip1 = _gamma_s1_tensor(
        cell,
        auxcell,
        omega,
        intor="int3c2e_ip1",
        precision=precision,
    ).reshape(3, nao, nao, naux)
    ip2 = _gamma_s1_tensor(
        cell,
        auxcell,
        omega,
        intor="int3c2e_ip2",
        precision=precision,
    ).reshape(3, nao, nao, naux)
    derivative = np.zeros_like(ip1)
    p0, p1 = _ao_slice(cell, atom_id)
    q0, q1 = _ao_slice(auxcell, atom_id)
    derivative[:, p0:p1, :, :] -= ip1[:, p0:p1, :, :]
    derivative[:, :, p0:p1, :] -= np.conj(
        np.swapaxes(ip1[:, p0:p1, :, :], 1, 2)
    )
    derivative[:, :, :, q0:q1] -= ip2[:, :, :, q0:q1]
    return _real_if_close(derivative.reshape(3, nao * nao, naux))


def _gamma_s1_tensor(cell, auxcell, omega, *, intor, precision=None):
    """Evaluate one exact Gamma block with unpacked AO-pair ownership."""

    gamma = np.zeros(3)
    dummy = 0.173 * np.asarray(cell.reciprocal_vectors())[0]
    pair_list = np.asarray([[gamma, gamma], [dummy, dummy]], dtype=float)
    temporary = lib.H5TmpFile()
    try:
        rsdf_helper._aux_e2_nospltbas(
            cell,
            auxcell,
            abs(float(omega)),
            temporary,
            intor=intor,
            aosym="s1",
            dataname="j3c",
            kptij_lst=pair_list,
            max_memory=2000,
            precision=(cell.precision if precision is None else precision),
        )
        group = temporary["j3c/0"]
        keys = sorted(group.keys(), key=lambda key: int(key))
        return np.concatenate([np.asarray(group[key]) for key in keys], axis=-1)
    finally:
        temporary.close()


def _helper_tensor(
    cell,
    auxcell,
    omega,
    *,
    kpti,
    kptj,
    intor,
    aosym,
    precision=None,
):
    temporary = lib.H5TmpFile()
    try:
        rsdf_helper._aux_e2_nospltbas(
            cell,
            auxcell,
            abs(float(omega)),
            temporary,
            intor=intor,
            aosym=aosym,
            dataname="j3c",
            kptij_lst=np.asarray([[kpti, kptj]], dtype=float),
            max_memory=2000,
            precision=(cell.precision if precision is None else precision),
        )
        group = temporary["j3c/0"]
        keys = sorted(group.keys(), key=lambda key: int(key))
        return np.concatenate([np.asarray(group[key]) for key in keys], axis=-1)
    finally:
        temporary.close()


def _unpack_packed_pair(tensor, nao):
    raw = np.asarray(tensor)
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"unexpected packed source shape {raw.shape}")
    full = lib.unpack_tril(raw[0].T).transpose(1, 2, 0)
    return full.reshape(1, int(nao) ** 2, raw.shape[-1])


def _ao_slice(cell, atom_id):
    slices = np.asarray(cell.aoslice_nr_by_atom(), dtype=int)
    return int(slices[atom_id, 2]), int(slices[atom_id, 3])


def _atom_id(cell, auxcell, atom_id):
    atom = int(atom_id)
    if atom < 0 or atom >= cell.natm or atom >= auxcell.natm:
        raise ValueError(f"atom_id out of range: {atom_id}")
    return atom


def _kpoint(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"k-point must have shape (3,), got {point.shape}")
    return point


def _absolute_gamma_pair(kpti, kptj):
    return np.linalg.norm(kpti) < 1e-10 and np.linalg.norm(kptj) < 1e-10


def _real_if_close(array, tolerance=1e-10):
    value = np.asarray(array)
    if value.size and np.max(np.abs(value.imag)) <= tolerance:
        return value.real
    return value
