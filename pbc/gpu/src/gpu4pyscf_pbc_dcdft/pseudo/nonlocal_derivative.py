"""GPU analytic matrix derivative of the nonlocal GTH pseudopotential.

GPU4PySCF constructs reciprocal GTH projector polynomials on the host and
transfers them before its device contractions. This module follows that same
documented boundary: small angular/radial projector tables are host setup;
Fourier AOs, projector--AO contraction, product-rule derivatives, and matrix
assembly are CuPy operations. No density-contracted CPU gradient is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.df.fft import get_SI
from gpu4pyscf.pbc.df.ft_ao import ft_ao
from pyscf import gto as molecular_gto
from pyscf.pbc.gto.pseudo import pp as gth_pp


@dataclass(frozen=True)
class NonlocalDerivativeResult:
    """Device-resident nonlocal GTH value and derivative channels."""

    matrix: cp.ndarray
    projector_center: cp.ndarray
    ao_center: cp.ndarray

    @property
    def derivative(self):
        return self.projector_center + self.ao_center


def _projector_molecule():
    molecule = molecular_gto.Mole()
    molecule._atm = np.zeros((1, molecular_gto.ATM_SLOTS), dtype=np.int32)
    molecule._bas = np.zeros((1, molecular_gto.BAS_SLOTS), dtype=np.int32)
    pointer = molecular_gto.PTR_ENV_START
    molecule._env = np.zeros(pointer + 10)
    molecule._bas[0, molecular_gto.NPRIM_OF] = 1
    molecule._bas[0, molecular_gto.NCTR_OF] = 1
    molecule._bas[0, molecular_gto.PTR_EXP] = pointer + 3
    molecule._bas[0, molecular_gto.PTR_COEFF] = pointer + 4
    return molecule, pointer


def _projectors_for_atom(
    cell,
    atom_id: int,
    g_plus_k_host,
    radial_host,
    helper,
    pointer: int,
):
    """Yield host-built projector tables and device coupling matrices."""

    symbol = cell.atom_symbol(atom_id)
    if symbol not in cell._pseudo:
        return
    for angular_momentum, channel in enumerate(cell._pseudo[symbol][5:]):
        radius, nprojectors, coupling = channel
        if nprojectors <= 0:
            continue
        helper._bas[0, molecular_gto.ANG_OF] = angular_momentum
        helper._env[pointer + 3] = 0.5 * radius**2
        helper._env[pointer + 4] = (
            radius ** (angular_momentum + 1.5) * np.pi**1.25
        )
        solid_harmonic = helper.eval_gto("GTOval", g_plus_k_host)
        projectors = np.empty(
            (nprojectors, 2 * angular_momentum + 1, len(g_plus_k_host)),
            dtype=np.complex128,
        )
        for projector_id in range(nprojectors):
            radial = gth_pp._qli(
                radial_host * radius,
                angular_momentum,
                projector_id,
            )
            projectors[projector_id] = solid_harmonic.T * radial
        yield cp.asarray(projectors), cp.asarray(coupling)


def _ao_center_derivative(cell, ao_fourier, g_plus_k, atom_id, axis):
    p0, p1 = np.asarray(cell.aoslice_by_atom(), dtype=int)[atom_id, 2:]
    derivative = cp.zeros_like(ao_fourier)
    derivative[:, p0:p1] = (
        -1j * g_plus_k[:, axis, None] * ao_fourier[:, p0:p1]
    )
    return derivative


def _differentiate(block, coupled, coupling, derivative):
    coupled_derivative = cp.einsum("ij,jmp->imp", coupling, derivative)
    return cp.einsum(
        "imp,imq->pq", derivative.conj(), coupled
    ) + cp.einsum(
        "imp,imq->pq", block.conj(), coupled_derivative
    )


def _one_kpoint_derivative(
    cell,
    kpt,
    gvectors_host,
    structure_factors,
    helper,
    pointer,
):
    g_plus_k_host = gvectors_host + kpt
    radial_host = np.linalg.norm(g_plus_k_host, axis=1)
    g_plus_k = cp.asarray(g_plus_k_host)
    gvectors = cp.asarray(gvectors_host)
    ao_fourier = ft_ao(cell, gvectors_host, kpt=kpt) / np.sqrt(cell.vol)
    nao = int(cell.nao_nr())
    matrix = cp.zeros((nao, nao), dtype=cp.complex128)
    projector_center = cp.zeros(
        (cell.natm, 3, nao, nao), dtype=cp.complex128
    )
    ao_center = cp.zeros_like(projector_center)

    for pseudo_atom in range(cell.natm):
        channels = _projectors_for_atom(
            cell,
            pseudo_atom,
            g_plus_k_host,
            radial_host,
            helper,
            pointer,
        )
        for projectors, coupling in channels:
            phased = structure_factors[pseudo_atom].conj() * projectors
            block = cp.einsum("nmg,gp->nmp", phased, ao_fourier)
            coupled = cp.einsum("ij,jmp->imp", coupling, block)
            matrix += cp.einsum("imp,imq->pq", block.conj(), coupled)

            for atom_id in range(cell.natm):
                for axis in range(3):
                    if atom_id == pseudo_atom:
                        projector_derivative = cp.einsum(
                            "nmg,gp->nmp",
                            1j * gvectors[:, axis] * phased,
                            ao_fourier,
                        )
                        projector_center[atom_id, axis] += _differentiate(
                            block,
                            coupled,
                            coupling,
                            projector_derivative,
                        )
                    ao_derivative = _ao_center_derivative(
                        cell, ao_fourier, g_plus_k, atom_id, axis
                    )
                    derivative_block = cp.einsum(
                        "nmg,gp->nmp", phased, ao_derivative
                    )
                    ao_center[atom_id, axis] += _differentiate(
                        block,
                        coupled,
                        coupling,
                        derivative_block,
                    )
    return NonlocalDerivativeResult(matrix, projector_center, ao_center)


def nonlocal_gth_derivative(cell, kpts):
    """Return GPU nonlocal GTH values and ``dV_nl/dR_Ax`` matrices.

    Returns
    -------
    NonlocalDerivativeResult
        ``matrix`` has shape ``(Nk,nao,nao)``. Derivative channels have shape
        ``(Natom,Nk,3,nao,nao)`` and units Hartree/Bohr.
    """

    kpoint_array = np.asarray(kpts, dtype=float).reshape(-1, 3)
    gvectors_host = cell.get_Gv(cell.mesh)
    structure_factors = get_SI(cell, mesh=cell.mesh)
    nao = int(cell.nao_nr())
    helper, pointer = _projector_molecule()
    matrices = cp.zeros(
        (len(kpoint_array), nao, nao), dtype=cp.complex128
    )
    projector_center = cp.zeros(
        (cell.natm, len(kpoint_array), 3, nao, nao), dtype=cp.complex128
    )
    ao_center = cp.zeros_like(projector_center)

    for k_index, kpt in enumerate(kpoint_array):
        result = _one_kpoint_derivative(
            cell,
            kpt,
            gvectors_host,
            structure_factors,
            helper,
            pointer,
        )
        matrices[k_index] = result.matrix
        projector_center[:, k_index] = result.projector_center
        ao_center[:, k_index] = result.ao_center
    scale = 1.0 / cell.vol
    return NonlocalDerivativeResult(
        matrices * scale,
        projector_center * scale,
        ao_center * scale,
    )
