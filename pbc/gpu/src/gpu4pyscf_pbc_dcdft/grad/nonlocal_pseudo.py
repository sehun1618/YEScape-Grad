"""Batched Gamma contraction of the nonlocal GTH derivative.

The projector assembly follows ``gpu4pyscf.pbc.grad.pp`` (Apache-2.0) while
retaining a leading density-set axis and a device-resident result.
"""

from __future__ import annotations

import warnings

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad
from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vnl
from pyscf.pbc.lib.kpts_helper import gamma_point


def contract_nonlocal_pseudo_batch(cell, densities, *, kpts=None):
    """Contract shared Gamma projector derivatives with several densities."""

    points = (
        np.zeros((1, 3))
        if kpts is None
        else np.asarray(kpts, dtype=float).reshape(-1, 3)
    )
    batch = cp.asarray(densities)
    if batch.ndim == 3:
        batch = batch[None]
    nao = int(cell.nao_nr())
    expected = (len(points), nao, nao)
    if batch.ndim != 4 or tuple(batch.shape[1:]) != expected:
        raise ValueError(
            "nonlocal pseudo density batch must have shape "
            f"(nset,{expected[0]},{nao},{nao}), got {batch.shape}"
        )
    if not gamma_point(points):
        return cp.stack(
            [cp.asarray(vppnl_nuc_grad(cell, dm, kpts=points)) for dm in batch]
        )
    return _contract_gamma_batch(cell, batch, points)


def _contract_gamma_batch(cell, densities, points):
    from gpu4pyscf.pbc.gto.pseudo.pp_int import _int_vnl_gpu

    fakecell, hl_blocks = fake_cell_vnl(cell)
    integral_names = (
        "int1e_ipovlp",
        "int1e_r2_origi_ip2",
        "int1e_r4_origi_ip2",
    )
    half = _int_vnl_gpu(cell, fakecell, hl_blocks, points)
    half_ip2 = _int_vnl_gpu(
        cell,
        fakecell,
        hl_blocks,
        points,
        integral_names,
        comp=3,
    )
    if len(half_ip2[0]) > 0:
        half_ip2[0][0] *= -1
    half = _to_device_integrals(half)
    half_ip2 = _to_device_integrals(half_ip2)

    dm = densities.real
    dm_hermitian = dm + dm.transpose(0, 1, 3, 2)
    gradient = cp.zeros(
        (len(dm), cell.natm, 3),
        dtype=cp.complex128,
    )
    derivative_matrix = cp.zeros(
        (1, 3, cell.nao_nr(), cell.nao_nr()),
        dtype=cp.complex128,
    )
    offsets = [0] * 3
    for block_id, coefficients in enumerate(hl_blocks):
        angular = fakecell.bas_angular(block_id)
        projector_count = 2 * angular + 1
        radial_count = coefficients.shape[0]
        value = cp.zeros(
            (radial_count, projector_count, cell.nao_nr()),
            dtype=cp.complex128,
        )
        derivative = cp.zeros(
            (radial_count, 3, projector_count, cell.nao_nr()),
            dtype=cp.complex128,
        )
        for radial_id in range(radial_count):
            start = offsets[radial_id]
            stop = start + projector_count
            if len(half[radial_id]) > 0:
                value[radial_id] = half[radial_id][0, start:stop]
            if len(half_ip2[radial_id]) > 0:
                derivative[radial_id] = half_ip2[radial_id][
                    0, :, start:stop
                ]
            offsets[radial_id] = stop
        derivative_block = cp.einsum(
            "idlp,ij,jlq->dpq",
            derivative.conj(),
            cp.asarray(coefficients),
            value,
        )
        derivative_matrix[0] += derivative_block
        atom_id = int(fakecell._bas[block_id, 0])
        gradient[:, atom_id] += cp.einsum(
            "dpq,sqp->sd",
            derivative_block,
            dm_hermitian[:, 0],
        )

    for atom_id, ao_slice in enumerate(cell.aoslice_by_atom()):
        start, stop = ao_slice[2:]
        gradient[:, atom_id] -= cp.einsum(
            "kdpq,skqp->sd",
            derivative_matrix[:, :, start:stop, :],
            dm_hermitian[:, :, :, start:stop],
        )
    maximum_imaginary = float(cp.max(cp.abs(gradient.imag)).get())
    if maximum_imaginary >= 1.0e-8:
        warnings.warn(
            "large imaginary nonlocal pseudopotential gradient component: "
            f"{maximum_imaginary:.3e}",
            RuntimeWarning,
            stacklevel=2,
        )
    return gradient.real


def _to_device_integrals(values):
    return [cp.asarray(value) if len(value) > 0 else value for value in values]
