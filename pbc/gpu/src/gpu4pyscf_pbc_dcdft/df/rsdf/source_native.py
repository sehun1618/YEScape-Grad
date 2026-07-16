"""Public orchestration for native RSDF source derivative contractions."""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter
from pyscf.pbc.tools.k2gamma import double_translation_indices
from pyscf_pbc_dcdft.df.gdf.storage import canonical_pairs

from . import source_reciprocal, source_short_range


def build_source_optimizer(cell, auxcell, parameters, *, bvk_kmesh=None):
    """Build an SR optimizer with the exact RSDF source split settings."""

    working_cell = cell.copy()
    working_auxcell = auxcell.copy()
    working_cell.precision = float(parameters.precision_r)
    working_auxcell.precision = float(parameters.precision_r)
    optimizer = SRInt3c2eOpt(
        working_cell,
        working_auxcell,
        float(parameters.source_omega),
        bvk_kmesh,
    )
    optimizer.mesh = np.asarray(parameters.source_mesh, dtype=int)
    return optimizer.build()


def contract_gamma_sources(optimizer, source_adjoints):
    """Contract several Gamma source adjoints in one shared traversal."""

    _require_gamma_optimizer(optimizer)
    sources = _gamma_source_batch(optimizer, source_adjoints)
    pair_addresses = optimizer.pair_and_diag_indices(
        cart=True,
        original_ao_order=False,
    )[0]
    short_range = source_short_range.contract_gamma_batch(
        optimizer,
        sources,
        pair_addresses,
    )
    reciprocal = source_reciprocal.contract_gamma_batch(
        optimizer,
        sources,
    )
    return short_range + reciprocal


def contract_kpoint_sources(optimizer, kpts, source_adjoint_groups):
    """Contract several sampled-k adjoint maps in one shared traversal."""

    points = _kpoint_array(kpts)
    groups = tuple(source_adjoint_groups)
    if not groups:
        raise ValueError("at least one sampled-k source adjoint map is required")
    ordered = cp.stack(
        [
            _ordered_sorted_sources(optimizer, points, group)
            for group in groups
        ]
    )
    momentum = _build_momentum_context(optimizer, points)
    pair_addresses = optimizer.pair_and_diag_indices(
        cart=True,
        original_ao_order=False,
    )[0]
    short_range = source_short_range.contract_kpoint_batch(
        optimizer,
        ordered,
        momentum,
        pair_addresses,
    )
    reciprocal = source_reciprocal.contract_kpoint_batch(
        optimizer,
        ordered,
        momentum,
        points,
    )
    return short_range + reciprocal


def contract_gamma_short_range_source(optimizer, source_adjoint):
    """Contract Gamma ``dB_SR`` with one source adjoint on the GPU."""

    _require_gamma_optimizer(optimizer)
    source = _sorted_gamma_source(optimizer, source_adjoint)
    pair_addresses = optimizer.pair_and_diag_indices(
        cart=True,
        original_ao_order=False,
    )[0]
    return source_short_range.contract_gamma(
        optimizer,
        source,
        pair_addresses,
    )


def contract_gamma_reciprocal_source(optimizer, source_adjoint):
    """Contract Gamma ``dB_LR + dB_vbar`` on the GPU.

    GPU4PySCF folds the zero-transfer ``vbar`` correction into the ``G=0``
    Coulomb weight.  The CPU reference stores it separately; their sum is
    the parity contract for this routine.
    """

    _require_gamma_optimizer(optimizer)
    source = _sorted_gamma_source(optimizer, source_adjoint)
    return source_reciprocal.contract_gamma(optimizer, source)


def contract_kpoint_short_range_source(
    optimizer,
    kpts,
    source_adjoints,
):
    """Contract sampled-k short-range source adjoints in one traversal."""

    points, ordered, momentum = _kpoint_inputs(
        optimizer,
        kpts,
        source_adjoints,
    )
    del points
    pair_addresses = optimizer.pair_and_diag_indices(
        cart=True,
        original_ao_order=False,
    )[0]
    return source_short_range.contract_kpoint(
        optimizer,
        ordered,
        momentum,
        pair_addresses,
    )


def contract_kpoint_reciprocal_source(
    optimizer,
    kpts,
    source_adjoints,
):
    """Contract sampled-k ``dB_LR + dB_vbar`` in one traversal."""

    points, ordered, momentum = _kpoint_inputs(
        optimizer,
        kpts,
        source_adjoints,
    )
    return source_reciprocal.contract_kpoint(
        optimizer,
        ordered,
        momentum,
        points,
    )


def _require_gamma_optimizer(optimizer):
    if int(np.prod(optimizer.bvk_kmesh)) != 1:
        raise ValueError(
            "Gamma source contraction requires a 1 x 1 x 1 BvK mesh"
        )


def _sorted_gamma_source(optimizer, source_adjoint):
    source = cp.asarray(source_adjoint)
    imaginary = float(cp.max(cp.abs(source.imag)).get())
    if imaginary > 1.0e-10:
        raise ValueError(
            "Gamma source adjoint must be real; maximum imaginary "
            f"component is {imaginary:.3e}"
        )
    return _sorted_source(optimizer, source.real)


def _gamma_source_batch(optimizer, source_adjoints):
    adjoints = tuple(source_adjoints)
    if not adjoints:
        raise ValueError("at least one Gamma source adjoint is required")
    return cp.stack(
        [_sorted_gamma_source(optimizer, adjoint) for adjoint in adjoints]
    )


def _sorted_source(optimizer, source_adjoint):
    nao = int(optimizer.cell.cell.nao_nr())
    naux = int(optimizer.auxcell.cell.nao_nr())
    source = cp.asarray(source_adjoint)
    if source.shape != (nao**2, naux):
        raise ValueError(
            "source adjoint must have shape "
            f"({nao**2}, {naux}), got {source.shape}"
        )
    source = source.reshape(nao, nao, naux).transpose(2, 0, 1)
    source = optimizer.cell.apply_C_mat_CT(source)
    source = optimizer.auxcell.apply_C_dot(source, axis=0)
    return cp.asarray(source.transpose(1, 2, 0), order="C")


def _kpoint_inputs(optimizer, kpts, source_adjoints):
    points = _kpoint_array(kpts)
    ordered = _ordered_sorted_sources(
        optimizer,
        points,
        source_adjoints,
    )
    return points, ordered, _build_momentum_context(optimizer, points)


def _ordered_sorted_sources(optimizer, kpts, source_adjoints):
    nkpts = len(kpts)
    sorted_nao = int(optimizer.cell.nao)
    sorted_naux = int(optimizer.auxcell.nao)
    ordered = cp.zeros(
        (nkpts, nkpts, sorted_nao, sorted_nao, sorted_naux),
        dtype=cp.complex128,
    )
    expected_pairs = set(canonical_pairs(nkpts))
    supplied_pairs = {
        (int(pair[0]), int(pair[1])) for pair in source_adjoints
    }
    if supplied_pairs != expected_pairs:
        raise ValueError(
            "source adjoints must contain every canonical k-point pair; "
            f"expected {sorted(expected_pairs)}, got {sorted(supplied_pairs)}"
        )
    for pair, adjoint in source_adjoints.items():
        first, second = int(pair[0]), int(pair[1])
        source = _sorted_source(optimizer, adjoint)
        if first == second:
            ordered[first, second] = source
        else:
            ordered[first, second] = 0.5 * source
            ordered[second, first] = (
                0.5 * source.conj().transpose(1, 0, 2)
            )
    return ordered


@dataclass(frozen=True)
class _MomentumContext:
    iterations: tuple
    unique_kpts: np.ndarray
    ijk_conserv: cp.ndarray
    exp_lk: cp.ndarray
    exp_lk_conj: cp.ndarray


def _build_momentum_context(optimizer, kpts):
    nkpts = len(kpts)
    bvk_ncells = len(optimizer.bvkmesh_Ls)
    if bvk_ncells != nkpts:
        raise ValueError(
            "sampled-k source contraction requires a complete BvK mesh: "
            f"{bvk_ncells} cells for {nkpts} k-points"
        )
    exp_lk = cp.exp(1j * cp.asarray(optimizer.bvkmesh_Ls.dot(kpts.T)))
    iterations = tuple(kk_adapted_iter(optimizer.bvk_kmesh))
    return _MomentumContext(
        iterations=iterations,
        unique_kpts=kpts[[item[0] for item in iterations]],
        ijk_conserv=cp.asarray(
            double_translation_indices(optimizer.bvk_kmesh)
        ),
        exp_lk=exp_lk,
        exp_lk_conj=exp_lk.conj(),
    )


def _kpoint_array(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"k-points must have shape (nkpts,3), got {points.shape}"
        )
    return points
