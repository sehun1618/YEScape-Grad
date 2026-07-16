"""Source-contracted native GPU GDF nuclear derivatives.

GPU4PySCF exposes periodic GDF CUDA derivative kernels through its ordinary
RHF/KRHF gradient dispatcher, even though ``GDF.get_j_e1`` and
``GDF.get_k_e1`` are not public methods.  This module gives those kernels a
small stable project-local interface and supplies the bilinear contraction
needed by a non-variational DC-DFT Z-vector gradient.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from types import SimpleNamespace

import cupy as cp
import numpy as np
from gpu4pyscf.gto import mole as gpu_mole
from gpu4pyscf.pbc.df.grad import krhf as gpu_gdf_grad_krhf
from gpu4pyscf.pbc.df.grad import rhf as gpu_gdf_grad_rhf
from gpu4pyscf.pbc.df.int3c2e import POOL_SIZE
from gpu4pyscf.pbc.grad.rhf import jk_energy_per_atom


_DERIVATIVE_BATCH_PATCH_LOCK = RLock()
_GAMMA_DERIVATIVE_OMEGA_SCALE = 1.5


def _build_gamma_derivative_optimizer(provider):
    """Build the exact SR/LR split tuned for Gamma derivative throughput."""

    from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
    from gpu4pyscf.pbc.df.rsdf_builder import _guess_omega

    omega = _GAMMA_DERIVATIVE_OMEGA_SCALE * float(
        _guess_omega(provider.cell)
    )
    return SRInt3c2eOpt(
        provider.cell,
        provider.auxcell,
        omega,
        None,
    ).build()


@contextmanager
def _bounded_sr_derivative_batches(provider, kpts):
    """Keep every native SR derivative task batch within ``POOL_SIZE``.

    ``PBCsr_ejk_int3c2e_ip1`` stores one task for each product of an AO shell
    pair, an auxiliary shell, and a BvK cell.  GPU4PySCF 1.7.4 allocates a
    fixed ``POOL_SIZE`` task buffer but its RHF/KRHF derivative wrappers call
    ``aggregate_shl_pairs(..., 1000000)`` without accounting for the other two
    factors.  The resulting buffer overrun is non-deterministic for larger
    systems (for example, 109800 tasks for water20 in a 16384-entry pool).

    This narrow context bounds both batching axes before the upstream helper
    builds its internal ``SRInt3c2eOpt``.  The limits are conservative because
    ``auxcell.nbas`` is an upper bound on every angular-momentum shell group.
    """

    bvk_ncells = 1 if kpts is None else len(np.asarray(kpts).reshape(-1, 3))
    auxiliary_shells = max(1, int(provider.auxcell.nbas))
    auxiliary_limit = max(1, min(auxiliary_shells, POOL_SIZE // bvk_ncells))
    shell_pair_limit = max(
        1,
        POOL_SIZE // (auxiliary_limit * bvk_ncells),
    )

    with _DERIVATIVE_BATCH_PATCH_LOCK:
        original_aggregate = gpu_mole.SortedCell.aggregate_shl_pairs
        modules = (gpu_gdf_grad_rhf, gpu_gdf_grad_krhf)
        original_splitters = tuple(
            module._split_l_ctr_pattern for module in modules
        )

        def bounded_aggregate(cell, bas_ij_cache=None, nsp_per_block=512):
            if isinstance(nsp_per_block, (int, np.integer)):
                nsp_per_block = min(int(nsp_per_block), shell_pair_limit)
            else:
                nsp_per_block = np.minimum(nsp_per_block, shell_pair_limit)
            return original_aggregate(cell, bas_ij_cache, nsp_per_block)

        def bounded_splitter(l_ctr_offsets, uniq_l_ctr, batch_size):
            return original_splitters[0](
                l_ctr_offsets,
                uniq_l_ctr,
                min(int(batch_size), auxiliary_limit),
            )

        try:
            gpu_mole.SortedCell.aggregate_shl_pairs = bounded_aggregate
            for module in modules:
                module._split_l_ctr_pattern = bounded_splitter
            yield
        finally:
            gpu_mole.SortedCell.aggregate_shl_pairs = original_aggregate
            for module, original in zip(modules, original_splitters):
                module._split_l_ctr_pattern = original


def coulomb_energy_derivative(provider, density):
    """Return ``d[Tr(D J[D])/2]/dR`` from native GPU GDF kernels."""

    return _quadratic_energy_derivative(
        provider,
        density,
        j_factor=1.0,
        k_factor=0.0,
        exxdiv=None,
    )


def exchange_energy_derivative(provider, density, *, exxdiv="ewald"):
    """Return the positive restricted exchange-energy derivative.

    The upstream dispatcher evaluates ``J-K`` combinations.  Setting the
    Coulomb coefficient to zero therefore produces the negative exchange
    contribution, which is negated here for a positive ``K`` API.
    """

    return -_quadratic_energy_derivative(
        provider,
        density,
        j_factor=0.0,
        k_factor=1.0,
        exxdiv=exxdiv,
    )


def restricted_hf_energy_derivative(provider, density, *, exxdiv="ewald"):
    """Return the fixed-density restricted ``J-K/2`` energy derivative."""

    return _quadratic_energy_derivative(
        provider,
        density,
        j_factor=1.0,
        k_factor=1.0,
        exxdiv=exxdiv,
    )


def restricted_hf_bilinear_derivative(
    provider,
    reference_density,
    response_density,
    *,
    exxdiv="ewald",
):
    """Contract the GDF derivative between two Hermitian densities.

    If ``Q(D)`` is the quadratic fixed-density RHF two-electron derivative,
    the DC-DFT Z-vector contribution obeys the exact polarization identity

    ``response = Q(D + T) - Q(D - T)``.

    Here ``T`` is the Hermitian occupied--virtual transition density built
    from the adjoint Z-vector.  This avoids materializing the full five-index
    ``dCDERI`` tensor while retaining auxiliary metric and basis responses.
    """

    reference = cp.asarray(reference_density)
    response = cp.asarray(response_density)
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )
    plus = restricted_hf_energy_derivative(
        provider,
        reference + response,
        exxdiv=exxdiv,
    )
    minus = restricted_hf_energy_derivative(
        provider,
        reference - response,
        exxdiv=exxdiv,
    )
    return plus - minus


def restricted_hf_bilinear_derivative_one_pass_gamma(
    provider,
    reference_density,
    response_density,
    *,
    exxdiv="ewald",
    memory_diagnostics=None,
):
    """Evaluate the production Gamma bilinear derivative in one traversal.

    Sampled k-points use their separately validated momentum-aware one-pass
    implementation.  The two-pass polarization route remains available as a
    component-level reference, not as the default Gamma production path.
    """

    if not getattr(provider, "uses_native_integrals", False):
        raise RuntimeError(
            "one-pass Gamma GDF derivative requires setup_mode='native'"
        )
    provider._ensure_built()
    points = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    if len(points) != 1 or float(np.max(np.abs(points))) > 1.0e-12:
        raise NotImplementedError(
            "one-pass signed GDF derivative currently supports Gamma only"
        )

    from .signed_gamma_native import gamma_signed_energy_derivative

    optimizer = _build_gamma_derivative_optimizer(provider)
    with _bounded_sr_derivative_batches(provider, None):
        value = gamma_signed_energy_derivative(
            optimizer,
            reference_density,
            response_density,
            exxdiv=exxdiv,
            omega=0.0,
            memory_diagnostics=memory_diagnostics,
        )
    if memory_diagnostics is not None:
        memory_diagnostics["derivative_omega_scale"] = (
            _GAMMA_DERIVATIVE_OMEGA_SCALE
        )
    return cp.asarray(value, dtype=cp.float64)


def restricted_hf_bilinear_derivative_one_pass(
    provider,
    reference_density,
    response_density,
    *,
    exxdiv="ewald",
    memory_diagnostics=None,
):
    """Evaluate the signed RHF bilinear derivative in one native traversal.

    Gamma and sampled-k inputs have separate kernel implementations because
    sampled k-points require momentum-transfer and conjugate-pair maps.  This
    dispatcher gives the gradient layer one explicit optimization API.
    """

    if not getattr(provider, "uses_native_integrals", False):
        raise RuntimeError(
            "one-pass GDF derivative requires setup_mode='native'"
        )
    provider._ensure_built()
    points = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    if len(points) == 1:
        return restricted_hf_bilinear_derivative_one_pass_gamma(
            provider,
            reference_density,
            response_density,
            exxdiv=exxdiv,
            memory_diagnostics=memory_diagnostics,
        )

    from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
    from gpu4pyscf.pbc.df.rsdf_builder import _guess_omega
    from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh

    from .signed_kpoint_native import kpoint_signed_energy_derivative

    kmesh = kpts_to_kmesh(
        provider.cell,
        points,
        rcut=provider.cell.rcut + 10,
        bound_by_supmol=False,
    )
    optimizer = SRInt3c2eOpt(
        provider.cell,
        provider.auxcell,
        _guess_omega(provider.cell),
        kmesh,
    ).build()
    with _bounded_sr_derivative_batches(provider, points):
        value = kpoint_signed_energy_derivative(
            optimizer,
            reference_density,
            response_density,
            points,
            exxdiv=exxdiv,
            omega=0.0,
        )
    return cp.asarray(value, dtype=cp.float64)


def target_j_plus_restricted_hf_bilinear_derivative_one_pass(
    provider,
    reference_density,
    response_density,
    *,
    exxdiv="ewald",
    memory_diagnostics=None,
):
    """Fuse target-J and Z-bilinear GDF derivatives into one traversal.

    Only the sum is returned.  A single derivative traversal cannot recover
    the two physical channels independently without a multi-output CUDA ABI.
    The ordinary :func:`coulomb_energy_derivative` plus
    :func:`restricted_hf_bilinear_derivative_one_pass` route remains
    available when component-resolved diagnostics are required.
    """

    if not getattr(provider, "uses_native_integrals", False):
        raise RuntimeError("fused GDF derivative requires setup_mode='native'")
    provider._ensure_built()
    points = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)

    from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt
    from gpu4pyscf.pbc.df.rsdf_builder import _guess_omega

    if len(points) == 1:
        from .signed_gamma_native import (
            gamma_target_j_plus_signed_energy_derivative,
        )

        optimizer = _build_gamma_derivative_optimizer(provider)
        with _bounded_sr_derivative_batches(provider, None):
            value = gamma_target_j_plus_signed_energy_derivative(
                optimizer,
                reference_density,
                response_density,
                exxdiv=exxdiv,
                omega=0.0,
                memory_diagnostics=memory_diagnostics,
            )
        if memory_diagnostics is not None:
            memory_diagnostics["derivative_omega_scale"] = (
                _GAMMA_DERIVATIVE_OMEGA_SCALE
            )
        return cp.asarray(value, dtype=cp.float64)

    from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh

    from .signed_kpoint_native import (
        kpoint_target_j_plus_signed_energy_derivative,
    )

    kmesh = kpts_to_kmesh(
        provider.cell,
        points,
        rcut=provider.cell.rcut + 10,
        bound_by_supmol=False,
    )
    optimizer = SRInt3c2eOpt(
        provider.cell,
        provider.auxcell,
        _guess_omega(provider.cell),
        kmesh,
    ).build()
    with _bounded_sr_derivative_batches(provider, points):
        value = kpoint_target_j_plus_signed_energy_derivative(
            optimizer,
            reference_density,
            response_density,
            points,
            exxdiv=exxdiv,
            omega=0.0,
        )
    return cp.asarray(value, dtype=cp.float64)


def _quadratic_energy_derivative(
    provider,
    density,
    *,
    j_factor,
    k_factor,
    exxdiv,
):
    if not getattr(provider, "uses_native_integrals", False):
        raise RuntimeError(
            "native contracted GDF derivatives require setup_mode='native'"
        )
    provider._ensure_built()
    dm, kpts = _format_density(provider, density)
    if kpts is None:
        optimizer = _build_gamma_derivative_optimizer(provider)
        with _bounded_sr_derivative_batches(provider, None):
            value = gpu_gdf_grad_rhf._jk_energy_per_atom(
                optimizer,
                dm,
                hermi=1,
                j_factor=float(j_factor),
                k_factor=float(k_factor),
                exxdiv=exxdiv,
                omega=0.0,
            )
        return cp.asarray(value, dtype=cp.float64)

    method = SimpleNamespace(with_df=provider, rsjk=None)
    with _bounded_sr_derivative_batches(provider, kpts):
        value = jk_energy_per_atom(
            method,
            dm,
            kpts=kpts,
            j_factor=float(j_factor),
            lr_factor=float(k_factor),
            sr_factor=float(k_factor),
            omega=0.0,
            exxdiv=exxdiv,
        )
    return cp.asarray(value, dtype=cp.float64)


def _format_density(provider, density):
    dm = cp.asarray(density)
    nkpts = len(np.asarray(provider.kpts).reshape(-1, 3))
    if nkpts == 1:
        if dm.ndim == 3 and dm.shape[0] == 1:
            dm = dm[0]
        if dm.ndim != 2:
            raise ValueError(f"Gamma density must be two-dimensional, got {dm.shape}")
        imaginary_max = float(cp.max(cp.abs(dm.imag)).item())
        if imaginary_max > 1.0e-10:
            raise NotImplementedError(
                "native Gamma GDF derivative requires a real Hermitian density; "
                f"max imaginary component is {imaginary_max:.3e}"
            )
        return cp.ascontiguousarray(dm.real), None
    if dm.ndim != 3 or dm.shape[0] != nkpts:
        raise ValueError(
            f"k-point density must have shape ({nkpts},nao,nao), got {dm.shape}"
        )
    return cp.ascontiguousarray(dm), np.asarray(provider.kpts)
