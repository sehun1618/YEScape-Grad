# Copyright 2026 Sehun.
# Licensed under the Apache License, Version 2.0.

"""Direct reverse-adjoint Gamma GDF derivative on native CUDA primitives."""

from __future__ import annotations

import numpy as np
import cupy as cp
from gpu4pyscf.lib.cupy_helper import get_avail_mem
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.grad.rhf import contract_h1e_dm
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.tools.pbc import madelung

from .signed_density import build_gamma_primitive_adjoint
from .reciprocal_memory import plan_gamma_reciprocal_blocks
from .signed_gamma_native import (
    _DevicePhaseTimeline,
    _QuadraticState,
    _ReciprocalContext,
    _build_short_range_zero_order,
    _long_range_derivative,
    _metric_inverse_in_primitive_basis,
    _prepare_state,
    _real_symmetric_density,
    _short_range_derivative,
    _working_block_sizes,
)


_REVERSE_RECIPROCAL_MAX_WORKING_BYTES = 8 * 1024**3


def gamma_reverse_adjoint_energy_derivative(
    int3c2e_opt,
    reference_density,
    response_density,
    *,
    include_target_j=False,
    exxdiv="ewald",
    omega=0.0,
    memory_diagnostics=None,
    component_diagnostics=None,
):
    """Evaluate the Gamma target-J/Z response from direct primitive adjoints.

    One identity-factor state constructs the primitive three-center tensor
    once.  After the metric solve, exact ``D/T`` cross terms replace the
    separate quadratic ``D+T`` and ``D-T`` states used by the validation
    route.  The same SR, LR, metric and Ewald derivative kernels are retained.
    """

    reference = _real_symmetric_density(reference_density, "reference_density")
    response = _real_symmetric_density(response_density, "response_density")
    if reference.shape != response.shape:
        raise ValueError(
            "reference and response densities must have the same shape, "
            f"got {reference.shape} and {response.shape}"
        )

    timeline = _DevicePhaseTimeline(memory_diagnostics is not None)
    timeline.mark("start")
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    sorted_reference = _sorted_density(cell, reference)
    sorted_response = _sorted_density(cell, response)
    nao = int(sorted_reference.shape[0])
    identity = cp.eye(nao, dtype=cp.float64)
    primitive_state = _QuadraticState(
        coulomb_weight=0.0,
        exchange_weight=0.0,
        factor_l=identity,
        factor_r=identity,
        original_density=reference,
        density=identity,
    )

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False
    )[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    naux = int(auxcell.ao_loc[-1])
    batch_size, block_size = _working_block_sizes(
        [primitive_state],
        nao_pair=nao_pair,
        naux=naux,
        bvk_ncells=len(int3c2e_opt.bvkmesh_Ls),
    )
    _build_short_range_zero_order(
        int3c2e_opt,
        [primitive_state],
        pair_addresses,
        i_addr,
        j_addr,
        batch_size=batch_size,
        block_size=block_size,
    )
    timeline.mark("short_range_zero_order")

    precision = auxcell.precision * 1.0e-6
    auxcell.rcut = _estimate_sr_2c2e_rcut(
        auxcell, int3c2e_opt.omega, precision
    )
    int2c2e_opt = Int2c2eOpt(auxcell)
    j2c = int2c2e_opt.int2c2e(sort_output=False)
    timeline.mark("two_center_metric")

    preflight = plan_gamma_reciprocal_blocks(
        int(np.prod(int3c2e_opt.mesh)),
        nao,
        naux,
        nao,
        free_device_bytes=int(get_avail_mem(exclude_memory_pool=True)),
    )
    max_reciprocal_block_size = max(
        1,
        _REVERSE_RECIPROCAL_MAX_WORKING_BYTES
        // preflight.bytes_per_grid,
    )
    reciprocal = _ReciprocalContext.build(
        int3c2e_opt,
        [primitive_state],
        j2c,
        i_addr=i_addr,
        j_addr=j_addr,
        omega=omega,
        max_block_size=max_reciprocal_block_size,
    )
    timeline.mark("reciprocal_zero_order")
    metric_inverse = _metric_inverse_in_primitive_basis(
        auxcell, reciprocal.j2c
    )
    metric_solved_source = cp.einsum(
        "uv,vij->uij",
        metric_inverse,
        primitive_state.j3c_oo,
        optimize=True,
    )
    primitive_bytes = int(metric_solved_source.nbytes)
    adjoint = build_gamma_primitive_adjoint(
        metric_solved_source,
        sorted_reference,
        sorted_response,
        include_target_j=include_target_j,
    )
    source_bytes = int(adjoint.source.nbytes)
    primitive_state.j3c_oo = None
    del metric_solved_source, metric_inverse, identity
    timeline.mark("reverse_adjoint_assembly")

    metric_derivative = -int2c2e_opt.energy_ip1_per_atom(adjoint.metric)
    result = metric_derivative
    timeline.mark("metric_derivative")
    long_range, derivative_execution = _long_range_derivative(
        int3c2e_opt,
        reciprocal,
        (),
        adjoint.metric,
        nao=nao,
        naux=naux,
        primitive_source=adjoint.source,
    )
    result += long_range
    timeline.mark("long_range_derivative")
    short_range_device_timing = {}
    short_range = _short_range_derivative(
        int3c2e_opt,
        (),
        pair_addresses,
        batch_size=batch_size,
        block_size=block_size,
        nao=nao,
        naux=naux,
        device_timing=short_range_device_timing,
        primitive_source=adjoint.source,
    )
    result += short_range
    timeline.mark("short_range_derivative")
    ewald = _direct_ewald_derivative(
        cell,
        reference,
        response,
        exxdiv=exxdiv,
        omega=omega,
    )
    result += ewald
    timeline.mark("ewald_derivative")

    host_result = np.asarray(result, dtype=float)
    if component_diagnostics is not None:
        component_diagnostics.clear()
        component_diagnostics.update(
            {
                "metric": np.asarray(metric_derivative, dtype=float),
                "long_range_and_auxiliary_center": np.asarray(
                    long_range, dtype=float
                ),
                "short_range": np.asarray(short_range, dtype=float),
                "ewald": np.asarray(ewald, dtype=float),
            }
        )
    if memory_diagnostics is not None:
        memory_diagnostics.clear()
        memory_diagnostics.update(
            {
                "route": "gamma_reverse_adjoint",
                "include_target_j": bool(include_target_j),
                "short_range_omega": float(int3c2e_opt.omega),
                "primitive_tensor_bytes": primitive_bytes,
                "primitive_source_bytes": source_bytes,
                "reciprocal_working_cap_bytes": (
                    _REVERSE_RECIPROCAL_MAX_WORKING_BYTES
                ),
                "signed_state_full_rank_bytes_avoided": (
                    2 * naux * nao * nao * 8
                ),
                "plan": reciprocal.memory_plan.metadata(),
                "zero_order_execution": (
                    reciprocal.zero_order_execution.metadata()
                ),
                "derivative_execution": derivative_execution.metadata(),
                "device_phase_timing": timeline.metadata(),
                "short_range_device_timing": short_range_device_timing,
            }
        )
    return host_result


def _sorted_density(cell, density):
    state = _prepare_state(
        cell,
        density,
        coulomb_weight=0.0,
        exchange_weight=0.0,
    )
    return state.density


def _direct_ewald_derivative(
    cell,
    reference,
    response,
    *,
    exxdiv,
    omega,
):
    if exxdiv != "ewald" or not (
        cell.dimension == 3
        or (cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum")
    ):
        return np.zeros((cell.natm, 3))
    overlap = int1e.int1e_ovlp(cell)
    overlap_derivative = int1e.int1e_ipovlp(cell)
    cross_product = (
        reference @ overlap @ response
        + response @ overlap @ reference
    )
    result = contract_h1e_dm(
        cell.cell,
        overlap_derivative,
        cross_product,
        hermi=1,
    )
    correction = madelung(cell, np.zeros((1, 3)), omega=-float(omega))
    return np.asarray(result * correction, dtype=float)
