"""Single import boundary to the established PBC DC-gradient response code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DCResponseState:
    context: object
    partitions: tuple[object, ...]
    z_vector: object
    target_fock: np.ndarray
    target_energy_weighted_density: np.ndarray


def build_dc_response_state(
    method, *, z_equation="adjoint", shared_z_vector=None,
) -> DCResponseState:
    """Reuse the validated nuclear-gradient Z-vector algebra read-only."""

    from pyscf_pbc_dcdft.grad.context import RestrictedGradientContext
    from pyscf_pbc_dcdft.grad.orbitals import (
        build_mo_partitions,
        target_fock_vo_blocks,
    )
    from pyscf_pbc_dcdft.grad.target import (
        target_energy_weighted_density,
        target_fock_kpts,
    )
    from pyscf_pbc_dcdft.grad.zvector import solve_z_vector

    context = RestrictedGradientContext.from_dc_method(method)
    partitions = tuple(build_mo_partitions(context.source_mf))
    target_fock = target_fock_kpts(context)
    target_vo = target_fock_vo_blocks(list(target_fock), list(partitions))
    z_vector = shared_z_vector
    if z_vector is None:
        z_vector = solve_z_vector(
            context.source_mf, list(partitions), target_vo, equation=z_equation
        )
    elif len(z_vector.blocks) != context.nkpts:
        raise ValueError("shared Z-vector k-point count does not match the method")
    weighted = target_energy_weighted_density(context, target_fock)
    return DCResponseState(
        context=context,
        partitions=partitions,
        z_vector=z_vector,
        target_fock=np.asarray(target_fock),
        target_energy_weighted_density=np.asarray(weighted),
    )


def contract_strain_response(state, h1_kpts, s1_kpts) -> float:
    """Contract one full source-Fock/overlap strain source with the Z-vector."""

    from pyscf_pbc_dcdft.grad.nuclear_response import (
        contract_z_vector,
        nuclear_response_source_vo,
    )

    source = nuclear_response_source_vo(
        state.context,
        list(state.partitions),
        np.asarray(h1_kpts),
        np.asarray(s1_kpts),
    )
    return contract_z_vector(
        state.z_vector.blocks, source, nkpts=state.context.nkpts
    )


def resolved_gdf_parameters(provider) -> dict[str, object]:
    """Report the local GDF builder parameters through the adapter boundary."""

    from pyscf_pbc_dcdft.df.gdf.pair_response import rsgdf_parameters

    if getattr(provider, "auxcell", None) is None:
        raise RuntimeError("build the GDF provider before resolving its parameters")
    settings = rsgdf_parameters(
        provider.cell,
        provider.auxcell,
        np.asarray(provider.kpts, dtype=float).reshape(-1, 3),
        source_mesh=getattr(provider, "mesh", None),
        linear_dep_threshold=float(provider.linear_dep_threshold),
    )
    return {
        "omega": float(settings.omega),
        "source_mesh": list(settings.source_mesh),
        "metric_mesh": list(settings.metric_mesh),
    }


def resolved_gdf_pp_parameters(cell, kpts) -> dict[str, object]:
    """Report the independent range split used by ``GDF.get_pp``."""

    from pyscf.pbc.df.rsdf_builder import _RSNucBuilder

    builder = _RSNucBuilder(
        cell, np.asarray(kpts, dtype=float).reshape(-1, 3)
    ).build()
    return {
        "omega": float(builder.omega),
        "mesh": [int(value) for value in builder.mesh],
    }


def load_gdf_ft_center_extension():
    """Load the established package-owned i/j Fourier derivative callbacks."""

    from pyscf_pbc_dcdft.df.gdf.ft_derivative import load_extension

    return load_extension()
