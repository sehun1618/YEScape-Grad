"""Fixed-density GPU DF strain contractions without derivative tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import cupy as cp
import numpy as np

from pyscf_pbc_dcdft_stress.strain import deform_cell, deform_kpoints


_RSDF_SETTINGS = (
    "use_bvk", "precision_R", "precision_G", "npw_max", "_omega_min",
    "omega", "ke_cutoff", "mesh_compact", "omega_j2c", "mesh_j2c",
    "precision_j2c", "j2c_eig_always",
)


@dataclass(frozen=True)
class ContractedDFStrain:
    """Target-J and adjoint RHF bilinear Voigt derivatives."""

    target_coulomb: cp.ndarray
    adjoint_restricted_hf: cp.ndarray
    timings_s: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "target_coulomb", cp.asarray(self.target_coulomb))
        object.__setattr__(
            self, "adjoint_restricted_hf",
            cp.asarray(self.adjoint_restricted_hf),
        )
        object.__setattr__(self, "timings_s", MappingProxyType(dict(self.timings_s)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def contracted_gpu_df_strain(
    provider, density, transition_density, *, exxdiv="ewald", step=1.0e-5,
):
    """Differentiate only the two DF scalar contractions required by DC stress.

    Each displaced cell rebuilds GDF/RSDF factors and J/K at the supplied fixed
    AO densities.  It does not rerun source HF, target DFA or the Z-vector.  The
    route preserves the selected DF representation while avoiding host 2c/3c
    derivative tensors.  It is intentionally explicit as an internal centered
    finite difference primitive.
    """

    delta = float(step)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("DF contracted strain step must be positive")
    reference = _density_array(density)
    transition = _density_array(transition_density)
    if reference.shape != transition.shape:
        raise ValueError(
            "reference and transition density shapes differ: "
            f"{reference.shape} and {transition.shape}"
        )
    cell = provider.cell
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    if reference.shape[0] != len(kpts):
        raise ValueError("density and provider k-point counts differ")

    target = cp.empty(6, dtype=cp.float64)
    adjoint = cp.empty(6, dtype=cp.float64)
    component_seconds = []
    started_total = perf_counter()
    for component in range(6):
        started = perf_counter()
        plus_cell = deform_cell(
            cell, component, delta, mesh=cell.mesh, preserve_mesh=True
        )
        minus_cell = deform_cell(
            cell, component, -delta, mesh=cell.mesh, preserve_mesh=True
        )
        plus = _fixed_density_observables(
            provider, plus_cell, deform_kpoints(cell, plus_cell, kpts),
            reference, transition, exxdiv=exxdiv,
        )
        minus = _fixed_density_observables(
            provider, minus_cell, deform_kpoints(cell, minus_cell, kpts),
            reference, transition, exxdiv=exxdiv,
        )
        target[component] = (plus[0]-minus[0])/(2.0*delta)
        adjoint[component] = (plus[1]-minus[1])/(2.0*delta)
        cp.cuda.Stream.null.synchronize()
        component_seconds.append(perf_counter()-started)
    total_seconds = perf_counter()-started_total
    return ContractedDFStrain(
        target_coulomb=target,
        adjoint_restricted_hf=adjoint,
        timings_s={
            "total": total_seconds,
            "component_sum": float(sum(component_seconds)),
            "component_max": float(max(component_seconds)),
            "component_min": float(min(component_seconds)),
        },
        metadata={
            "owner": "gpu_fixed_density_df_energy_contracted",
            "derivative_kind": "centered_fixed_density_finite_difference",
            "analytic_outer_stress": False,
            "backend": provider.backend_name,
            "step": delta,
            "component_count": 6,
            "component_timings_s": [float(value) for value in component_seconds],
            "deformed_cell_build_count": 12,
            "kpoint_deformation_count": 12,
            "provider_build_count": 12,
            "jk_call_count": 12,
            "fixed_density_count": 2,
            "fixed_density_payload_bytes": int(reference.nbytes+transition.nbytes),
            "estimated_jk_result_payload_bytes": int(24*reference.nbytes),
            "source_scf_recomputed": False,
            "target_dfa_recomputed": False,
            "z_vector_recomputed": False,
            "fixed_fractional_coordinates": True,
            "fixed_fractional_kpoints": True,
            "fixed_integer_mesh": True,
        },
    )


def _fixed_density_observables(
    template, cell, kpts, density, transition, *, exxdiv,
):
    provider = _clone_provider(template, cell, kpts)
    dm = density[0] if len(kpts) == 1 else density
    j_matrix, k_matrix = provider.get_jk(
        dm, hermi=1, kpts=kpts, with_j=True, with_k=True,
        exxdiv=exxdiv,
    )
    j_value = _matrix_kpoint_array(j_matrix)
    k_value = _matrix_kpoint_array(k_matrix)
    nkpts = len(kpts)
    target = 0.5*cp.einsum(
        "kij,kji->", density, j_value, optimize=True
    ).real/nkpts
    source_hf = j_value-0.5*k_value
    adjoint = 2.0*cp.einsum(
        "kij,kji->", transition, source_hf, optimize=True
    ).real/nkpts
    cp.cuda.Stream.null.synchronize()
    del provider, j_matrix, k_matrix, j_value, k_value, source_hf
    return target, adjoint


def _clone_provider(template, cell, kpts):
    backend = getattr(template, "backend_name", None)
    common = {"auxbasis": getattr(template, "auxbasis", None)}
    if backend == "gdf":
        provider = template.__class__(
            cell, kpts=kpts, setup_mode=template.setup_mode, **common
        )
    elif backend == "rsdf":
        provider = template.__class__(
            cell, kpts=kpts, central_setup=template.central_setup,
            compressed_device_cache=template.compressed_device_cache, **common,
        )
        for name in _RSDF_SETTINGS:
            value = getattr(template, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            setattr(provider, name, value)
    else:
        raise TypeError(f"unsupported contracted DF backend {backend!r}")
    for name in (
        "mesh", "linear_dep_threshold", "max_memory", "verbose",
        "exp_to_discard",
    ):
        if hasattr(template, name):
            value = getattr(template, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            setattr(provider, name, value)
    return provider


def _density_array(value):
    array = cp.asarray(value)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[-1] != array.shape[-2]:
        raise ValueError(
            "density must have shape (nao,nao) or (nkpts,nao,nao); "
            f"got {array.shape}"
        )
    return array


def _matrix_kpoint_array(value):
    array = cp.asarray(value)
    return array[None] if array.ndim == 2 else array


__all__ = ["ContractedDFStrain", "contracted_gpu_df_strain"]
