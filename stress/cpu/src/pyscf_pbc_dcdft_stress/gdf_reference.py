"""GDF matrix-level micro-FD reference for cell-response development.

This route does not rerun HF and is substantially narrower than total-energy
FD, but it is still numerical differentiation.  It exists to validate future
analytic GDF pseudopotential, metric, and three-center strain derivatives and
must never be advertised as the production analytic stress kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
from pyscf.pbc.df import aft
from pyscf.pbc.gto.pseudo import pp_int

from .conventions import VOIGT_LABELS, component_index
from .gdf_settings import GDFProviderSettings
from .gradient_adapter import resolved_gdf_parameters, resolved_gdf_pp_parameters
from .strain import deform_cell


@dataclass(frozen=True)
class GDFMatrixFDResult:
    """Derivative AO matrices for GDF PP, J, and optionally K."""

    pseudopotential_ha: np.ndarray
    coulomb_ha: np.ndarray
    exchange_ha: np.ndarray | None
    energy_derivatives_ha: Mapping[str, np.ndarray]
    step: float
    components: tuple[int, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "pseudopotential_ha", np.asarray(self.pseudopotential_ha))
        object.__setattr__(self, "coulomb_ha", np.asarray(self.coulomb_ha))
        if self.exchange_ha is not None:
            object.__setattr__(self, "exchange_ha", np.asarray(self.exchange_ha))
        object.__setattr__(
            self,
            "energy_derivatives_ha",
            MappingProxyType(
                {key: np.asarray(value, dtype=float) for key, value in self.energy_derivatives_ha.items()}
            ),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def as_dict(self) -> dict[str, object]:
        return {
            "matrix_shapes": {
                "pseudopotential": list(self.pseudopotential_ha.shape),
                "coulomb": list(self.coulomb_ha.shape),
                "exchange": None if self.exchange_ha is None else list(self.exchange_ha.shape),
            },
            "energy_derivatives_ha": {
                key: value.tolist() for key, value in self.energy_derivatives_ha.items()
            },
            "step": float(self.step),
            "components": [VOIGT_LABELS[index] for index in self.components],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GDFPseudopotentialFDResult:
    """Gamma GDF pseudopotential matrix micro-FD reference."""

    derivative_ha: np.ndarray
    energy_derivative_ha: np.ndarray
    step: float
    components: tuple[int, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "derivative_ha", np.asarray(self.derivative_ha))
        object.__setattr__(
            self, "energy_derivative_ha",
            np.asarray(self.energy_derivative_ha, dtype=float),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class GDFResidualPseudopotentialFDResult:
    """GDF local-part2 plus nonlocal-projector matrix micro-FD."""

    derivative_ha: np.ndarray
    energy_derivative_ha: np.ndarray
    step: float
    components: tuple[int, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "derivative_ha", np.asarray(self.derivative_ha))
        object.__setattr__(
            self, "energy_derivative_ha",
            np.asarray(self.energy_derivative_ha, dtype=float),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def gdf_residual_pseudopotential_finite_difference(
    method, *, step=2.5e-4, components=tuple(range(6)),
):
    """Differentiate GTH local-part2 and nonlocal projectors only."""

    from pyscf_pbc_dcdft.dft.state import require_current_state

    require_current_state(method)
    _validate_method(method)
    step = float(step)
    if step <= 0:
        raise ValueError("finite-difference strain step must be positive")
    indices = tuple(component_index(value) for value in components)
    if len(indices) != len(set(indices)):
        raise ValueError("finite-difference components must be unique")
    cell = method.cell
    density = np.asarray(method.source_scf.make_rdm1())
    nao = int(cell.nao_nr())
    derivative = np.full((6, nao, nao), np.nan, dtype=np.complex128)
    energy = np.full(6, np.nan)
    mesh = tuple(int(value) for value in cell.mesh)
    for index in indices:
        plus_cell = deform_cell(cell, index, step, mesh=mesh, preserve_mesh=True)
        minus_cell = deform_cell(cell, index, -step, mesh=mesh, preserve_mesh=True)
        plus = _residual_pseudopotential(plus_cell)
        minus = _residual_pseudopotential(minus_cell)
        derivative[index] = (plus - minus) / (2.0 * step)
        energy[index] = _real_scalar(
            np.einsum("ij,ji->", derivative[index], density),
            "GDF residual PP derivative",
        )
    return GDFResidualPseudopotentialFDResult(
        derivative_ha=_real_if_close(derivative),
        energy_derivative_ha=energy,
        step=step,
        components=indices,
        metadata={
            "method": "local_part2_plus_nonlocal_matrix_central_fd",
            "production": False,
            "source_hf_rerun": False,
            "density_relaxed": False,
            "df_backend": "gdf",
            "fixed_integer_mesh": True,
            "numerical_components": [
                "gth_local_part2_matrix",
                "gth_nonlocal_projector_matrix",
            ],
            "purpose": "oracle_for_remaining_analytic_gdf_pp_cell_response",
        },
    )


def gdf_pseudopotential_finite_difference(
    method, *, step=2.5e-4, components=tuple(range(6)),
    require_matching_meshes=True,
):
    """Differentiate only ``GDF.get_pp`` without constructing J or K."""

    from pyscf_pbc_dcdft.dft.state import require_current_state

    require_current_state(method)
    _validate_method(method)
    step = float(step)
    if step <= 0:
        raise ValueError("finite-difference strain step must be positive")
    indices = tuple(component_index(value) for value in components)
    if len(indices) != len(set(indices)):
        raise ValueError("finite-difference components must be unique")
    cell = method.cell
    density = np.asarray(method.source_scf.make_rdm1())
    nao = int(cell.nao_nr())
    derivative = np.full((6, nao, nao), np.nan, dtype=np.complex128)
    energy = np.full(6, np.nan)
    provider_settings = GDFProviderSettings.from_provider(method.with_df)
    mesh = tuple(int(value) for value in cell.mesh)
    parameter_pairs = {}
    for index in indices:
        plus_cell = deform_cell(cell, index, step, mesh=mesh, preserve_mesh=True)
        minus_cell = deform_cell(cell, index, -step, mesh=mesh, preserve_mesh=True)
        plus_provider = provider_settings.build(plus_cell)
        minus_provider = provider_settings.build(minus_cell)
        plus = np.asarray(plus_provider.get_pp(np.zeros(3)))
        minus = np.asarray(minus_provider.get_pp(np.zeros(3)))
        plus_parameters = resolved_gdf_pp_parameters(
            plus_provider.cell, plus_provider.kpts
        )
        minus_parameters = resolved_gdf_pp_parameters(
            minus_provider.cell, minus_provider.kpts
        )
        parameter_pairs[VOIGT_LABELS[index]] = {
            "plus": plus_parameters,
            "minus": minus_parameters,
        }
        if (
            require_matching_meshes
            and plus_parameters["mesh"] != minus_parameters["mesh"]
        ):
            raise RuntimeError(
                f"GDF PP matrix FD component {VOIGT_LABELS[index]} changed "
                f"mesh: plus={plus_parameters['mesh']}, "
                f"minus={minus_parameters['mesh']}"
            )
        derivative[index] = (plus - minus) / (2.0 * step)
        energy[index] = _real_scalar(
            np.einsum("ij,ji->", derivative[index], density),
            "GDF PP derivative",
        )
    return GDFPseudopotentialFDResult(
        derivative_ha=_real_if_close(derivative),
        energy_derivative_ha=energy,
        step=step,
        components=indices,
        metadata={
            "method": "pseudopotential_matrix_central_finite_difference",
            "production": False,
            "source_hf_rerun": False,
            "density_relaxed": False,
            "df_backend": "gdf",
            "fixed_integer_mesh": True,
            "resolved_parameter_pairs": parameter_pairs,
            "matching_discrete_meshes_required": bool(require_matching_meshes),
            "purpose": "oracle_for_future_analytic_gdf_pp_cell_response",
        },
    )


def gdf_matrix_finite_difference(
    method, *, step=2.5e-4, components=tuple(range(6)), with_exchange=False,
    require_matching_meshes=True,
):
    """Differentiate GDF-owned AO matrices at the central source density."""

    from pyscf_pbc_dcdft.dft.state import require_current_state

    require_current_state(method)
    _validate_method(method)
    step = float(step)
    if step <= 0:
        raise ValueError("finite-difference strain step must be positive")
    indices = tuple(component_index(value) for value in components)
    if len(indices) != len(set(indices)):
        raise ValueError("finite-difference components must be unique")

    cell = method.cell
    density = np.asarray(method.source_scf.make_rdm1())
    nao = int(cell.nao_nr())
    pp_derivatives = np.full((6, nao, nao), np.nan, dtype=np.complex128)
    j_derivatives = np.full((6, nao, nao), np.nan, dtype=np.complex128)
    k_derivatives = (
        np.full((6, nao, nao), np.nan, dtype=np.complex128)
        if with_exchange else None
    )
    parameter_pairs = {}
    provider_settings = GDFProviderSettings.from_provider(method.with_df)
    exxdiv = method.exxdiv if with_exchange else None
    mesh = tuple(int(value) for value in cell.mesh)

    for index in indices:
        plus_cell = deform_cell(cell, index, step, mesh=mesh, preserve_mesh=True)
        minus_cell = deform_cell(cell, index, -step, mesh=mesh, preserve_mesh=True)
        plus = _gdf_matrices(
            provider_settings.build(plus_cell), density,
            with_exchange=with_exchange, exxdiv=exxdiv,
        )
        minus = _gdf_matrices(
            provider_settings.build(minus_cell), density,
            with_exchange=with_exchange, exxdiv=exxdiv,
        )
        parameter_pairs[VOIGT_LABELS[index]] = {
            "plus": plus[3],
            "minus": minus[3],
        }
        if require_matching_meshes:
            mesh_paths = (
                ("cderi", "source_mesh"),
                ("cderi", "metric_mesh"),
                ("pseudopotential", "mesh"),
            )
            for owner, name in mesh_paths:
                if plus[3][owner][name] != minus[3][owner][name]:
                    raise RuntimeError(
                        f"GDF matrix FD component {VOIGT_LABELS[index]} changed "
                        f"{owner}.{name}: plus={plus[3][owner][name]}, "
                        f"minus={minus[3][owner][name]}"
                    )
        pp_derivatives[index] = (plus[0] - minus[0]) / (2.0 * step)
        j_derivatives[index] = (plus[1] - minus[1]) / (2.0 * step)
        if with_exchange:
            k_derivatives[index] = (plus[2] - minus[2]) / (2.0 * step)

    pp_energy = np.full(6, np.nan)
    j_energy = np.full(6, np.nan)
    k_energy = np.full(6, np.nan) if with_exchange else None
    for index in indices:
        pp_energy[index] = _real_scalar(
            np.einsum("ij,ji->", pp_derivatives[index], density), "GDF PP derivative"
        )
        j_energy[index] = 0.5 * _real_scalar(
            np.einsum("ij,ji->", j_derivatives[index], density), "GDF J derivative"
        )
        if with_exchange:
            k_energy[index] = -0.25 * _real_scalar(
                np.einsum("ij,ji->", k_derivatives[index], density),
                "GDF K derivative",
            )
    energies = {"pseudopotential": pp_energy, "coulomb": j_energy}
    if with_exchange:
        energies["restricted_hf_exchange"] = k_energy
    return GDFMatrixFDResult(
        pseudopotential_ha=_real_if_close(pp_derivatives),
        coulomb_ha=_real_if_close(j_derivatives),
        exchange_ha=None if k_derivatives is None else _real_if_close(k_derivatives),
        energy_derivatives_ha=energies,
        step=step,
        components=indices,
        metadata={
            "method": "matrix_level_central_finite_difference",
            "production": False,
            "source_hf_rerun": False,
            "density_relaxed": False,
            "df_backend": "gdf",
            "fixed_integer_mesh": True,
            "exchange_divergence": exxdiv,
            "gdf_settings": provider_settings.as_dict(),
            "resolved_parameter_pairs": parameter_pairs,
            "matching_discrete_meshes_required": bool(require_matching_meshes),
            "purpose": "oracle_for_future_analytic_gdf_cell_response",
        },
    )


def _gdf_matrices(provider, density, *, with_exchange, exxdiv):
    provider.verbose = 0
    pseudopotential = np.asarray(provider.get_pp(np.zeros(3)))
    coulomb, exchange = provider.get_jk(
        density, hermi=1, kpts=np.zeros(3), with_j=True,
        with_k=bool(with_exchange), exxdiv=exxdiv,
    )
    parameters = {
        "cderi": resolved_gdf_parameters(provider),
        "pseudopotential": resolved_gdf_pp_parameters(
            provider.cell, provider.kpts
        ),
    }
    return (
        pseudopotential,
        np.asarray(coulomb),
        None if exchange is None else np.asarray(exchange),
        parameters,
    )


def _residual_pseudopotential(cell):
    kpoints = np.zeros((1, 3))
    local_part2 = np.asarray(
        aft._IntPPBuilder(cell, kpoints).get_pp_loc_part2()
    )[0]
    nonlocal_part = np.asarray(pp_int.get_pp_nl(cell, kpoints))[0]
    return local_part2 + nonlocal_part


def _validate_method(method):
    if bool(getattr(method, "kpoint_mode", False)):
        raise NotImplementedError("GDF matrix FD reference supports Gamma only")
    if getattr(getattr(method, "with_df", None), "backend_name", None) != "gdf":
        raise NotImplementedError("GDF matrix FD reference requires GDF")
    if getattr(method, "dispersion", None) is not None:
        raise NotImplementedError("dispersion stress is outside the GDF matrix FD reference")


def _real_scalar(value, label, *, imag_tol=1e-9):
    scalar = complex(np.asarray(value).item())
    if abs(scalar.imag) > imag_tol:
        raise ValueError(f"{label} has an imaginary residual {scalar.imag:.3e}")
    return float(scalar.real)


def _real_if_close(value, *, imag_tol=1e-9):
    array = np.asarray(value)
    finite = np.isfinite(array)
    if np.any(finite) and np.max(np.abs(array.imag[finite])) > imag_tol:
        return array
    return array.real
