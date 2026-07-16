"""Restricted semilocal DC-DFT analytic nuclear gradient with CPU FFTDF.

This Stage-2 orchestration layer combines independently testable derivative
providers: GTH one-electron terms, target-DFA Hartree/XC terms, target-Fock
overlap/Pulay terms, the Ewald nuclear gradient, and the HF Z-vector response.
Gamma is represented as ``Nk=1``; sampled k-points use equal PySCF weights.
Returned arrays are energy derivatives in Hartree/Bohr, not forces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf.pbc.grad import krhf as pyscf_krhf_grad

from ..df.fft import fft_jk_derivative
from .assembly import GradientContributions, contract_one_electron
from .context import RestrictedGradientContext
from .nuclear_response import contract_z_vector, nuclear_response_source_vo
from .one_electron import build_one_electron_derivatives
from .orbitals import build_mo_partitions, target_fock_vo_blocks
from .pulay import contract_target_overlap
from .target import (
    contract_target_potential,
    target_energy_weighted_density,
    target_fock_kpts,
    target_potential_row_derivative,
)
from .zvector import ZVectorResult, solve_z_vector


@dataclass(frozen=True)
class FFTDFGradientResult:
    """Complete CPU FFTDF gradient and its physical decomposition."""

    contributions: GradientContributions
    one_electron: np.ndarray
    target_coulomb_xc: np.ndarray
    target_overlap_pulay: np.ndarray
    nuclear: np.ndarray
    z_vector: ZVectorResult
    mesh: tuple[int, int, int]
    ke_cutoff_ha: float
    nkpts: int

    @property
    def gradient(self) -> np.ndarray:
        return self.contributions.total

    def metadata(self) -> dict[str, object]:
        """Return compact numerical diagnostics for reports and examples."""

        return {
            "backend": "fftdf",
            "device": "cpu",
            "units": "Ha/Bohr",
            "nkpts": self.nkpts,
            "mesh": list(self.mesh),
            "ke_cutoff_ha": self.ke_cutoff_ha,
            "gradient_shape": list(self.gradient.shape),
            "z_vector_dimension": self.z_vector.dimension,
            "z_vector_residual_max_abs": self.z_vector.residual_max_abs,
            "z_vector_condition_number": self.z_vector.condition_number,
            "z_vector_equation": self.z_vector.equation,
            "component_max_abs": {
                "one_electron": float(np.max(np.abs(self.one_electron))),
                "target_coulomb_xc": float(
                    np.max(np.abs(self.target_coulomb_xc))
                ),
                "target_overlap_pulay": float(
                    np.max(np.abs(self.target_overlap_pulay))
                ),
                "nuclear": float(np.max(np.abs(self.nuclear))),
                "orbital_response": float(
                    np.max(np.abs(self.contributions.response))
                ),
            },
        }


def fftdf_cpu_gradient(dc_mf, *, z_equation: str = "adjoint") -> FFTDFGradientResult:
    """Evaluate the restricted CPU FFTDF analytic gradient.

    Parameters
    ----------
    dc_mf
        Converged ``DCRKS`` or ``DCKRKS`` object.
    z_equation
        ``"adjoint"`` solves the general Lagrange-multiplier equation.
        ``"direct"`` is available as a Hessian-symmetry diagnostic.
    """

    if getattr(dc_mf.with_df, "backend_name", None) != "fftdf":
        raise NotImplementedError(
            "fftdf_cpu_gradient requires df.FFTDF; GDF has a separate "
            "metric-response completion gate"
        )
    context = RestrictedGradientContext.from_dc_method(dc_mf)
    partitions = build_mo_partitions(context.source_mf)
    target_fock = target_fock_kpts(context)
    target_vo = target_fock_vo_blocks(list(target_fock), partitions)
    z_result = solve_z_vector(
        context.source_mf,
        partitions,
        target_vo,
        equation=z_equation,
    )

    one_electron_sources = build_one_electron_derivatives(
        context.cell,
        context.kpts,
    )
    hcore_by_atom = np.asarray(
        [
            one_electron_sources.hcore(atom_id)
            for atom_id in range(context.cell.natm)
        ]
    )
    one_electron = contract_one_electron(context, hcore_by_atom)

    target_rows = target_potential_row_derivative(context)
    target_coulomb_xc = contract_target_potential(context, target_rows)
    target_weighted_density = target_energy_weighted_density(
        context,
        target_fock,
    )
    target_overlap = contract_target_overlap(
        context,
        one_electron_sources.overlap_rows,
        target_weighted_density,
    )
    nuclear = np.asarray(
        pyscf_krhf_grad.grad_nuc(
            context.cell,
            range(context.cell.natm),
        ),
        dtype=float,
    )
    explicit = one_electron + target_coulomb_xc + target_overlap + nuclear

    source_jk = fft_jk_derivative(
        context.cell,
        context.kpts,
        context.density,
        exxdiv=context.source_mf.exxdiv,
        with_df=dc_mf.with_df,
    ).restricted_hf
    response = np.zeros((context.cell.natm, 3), dtype=float)
    for atom_id in range(context.cell.natm):
        for axis in range(3):
            h1_kpts = (
                hcore_by_atom[atom_id, axis]
                + source_jk[atom_id, :, axis]
            )
            s1_kpts = one_electron_sources.overlap(atom_id, axis)
            source_vo = nuclear_response_source_vo(
                context,
                partitions,
                h1_kpts,
                s1_kpts,
            )
            response[atom_id, axis] = contract_z_vector(
                z_result.blocks,
                source_vo,
                nkpts=context.nkpts,
            )

    return FFTDFGradientResult(
        contributions=GradientContributions.assemble(explicit, response),
        one_electron=one_electron,
        target_coulomb_xc=target_coulomb_xc,
        target_overlap_pulay=target_overlap,
        nuclear=nuclear,
        z_vector=z_result,
        mesh=tuple(int(value) for value in np.asarray(context.cell.mesh)),
        ke_cutoff_ha=float(context.cell.ke_cutoff),
        nkpts=context.nkpts,
    )
