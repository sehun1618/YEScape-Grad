"""HF orbital-Hessian action in virtual--occupied space.

For a trial occupied--virtual rotation ``X_ai`` the spin-summed first-order AO
density at k-point ``k`` is

``D1_k = C_v,k X_k C_o,k^H + h.c.``.

PySCF's restricted response function returns ``G[D1] = J[D1] - 1/2 K[D1]``.
The extra factor two below follows the restricted spatial-orbital Hessian
convention used by the CPHF equation in this package.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
from pyscf import lib

from .orbitals import MOPartition


def apply_hf_response(source_mf, density_blocks) -> list[np.ndarray]:
    """Apply ``G[D] = J[D] - K[D]/2`` to k-resolved AO densities.

    PySCF's Gamma RHF and sampled-k-point KRHF response call signatures differ
    even though the physical operator is identical.  This adapter localizes
    that representation detail and always returns one AO matrix per k-point.
    No extra restricted factor is applied here.
    """

    blocks = [np.asarray(block) for block in density_blocks]
    kpoint_mode = np.asarray(source_mf.mo_coeff).ndim == 3
    with _response_environment(source_mf):
        if kpoint_mode:
            response = source_mf.gen_response(
                mo_coeff=source_mf.mo_coeff,
                mo_occ=source_mf.mo_occ,
                hermi=1,
            )
            value = np.asarray(response(np.asarray(blocks)))
            return [np.asarray(block) for block in value]

        response = source_mf.gen_response(
            mo_coeff=[source_mf.mo_coeff],
            mo_occ=[source_mf.mo_occ],
            hermi=1,
        )
        return [np.asarray(response(blocks[0]))]


def _response_environment(source_mf):
    """Select the robust exact-DM GDF contraction for transition densities."""

    provider = getattr(source_mf, "with_df", None)
    if getattr(provider, "backend_name", None) not in {"gdf", "rsdf"}:
        return nullcontext()
    # PySCF's low-rank MO optimization can create an empty imaginary factor
    # for a real rank-one transition density and then call ``max`` on it.
    # ``force_dm_kbuild`` selects the algebraically identical CDERI-DM route
    # and avoids that representation-only failure.
    return lib.temporary_env(provider, force_dm_kbuild=True)


class HFOrbitalHessian:
    """Apply the restricted HF CPHF operator to k-resolved VO blocks."""

    def __init__(self, source_mf, partitions: Sequence[MOPartition]):
        self.source_mf = source_mf
        self.partitions = tuple(partitions)

    def _validate(self, trial_blocks: Sequence[np.ndarray]) -> None:
        if len(trial_blocks) != len(self.partitions):
            raise ValueError("trial and MO partition k-point counts differ")
        for k_index, (trial, partition) in enumerate(
            zip(trial_blocks, self.partitions, strict=True)
        ):
            if np.asarray(trial).shape != partition.vo_shape:
                raise ValueError(
                    f"trial block {k_index} has shape {np.asarray(trial).shape}; "
                    f"expected {partition.vo_shape}"
                )

    def coupling(self, trial_blocks: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Return only the two-electron Hessian coupling for ``trial_blocks``."""

        self._validate(trial_blocks)
        density_blocks: list[np.ndarray] = []
        for trial, partition in zip(trial_blocks, self.partitions, strict=True):
            one_sided = (
                partition.virtual_coeff
                @ np.asarray(trial)
                @ partition.occupied_coeff.conj().T
            )
            density_blocks.append(one_sided + one_sided.conj().T)

        response_blocks = [
            2.0 * block
            for block in apply_hf_response(self.source_mf, density_blocks)
        ]
        return [
            partition.virtual_coeff.conj().T
            @ response
            @ partition.occupied_coeff
            for response, partition in zip(
                response_blocks,
                self.partitions,
                strict=True,
            )
        ]

    def __call__(self, trial_blocks: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Apply energy gaps plus the two-electron HF response coupling."""

        coupling = self.coupling(trial_blocks)
        return [
            partition.energy_gap * np.asarray(trial) + response
            for trial, response, partition in zip(
                trial_blocks,
                coupling,
                self.partitions,
                strict=True,
            )
        ]
