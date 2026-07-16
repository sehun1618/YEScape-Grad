r"""Nuclear CPHF source and Z-vector contraction.

For each nuclear perturbation, the restricted HF orbital-response source is

.. math::

   B^{A_x}_{ai} = H^{A_x}_{ai}
       - S^{A_x}_{ai}\epsilon_i
       - \frac{1}{2}\left[(A+B)^{\rm HF}S^{A_x}_{oo}\right]_{ai}.

The first term contains the complete derivative of the source HF Fock matrix,
including kinetic, local/nonlocal GTH, J, and K pieces.  The last term is the
metric-density response required because the AO basis is nonorthogonal and
moves with the nuclei.
"""

from __future__ import annotations

import numpy as np

from .response import apply_hf_response


def nuclear_response_source_vo(
    context,
    partitions,
    h1_kpts: np.ndarray,
    s1_kpts: np.ndarray,
) -> list[np.ndarray]:
    """Project one nuclear CPHF source into virtual--occupied blocks."""

    positive_metric_densities = []
    for partition, overlap in zip(partitions, s1_kpts, strict=True):
        occupied = partition.occupied_coeff
        overlap_oo = occupied.conj().T @ overlap @ occupied
        one_sided = occupied @ overlap_oo @ occupied.conj().T
        # For Hermitian S_oo this is 2 C_o S_oo C_o^H.  The later factor
        # ``2 G[D_plus]`` and ``-1/2`` reproduce the physical restricted
        # metric density -2 C_o S_oo C_o^H.
        positive_metric_densities.append(one_sided + one_sided.conj().T)

    ax_metric = [
        2.0 * matrix
        for matrix in apply_hf_response(
            context.source_mf,
            positive_metric_densities,
        )
    ]
    output = []
    for k_index, (partition, h1, overlap, ax) in enumerate(
        zip(
            partitions,
            h1_kpts,
            s1_kpts,
            ax_metric,
            strict=True,
        )
    ):
        occupied = partition.occupied_coeff
        virtual = partition.virtual_coeff
        h1_vo = virtual.conj().T @ h1 @ occupied
        overlap_vo = virtual.conj().T @ overlap @ occupied
        ax_vo = virtual.conj().T @ ax @ occupied
        occupied_energies = context.mo_energy[k_index][partition.occupied_mask]
        output.append(
            h1_vo
            - overlap_vo * occupied_energies[None, :]
            - 0.5 * ax_vo
        )
    return output


def contract_z_vector(z_blocks, source_blocks, *, nkpts: int) -> float:
    """Return ``4/Nk Re sum_k <Z_k|B_k>`` in Hartree/Bohr."""

    raw = sum(
        np.vdot(np.asarray(z), np.asarray(source)).real
        for z, source in zip(z_blocks, source_blocks, strict=True)
    )
    # Four combines the two restricted occupations and the Hermitian
    # occupied--virtual plus virtual--occupied rotation channels.
    return float(4.0 * raw / nkpts)
