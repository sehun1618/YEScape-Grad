"""CuPy restricted-HF orbital-Hessian action for PBC DC-DFT."""

from __future__ import annotations

import cupy as cp


def apply_hf_response(source_mf, density_blocks):
    """Apply ``G[D]=J[D]-K[D]/2`` through the selected GPU DF provider."""

    blocks = [cp.asarray(block, order="C") for block in density_blocks]
    kpoint_mode = cp.asarray(source_mf.mo_coeff).ndim == 3
    if kpoint_mode:
        density = cp.stack(blocks)
        vj, vk = source_mf.get_jk(
            dm_kpts=density,
            hermi=1,
            kpts=source_mf.kpts,
            with_j=True,
            with_k=True,
        )
        value = vj - 0.5 * vk
        return [block for block in value]
    backend = getattr(source_mf.with_df, "backend_name", None)
    if backend == "gdf" and cp.iscomplexobj(blocks[0]):
        vj, vk = _gamma_gdf_complex_jk(source_mf, blocks[0])
    else:
        vj, vk = source_mf.get_jk(
            dm=blocks[0], hermi=1, with_j=True, with_k=True
        )
    return [vj - 0.5 * vk]


def apply_hf_response_sets(source_mf, density_sets):
    """Apply the HF response to multiple density sets in one DF traversal.

    ``density_sets[k]`` has shape ``(nset, nao, nao)``.  Sampled-k inputs are
    assembled as ``(nset, nkpts, nao, nao)`` for GPU4PySCF, while Gamma inputs
    keep the leading density-set axis.  Batching is particularly important for
    nuclear CPHF sources because every Cartesian component uses the same CDERI.
    """

    blocks = [cp.asarray(block, order="C") for block in density_sets]
    if not blocks or any(block.ndim != 3 for block in blocks):
        raise ValueError("density sets must contain (nset,nao,nao) blocks")
    nsets = int(blocks[0].shape[0])
    if nsets < 1 or any(int(block.shape[0]) != nsets for block in blocks):
        raise ValueError("density-set blocks must have one common nonzero size")

    kpoint_mode = cp.asarray(source_mf.mo_coeff).ndim == 3
    if kpoint_mode:
        density = cp.stack(blocks, axis=1)
        vj, vk = source_mf.get_jk(
            dm_kpts=density,
            hermi=1,
            kpts=source_mf.kpts,
            with_j=True,
            with_k=True,
        )
        value = cp.asarray(vj) - 0.5 * cp.asarray(vk)
        return [value[:, k_index] for k_index in range(value.shape[1])]

    density = blocks[0]
    backend = getattr(source_mf.with_df, "backend_name", None)
    if backend == "gdf" and cp.iscomplexobj(density):
        vj, vk = _gamma_gdf_complex_jk(source_mf, density)
    else:
        vj, vk = source_mf.get_jk(
            dm=density,
            hermi=1,
            with_j=True,
            with_k=True,
        )
    return [cp.asarray(vj) - 0.5 * cp.asarray(vk)]


def _gamma_gdf_complex_jk(source_mf, density):
    """Apply the real-integral Gamma GDF kernel to a complex Hermitian DM.

    GPU4PySCF's optimized Gamma GDF entry point accepts only ``float64``.
    The Coulomb and exchange maps are real-linear, so a complex transition
    density is evaluated as ``G[Re D] + i G[Im D]``.  ``hermi=0`` is required
    because the imaginary component of a Hermitian matrix is antisymmetric.
    """

    kwargs = {
        "hermi": 0,
        "with_j": True,
        "with_k": True,
    }
    vj_real, vk_real = source_mf.get_jk(dm=density.real, **kwargs)
    vj_imag, vk_imag = source_mf.get_jk(dm=density.imag, **kwargs)
    return vj_real + 1j * vj_imag, vk_real + 1j * vk_imag


class HFOrbitalHessian:
    """Apply energy gaps and the restricted HF response on the GPU."""

    def __init__(self, source_mf, partitions):
        self.source_mf = source_mf
        self.partitions = tuple(partitions)

    def coupling(self, trial_blocks):
        density_blocks = []
        for trial, partition in zip(
            trial_blocks, self.partitions, strict=True
        ):
            trial = cp.asarray(trial)
            if trial.shape != partition.vo_shape:
                raise ValueError(
                    f"VO trial shape {trial.shape}; expected {partition.vo_shape}"
                )
            one_sided = (
                partition.virtual_coeff
                @ trial
                @ partition.occupied_coeff.conj().T
            )
            density_blocks.append(one_sided + one_sided.conj().T)
        response_blocks = [
            2.0 * value
            for value in apply_hf_response(self.source_mf, density_blocks)
        ]
        return [
            part.virtual_coeff.conj().T @ response @ part.occupied_coeff
            for response, part in zip(
                response_blocks, self.partitions, strict=True
            )
        ]

    def __call__(self, trial_blocks):
        coupling = self.coupling(trial_blocks)
        return [
            part.energy_gap * cp.asarray(trial) + response
            for trial, response, part in zip(
                trial_blocks, coupling, self.partitions, strict=True
            )
        ]
