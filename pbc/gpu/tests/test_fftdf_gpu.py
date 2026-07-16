"""Small end-to-end gates for the restricted GPU FFTDF route.

These tests deliberately compare independent CPU and GPU calculations.  They
therefore detect a mislabeled CPU object, a silent CPU fallback, and numerical
drift in the GPU FFTDF derivative route.  The larger distorted-H2O and full-FD
matrix is kept in ``validation/validate_fftdf_gpu.py`` so that the unit suite
remains reasonably quick.
"""

from __future__ import annotations

import unittest

import cupy as cp
import numpy as np
from pyscf.pbc import gto
from pyscf_pbc_dcdft import dft as cpu_dft

from gpu4pyscf_pbc_dcdft import dft as gpu_dft


def _gpu_available() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def _cell(*, basis: str = "gth-szv"):
    return gto.Cell(
        atom="H 0 0 0; H 0 0 1.4",
        a=np.eye(3) * 6.0,
        unit="Bohr",
        basis=basis,
        pseudo="gth-hf",
        ke_cutoff=15.0,
        verbose=0,
    ).build()


def _run_pair(*, kmesh=(1, 1, 1), basis="gth-szv"):
    cell = _cell(basis=basis)
    if kmesh == (1, 1, 1):
        cpu = cpu_dft.DCRKS(cell, xc="PBE")
        gpu = gpu_dft.DCRKS(cell, xc="PBE")
    else:
        kpts = cell.make_kpts(kmesh)
        cpu = cpu_dft.DCKRKS(cell, kpts=kpts, xc="PBE")
        gpu = gpu_dft.DCKRKS(cell, kpts=kpts, xc="PBE")
    cpu.conv_tol = gpu.conv_tol = 1.0e-11
    cpu_energy = cpu.kernel()
    cpu_gradient = cpu.nuc_grad_method().kernel()
    gpu_energy = gpu.kernel()
    gpu_gradient_object = gpu.nuc_grad_method()
    gpu_gradient = gpu_gradient_object.kernel()
    return cpu_energy, cpu_gradient, gpu_energy, gpu_gradient, gpu_gradient_object


@unittest.skipUnless(_gpu_available(), "CUDA device is required")
class FFTDFGPUParityTest(unittest.TestCase):
    """Check public Gamma and non-Gamma GPU APIs against the CPU package."""

    def assert_pair_parity(self, *, kmesh, basis="gth-szv"):
        cpu_energy, cpu_gradient, gpu_energy, gpu_gradient, gradient_object = (
            _run_pair(kmesh=kmesh, basis=basis)
        )
        self.assertLess(abs(cpu_energy - gpu_energy), 1.0e-8)
        self.assertLess(
            float(np.max(np.abs(cpu_gradient - gpu_gradient))),
            1.0e-7,
        )
        self.assertLess(
            gradient_object.result.z_vector.residual_max_abs,
            1.0e-9,
        )
        self.assertIn("gradient_total", gradient_object.result.timings_s)
        self.assertEqual(
            gradient_object.base.with_df.capability()["device"],
            "gpu",
        )

    def test_gamma_h2_cpu_gpu_parity(self):
        self.assert_pair_parity(kmesh=(1, 1, 1))

    def test_sampled_kpoint_h2_cpu_gpu_parity(self):
        self.assert_pair_parity(kmesh=(2, 1, 1), basis="gth-dzvp")


if __name__ == "__main__":
    unittest.main()
