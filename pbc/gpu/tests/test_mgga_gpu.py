"""Focused GPU SCAN/r2SCAN gradient and CPU-parity regressions."""

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


def _cell():
    return gto.Cell(
        atom="H 0 0 0; H 0 0 1.4",
        a=np.eye(3) * 6.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-hf",
        ke_cutoff=20.0,
        verbose=0,
    ).build()


def _methods(*, xc, kmesh, backend):
    cell = _cell()
    if kmesh == (1, 1, 1):
        cpu = cpu_dft.DCRKS(cell, xc=xc)
        gpu = gpu_dft.DCRKS(cell, xc=xc)
    else:
        kpts = cell.make_kpts(kmesh)
        cpu = cpu_dft.DCKRKS(cell, kpts=kpts, xc=xc)
        gpu = gpu_dft.DCKRKS(cell, kpts=kpts, xc=xc)
    if backend == "gdf":
        cpu.density_fit("gth-szv")
        gpu.density_fit("gth-szv")
    elif backend == "rsdf":
        cpu.rs_density_fit("gth-szv")
        gpu.rs_density_fit("gth-szv")
    for method in (cpu, gpu):
        method.conv_tol = 1.0e-11
        method.conv_tol_grad = 1.0e-9
    return cpu, gpu


@unittest.skipUnless(_gpu_available(), "CUDA device is required")
class MGGAGPUGradientTest(unittest.TestCase):
    def assert_cpu_gpu_parity(self, *, xc, kmesh, backend):
        cpu, gpu = _methods(xc=xc, kmesh=kmesh, backend=backend)
        cpu_energy = cpu.kernel()
        gpu_energy = gpu.kernel()
        cpu_gradient = cpu.nuc_grad_method().kernel()
        gpu_gradient = gpu.nuc_grad_method().kernel()
        self.assertLess(abs(cpu_energy - gpu_energy), 1.0e-8)
        self.assertLess(
            float(np.max(np.abs(cpu_gradient - gpu_gradient))),
            1.0e-7,
        )

    def test_scan_gamma_fftdf_cpu_gpu(self):
        self.assert_cpu_gpu_parity(
            xc="SCAN", kmesh=(1, 1, 1), backend="fftdf"
        )

    def test_r2scan_kpoint_gdf_cpu_gpu(self):
        self.assert_cpu_gpu_parity(
            xc="R2SCAN", kmesh=(2, 1, 1), backend="gdf"
        )

    def test_scan_gamma_rsdf_cpu_gpu(self):
        self.assert_cpu_gpu_parity(
            xc="SCAN", kmesh=(1, 1, 1), backend="rsdf"
        )

    def test_r2scan_kpoint_rsdf_cpu_gpu(self):
        self.assert_cpu_gpu_parity(
            xc="R2SCAN", kmesh=(2, 1, 1), backend="rsdf"
        )


if __name__ == "__main__":
    unittest.main()
