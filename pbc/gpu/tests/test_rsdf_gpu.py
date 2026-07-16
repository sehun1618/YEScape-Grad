"""GPU RSDF energy, gradient, conversion, and device-parity gates."""

from __future__ import annotations

import unittest

import numpy as np
from pyscf.pbc import gto
from pyscf_pbc_dcdft import df as cpu_df
from pyscf_pbc_dcdft import dft as cpu_dft

from gpu4pyscf_pbc_dcdft import df as gpu_df
from gpu4pyscf_pbc_dcdft import dft as gpu_dft


def _cell():
    return gto.Cell(
        atom="H 0 0 0; H 0 0 1.4",
        a=np.eye(3) * 6.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-hf",
        ke_cutoff=15.0,
        precision=1.0e-9,
        verbose=0,
    ).build()


def _method(cell, kmesh, *, device):
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    dft_module = gpu_dft if device == "gpu" else cpu_dft
    df_module = gpu_df if device == "gpu" else cpu_df
    if tuple(kmesh) == (1, 1, 1):
        method = dft_module.DCRKS(cell, xc="PBE")
    else:
        method = dft_module.DCKRKS(cell, kpts=kpts, xc="PBE")
    method.with_df = df_module.RSDF(
        cell, kpts=kpts, auxbasis="gth-szv"
    )
    method.conv_tol = 1.0e-11
    method.conv_tol_grad = 1.0e-9
    method.kernel()
    return method


class RSDFGPUParityTest(unittest.TestCase):
    def test_gamma_energy_gradient_parity(self):
        self._assert_parity((1, 1, 1))

    def test_kpoint_energy_gradient_parity(self):
        self._assert_parity((2, 1, 1))

    def test_cpu_to_gpu_and_back_preserves_backend(self):
        cell = _cell()
        kpts = cell.make_kpts((2, 1, 1), wrap_around=True)
        cpu = cpu_dft.DCKRKS(cell, kpts=kpts, xc="PBE")
        cpu.with_df = cpu_df.RSDF(
            cell, kpts=kpts, auxbasis="gth-szv"
        )
        gpu = cpu.to_gpu()
        self.assertEqual(gpu.with_df.backend_name, "rsdf")
        self.assertEqual(gpu.to_cpu().with_df.backend_name, "rsdf")

    def _assert_parity(self, kmesh):
        cell = _cell()
        cpu = _method(cell, kmesh, device="cpu")
        gpu = _method(cell, kmesh, device="gpu")
        cpu_gradient = cpu.nuc_grad_method().kernel()
        gpu_gradient_object = gpu.nuc_grad_method()
        gpu_gradient = gpu_gradient_object.kernel()
        self.assertLess(abs(cpu.e_tot - gpu.e_tot), 1.0e-8)
        self.assertLess(np.max(np.abs(cpu_gradient - gpu_gradient)), 1.0e-7)
        metadata = gpu_gradient_object.result.metadata()
        self.assertEqual(metadata["backend"], "rsdf")
        self.assertFalse(metadata["silent_cpu_fallback"])
        self.assertLess(metadata["cderi_value_parity_max_abs"], 1.0e-7)


if __name__ == "__main__":
    unittest.main()
