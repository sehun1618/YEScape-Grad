#!/usr/bin/env python
"""Compare CPU and GPU FFTDF energy/gradient for exactly one input."""

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft as cpu_dft


cell = gto.Cell(
    atom="H 0 0 0; H 0 0 1.4",
    a=np.eye(3) * 6.0,
    unit="Bohr",
    basis="gth-szv",
    pseudo="gth-hf",
    ke_cutoff=20.0,
    verbose=0,
).build()

cpu = cpu_dft.DCRKS(cell, xc="PBE")
cpu.conv_tol = 1.0e-11
cpu_energy = cpu.kernel()
cpu_gradient = cpu.nuc_grad_method().kernel()

# ``to_gpu`` constructs GPU RHF, target RKS, FFTDF, response and gradient
# providers. It does not return a CPU object with a device label.
gpu = cpu.to_gpu()
gpu_energy = gpu.kernel()
gpu_gradient = gpu.nuc_grad_method().kernel()

print("|E_cpu - E_gpu| (Ha):", abs(cpu_energy - gpu_energy))
print(
    "max|grad_cpu - grad_gpu| (Ha/Bohr):",
    np.max(np.abs(cpu_gradient - gpu_gradient)),
)
