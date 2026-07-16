"""Sampled-k-point DC-PBE energy and analytic gradient with FFTDF CPU."""

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


cell = gto.Cell()
cell.atom = "H 0 0 0; H 0 0 1.4"
cell.a = np.eye(3) * 6.0
cell.unit = "Bohr"
cell.basis = "gth-dzvp"
cell.pseudo = "gth-hf"
cell.ke_cutoff = 15.0
cell.verbose = 3
cell.build()

kpts = cell.make_kpts([2, 1, 1])
mf = dft.DCKRKS(cell, kpts=kpts, xc="PBE")
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()

print("k-points (1/Bohr):")
print(kpts)
print(f"DC-PBE energy: {energy:.12f} Ha")
print("Analytic gradient (Ha/Bohr):")
print(gradient)
