"""Periodic DC-PBE with the published D2C-D4 correction."""

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


cell = gto.Cell(
    atom="O 0 0 0; H 0 0 1.8; H 0 1.7 0",
    a=np.eye(3) * 12.0,
    unit="Bohr",
    basis="gth-szv",
    pseudo="gth-hf",
    ke_cutoff=30.0,
    verbose=0,
).build()

mf = dft.DCRKS(cell, xc="PBE").set_dispersion("d2c")
energy = mf.kernel()
gradient_object = mf.nuc_grad_method()
gradient = gradient_object.kernel()

print(f"DC-PBE+D2C energy: {energy:.12f} Ha")
print(f"D2C energy: {mf.dispersion_result.energy:.12f} Ha")
print("Total gradient (Ha/Bohr):")
print(gradient)
print("D2C gradient (Ha/Bohr):")
print(gradient_object.dispersion_result.gradient)
