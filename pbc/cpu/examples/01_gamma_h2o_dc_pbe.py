"""Gamma-point H2O DC-PBE energy and analytic gradient.

Coordinates and cubic lattice vectors are in Bohr.  The gradient printed at
the end is ``dE/dR`` in Hartree/Bohr; force would have the opposite sign.
The modest cutoff is for demonstration, not a converged production result.
"""

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


cell = gto.Cell()
cell.atom = "O 0 0 0; H 1.43 0 1.10; H -1.20 0.20 1.00"
cell.a = np.eye(3) * 7.0
cell.unit = "Bohr"
cell.basis = "gth-szv"
cell.pseudo = "gth-hf"
cell.ke_cutoff = 25.0
cell.verbose = 3
cell.build()

mf = dft.DCRKS(cell, xc="PBE")
energy = mf.kernel()
gradient_object = mf.nuc_grad_method()
gradient = gradient_object.kernel()

print(f"DC-PBE energy: {energy:.12f} Ha")
print("Analytic gradient (Ha/Bohr):")
print(gradient)
print("Gradient metadata:")
print(gradient_object.result.metadata())
