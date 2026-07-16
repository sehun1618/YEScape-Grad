"""Compute restricted DC-PBE energy and analytic gradient with CPU GDF.

This example uses a distorted periodic H2O cell so that all Cartesian force
components are nontrivial.  Coordinates and lattice vectors are in Bohr,
energy is in Hartree, and the returned nuclear gradient is in Hartree/Bohr.
``auxbasis`` controls the Gaussian fitting basis independently of the primary
AO basis.
"""

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


cell = gto.Cell(
    atom=(
        "O 0.1 -0.2 0.3; "
        "H 1.55 0.15 0.05; "
        "H -0.35 1.25 -0.25"
    ),
    a=np.eye(3) * 7.0,
    unit="Bohr",
    basis="gth-szv",
    pseudo="gth-hf",
    ke_cutoff=25.0,
    verbose=3,
).build()

mf = dft.DCRKS(cell, xc="PBE").density_fit(auxbasis="gth-szv")
energy = mf.kernel()
gradient_object = mf.nuc_grad_method()
gradient = gradient_object.kernel()

print(f"DC-PBE energy: {energy:.12f} Ha")
print("GDF analytic gradient (Ha/Bohr):")
print(gradient)
print("GDF response metadata:")
print(gradient_object.result.metadata())
