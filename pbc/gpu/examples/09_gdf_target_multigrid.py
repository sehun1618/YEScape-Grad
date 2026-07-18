#!/usr/bin/env python
"""Use GDF source HF with opt-in multigrid target XC.

Coordinates and lattice vectors are in Bohr.  The returned array is the
analytic energy gradient in Hartree/Bohr, not a force.
"""

import numpy as np
from pyscf.pbc import gto

from gpu4pyscf_pbc_dcdft import df, dft


cell = gto.Cell(
    atom="H 0 0 -0.7; H 0 0 0.7",
    a=np.eye(3) * 6.0,
    unit="Bohr",
    basis="gth-szv",
    pseudo="gth-hf",
    ke_cutoff=20.0,
    verbose=0,
).build()

mf = dft.DCRKS(cell, xc="R2SCAN")
mf.with_df = df.GDF(cell, auxbasis="gth-szv")
mf.set_target_integrator("multigrid_xc")
mf.conv_tol = 1.0e-10
mf.conv_tol_grad = 1.0e-8

energy = mf.kernel()
gradient_object = mf.nuc_grad_method()
gradient = gradient_object.kernel()

print(f"DC-r2SCAN energy: {energy:.12f} Ha")
print("Analytic gradient (Ha/Bohr):")
print(gradient)
print("Target integration:")
print(gradient_object.result.metadata()["target_integration"])
