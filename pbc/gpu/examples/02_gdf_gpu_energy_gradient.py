#!/usr/bin/env python
"""Gamma-point GPU GDF DC-PBE energy and analytic gradient.

The periodic two-/three-center integrals are generated once on the host and
uploaded as canonical CDERI factors.  Source HF, target PBE, response, and
gradient contractions then use the project-local CuPy GDF route.

Coordinates and lattice vectors are in Bohr.  The returned derivative is an
energy gradient in Hartree/Bohr, not a force.
"""

import numpy as np
from pyscf.pbc import gto

from gpu4pyscf_pbc_dcdft import df, dft


cell = gto.Cell(
    atom=[
        ("O", (0.10, -0.20, 0.30)),
        ("H", (1.55, 0.15, 0.05)),
        ("H", (-0.35, 1.25, -0.25)),
    ],
    a=np.eye(3) * 7.0,
    unit="Bohr",
    basis="gth-szv",
    pseudo="gth-hf",
    ke_cutoff=25.0,
    verbose=0,
).build()

mf = dft.DCRKS(cell, xc="PBE")
mf.with_df = df.GDF(cell, auxbasis="gth-szv")
mf.conv_tol = 1.0e-11
mf.conv_tol_grad = 1.0e-9

energy = mf.kernel()
gradient_object = mf.nuc_grad_method()
gradient = gradient_object.kernel()

print(f"DC-PBE energy: {energy:.12f} Ha")
print("Analytic gradient (Ha/Bohr):")
print(gradient)
print("Energy timings (s):", mf.timings_s)
print("Gradient timings (s):", gradient_object.result.timings_s)
print("GDF capability:", mf.with_df.capability())
print("Gradient route metadata:", gradient_object.result.metadata())
