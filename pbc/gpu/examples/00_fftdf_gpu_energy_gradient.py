#!/usr/bin/env python
"""Gamma-point GPU FFTDF DC-PBE energy and analytic gradient.

Coordinates and lattice vectors are in Bohr. Energy is Hartree and the
returned array is an energy gradient in Hartree/Bohr, not a force.
"""

import numpy as np
from pyscf.pbc import gto

from gpu4pyscf_pbc_dcdft import dft


cell = gto.Cell(
    atom=[
        ("O", (0.0, 0.0, 0.0)),
        ("H", (1.43, 0.0, 1.10)),
        ("H", (-1.20, 0.20, 1.00)),
    ],
    a=np.eye(3) * 7.0,
    unit="Bohr",
    basis="gth-szv",
    pseudo="gth-hf",
    ke_cutoff=25.0,
    verbose=0,
).build()

mf = dft.DCRKS(cell, xc="PBE")
mf.conv_tol = 1.0e-11
energy = mf.kernel()
gradient_object = mf.nuc_grad_method()
gradient = gradient_object.kernel()

print(f"DC-PBE energy: {energy:.12f} Ha")
print("Analytic gradient (Ha/Bohr):")
print(gradient)
print("Energy timings (s):", mf.timings_s)
print("Gradient timings (s):", gradient_object.result.timings_s)
