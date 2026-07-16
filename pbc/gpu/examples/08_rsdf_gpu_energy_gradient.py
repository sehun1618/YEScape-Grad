#!/usr/bin/env python
"""Compute sampled-k-point DC-PBE energy and analytic gradient with GPU RSDF.

PySCF generates the periodic RSDF integral primitives at the explicit host
boundary.  Canonical CDERI storage, source/target J/K contractions, metric and
CDERI response algebra, AX/Z-vector, and final electronic assembly are owned
by the GPU package.  No CPU DC-DFT energy or gradient routine is called.

Coordinates and lattice vectors are in Bohr.  The gradient is ``dE/dR`` in
Hartree/Bohr; force would have the opposite sign.
"""

import numpy as np
from pyscf.pbc import gto

from gpu4pyscf_pbc_dcdft import dft


cell = gto.Cell(
    atom="H 0 0 0; H 0 0 1.4",
    a=np.eye(3) * 6.0,
    unit="Bohr",
    basis="gth-dzvp",
    pseudo="gth-hf",
    ke_cutoff=15.0,
    verbose=0,
).build()
kpts = cell.make_kpts([2, 1, 1])

mf = dft.DCKRKS(cell, kpts=kpts, xc="PBE").rs_density_fit(
    auxbasis="gth-szv"
)
mf.conv_tol = 1.0e-11
mf.conv_tol_grad = 1.0e-9

energy = mf.kernel()
gradient_object = mf.nuc_grad_method()
gradient = gradient_object.kernel()

print(f"DC-PBE energy: {energy:.12f} Ha")
print("GPU RSDF analytic gradient (Ha/Bohr):")
print(gradient)
print("Energy timings (s):", mf.timings_s)
print("Gradient timings (s):", gradient_object.result.timings_s)
print("RSDF capability:", mf.with_df.capability())
print("Gradient route metadata:", gradient_object.result.metadata())
