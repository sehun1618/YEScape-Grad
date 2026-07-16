#!/usr/bin/env python
"""Sampled-k-point GPU GDF DC-PBE energy and analytic gradient.

The ``(2, 1, 1)`` mesh exercises the KRHF/KRKS path and all off-diagonal
k-pair CDERI blocks.  Gamma is therefore a supported special case, not the
only implemented route.
"""

import numpy as np
from pyscf.pbc import gto

from gpu4pyscf_pbc_dcdft import df, dft


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

mf = dft.DCKRKS(cell, kpts=kpts, xc="PBE")
mf.with_df = df.GDF(
    cell,
    kpts=kpts,
    auxbasis="gth-szv",
)
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
