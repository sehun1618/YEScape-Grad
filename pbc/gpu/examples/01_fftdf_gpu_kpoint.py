#!/usr/bin/env python
"""Sampled-k-point GPU FFTDF DC-PBE calculation.

The ``(2,1,1)`` mesh demonstrates that Gamma is represented as a special
``Nk=1`` case rather than the only supported route.
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

mf = dft.DCKRKS(cell, kpts=kpts, xc="PBE")
mf.conv_tol = 1.0e-11
print(f"DC-PBE energy: {mf.kernel():.12f} Ha")
print("Analytic gradient (Ha/Bohr):")
print(mf.nuc_grad_method().kernel())
