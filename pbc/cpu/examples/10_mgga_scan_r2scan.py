#!/usr/bin/env python
"""CPU DC-SCAN and DC-r2SCAN energy/analytic-gradient examples.

SCAN uses FFTDF at Gamma.  r2SCAN uses GDF on a ``(2,1,1)`` k-point mesh so
that the same MGGA implementation is demonstrated across both completed DF
backends and both k-point modes.  A second sampled-k r2SCAN calculation uses
RSDF to demonstrate that MGGA target evaluation is independent of the source
HF J/K provider.  Coordinates/lattice are in Bohr and the printed derivative
is in Hartree/Bohr.
"""

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


cell = gto.Cell(
    atom="H 0 0 0; H 0 0 1.4",
    a=np.eye(3) * 6.0,
    unit="Bohr",
    basis="gth-szv",
    pseudo="gth-hf",
    ke_cutoff=20.0,
    verbose=0,
).build()

scan = dft.DCRKS(cell, xc="SCAN")
scan.conv_tol = 1.0e-11
scan.conv_tol_grad = 1.0e-9

kpts = cell.make_kpts([2, 1, 1])
r2scan = dft.DCKRKS(cell, kpts=kpts, xc="R2SCAN")
r2scan.density_fit("gth-szv")
r2scan.conv_tol = 1.0e-11
r2scan.conv_tol_grad = 1.0e-9

r2scan_rsdf = dft.DCKRKS(cell, kpts=kpts, xc="R2SCAN")
r2scan_rsdf.rs_density_fit("gth-szv")
r2scan_rsdf.conv_tol = 1.0e-11
r2scan_rsdf.conv_tol_grad = 1.0e-9

for label, method in (
    ("DC-SCAN/FFTDF", scan),
    ("DC-r2SCAN/GDF", r2scan),
    ("DC-r2SCAN/RSDF", r2scan_rsdf),
):
    print(label)
    print(f"Energy: {method.kernel():.12f} Ha")
    print("Analytic gradient (Ha/Bohr):")
    print(method.nuc_grad_method().kernel())
