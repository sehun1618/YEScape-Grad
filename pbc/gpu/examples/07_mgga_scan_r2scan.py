#!/usr/bin/env python
"""GPU DC-SCAN and DC-r2SCAN with FFTDF, GDF, and RSDF.

The MGGA kinetic-energy-density contribution and its nuclear derivative remain
on the GPU.  GDF periodic integral generation is the explicit host setup
boundary described by ``mf.with_df.capability()``.  The RSDF example verifies
that the MGGA target path also composes with a range-separated source-HF J/K
provider.
"""

import numpy as np
from pyscf.pbc import gto

from gpu4pyscf_pbc_dcdft import dft


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
    gradient_object = method.nuc_grad_method()
    print("Analytic gradient (Ha/Bohr):")
    print(gradient_object.kernel())
    print("Gradient timings (s):", gradient_object.result.timings_s)
