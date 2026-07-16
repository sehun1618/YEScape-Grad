#!/usr/bin/env python
"""Print synchronized wall times for GPU energy and gradient stages."""

import json
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
mf = dft.DCRKS(cell, xc="PBE")
mf.conv_tol = 1.0e-11
mf.kernel()
gradient_object = mf.nuc_grad_method()
gradient_object.kernel()

report = {
    "energy_timings_s": mf.timings_s,
    "gradient_timings_s": gradient_object.result.timings_s,
}
print(json.dumps(report, indent=2))
