"""Explicitly select FFTDF and inspect the gradient decomposition."""

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import df, dft


cell = gto.Cell(
    atom="H 0 0 0; H 0 0 1.4",
    a=np.eye(3) * 6.0,
    unit="Bohr",
    basis="gth-szv",
    pseudo="gth-hf",
    ke_cutoff=20.0,
    verbose=3,
).build()

mf = dft.DCRKS(cell, xc="PBE")
mf.with_df = df.FFTDF(cell, mesh=cell.mesh)
energy = mf.kernel()
grad = mf.nuc_grad_method()
gradient = grad.kernel()

print(f"Energy: {energy:.12f} Ha")
print("Explicit gradient (Ha/Bohr):")
print(grad.result.contributions.explicit)
print("Z-vector response gradient (Ha/Bohr):")
print(grad.result.contributions.response)
print("Total analytic gradient (Ha/Bohr):")
print(gradient)
