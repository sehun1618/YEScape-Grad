"""Compare Gamma and sampled-k-point DC-PBE energy/gradient results."""

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

for kmesh in ((1, 1, 1), (2, 1, 1), (2, 2, 1)):
    if kmesh == (1, 1, 1):
        mf = dft.DCRKS(cell, xc="PBE")
    else:
        mf = dft.DCKRKS(cell, kpts=cell.make_kpts(kmesh), xc="PBE")
    energy = mf.kernel()
    gradient = mf.nuc_grad_method().kernel()
    print(
        f"kmesh={kmesh}  E={energy: .12f} Ha  "
        f"grad[1,z]={gradient[1, 2]: .9e} Ha/Bohr"
    )
