"""Inspect how PySCF maps ``ke_cutoff`` to mesh and gradient convergence."""

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


for cutoff_ha in (15.0, 25.0, 40.0):
    cell = gto.Cell(
        atom="H 0 0 0; H 0 0 1.4",
        a=np.eye(3) * 6.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-hf",
        ke_cutoff=cutoff_ha,
        verbose=0,
    ).build()
    mf = dft.DCRKS(cell, xc="PBE")
    energy = mf.kernel()
    gradient = mf.nuc_grad_method().kernel()
    print(
        f"cutoff={cutoff_ha:6.1f} Ha  mesh={tuple(cell.mesh)}  "
        f"E={energy: .12f} Ha  "
        f"|sum grad|={np.linalg.norm(gradient.sum(axis=0)): .3e} Ha/Bohr"
    )
