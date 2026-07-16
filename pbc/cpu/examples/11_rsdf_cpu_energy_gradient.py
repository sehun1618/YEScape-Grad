"""Compute sampled-k-point DC-PBE energy and analytic gradient with CPU RSDF.

The example uses the public PySCF ``RSGDF`` numerical convention through the
project-local :class:`pyscf_pbc_dcdft.df.RSDF` provider.  The three-center
source and two-center metric may use different range-separation parameters
and meshes; ``response_settings`` reports the values actually differentiated.

Coordinates and lattice vectors are in Bohr.  Energy is returned in Hartree
and ``gradient`` is ``dE/dR`` in Hartree/Bohr, not the force ``-dE/dR``.
"""

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


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
print("RSDF analytic gradient (Ha/Bohr):")
print(gradient)
print("Resolved RSDF source/metric settings:")
print(mf.with_df.response_settings())
print("Gradient metadata:")
print(gradient_object.result.metadata())
