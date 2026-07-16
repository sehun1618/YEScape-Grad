"""Minimal Gamma-point ``E_PBE[n_RHF]`` calculation.

Coordinates and lattice vectors are given in Angstrom. Energy is printed in
Hartree. The deliberately small basis/cutoff makes this a package smoke
example, not a converged production calculation.
"""

from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


cell = gto.Cell()
cell.atom = "H 0 0 0; H 0 0 0.74"
cell.a = "6 0 0; 0 6 0; 0 0 6"
cell.unit = "Angstrom"
cell.basis = "gth-szv"
cell.pseudo = "gth-hf"
cell.ke_cutoff = 40.0
cell.verbose = 4
cell.build()

mf = dft.DCRKS(cell, xc="PBE")
energy = mf.kernel()

print(f"HF reference energy : {mf.e_hf:.12f} Ha")
print(f"DC-PBE energy       : {energy:.12f} Ha")
print(mf.dc_energy_result.as_dict())
