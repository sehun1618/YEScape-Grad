"""Inspect the nonvariational RHS and Z-vector for a two-k-point H2 cell.

This educational example deliberately calls the response building blocks one
by one.  A normal user will obtain the same objects through the completed
``nuc_grad_method()`` route.
"""

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft
from pyscf_pbc_dcdft.grad import (
    build_mo_partitions,
    solve_z_vector,
    target_fock_ao_blocks,
    target_fock_vo_blocks,
)


cell = gto.Cell()
cell.atom = "H 0 0 0; H 0 0 0.74"
cell.a = np.eye(3) * 6.0
cell.unit = "Angstrom"
cell.basis = "gth-szv"
cell.pseudo = "gth-hf"
cell.ke_cutoff = 30.0
cell.verbose = 0
cell.build()

kpts = cell.make_kpts([2, 1, 1])
mf = dft.DCKRKS(cell, kpts=kpts, xc="PBE").run()

partitions = build_mo_partitions(mf.source_scf)
target_fock_vo = target_fock_vo_blocks(target_fock_ao_blocks(mf), partitions)
z_result = solve_z_vector(
    mf.source_scf,
    partitions,
    target_fock_vo,
    equation="adjoint",
)

print(f"E_PBE[n_HF]             = {mf.e_tot:.12f} Ha")
print(
    "max |F_target,vo|      = "
    f"{max(np.max(np.abs(block)) for block in target_fock_vo):.6e} Ha"
)
print(f"real-linear dimension   = {z_result.dimension}")
print(f"Z residual max          = {z_result.residual_max_abs:.6e}")
print(f"Hessian condition number= {z_result.condition_number:.6e}")
print(f"Hessian asymmetry max   = {z_result.matrix_asymmetry_max_abs:.6e}")
