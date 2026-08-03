'''GPU GDF response-route and reciprocal-storage options.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import RHFDFT

cell = gto.Cell()
cell.atom = '''
    O  0.10 -0.20  0.30
    H  1.55  0.15  0.05
    H -0.35  1.25 -0.25
'''
cell.a = np.eye(3) * 7.
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-hf'
cell.ke_cutoff = 100
cell.precision = 1e-8
cell.verbose = 0
cell.build()


#
# Case 1: production GDF settings
#
mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')

# These four values bound temporary reciprocal storage.  They do not change
# the GDF Hamiltonian, cell.mesh, or any gradient term.
mf.with_df.reciprocal_memory_fraction = .10
mf.with_df.reciprocal_minimum_reserve_bytes = 512 * 1024**2
mf.with_df.reciprocal_max_block_size = 512
mf.with_df.reciprocal_block_alignment = 32

# Real Gamma uses the exact X(-G)=X(G)* relation.  True evaluates one member
# of each pair with its multiplicity; False is the slower full-grid oracle.
mf.with_df.reciprocal_gamma_conjugate_symmetry = True

# None asks the provider to resolve the local-GTH SR/LR work split from the
# Cell and precision.  It is not the physical target-XC range separation.
mf.with_df.pp_loc_part1_omega = None
mf.run()

mf_grad = mf.nuc_grad_method().set(
    cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8,
    cphf_max_cycle=200,
    cphf_refinement_solver='minres',
    gdf_response_mode='auto',
    hcore_method='auto',
    hcore_batch_size=4,
).run()

print('Accuracy-defining inputs')
print(f'  cell.ke_cutoff = {cell.ke_cutoff:g} Ha')
print(f'  cell.precision = {cell.precision:.1e}')
print(f'  cell.mesh = {cell.mesh.tolist()}')
print('Resolved GDF settings')
print(f'  local GTH part-1 plan = {mf.with_df.get_pp_loc_part1_plan()}')
print('Implementation/storage settings actually used')
print(f'  GDF response partition = {mf_grad.gdf_term_partition}')
print(f'  Hcore backend = {mf_grad.hcore_backend}')
print(f'  reciprocal memory plan = {mf_grad.gdf_reciprocal_memory_plan}')
print('Result invariants')
print(f'  E_PBE@HF = {mf.e_dft:.15f} Ha')
print(f'  Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print('  term recomposition max abs = '
      f'{np.max(np.abs(mf_grad.de - mf_grad.de_terms)):.3e} Ha/Bohr')


#
# Case 2: full-reciprocal-grid numerical oracle
#
# The complete case differs from Case 1 only in this assignment.  Recreate and
# rerun the workflow so both timings start from clean provider state.
#
# mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
# mf.with_df.reciprocal_gamma_conjugate_symmetry = False
# mf.run()
# mf_grad = mf.nuc_grad_method().run()


#
# Case 3: pin alternative, mathematically equivalent response routes
#
# mf_grad = mf.nuc_grad_method()
# mf_grad.gdf_response_mode = 'signed'  # quadratic-state oracle
# mf_grad.hcore_method = 'grid'         # finite-grid Hcore oracle
# mf_grad.run()
