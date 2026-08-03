'''GPU RSDF resolved source/metric settings and storage options.'''

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
# Case 1: production RSDF settings
#
mf = RHFDFT(cell, xc='PBE').rs_density_fit(auxbasis='gth-szv')

# These values control only the native source-response workspace.
mf.with_df.source_reciprocal_memory_fraction = .15
mf.with_df.source_reciprocal_minimum_reserve_bytes = 512 * 1024**2
mf.with_df.source_reciprocal_max_block_size = 512
mf.with_df.source_reciprocal_block_alignment = 32
mf.with_df.source_reciprocal_gamma_conjugate_symmetry = True

# Release unused CuPy pool blocks at response-stage ownership boundaries.
# Live arrays and the mathematical order of terms are unchanged.
mf.with_df.response_release_cached_blocks = True

# Local-GTH part 1 is independent of both RSDF range separations.
mf.with_df.pp_loc_part1_omega = None
mf.run()

mf_grad = mf.nuc_grad_method().set(
    cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8,
    cphf_max_cycle=200,
    cphf_refinement_solver='minres',
    hcore_method='auto',
).run()

print('Accuracy-defining inputs')
print(f'  cell.ke_cutoff = {cell.ke_cutoff:g} Ha')
print(f'  cell.precision = {cell.precision:.1e}')
print(f'  cell.mesh = {cell.mesh.tolist()}')
print('Resolved RSDF settings')
print(f'  source/metric = {mf.with_df.response_settings()}')
print(f'  local GTH part-1 plan = {mf.with_df.get_pp_loc_part1_plan()}')
print('Implementation/storage settings actually used')
print(f'  Hcore backend = {mf_grad.hcore_backend}')
print(f'  source reciprocal = {mf_grad.rsdf_source_reciprocal}')
print('  response cache releases = '
      f'{mf_grad.rsdf_response_metadata["memory_releases"]}')
print('Result invariants')
print(f'  E_PBE@HF = {mf.e_dft:.15f} Ha')
print(f'  Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print('  term recomposition max abs = '
      f'{np.max(np.abs(mf_grad.de - mf_grad.de_terms)):.3e} Ha/Bohr')


#
# Case 2: full-reciprocal-grid source-response oracle
#
# mf = RHFDFT(cell, xc='PBE').rs_density_fit(auxbasis='gth-szv')
# mf.with_df.source_reciprocal_gamma_conjugate_symmetry = False
# mf.run()
# mf_grad = mf.nuc_grad_method().run()


#
# Case 3: manual RSDF split override for diagnosis or legacy reproduction
#
# PySCF normally resolves source omega/mesh and metric omega_j2c/mesh_j2c
# independently.  Manual values are not ordinary accuracy controls.  If an
# override is required, apply it before build/kernel and record all four values.
#
# mf = RHFDFT(cell, xc='PBE').rs_density_fit(auxbasis='gth-szv')
# mf.with_df.omega = .3
# mf.with_df.omega_j2c = .4
# mf.run()
