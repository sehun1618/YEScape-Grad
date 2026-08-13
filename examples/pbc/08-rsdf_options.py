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

# Production ``auto`` keeps central CDERI and reverse adjoints in unique Gamma
# AO-pair storage.  It fuses target-J/source-Fock only when both the live-VRAM
# preflight and the benchmark-calibrated persistent-work ceiling permit it;
# otherwise it evaluates the two observables sequentially.  This changes only
# storage/traversal, not the RSDF source, metric, or resulting gradient.
mf.with_df.response_mode = 'auto'
mf.with_df.response_fused_max_bytes = 4 * 1024**3
mf.with_df.response_pair_block_size = 8192
mf.with_df.response_auxiliary_block_size = 32
# Explicit ``'fused_packed'`` overrides the promotion ceiling for experiments;
# ``'sequential_packed'`` is the low-memory oracle and ``'dense'`` is the
# intentionally high-memory small-system algebra oracle.

# mapped_host is the fast GPU4PySCF central storage.  hdf5 preserves the same
# factors and streams auxiliary blocks when pinned host RAM is the limiting
# resource.  This does not by itself reduce the temporary GPU response source.
mf.with_df.cderi_storage = 'mapped_host'  # or 'hdf5'
mf.with_df.cderi_storage_path = None
mf.with_df.cderi_storage_block_size = 256
mf.with_df.cderi_preflight_max_bytes = None
mf.with_df.release_cderi_after_response = False

# Local-GTH part 1 is independent of both RSDF range separations.
mf.with_df.pp_loc_part1_omega = None
mf.run()

mf_grad = mf.nuc_grad_method().set(
    cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8,
    cphf_max_cycle=200,
    cphf_refinement_solver='minres',
    df_cphf_response_mode='direct_vo',
    # ``auto`` chooses recompute/occupied/CDERI/all from the live budget.
    # An allocation failure falls back to exact recompute, without changing
    # the response equation.  Explicit modes remain algebra or memory oracles.
    df_cphf_response_cache='auto',
    hcore_method='auto',
).run()

# Diagnostics-only synchronized component timings are opt-in because their
# CUDA synchronization must not be mixed with production timing.
# mf.with_df.response_profile_components = True

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
print(f'  central CDERI = {mf.with_df.cderi_memory_report()}')
print('  response cache releases = '
      f'{mf_grad.rsdf_response_metadata["memory_releases"]}')
print('  response mode = '
      f'{mf_grad.rsdf_response_metadata["response_mode"]}')
print('  auto memory plan = '
      f'{mf_grad.rsdf_response_metadata.get("auto_memory_plan")}')
print('  CPHF cache = '
      f'{mf_grad.z_response_profile["resolved_cache_mode"]}')
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


#
# Case 4: dense response oracle or HDF5 central storage
#
# mf = RHFDFT(cell, xc='PBE').rs_density_fit(auxbasis='gth-szv')
# mf.with_df.response_mode = 'dense'
# mf.with_df.cderi_storage = 'hdf5'
# mf.run()
# mf_grad = mf.nuc_grad_method().run()
