'''GPU GDF central-build, response-route, and storage options.'''

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

# The native central CDERI build first reads the actual free VRAM, then uses
# the smallest of (free*fraction), (free-reserve), and max_working_bytes.
# The default 25%/2-GiB/4-GiB policy therefore scales down on a small device
# but does not let a large-memory device request one giant AO-pair batch.
# max_working_bytes is the budget reported to GPU4PySCF's internal planner,
# not an exact peak-VRAM promise.  Set it to None to remove only this ceiling.
mf.with_df.native_build_memory_fraction = .25
mf.with_df.native_build_minimum_reserve_bytes = 2 * 1024**3
mf.with_df.native_build_max_working_bytes = 4 * 1024**3

# These four values instead bound temporary *gradient-response* reciprocal
# storage.  Neither option family changes the GDF Hamiltonian, cell.mesh, or
# any mathematical gradient term.
mf.with_df.reciprocal_memory_fraction = .10
mf.with_df.reciprocal_minimum_reserve_bytes = 512 * 1024**2
mf.with_df.reciprocal_max_block_size = 512
mf.with_df.reciprocal_block_alignment = 32

# A large-memory Water64 experiment used fraction=.40 and max_block_size=160.
# It reduced the complete GDF response by only 2.21% while requesting a much
# more aggressive live-VRAM budget, so the portable production defaults above
# remain .10/512.  Treat larger fractions as system/device-specific timing
# options, not accuracy controls or universal faster defaults.

# Real Gamma uses the exact X(-G)=X(G)* relation.  True evaluates one member
# of each pair with its multiplicity; False is the slower full-grid oracle.
mf.with_df.reciprocal_gamma_conjugate_symmetry = True

# The explicit packed oracle/fallback uses the exact transpose identity
# T W P = (P W T)^T and expands 128 auxiliary rows at once.  When that route
# is selected, it caches the first-pass packed source in CUDA-mapped host
# memory, avoiding a second dense source-algebra pass.  The cache is an
# execution policy only: it does not change the density, GDF metric or
# gradient.  The live planner may use at most 80% of currently available host
# memory while preserving 4 GiB.  If the source does not fit, its fallback
# repeats the exact algebra in two passes instead of risking host OOM.
#
# This option family applies only to the real one-Gamma native response.
# Sampled-k response retains its complex native path.
mf.with_df.adjoint_auxiliary_block_size = 128
mf.with_df.adjoint_storage = 'host_cache'  # or 'double_pass'
mf.with_df.adjoint_host_cache = 'mapped'  # or 'pageable'
mf.with_df.adjoint_host_memory_fraction = .80
mf.with_df.adjoint_host_minimum_reserve_bytes = 4 * 1024**3
mf.with_df.adjoint_host_max_bytes = None
mf.with_df.adjoint_host_cache_fallback = True

# Central-factor storage is independent of the build workspace above.
# mapped_host is GPU4PySCF's fast default.  hdf5 keeps the same compressed
# factors on disk and streams auxiliary blocks through the inherited iterator;
# it lowers pinned-host residency at the cost of file I/O.
mf.with_df.cderi_storage = 'mapped_host'  # or 'hdf5'
mf.with_df.cderi_storage_path = None      # temp file; a directory is accepted
mf.with_df.cderi_storage_block_size = 256

# Optional safety gate evaluated before the native build.  None only reports
# the estimate.  A byte limit raises instead of silently changing auxbasis.
mf.with_df.cderi_preflight_max_bytes = None

# False preserves immediate same-object energy reuse.  True computes the last
# overlap-response action after Z-vector, then releases central CDERI before
# native explicit GDF response.  A later SCF call rebuilds the factors.
mf.with_df.release_cderi_after_response = False

# None asks the provider to resolve the local-GTH SR/LR work split from the
# Cell and precision.  It is not the physical target-XC range separation.
mf.with_df.pp_loc_part1_omega = None
mf.run()

mf_grad = mf.nuc_grad_method().set(
    cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8,
    cphf_max_cycle=200,
    cphf_refinement_solver='minres',
    df_cphf_response_mode='direct_vo',
    # ``auto`` is already the production default; it is written explicitly
    # here only because this file documents the available controls.  It may
    # retain the invariant A_P=L_P C_occ, H_P=A_P^T C_occ and/or compressed
    # CDERI on device for this Z solve only.  The planner accounts for the
    # action workspace plus a 4-GiB reserve and safely falls back to
    # recompute.  No cache survives into the native nuclear-response stage or
    # a later scanner geometry.
    #
    # In a 96-GiB Water64 GTH-DZVP/Weigend test, auto resolved to ``all``:
    #
    #   cache       Z-vector / s    complete gradient / s
    #   recompute        177.21                  433.51
    #   occupied         112.58                  368.83
    #   cderi            171.70                  428.26
    #   all              103.90                  360.49
    #   auto -> all      103.72                  359.93
    #
    # This is a memory-dependent execution result, not a reason to hard-code
    # ``all``.  A larger system or smaller GPU can resolve auto to occupied,
    # cderi, or recompute without changing the GDF equations.
    df_cphf_response_cache='auto',
    df_cphf_response_memory_fraction=.80,
    df_cphf_response_minimum_reserve_bytes=4 * 1024**3,
    # Validation-only synchronized action timings.  Leave False for ordinary
    # production timing; queue record_stage_timings also enables this ledger.
    profile_df_cphf_response=False,
    gdf_response_mode='auto',
    # For supported closed-shell real-Gamma GDF, auto selects the exact
    # packed-source-free MO-direct response.  Set 'packed' to pin the former
    # density-packed numerical oracle.  'mo_factorized' is the packed-backed
    # algebra experiment; explicit 'mo_direct' is useful for A/B validation.
    gdf_adjoint_backend='auto',
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
print(f'  native build memory plan = '
      f'{mf.with_df.native_build_memory_plan}')
print(f'  central CDERI = {mf.with_df.cderi_memory_report()}')
print(f'  GDF response partition = {mf_grad.gdf_term_partition}')
print(f'  GDF adjoint requested/used = '
      f'{mf_grad.gdf_adjoint_backend}/'
      f'{mf_grad.gdf_adjoint_backend_used}')
print(f'  GDF adjoint fallback = '
      f'{mf_grad.gdf_adjoint_backend_fallback_reason}')
print(f'  Hcore backend = {mf_grad.hcore_backend}')
print(f'  reciprocal memory plan = {mf_grad.gdf_reciprocal_memory_plan}')
print(f'  packed-adjoint host plan = '
      f'{mf.with_df.adjoint_host_cache_plan}')
print(f'  Z-response cache/action plan = {mf_grad.z_response_profile}')
print(f'  Z cache requested/resolved = '
      f'{mf_grad.df_cphf_response_cache}/'
      f'{mf_grad.z_response_profile["resolved_cache_mode"]}')
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
# mf_grad.df_cphf_response_mode = 'ao_oracle'  # former full-AO J/K action
# mf_grad.gdf_response_mode = 'signed'  # quadratic-state oracle
# mf_grad.gdf_adjoint_backend = 'packed'  # fused AO-pair oracle
# mf_grad.hcore_method = 'grid'         # finite-grid Hcore oracle
# mf_grad.run()


#
# Case 4: reproduce one Z-response storage mode
#
# Production code should normally retain ``auto``.  Explicit modes are useful
# for profiling, memory diagnosis, and like-for-like regression tests:
#
# mf_grad = mf.nuc_grad_method()
# mf_grad.df_cphf_response_cache = 'recompute'  # minimum persistent cache
# mf_grad.df_cphf_response_cache = 'occupied'   # cache A_P and H_P
# mf_grad.df_cphf_response_cache = 'cderi'      # cache compressed CDERI
# mf_grad.df_cphf_response_cache = 'all'        # cache both when they fit
# mf_grad.run()
# print(mf_grad.z_response_profile['resolved_cache_mode'])
#
# Even an explicit request is checked against the bounded-memory plan.  If it
# cannot fit, the recorded resolved mode is recompute rather than an unsafe
# allocation attempt.


#
# Case 5: pinned-host-limited out-of-core calculation
#
# mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
# mf.with_df.cderi_storage = 'hdf5'
# mf.with_df.cderi_storage_path = '/scratch/my-job'
# mf.with_df.release_cderi_after_response = True
# mf.run()
# mf_grad = mf.nuc_grad_method().run()
# print(mf_grad.cderi_release_metadata)


#
# Case 6: calculation-wide policy instead of individual storage controls
#
# ``custom`` is the compatibility default and preserves every fine-grained
# assignment above.  The named policies all retain the same equations:
#
#   fast       : all Z cache + mapped-host packed-adjoint cache
#   balanced   : bounded auto Z cache + mapped-host adjoint cache
#   low_memory : recomputed Z action + exact double-pass adjoint
#
# Declare the complete property sequence before starting.  A low-memory
# gradient-only plan may release CDERI after response; a declared following
# stress keeps it alive and reuses the compatible target/Z state.
# mf_grad = mf.nuc_grad_method()
# mf_grad.memory_policy = 'balanced'
# mf_grad.set_property_sequence('gradient', 'stress')
# mf_grad.run()
# stress = mf_grad.get_stress()  # requires an XC-only-MG stress workflow
# print(mf_grad.resolved_memory_policy)
