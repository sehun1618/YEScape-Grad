# Changelog

All notable changes to the public packages will be documented here.

## Unreleased

This development branch is compared against GitHub `main` at `0f286a1`.

### Packaging and scope

- Separate stable user-run benchmarks (`pbc/benchmark/user`) from Codex
  implementation and profiling benchmarks (`pbc/benchmark/development`).
  Managed queues declare their scope, and the runner rejects cross-scope
  output paths before starting a calculation.
- Refresh every user queue for the current production profile: native
  `optimized_fused` GDF, exact GPU-source RSDF with mapped-host compressed
  CDERI, derivative omega scales fixed at 1.0, and matrix-free Z-vector.
  Keep Water64/DZ RSDF disabled because its latest same-input runs still OOM.
- Curate molecular CPU/GPU and periodic CPU/GPU DC-DFT gradient packages with
  reproducible examples, validation scripts, and benchmark inputs.
- Align the repository and all four wheel distributions with the Apache-2.0
  licenses used by PySCF and GPU4PySCF, including upstream notices and
  file-level attribution for adapted implementations.
- Keep the active public scope restricted to neutral, closed-shell periodic
  DC-DFT with GTH pseudopotentials and fixed-cell nuclear gradients; reject
  unsupported spin, occupation, and stale-driver states explicitly.

### GDF and RSDF GPU response

- Build the production single-Gamma RSDF fused CDERI adjoint directly in
  lower-triangle AO-pair storage.  Source reversal overwrites that allocation,
  while the metric adjoint contracts weighted packed pairs in bounded blocks;
  the former full `(naux,nao,nao)` adjoint and packed RHS copy are absent.
  Add a paired Water64/DZ development queue for user-run GDF and RSDF memory
  confirmation; the large calculation is not part of the automated gate.
- Repair the Water64/DZ Gamma GDF reverse-adjoint high-water path: keep the
  metric solve real before allocating its AO-square stack, retire the
  zero-order primitive tensor before source assembly, reduce imaginary-part
  diagnostics in bounded slices, and assemble the exact primitive source in
  transactional auxiliary blocks.  Add a guarded development queue for the
  user-run scaling confirmation; Water64 is not part of the automated gate.
- Fuse the Gamma explicit and Z-vector GTH pseudopotential derivative through
  the exact linear source `D + 2 T_Z`.  Water20 PP time is halved; current GDF
  and RSDF workflows improve by 7.36% and 6.68% with maximum gradient changes
  below `6.5e-14 Ha/Bohr` and selected-FD errors below `5.2e-9 Ha/Bohr`.
- Promote native GDF `optimized_fused` to the public GPU gradient default.
  Keep both derivative omega scales at 1.0 so PySCF-resolved partitions remain
  the production default.  Water64 scale-1.0 selected-FD errors stay below
  `7.4e-9 Ha/Bohr`.
- On a fresh, same-input Water64/SZ paired gate at omega scale 1.0, the current
  GDF and RSDF routes reduce workflow time by 22.10% and 14.81% and sampled
  peak VRAM by 76.71% and 62.55%.  Complete gradients remain within
  `2.4e-13 Ha/Bohr` of their pre-optimization references.
- Add source-HF CUDA-event profiling for exact-central setup, first/repeated
  J/K, diagonalization and density updates.  Mapped-host compressed-CDERI
  storage remains the production default.

- Route exact GPU-source RSDF SCF, target J/K, and matrix-free AX calls through
  GPU4PySCF's compressed J/K implementation.  Canonical AO-pair factors are
  now expanded only when analytic derivatives request them, after which the
  compressed storage is retired so the two large representations do not
  remain resident together.  On the recorded Water/Ethanol-SZ Gamma case this
  reduced source HF by 12.5%, matrix-free Z by 48.1%, and analytic total time
  by 13.5%; energy and gradient changed by only `4.55e-13 Ha` and
  `1.51e-13 Ha/Bohr`, respectively.
- Stage the multigrid target gradient before the large RSDF derivative
  response, then retire completed multigrid tasks, unused CuPy pool blocks,
  FFT plans, and rebuildable canonical CDERI at their final-use boundaries.
  Bound the local-part-1 reciprocal derivative to 10% of available VRAM and
  at most 1024 G-vectors per exact block.  Retire Gamma canonical CDERI after
  its final source/metric-adjoint use and cap the subsequent source-reciprocal
  derivative at 15% of free VRAM and 512 G-vectors.  On Water/Ethanol-SZ with
  a `240^3` r2SCAN multigrid mesh the combined changes reduced sampled peak
  VRAM from 62.250 to 39.192 GiB (`-37.04%`) while preserving energy and
  gradient within `4.55e-13 Ha` and `1.26e-13 Ha/Bohr`.
- Add bounded-memory Gamma GDF reciprocal planning, transactional OOM
  rebatching, workspace reuse, reportable phase metadata, and a profiled
  Gamma derivative short-/long-range workload balance.
- Add GPU-native RSDF source and metric derivatives, all-atom primitive reuse,
  native one-electron H1 streaming, and source-contracted target-J/Z-vector
  response paths that avoid retaining full dCDERI, dJ, or dK tensors.
- Separate RSDF full-tensor validation, production scalar contraction, and
  bounded GPU memory/OOM scheduling into independently testable modules.
- Preserve complex-valued sampled-k RSDF response data instead of silently
  casting it to real values.
- Fix larger-basis native GDF derivatives by using the transformed Cartesian
  AO dimension of GPU4PySCF's sorted cell.  This covers the spherical-to-
  Cartesian padding introduced by d shells in DZVP-GTH calculations.
- Replace the canonical full-auxiliary GDF/RSDF exchange contraction with an
  auxiliary-blocked implementation.  The planner reserves 2 GiB, uses a
  bounded fraction of currently free VRAM, and retries the same range with a
  smaller block after a recoverable allocation failure.
- Reuse the transactional pseudopotential local-part-1 grid builder for
  native GDF as well as RSDF, and cap the free-memory value reported to the
  upstream GPU4PySCF CDERI batch planner at 512 MiB.  This prevents large
  cells from selecting AO-pair/grid batches whose explicit arrays fit alone
  but whose contiguous contraction operands do not.

### Target integration and gradients

- Add explicit `uniform`, `multigrid_xc`, and `multigrid_j_xc` target
  integrator modes.  `multigrid_xc` retains GDF/RSDF ownership of target J and
  moves only semilocal XC energy, potential, and explicit XC gradient work to
  GPU4PySCF multigrid.
- Thread target-integrator ownership, mesh identity, workspace compatibility,
  and gradient metadata through the public GPU driver.
- Keep CPU target multigrid unsupported explicitly rather than falling back to
  an unreported implementation.

### Benchmark evidence and diagnostics

- Add configurable VRAM high-water sampling, live KST progress output, failure
  reports, attempt records, queue locking, and summary CSV files.
- Fingerprint effective inputs, resolved structures, and source state so a
  completed report is skipped only when it belongs to the current calculation.
- Separate execution completion from FD/translation-sum accuracy gates and
  record software provenance plus source-HF array identities in reports.
- Release finite-difference energy objects after each displacement so FD loops
  do not retain complete SCF methods and their GPU workspaces.
- Diagnose the observed translation sum as target-XC uniform-grid egg-box
  error: rigid-translation energy FD reproduces the analytic net gradient,
  while GDF and RSDF show the same drift.  The implementation does not project
  this physical discretization error out of reported gradients.

### Validation and known limitations

- Add regression coverage for driver-state invalidation, benchmark evidence,
  multigrid ownership, complex-k response, DZVP sorted-cell AO transforms, and
  bounded exchange contraction/OOM retry behavior.
- Water64/DZVP GDF now completes with both uniform and XC-only multigrid target
  integration.  On the recorded test, XC-only multigrid reduced end-to-end
  time from about 2363 s to 1227 s while changing energy by `9.1e-8 Ha` and the
  largest gradient component by `4.7e-8 Ha/Bohr`.
- The first Water64/DZVP RSDF run exceeded available VRAM in the
  source-adjoint stage because each complex `(naux, nao, nao)` array at
  `naux = nao = 1472` required 47.5 GiB.  The follow-up Gamma candidate builds
  the production observable directly as one real 23.8 GiB adjoint, performs
  source/metric reverse contractions from that shared array, and consumes
  obsolete basis-transform generations instead of stacking them.  Small
  Gamma, sampled-k, DZVP-transform, and full-response parity gates pass.  The
  first rerun then exposed one remaining full-size outer-product temporary;
  fused assembly now uses transactional auxiliary blocks capped at 32 and
  halves the current block after an allocation failure.  The reverse
  triangular solve overwrites that fused storage, and memory pressure may
  evict the now-consumed central CDERI before source-basis sorting; the next
  J/K request rebuilds it.  The second rerun passed blocked assembly and then
  exposed a full-size common-dtype copy in the metric adjoint.  Gamma density
  containers whose imaginary part is at most `1e-10` now use zero-copy real
  views, while genuine mixed real/complex metric products assemble only their
  small `naux x naux` result from separate real and imaginary contractions.
  A post-v4 Water64/DZVP rerun is still required before declaring this case
  complete.
- `multigrid_j_xc` changes the numerical representation of target Coulomb J;
  it is an explicit experimental route, not a drop-in accuracy-equivalent
  replacement for finite-auxiliary GDF/RSDF J.
- Machine-specific `benchmark/results/` directories remain ignored.  Versioned
  queue files and compact evidence records reproduce and describe those runs.
