# Changelog

All notable changes to the public packages will be documented here.

## Unreleased

This development branch is compared against GitHub `main` at `0f286a1`.

### Packaging and scope

- Curate molecular CPU/GPU and periodic CPU/GPU DC-DFT gradient packages with
  reproducible examples, validation scripts, and benchmark inputs.
- Align the repository and all four wheel distributions with the Apache-2.0
  licenses used by PySCF and GPU4PySCF, including upstream notices and
  file-level attribution for adapted implementations.
- Keep the active public scope restricted to neutral, closed-shell periodic
  DC-DFT with GTH pseudopotentials and fixed-cell nuclear gradients; reject
  unsupported spin, occupation, and stale-driver states explicitly.

### GDF and RSDF GPU response

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
  J/K request rebuilds it.  A post-v3 Water64/DZVP rerun is still required
  before declaring this case complete.
- `multigrid_j_xc` changes the numerical representation of target Coulomb J;
  it is an explicit experimental route, not a drop-in accuracy-equivalent
  replacement for finite-auxiliary GDF/RSDF J.
- Machine-specific `benchmark/results/` directories remain ignored.  Versioned
  queue files and compact evidence records reproduce and describe those runs.
