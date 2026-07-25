# Stress Optimization Release Audit

Date: 2026-07-23

## Scope

This audit covers restricted PBC DC-PBE/DC-r2SCAN analytic stress with GTH
pseudopotentials, GDF/RSDF, Gamma and sampled k-points. It does not promote
multigrid, FFTDF, dispersion, open-shell, or finite-difference stress to a
production backend.

## Exact optimized owners

| Contribution | Selected owner | Numerical cell displacement |
|---|---|---:|
| Nonlocal GTH PP response | `gpu4pyscf_native_contracted_analytic_nonlocal_device` | no |
| GDF/RSDF reciprocal AO-pair source | `host_analytic_source_adjoint_gpu_ft` | no |
| Short-range source, vbar, auxiliary/kernel response, metric | established exact analytic owners | no |
| Target Fock/XC, AX and Z-vector | GPU PBC DC-DFT gradient state | no |
| One-electron and remaining integral primitives | declared host analytic boundary | no |

The DF route applies the reverse Cholesky adjoint before the blocked GPU
Fourier strain contraction. This is an exact reassociation of the same GDF or
RSDF derivative. It does not change mesh, cutoff, omega, precision, screening,
auxiliary basis, or convergence tolerance.

## Closed gates

| Gate | Status | Evidence |
|---|---|---|
| H2 SZ GDF/RSDF Gamma/`2x1x1` FD | pass | maximum below `2.83e-11 Eh/Bohr^3` |
| Water8 GDF/RSDF Gamma/`2x1x1` FD | pass | maximum below `6.90e-11 Eh/Bohr^3` |
| H2 `DZVP-GTH` GDF/RSDF Gamma/`2x1x1` FD | pass | maximum below `2.95e-11 Eh/Bohr^3` |
| Water8 Gamma three-run median | pass | RSDF 8.057 s, GDF 12.209 s |
| Host-memory reduction against bounded forward | pass | RSDF -64.7%, GDF -71.3% RSS |
| Water20 GDF/RSDF system scaling | pass | 61.52/52.51 s stress, 5.43/5.20 GiB RSS |
| API owner metadata and optimized example | pass | capability fields and example 01 |
| User/development queue isolation | pass | managed scope tests |

Machine-readable measurements are in
`../validation/results/2026-07-23_o4_gpu_ft_source_adjoint.json`. The derivation
and ownership map are in `S9_GPU_FT_Source_Design_2026-07-23.md` and
`Stress_Optimization_Reference_Survey.md`.

## Verification at this revision

- CPU stress tests: 73 passed in 162.6 s.
- GPU stress tests: 21 passed in 14.1 s, including Gamma/sampled-k dispatch and
  large-AO automatic block-planner tests.
- Benchmark tests: 10 passed, including queue-policy and scaling-gate tests.
- User/development/evidence JSON: 87 files parsed.
- GPU package and examples compile successfully.
- CPU and GPU wheels build without dependency resolution and import from a
  clean temporary installation. The CPU extension records direct `libcgto`
  and `libcint` dependencies and resolves them through a wheel-relative PySCF
  library RPATH.
- The optimized H2 wheel-installed example completed end to end and wrote
  JSON/extxyz frames carrying `production=true`, the PP/DF response owners,
  source-HF/Z owners, stress/virial conventions, and units.
- Both wheels contain Apache-2.0 `LICENSE`, project `NOTICE`, the original
  PySCF notice, and a provenance summary. The GPU wheel declares its direct
  `gpu4pyscf-pbc-dcdft` dependency rather than relying on the active
  environment.

## System-scaling gate

The repaired Water20/SZV-GTH/Gamma user queue completed both GDF and RSDF.
`validate_scaling_gate.py` returned `pass=true` with no errors. GDF/RSDF stress
walls are 61.52/52.51 s, peak RSS 5.43/5.20 GiB, and global device peaks
12.99/13.62 GiB. The exact Gamma GPU FT DF owner, exact analytic nonlocal PP
owner, `production=true`, empty numerical-component list and no silent fallback
are present in both reports.

The block repair reduced the DF-stage global device peak from 16.47/15.88 GiB
in the failed attempts to 7.45/7.47 GiB. The overall peak now belongs to the PP
stage. Machine-readable evidence is in
`../validation/results/2026-07-23_water20_scaling_gate.json`.

## Decision

The exact optimized PP and DF routes passed the required correctness,
performance, ownership and Water20 system-scaling gates. They remain explicit
production opt-ins selected by current GPU user queues. The Python API default
is intentionally unchanged here; changing it is a separate compatibility and
release-policy decision rather than an unresolved correctness or scaling gate.
