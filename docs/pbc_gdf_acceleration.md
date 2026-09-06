# PBC GDF acceleration audit

This document records which GDF acceleration paths were migrated from
`/home/ksh/2026_project/dc_forge_repo/src/dcforge`, where they live now, and
whether they are production defaults or explicit experiments.  The comparison
uses the source snapshot recorded in `pbc_migration.md`; old timing numbers are
evidence for that snapshot, not fresh benchmarks of this refactor.

The precision profiles themselves, including their information-versus-
arithmetic distinction and complete FP64 reference recipe, are described in
`pbc_gdf_mixed_precision.md`.

## Code ownership

The current split follows the calculation rather than the chronological order
in which the optimizations were developed.

| Layer | Current files | Responsibility |
|---|---|---|
| Workflow | `gpu/pbc/energy/rhf_dft.py` | Shared source/target provider, precision-profile selection, target energy |
| Provider and build | `gpu/pbc/df/gdf/gdf.py`, `build_memory.py`, `precision.py`, `residency.py` | CDERI construction, representation identity, storage and VRAM policy |
| Source response | `gpu/pbc/df/rhf_response.py`, `rhf_mixed_response.py`, `scf/gdf_mixed.py` | Direct-VO CPHF action and optional mixed-precision source Fock |
| GDF adjoint | `gpu/pbc/df/gdf/native_gamma.py`, `adjoint.py`, `rhf_mo_response.py` | Density-contracted source/metric derivative without full AO derivative tensors |
| Gradient driver | `gpu/pbc/grad/rhf_dft.py`, `rhf_zvec.py` | Z solve, term assembly, physical residual and backend selection |
| Shared Gaussian primitives | `cpu/pbc/df/gaussian/`, `gpu/pbc/df/gaussian/` | Hcore, pseudopotential, reciprocal and short-/long-range derivative primitives |

`gpu/pbc/df/cderi.py` and `compatibility.py` stay directly below `df` because
they describe the GPU4PySCF CDERI representation and version boundary, not a
single response algorithm.  GPU4PySCF calls its central GDF builder
`rsdf_builder`; imports bearing that upstream name are required by GDF and do
not mean that DC-Forge exposes RSDF.

## What was migrated

The central acceleration modules are unchanged from the original source apart
from import formatting and one unused local-name cleanup.  This includes the
build-memory policy, direct-VO response, mixed response, Gamma adjoint,
MO-direct response, device residency and precision profiles.  The refactor
narrows their public boundary to FFTDF/GDF and removes unused RSDF gradient
assembly.

| Optimization | Mathematical effect | Current default | Reference or fallback |
|---|---|---:|---|
| Adaptive native-build budget | None; changes only GPU4PySCF CDERI batch planning | Yes | Same builder with configurable fraction/reserve/cap |
| Gamma `G,-G` conjugate symmetry | None for real Gamma; evaluates one member with multiplicity | Yes | `reciprocal_gamma_conjugate_symmetry=False` |
| Bounded reciprocal blocks | None; changes response workspace and retries OOM with a smaller block | Yes | Same contraction at a different block size |
| Direct-VO CPHF action | Contracts the response directly in occupied-virtual space | Yes | AO-density response path |
| Stage-local response cache | Reuses occupied/CDERI intermediates within the Z solve | `auto` | `df_cphf_response_cache='recompute'` |
| Fused target-J/source-Fock response | Traverses shared GDF derivative primitives once | `auto` | `signed` or `separate` response modes |
| MO-direct adjoint | Uses occupied and response-orbital factors without dense AO-pair adjoints | `auto` | `gdf_adjoint_backend='packed'` |
| Paired Hcore contraction | Contracts source density and Z density in one primitive traversal | `auto` | Grid/full-matrix diagnostic route |
| Z3 mixed response | FP32 current actions with FP64 audit/refinement and an independent physical residual | Yes for real-Gamma GDF gradients | `mf_grad.precision_policy='fp64'` |
| Mapped-host CDERI | Keeps the large factor in host-mapped storage | Yes | Device residency is explicit and optional |

The default source Hamiltonian remains conservative: CDERI build/storage,
source RHF and GDF derivative contractions are FP64.  The guarded Z action is
the only mixed-precision default.  Consequently, `set_gdf_precision_profile`
and `mf_grad.precision_policy` are separate controls.

Most entries in this table are real-Gamma accelerations.  Sampled-k GDF uses
the same provider and adaptive native-build budget, but its complex response
uses GPU4PySCF's native density-contracted combined derivative and DC-Forge's
matrix-free complex Z solver.  Gamma conjugate-pair, MO-direct adjoint and GDF
precision-profile claims must not be extrapolated to sampled k-points.

## Explicit, non-default candidates

- `set_gdf_precision_profile('mixed')`: build CDERI in FP64, store it in FP32,
  and use staged mixed-precision source SCF, Z response and derivative
  contractions.  It emits a warning because the stored Hamiltonian differs
  from the FP64 reference.
- `set_gdf_precision_profile('fp32_build')`: also performs the large CDERI
  transform/reciprocal work in FP32/complex64.  This is the aggressive route;
  convergence is convergence of the approximate Hamiltonian, not an FP64
  error bound.
- `provider.configure_candidate(device_residency='auto', ...)`: keeps a safe
  prefix of CDERI on the GPU when sufficient VRAM remains.  The default is
  `off`, because the useful fraction is hardware- and system-dependent.
- `provider.configure_candidate(partition_omega=...)`: fixes the numerical
  short-/long-range partition.  `None` preserves GPU4PySCF's system-dependent
  resolver and is the production default.
- Native build/transfer/gradient profilers synchronize CUDA and are diagnostic
  controls, not timing defaults.
- `mf_grad.terminal_response_reuse=True` retains the final audited Z action's
  occupied projection for the following Pulay contraction.  It is the Gamma
  GDF default after direct `False`/`True` force parity validation; setting it
  to `False` retains the redundant-action numerical-reference route.  Exact
  response-amplitude identity is required and any mismatch falls back safely.
- `two_channel.py` is an adapter for an optional external multi-observable
  library.  No such binary is bundled, so production uses the stable fused
  one-channel implementation.

## Original features deliberately not migrated

| Original feature | Current status | Reason |
|---|---|---|
| RSDF and its source/metric/Ewald ledger | Excluded | This migration was explicitly limited to FFTDF and GDF |
| Stress/strain response | Excluded | Stress is a later capability, not part of fixed-cell nuclear gradients |
| Provider CDERI checkpoint restore | Not migrated | It is a lifecycle/performance feature, not required for one calculation; adding it needs an explicit identity and stale-cache audit |
| Large benchmark/queue framework | Not migrated | Validation artifacts belong outside the small package API |
| Scanner-based frame reuse | Not migrated for sampled k-points | The current requested workflow does not need scanner lifecycle semantics |

The missing CDERI checkpoint is the only original GDF acceleration that could
materially change repeated cold-start cost and is absent for scope rather than
because PySCF already replaces it.  It should be reconsidered only when a
multi-frame production runner is requested.

## Evidence inherited from the original source

These measurements describe earlier hardware and source revisions and must not
be presented as current performance results.

- Direct-VO caching reduced recorded Water8/20/64 Z-action time by roughly
  41--44%, at the cost of substantially more VRAM for the large case.
- MO-direct GDF response reduced the recorded response time by about 24% for
  Water20 and 28% for Water64; the corresponding end-to-end gradient stages
  improved by about 15% and 19%.
- The Water64/TZVP mixed profile was reported near 1657 s versus about 2743 s
  for its FP64 reference, with force RMS difference `4.348e-7 Ha/Bohr`
  (`2.236e-5 eV/Angstrom`).
- The aggressive FP32-build route was faster in that experiment, but shifted
  the HF energy by about `1.11e-3 Ha`; it therefore remains explicitly marked
  as an approximation.

The original work also rejected several candidates: a mismatched central-factor
reuse, streaming packed adjoints, a persistent reciprocal AO-pair cache,
full-primary delta-density J/K and HDF5 as a build-time acceleration.  They are
not silently revived here.

## Current validation contract

`tests/gpu/test_pbc_gdf_acceleration.py` checks the adaptive memory policy,
default/explicit option boundary and warning-bearing precision profiles.
`tests/gpu/test_pbc.py` compares the default Z3 gradient against an explicit
FP64 response on the same small GDF source calculation.  Broader sampled-k and
MultiGrid tests verify provider ownership, term recomposition, physical Z
residuals and relaxed-source finite differences.

No new Water20/Water64 performance claim is made by this audit.  A current
benchmark must record the GPU, PySCF/GPU4PySCF versions, source revision,
resolved auxiliary basis and omega, mesh, precision profile, cold/restored
lifecycle and synchronized-versus-production timing mode.

The CPU GDF directory is retained as the readable numerical/reference path for
the same source, metric and pseudopotential derivatives.  It does not expose
the GPU mixed-precision, device-residency or MO-direct acceleration policies;
those are not silently emulated on the CPU.
