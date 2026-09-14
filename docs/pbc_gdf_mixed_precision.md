# GPU Gamma GDF mixed precision

This document is the user-facing map of every mixed-precision boundary in the
GPU real-Gamma GDF workflow.  The implementation stays next to the calculation
stage it modifies; the public profile API in `RHFDFT` coordinates those stages.

## Why the code is not one package

Mixed precision is not a separate electronic-structure method.  It changes
the representation or arithmetic used inside several existing stages:

| Owner | File | Precision decision |
|---|---|---|
| GDF provider | `gpu/pbc/df/gdf/precision.py` | CDERI build and persistent storage |
| Source RHF | `gpu/pbc/scf/gdf_mixed.py` | J/K contraction schedule during SCF |
| Z-vector | `gpu/pbc/df/rhf_mixed_response.py` | Low-cost response action, refinement and physical residual |
| Nuclear response | `gpu/pbc/df/gdf/rhf_factorized_consumers.py` | MO-source contraction dtype |
| Workflow coordinator | `gpu/pbc/energy/rhf_dft.py` | Public profile selection and propagation |
| Gradient coordinator | `gpu/pbc/grad/rhf_dft.py` | Z and derivative precision selection |

Moving these implementations into one `mixed_precision/` directory would make
that directory depend back on the provider, SCF and gradient layers.  The
current arrangement preserves stage ownership and keeps the public choice in
one method:

```python
mf.set_gdf_precision_profile('fp64')
mf.set_gdf_precision_profile('mixed')
mf.set_gdf_precision_profile('fp32_build')
```

## The three public profiles

| Stage | `fp64` | `mixed` | `fp32_build` |
|---|---|---|---|
| Primitive SR/LR integrals | FP64/complex128 | FP64/complex128 | FP64/complex128 |
| Auxiliary metric and factorization | FP64 | FP64 | FP64, then factor cast to FP32 |
| CDERI metric transformation | FP64/complex128 | FP64/complex128 | FP32/complex64 |
| Persistent CDERI | mapped-host FP64 | mapped-host FP32 | mapped-host FP32 |
| Source RHF | native FP64 GDF | staged mixed, final FP64 arithmetic over FP32 CDERI | staged mixed, final FP64 arithmetic over FP32 CDERI |
| Target GDF Hartree evaluation | FP64 arithmetic on FP64 CDERI | FP64 arithmetic on quantized FP32 CDERI | FP64 arithmetic on quantized FP32 CDERI |
| Z-vector default | guarded Z3 mixed with FP64-CDERI physical audit | guarded Z3 mixed with FP32-CDERI/FP64-arithmetic audit | guarded Z3 mixed with FP32-CDERI/FP64-arithmetic audit |
| GDF nuclear response | FP64 | mixed MO products, FP64 primitive/metric/reduction | mixed MO products, FP64 primitive/metric/reduction |

`fp32_build` does not make every integral kernel FP32.  Its primitive
three-center integrals and auxiliary metric remain FP64.  The additional
approximation appears when the metric factor and the large SR/LR
transformations are lowered before the final CDERI is formed.

## Source SCF precision promotion budgets

The two intermediate source-SCF precision tiers each have a **20-cycle maximum
by default**. Existing density-change and AO-commutator gates can promote sooner;
reaching the budget also promotes, even if those residual gates are not met.
This prevents indefinite residence at a low-precision plateau, but does not
guarantee faster SCF or convergence for difficult electronic structures.

```python
mf.set_gdf_precision_profile('mixed')  # Also applies to 'fp32_build'.
mf.source_precision_coarse_max_cycle = 20  # First tier; customize independently.
mf.source_precision_fine_max_cycle = 20    # Second tier, not global cycle 20.
# None disables the budget for that tier (original residual-only promotion).
```

The source-only `mf.source_mixed_precision(...)` route uses the same budgets.
Native `fp64` source SCF is unchanged. Direct `run_mixed_source_scf` callers can
set `thresholds={'coarse': {'max_cycle': 12}, 'fine': {'max_cycle': 25}}`;
omitted gates inherit defaults. Limits must be positive integers or `None`.

Counts follow completed SCF callbacks within each tier, survive native SCF
kernel restarts inside the same controller, and reset only on promotion. A new
controller starts fresh. The pre-existing `minimum_cycle` gates still use native
SCF cycle numbering; they do not override a maximum residence budget. Promotion
clears DIIS and requests a full, non-incremental Fock rebuild on the next action.
With both default budgets, at most 40 intermediate-tier callbacks occur before
the final tier; this is **not** a promise of convergence in 40 SCF iterations.

The final tier has no additional residence cap: ordinary SCF iteration limits,
convergence tolerances and final physical audits still apply. For persistent
FP32 profiles this tier is FP64 **arithmetic over FP32 CDERI**, not reconstruction
of the discarded FP64 factor. Source logs record promotion reason and tier
cycle; `source_precision_report['schedule']` preserves `thresholds`, per-cycle
`tier_cycle`, and transition `reason` (`stationarity` or `max_cycle`).

한국어 요약: 1·2단계에서 각각 최대 20회 계산 후 다음 정밀도로 승격합니다.
잔차 조건을 먼저 만족하면 더 일찍 승격합니다. 마지막 단계의 수렴 기준은
완화하지 않으며, FP32 저장 과정에서 잃은 정보를 복원하는 기능도 아닙니다.
위 두 옵션으로 횟수를 각각 지정하고, `None`으로 횟수 제한을 끌 수 있습니다.

## Information precision versus arithmetic precision

Two separate questions must always be answered.

1. What information is stored in CDERI?
2. In what dtype is a contraction evaluated?

For example, a block read from an FP32 CDERI may be promoted to FP64 before a
matrix multiplication.  This improves accumulation and solver stability, but
it does not recreate the discarded CDERI bits:

```text
FP64 CDERI -> FP32 storage -> FP64 arithmetic
                              ^
                              arithmetic is FP64,
                              stored information is still FP32
```

Therefore an FP64 residual evaluated over FP32 storage is a rigorous residual
of the quantized Hamiltonian.  It is not an error estimate relative to the
native FP64 Hamiltonian.

## Recommended usage

### Conservative production default

```python
mf = RHFDFT(cell, xc='R2SCAN').density_fit(auxbasis=None)
mf = mf.multigrid_numint(with_j=False)
mf.set_gdf_precision_profile('fp64')
mf.run()

mf_grad = mf.nuc_grad_method()
gradient = mf_grad.run().de
```

This keeps CDERI, source RHF and the GDF nuclear response in FP64.  The
real-Gamma gradient still uses the guarded Z3 mixed response by default.

### Complete FP64 reference

```python
mf.set_gdf_precision_profile('fp64')
mf.run()

mf_grad = mf.nuc_grad_method()
mf_grad.precision_policy = 'fp64'
mf_grad.gdf_derivative_precision = 'fp64'
gradient = mf_grad.run().de
```

Both settings matter.  `precision_policy='fp64'` alone cannot recover a native
FP64 Hamiltonian if `mf` was already built with `mixed` or `fp32_build`.
Changing the workflow profile causes the GDF factor and source orbitals to be
invalidated and rebuilt.

### Moderate persistent-FP32 profile

```python
mf.set_gdf_precision_profile('mixed')
mf.run()
gradient = mf.nuc_grad_method().run().de
```

The complete CDERI transformation remains FP64, but each finished factor block
is written directly to mapped-host FP32 storage.  Source SCF, Z-vector and
selected derivative contractions then reuse that factor.  This profile emits
`GDFMixedPrecisionWarning` and must be compared with the complete FP64 reference
on representative structures.

### Aggressive FP32-build profile

```python
mf.set_gdf_precision_profile('fp32_build')
mf.run()
gradient = mf.nuc_grad_method().run().de
```

This additionally lowers the large CDERI transformation and reciprocal
contraction.  Accumulated build roundoff can bias energies and forces even
when SCF and CPHF residuals are small.  It emits `GDFBuildPrecisionWarning` and
should remain an explicitly labelled dataset-generation approximation.

## Z3 response-only mixed precision

The default `fp64` workflow uses Z3 only as an iterative accelerator:

1. Build or reuse an FP32 view of the GDF factor.
2. Solve the first orbital-Hessian problem with the FP32 direct-VO action.
3. Measure the residual of that quantized action.
4. Reconstruct a physical residual using FP64 arithmetic and the active
   provider's highest-information CDERI.
5. Apply bounded defect corrections.
6. If a native FP64 provider still fails the physical gate, solve again with
   its FP64 operator.

The quantized residual controls cleanup scheduling only.  The physical
residual is the acceptance gate.  For a native FP64 provider it tests the
native Hamiltonian; for a persistent-FP32 provider it tests the quantized
Hamiltonian using FP64 arithmetic.

The main controls are:

| Attribute | Default | Meaning |
|---|---:|---|
| `precision_policy` | `mixed` for real-Gamma GDF | Z response arithmetic |
| `mixed_z_inner_tol` | `1e-6` | Initial low-operator Krylov target |
| `mixed_z_correction_tol` | `1e-5` | Defect-correction solve target |
| `mixed_z_quantized_residual_tol` | `1e-7` | Advisory low-operator cleanup target |
| `mixed_z_quantized_hard_reject_tol` | `1e-5` | Skip further low corrections above this value |
| `mixed_z_max_corrections` | `3` | Maximum cleanup steps per stage |
| `cphf_residual_tol` | `1e-8` | Final physical acceptance threshold |

Changing these values defines a numerical experiment.  A Krylov solver's own
reported norm is not a substitute for `z_residual_max_abs`, which is obtained
by substituting the accepted Z-vector back into the physical equation.

## Source-only guarded mixed precision

`RHFDFT.source_mixed_precision()` predates the persistent profiles and has a
different contract:

- it starts with a native FP64 CDERI;
- it creates a temporary FP32 sidecar for early source-SCF J/K actions;
- it promotes through increasingly accurate contraction tiers;
- it must perform a native-FP64 Fock audit and can resume with native FP64;
- the temporary sidecar is released before target energy and gradient.

This is an advanced source-SCF experiment, not an alias for
`set_gdf_precision_profile('mixed')`.  The latter keeps only the FP32 central
factor and therefore cannot fall back to the discarded native-FP64 factor.

## Mixed GDF nuclear response

For persistent `mixed` and `fp32_build` profiles, the gradient factory selects
`gdf_derivative_precision='mixed'`.  This lowers only the large MO-source
products used by the MO-direct GDF adjoint:

- occupied coefficients and response orbitals are cached in FP32;
- transaction-local GEMMs use FP32;
- transaction outputs and per-atom accumulation use FP64;
- primitive SR/LR derivative integrals remain FP64/complex128;
- auxiliary metric derivatives remain FP64.

The mixed derivative path requires the MO-direct response backend.  The packed
reference remains FP64.

## Warnings and failure semantics

- A non-FP64 profile is restricted to single-GPU, three-dimensional,
  positive-metric real-Gamma GDF with mapped-host CDERI.
- Sampled k-points, stress, UHF/ROHF, hybrid targets and NLC/VV10 are outside
  this precision-profile contract.
- Changing profiles after CDERI construction resets the provider and source
  state; relabelling or casting an existing factor is forbidden.
- An OOM or failed physical residual must not be reported as successful mixed
  precision.  Native-FP64 response-only Z3 may fall back.  A persistent-FP32
  source calculation fails if its mandatory final quantized-Hamiltonian audit
  cannot be satisfied.
- `fp32_build` convergence says that the approximate Hamiltonian is
  stationary.  It says nothing by itself about the FP64 energy or force bias.

## What to record in validation

At minimum, retain:

- requested and effective GDF profile;
- provider build ledger and persistent CDERI dtype;
- requested and effective source-SCF policy;
- source energy, target HF-DFT energy and SCF residuals;
- requested and effective Z precision, quantized residual and physical
  residual;
- requested and effective GDF derivative precision;
- force RMS and maximum difference in both Ha/Bohr and eV/Angstrom;
- structure identity, basis, pseudopotential, auxiliary basis, mesh,
  `cell.precision`, resolved omega and software versions.

For MLIP data, inspect both a diverse structural sample and consecutive frames.
Report constant energy offsets separately from structure-dependent relative
energy errors, and inspect the distribution and outliers of force errors.  A
small residual for one structure is not a transferable accuracy guarantee.

The runtime reports are available as:

```python
mf.gdf_precision_report
mf.source_precision_report
mf.with_df.gdf_precision_report
mf_grad.mixed_precision_report
mf_grad.precision_policy_effective
mf_grad.z_residual_max_abs
mf_grad.gdf_derivative_precision
```

## Related material

- `pbc_gdf_acceleration.md`: all GDF acceleration paths, including those not
  based on precision reduction
- `pbc_migration.md`: supported PBC capability and release validation history
- `examples/pbc/gpu/04-gdf-acceleration.py`: runnable small-system example
