# RSDF Water64/SZ Performance Record

## Scope

This record tracks exact, fixed-cell restricted DC-DFT RSDF performance
changes. The user explicitly permits Water64 runs with the SZV-GTH basis for
this task. Water64/DZ remains outside the development loop.

The optimizations must preserve the following contracts:

- provider-owned central CDERI values remain authoritative;
- source and metric derivatives remain analytic;
- reciprocal sums retain every grid vector;
- no CPU DC-DFT gradient or finite difference enters the analytic route;
- Gamma and sampled-k behavior share the same algebra;
- OOM retries repeat the failed interval without dropping or double-counting
  work.

## Baseline

Reference result:

`benchmark/results/water64_rsdf_benchmark_SZ/000_water64_selected_fd_check_gpu_rsdf/report.json`

The input is Water64, SZV-GTH orbital and auxiliary bases, Gamma sampling,
500 Ha cutoff, r2SCAN, matrix-free Z-vector, and no finite difference.

| Quantity | Baseline |
| --- | ---: |
| HF SCF | 41.4706 s |
| target DFA energy | 63.9762 s |
| energy total | 105.4484 s |
| analytic gradient | 128.9630 s |
| analytic total | 234.4114 s |
| host peak RSS | 42.344 GiB |
| sampled device peak | 92.873 GiB |
| host central source rebuild | 9.8234 s |
| reciprocal source response | 18.5286 s |

The host peak is caused by independently rebuilding the central three-center
source after the provider has already supplied the central CDERI. The device
peak is caused by a reciprocal block that keeps an AO-pair grid tensor and an
equally large source-contracted grid tensor alive at the same time.

## Optimization R1: Provider-Owned Central CDERI

Production response assembly now factors the reconstructed metric and reuses
the provider-owned central CDERI directly. It no longer rebuilds the central
three-center source merely to reproduce the same CDERI. Independent source
reconstruction and value-parity checking remain available in the reference
and diagnostic route.

Expected effect:

- remove the 9.8234 s source-value rebuild from the production gradient;
- remove the approximately 37 GiB transient host source allocation;
- preserve source and metric adjoint equations exactly.

## Optimization R2: Bounded Reciprocal Source Blocks

The native reciprocal source response now uses at most 25% of currently free
device memory and caps one block at 4096 reciprocal vectors. AO-pair and
source-contracted grid tensors are evaluated in non-overlapping lifetimes.
The plan and completed adaptive execution are recorded in gradient metadata.

Expected effect:

- reduce Water64/SZ response VRAM substantially below the 92.873 GiB baseline;
- retain all 15,625 source-mesh reciprocal vectors in several exact blocks;
- preserve transactional OOM rebatching as a final safety mechanism.

## Acceptance Gates

1. Unit tests cover provider-CDERI factorization, bounded planning, and
   transactional rebatching.
2. Gamma and sampled-k CPU/GPU gradient parity remains within
   `1e-7 Ha/Bohr`.
3. Selected analytic/FD parity remains within `1e-5 Ha/Bohr` on the release
   fixture.
4. Water64/SZ completes without OOM.
5. Water64/SZ host RSS and sampled VRAM both decrease from baseline.
6. Timing and memory deltas are appended below using an independent output
   directory; the baseline result is never overwritten.

## Results

### R1 and R2

Post-change result:

`benchmark/results/water64_rsdf_benchmark_SZ_opt_r1_r2/000_water64_selected_fd_check_gpu_rsdf/report.json`

| Quantity | Baseline | R1 + R2 | Change |
| --- | ---: | ---: | ---: |
| analytic response stage | 37.0692 s | 26.3096 s | -29.03% |
| analytic gradient | 128.9630 s | 118.3859 s | -8.20% |
| analytic total | 234.4114 s | 223.6650 s | -4.58% |
| host peak RSS | 42.344 GiB | 6.697 GiB | -84.18% |
| sampled device peak | 92.873 GiB | 23.332 GiB | -74.88% |
| host source-value build | 9.8234 s | 0.0000 s | -100% |
| host primitive transfer | 448.172 MiB | 16.172 MiB | -96.39% |

The reciprocal execution completed all 15,625 vectors in four blocks of at
most 4096 vectors with zero OOM rebatches. Energy differs by
`2.27e-13 Ha`; the analytic gradient differs by at most
`1.60e-11 Ha/Bohr` with RMS `3.88e-12 Ha/Bohr`.

The remaining end-to-end bottleneck is the 105 s energy stage. R1/R2 do not
change it. The next exact optimization caches the CPU-witness GTH
pseudopotential value for repeated HF and target-DFA hcore requests at the
same geometry and k-point argument. Geometry or provider reset invalidates
the cache.

### R3: Pseudopotential Value Cache

Post-change result:

`benchmark/results/water64_rsdf_benchmark_SZ_opt_r1_r2_r3/000_water64_selected_fd_check_gpu_rsdf/report.json`

The first HF hcore request computes the CPU-witness GTH pseudopotential value
as before. The target-DFA hcore request reuses the cached value. Each caller
receives a device copy because PySCF hcore assembly may modify its returned
matrix in place. The cache therefore avoids the periodic integral rebuild
without sharing mutable hcore storage.

| Quantity | Baseline | R1 + R2 + R3 | Change |
| --- | ---: | ---: | ---: |
| HF SCF | 41.4706 s | 42.9179 s | +3.49% run variation |
| target DFA energy | 63.9762 s | 33.2045 s | -48.10% |
| energy total | 105.4484 s | 76.1238 s | -27.81% |
| analytic gradient | 128.9630 s | 118.4118 s | -8.18% |
| analytic total | 234.4114 s | 194.5356 s | -17.01% |
| host peak RSS | 42.344 GiB | 6.696 GiB | -84.19% |
| sampled device peak | 92.873 GiB | 23.363 GiB | -74.84% |

The final run reports one PP-cache miss and one hit. Its energy differs from
baseline by `2.27e-13 Ha`; its analytic gradient differs by at most
`1.60e-11 Ha/Bohr` with RMS `3.88e-12 Ha/Bohr`.

The Water64 timings are one independent run per version rather than a
three-run median, and both Water64 jobs used `run_fd=false`. Numerical
equivalence here is therefore a before/after analytic comparison; absolute
FD gates remain covered by the smaller release fixtures.

Validation completed with 16 RSDF tests, 4 MGGA tests, 7 reciprocal-memory
tests, 12 benchmark lifecycle tests, and 2 reference-GDF regression tests.
The RSDF suite covers Gamma and sampled-k CPU/GPU energy and gradient parity,
full-tensor/reference response parity, transactional OOM retry, cache
mutation isolation, and cache reset. The MGGA suite covers SCAN and r2SCAN.

The largest remaining RSDF cost is the first HF setup, including host central
CDERI and the first pseudopotential value generation. Replacing it with a
native GPU central builder is a separate numerical-contract change: the
GPU4PySCF builder chooses its own range-separation representation, while the
analytic response currently follows PySCF's independently resolved source
and metric meshes. It should be promoted only after primitive and full-force
parity tests, not as part of this exact cache/memory patch.

## Optimization R4: GPU RSDF Pseudopotential Value

The first PP value is evaluated with GPU4PySCF's RSDF PP builder. Local part 1
and the nonlocal projector are device kernels; upstream local part 2 remains a
host primitive. The immutable result is then reused by the R3 cache.

Small-system matrix parity before promotion:

| Sampling | Maximum matrix difference |
| --- | ---: |
| Gamma | `1.32e-11 Ha` |
| `2 x 1 x 1` | `1.31e-11 Ha` |

The first direct upstream implementation reduced HF setup but allocated 80%
of free device memory for one local-part-1 Fourier block. Its Water64/SZ peak
was 76.221 GiB. The retained implementation uses a project-local bounded
wrapper with a 20% memory budget and maximum block size of 4096 grid vectors.
OOM retries are transactional.

Final result:

`benchmark/results/water64_rsdf_benchmark_SZ_opt_r1_r2_r3_r4_bounded/000_water64_selected_fd_check_gpu_rsdf/report.json`

| Quantity | Baseline | R1 + R2 + R3 + bounded R4 | Change |
| --- | ---: | ---: | ---: |
| HF SCF | 41.4706 s | 12.8725 s | -68.96% |
| target DFA energy | 63.9762 s | 33.2753 s | -47.99% |
| energy total | 105.4484 s | 46.1494 s | -56.24% |
| analytic gradient | 128.9630 s | 118.2486 s | -8.31% |
| analytic total | 234.4114 s | 164.3980 s | -29.87% |
| host peak RSS | 42.344 GiB | 6.691 GiB | -84.20% |
| sampled device peak | 92.873 GiB | 23.201 GiB | -75.02% |

The PP mesh contains 4,913 grid vectors. It completed in blocks of 4096 and
817 vectors with zero OOM rebatches. Relative to baseline, energy changes by
`2.16e-9 Ha`; the analytic gradient changes by at most
`4.30e-11 Ha/Bohr` with RMS `6.73e-12 Ha/Bohr`.

The final bounded implementation passed 18 RSDF tests and 4 MGGA tests,
including Gamma and sampled-k CPU/GPU energy and gradient parity.

## Native Central CDERI Probe

GPU4PySCF's unmodified native central builder was compared against the
PySCF-owned RSDF factors required by the analytic response contract.

| Sampling | Direct CDERI maximum difference |
| --- | ---: |
| Gamma | `7.20e-6` |
| `2 x 1 x 1` | `5.00e-6` to `7.20e-6` |

These gaps exceed the `1e-7` central-value contract. The native builder uses
its own range-separation representation rather than the independently
resolved PySCF source and metric settings. It is therefore not connected to
production. A future native central route must expose and reproduce those
exact source/metric settings, followed by primitive and full-gradient parity.

## Optimization R5: Exact GPU-Source Central CDERI

The direct probe above was decomposed into the source and metric before
changing production ownership. With PySCF's resolved source settings
(`omega=0.3`, source mesh, and `precision_R`) injected into GPU4PySCF, the
three-center source reconstructed from the GPU CDERI agreed with PySCF to
`7.74e-12`. The remaining CDERI error came from the screened two-center
metric: the GPU and PySCF short-range metrics differed by `1.43e-4` on H2.
Their reciprocal long-range metric terms agreed to machine precision.

The retained exact route therefore has the following ownership:

- PySCF resolves independent source and metric omega/mesh values without
  building host CDERI;
- GPU4PySCF builds the large three-center source;
- PySCF evaluates only the screened `O(naux**2)` metric primitive;
- GPU4PySCF evaluates the reciprocal metric correction and CuPy performs the
  Cholesky/CDERI transformation.

The original CPU witness has `_cderi=None` in this mode. The former host
object proportional to `naux*nao**2` is not generated. The exact central
adapter is serialized by a reentrant lock because GPU4PySCF 1.7.4 does not
yet expose source and metric RSDF settings as independent builder arguments;
all temporary upstream bindings are restored in `finally`.

### Correctness gates

| Case | Maximum central-CDERI difference |
| --- | ---: |
| H2 Gamma | `1.17e-11` |
| H2 `2 x 1 x 1` | `1.42e-11` |
| heterogeneous H2O auxiliary basis | `< 1e-9` |

The initial heterogeneous-basis probe exposed an invalid call from PySCF's
screened metric driver on GPU4PySCF's internal `SortedGTO`. The fixed route
evaluates the host metric on the original AuxCell and uses the sorted object
only for GPU Fourier work; both return the final original AO order.

Gamma and `2 x 1 x 1` end-to-end CPU/GPU energy-gradient tests passed. The
Water64 analytic energy and gradient differ from the R4 reference by
`1.11e-10 Ha` and `5.23e-11 Ha/Bohr` maximum, respectively.

### Bounded source-builder memory

GPU4PySCF's central builder sizes AO-pair and reciprocal buffers from nearly
all free VRAM. The adapter now reports at most 512 MiB of working memory to
that builder while preserving enough space for its full auxiliary Fourier
cache plus 64 MiB of dynamic work. This changes batch sizes only. No source
pair or reciprocal vector is removed.

On Water20, this policy reduced the exact route's HF-stage sampled peak from
2.45 to 1.88 GiB without changing central parity. On Water64 it reduced the
same peak from 19.52 to 10.17 GiB. The later gradient stages still determine
the end-to-end device peak.

### Performance evidence and promotion decision

Water8 values are independent-process medians over three runs. Water20 and
Water64 values are one run per route and are scaling evidence rather than
timing medians.

| System | Quantity | host reference | exact GPU source | Change |
| --- | --- | ---: | ---: | ---: |
| Water8 | HF SCF | 0.8889 s | 0.7707 s | -13.30% |
| Water8 | analytic total | 5.8288 s | 5.7085 s | -2.06% |
| Water8 | host peak RSS | 2.967 GiB | 2.737 GiB | -7.75% |
| Water20 | analytic total | 24.4391 s | 24.3936 s | -0.19% |
| Water20 | host peak RSS | 3.107 GiB | 2.911 GiB | -6.30% |
| Water64 | energy wall | 46.1494 s | 45.8851 s | -0.57% |
| Water64 | analytic gradient | 118.2486 s | 123.9730 s | +4.84% |
| Water64 | analytic total | 164.3980 s | 169.8581 s | +3.32% |
| Water64 | host peak RSS | 6.691 GiB | 6.006 GiB | -10.24% |
| Water64 | sampled device peak | 23.201 GiB | 24.289 GiB | +4.69% |

The exact route is retained as explicit
`central_setup="exact_gpu_source"` / benchmark
`rsdf_central_setup="exact_gpu_source"`. It is useful when host memory is the
constraint and removes the large host central source. It is not promoted to
the default because the current evidence does not show a 5% end-to-end speed
improvement and does not reduce the whole-workflow VRAM peak. The default
remains `reference`; this is an explicit measured policy, not a silent CPU
fallback.

### Final regression record

After the bounded exact-central implementation, the complete RSDF GPU suite
passed 25 tests in 431.112 s. The SCAN/r2SCAN MGGA suite passed 4 tests, the
benchmark lifecycle suite passed 12 tests, and the CPU/GPU package-boundary
checks passed 4 tests. A public sampled-k exact-central example also completed
without an error. These gates cover Gamma and sampled-k central-factor parity,
heterogeneous auxiliary bases, CPU/GPU energy-gradient parity, memory-cap
metadata, and the requirement that the parameter-only CPU witness never builds
`_cderi`.

No new Water64 finite-difference calculation was performed for R5. Water64
correctness is the analytic exact-central/reference parity reported above;
finite-difference correctness remains covered by the smaller release-gate
systems where the same response and assembly paths are exercised.
