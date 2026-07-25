# Stress Optimization Log

## 2026-07-22: O0/O1 start

### Scope

- Work is confined to `stress/`.
- `grad/`, `pbc/cpu/`, and `pbc/gpu/` are read-only dependencies.
- Water20/Water64 are user-run scaling gates and are not launched automatically.

### Existing evidence

The current contracted Water8 Gamma route has historical dedicated-GPU medians
near 16 s for stress. PP takes roughly 13.5 s and DF roughly 2 s. At sampled
`2x1x1` k-points, total stress grows to roughly 52 s and PP to roughly 49 s.
These records identify nonlocal PP projector work as the first optimization
target, but they do not replace a current-source O0 baseline.

The contracted route is not yet production analytic:

- nonlocal PP uses centered cell displacement of projector factors,
- contracted GDF/RSDF uses 12 fixed-density provider builds,
- metadata therefore keeps `production=false`.

### O0 artifacts

- `optimization_o0_h2_correctness.json`: exact and contracted, Gamma/k211,
  GDF/RSDF, six-component full-energy FD.
- `optimization_o0_water8_correctness.json`: exact and contracted Gamma
  GDF/RSDF with selected `xx/xy` FD.
- `optimization_o0_water8_timing.json`: three explicit repeats per route,
  no FD, intended for an otherwise idle GPU.

No Water8 timing in this log is promoted until all repeated jobs use the same
source digest and checkpoint reuse is absent.

### O1 instrumentation

The first instrumentation increment records exact work counts and explicit
payload estimates without changing numerical arrays:

- PP grid blocks, AO evaluations, FFTs, displaced cells, projector templates,
  atom contractions, and known host/device payloads,
- contracted DF cell/provider/J-K build counts and returned matrix payload,
- exact DF canonical-pair, factorization, derivative evaluation, reverse-pair
  reuse, and derivative-transfer payload estimates.

CUDA-event timing remains a separate opt-in increment because host-heavy stages
cannot be represented honestly by a single device event.

### O1 first current-source measurement

One Water8 Gamma/GDF contracted timing job completed without FD:

- source HF and target energy: 2.08 s
- analytic stress: 17.66 s
- PP response: 15.01 s
- nonlocal PP projector: 7.84 s
- contracted DF response: 2.33 s
- peak sampled device-use delta: 11.67 GB

The work profile exposed 96 atom/projector contractions and about 7.95 GB of
explicit host-to-device projector workspace traffic. This is repeated transfer,
not required mathematical information.

### O2 increment 1: device-resident PP phase data

The strain-invariant structure factor is now uploaded once. For each displaced
cell and k-point, an element projector template is uploaded once and atom phases
are applied on the GPU. The previous path formed and uploaded one full
`(nprojector, nG)` workspace per atom. The new path keeps the same equations,
six centered nonlocal directions, and source/adjoint fused contraction.

Explicit live-workspace estimates intentionally exclude hidden cuBLAS, FFT, and
CUDA allocator workspaces. They are compared with the sampled process/global
VRAM peak rather than presented as a complete peak predictor.

Measured result:

- explicit PP H2D payload: 7.95 GB -> 1.66 GB (`-79.2%`)
- nonlocal PP: 7.84 s -> 5.02 s optimized median (`-35.9%`)
- total PP: 15.01 s -> 12.19 s optimized median (`-18.8%`)
- analytic stress: 17.66 s -> 14.81 s optimized median (`-16.1%`)
- host peak RSS: unchanged within measurement resolution
- sampled global VRAM delta: 11.67 GB -> 12.33 GB (`+5.7%`)
- selected `xx/xy` energy-FD maximum error: `9.37e-11 Eh/Bohr^3`

The performance comparison uses one paired pre-change run and three post-change
runs. It passes the O2 Gamma performance and selected-FD gates, but sampled-k
parity and an exact PP contribution comparison remain open. The machine-readable
record is `optimization_o2_pp_device_cache_summary_2026-07-22.json`.

### O1 detailed profile and sampled-k O2 gate

The opt-in CUDA event profiler separates Python wall time from current-stream
elapsed time. On Water8 Gamma/GDF it explains 7.64 s of a 12.17 s PP wall:

- target XC stream interval: 4.19 s
- local density response: 1.34 s
- nonlocal FT-AO: 1.41 s
- nonlocal projector-AO contraction: 0.27 s
- CPU projector-template construction: 1.08 s wall
- scalar device-to-host synchronization: 0.26 s wall

The remaining difference consists of cell/G-vector preparation, host array work,
allocator/library workspace, and synchronization boundaries. CUDA event values
are stream intervals, not a hardware activity trace.

For Water8 `kmesh=[2,1,1]` GDF, the same O2 change gave:

- analytic stress: 52.28 s -> 35.73 s (`-31.7%`)
- PP response: 49.52 s -> 33.00 s (`-33.4%`)
- nonlocal PP: 15.55 s -> 9.76 s (`-37.3%`)
- sampled global VRAM delta: `+4.0%`
- host peak RSS: `+7.7%`
- selected `xx` energy-FD error: `6.92e-11 Eh/Bohr^3`

This closes the sampled-k GDF performance/FD part of O2. RSDF and exact PP
component parity remain open.

The corresponding sampled-k RSDF result also passed:

- analytic stress: 52.61 s -> 36.46 s (`-30.7%`)
- PP response: `-32.8%`
- nonlocal PP: `-36.0%`
- sampled global VRAM delta: `+4.3%`
- host peak RSS: `+7.5%`
- selected `xx` energy-FD error: `6.66e-11 Eh/Bohr^3`

The machine-readable GDF/RSDF record is
`optimization_o2_sampled_k_summary_2026-07-22.json`.

### H2 O0 input correction

The first H2 contracted matrix exposed two independent issues:

1. a PP with no nonlocal projector channel passed an empty atom list to
   `cell.get_SI`; this now returns an explicit zero tensor and zero-work profile,
2. `mesh=[9,9,9]` was too coarse for the contracted full local GTH potential.

The H2 O0 queue now uses `ke_cutoff_ha=400.0`. Exact and contracted routes must
be compared on that common converged uniform mesh; the coarse-mesh runs are not
accepted as cross-route parity evidence.

The corrected H2 matrix completed all eight Gamma/k211, GDF/RSDF,
exact/contracted cases. Every six-component FD maximum was about
`8.7e-11 Eh/Bohr^3`; exact/contracted stress differences were at most
`2.9e-13 Eh/Bohr^3`, and PP raw derivative differences were about `1e-12 Eh`.
The machine-readable record is `optimization_o0_h2_summary_2026-07-22.json`.

### O1 status

O1 is complete for the current stress boundary:

- PP wall and CUDA stream intervals are separated,
- target Fock/RHS, Z solver, target XC, PP local/nonlocal and DF timings are
  reported,
- Z iterations, mean iteration time and residual are visible,
- PP AO/grid/projector and DF pair/provider work counts are recorded,
- explicit transfer payload and live-workspace estimates are recorded,
- benchmark stage peaks retain process/global VRAM and host RSS evidence.

The profiler remains opt-in and does not change the default route.

### Rejected O3 experiment: batching the two density contractions

A trial combined source/adjoint `D*AO`, FFT, and reciprocal contractions along
the density axis. It was numerically equivalent within `3.7e-14 Eh/Bohr^3` and
reduced FFT calls from two to one and reciprocal contractions from 18 to 9.
However, CuPy selected a less favorable batched contraction path:

- local AO-density stage: 1.60 s -> 2.77 s (`+72.9%`)
- total PP: `+9.3%`
- total stress: `+7.5%`
- VRAM delta: `+0.6%`

The experiment was rejected and the per-density GEMM route restored. Lower
launch count alone is not a valid optimization metric for this small density
batch.

### O4 increment 1: analytic streaming direct-contracted DF

`host_analytic_direct_contracted` now processes one canonical pair at a time.
Its six differentiated factors are contracted directly with the reference and
adjoint transition densities and released. It retains no pair derivative,
central factor, or six-component AO J/K response.

H2 Gamma/k211 GDF/RSDF results:

- exact matrix-route maximum difference: `<= 1.0e-18 Eh/Bohr^3`
- six-component energy-FD maximum error: about `8.8e-11 Eh/Bohr^3`
- `production=true` with host analytic PP
- low-level numerical component list: empty

The machine-readable record is
`optimization_o4_h2_direct_df_summary_2026-07-22.json`. The current host builder
still materializes all directions for the active pair, so Water8 scaling and
pair-internal directional streaming remain open.

### O4 increment 2: Water8 RSDF memory and accuracy gate

The Water8/SZV-GTH/r2SCAN/Gamma RSDF direct-contracted route completed with a
selected `xx` energy-FD error of `6.93e-11 Eh/Bohr^3`. The direct DF contraction
retained zero bytes for pair derivatives, central device factors, and six AO
J/K response matrices. Its estimated active-pair device payload was 1.80 MB,
with 9.03 MB of host primitive payload and 10.8 kB of global auxiliary
accumulators.

This increment is primarily a memory result, not yet a speed result:

- total analytic stress: 129.64 s
- PP response: 12.22 s
- DF response: 117.13 s
- host RSDF source primitive sum: 116.58 s
- host metric primitive sum: 0.50 s
- device direct contraction: 0.017 s

The dominant cost is therefore the CPU RSDF source builder, which still creates
all nine Cartesian strain directions for the active pair before the six Voigt
directions are consumed. The next exact optimization target is pair-internal
directional or density-contracted source streaming. It must preserve PySCF's
RSDF parameters and the existing analytic equations. CP2K/QE-style task and
block organization may inform the scheduling, but no GPL implementation code
is copied.

This Water8 run reports `production=false` only because the selected fast PP
backend still uses a low-level centered cell displacement. The analytic DF
route itself contains no numerical DF displacement. The machine-readable record
is `optimization_o4_water8_direct_df_summary_2026-07-22.json`.

The matching GDF case also completed and passed its selected `xx` FD gate:

| Backend | Stress (s) | DF (s) | Host source (s) | Host metric (s) | Device contraction (s) | Peak RSS (GiB) | FD max error |
|---|---:|---:|---:|---:|---:|---:|---:|
| GDF | 653.41 | 640.84 | 639.02 | 1.77 | 0.018 | 12.67 | `7.12e-11` |
| RSDF | 129.64 | 117.13 | 116.58 | 0.50 | 0.017 | 9.03 | `6.93e-11` |

GDF is 5.04 times slower overall and its host source builder is 5.48 times
slower than RSDF for this exact Water8 gate. Both routes have identical
zero-retention contracts for pair derivatives, central factors, and AO response
matrices. Consequently, the next optimization should not alter the already
negligible GPU scalar contraction. It should reduce the GDF host source builder's
active-pair work and peak host memory, then apply the same design to RSDF.

### O4 increment 3: compact symmetric-Voigt source contraction

The direct route now accumulates the six symmetric engineering-Voigt source
directions inside each reciprocal block. It no longer forms separate `xy/yx`,
`xz/zx`, and `yz/zy` final contractions and projects them later. AO-pair,
auxiliary, and kernel response contractions therefore require 18 rather than 27
matrix products per block. The public Cartesian source API remains unchanged;
only the direct-contracted internal route requests compact output.

H2 Gamma/k211 GDF/RSDF remained equal to the Cartesian reference within
`4.34e-19 Eh/Bohr^3`; all six-component FD errors stayed near `8.8e-11` and the
host-analytic-PP gate remained `production=true`.

Water8 results against the immediately preceding O4 direct route:

| Backend | Stress before/after (s) | Source before/after (s) | Source payload change | Peak RSS change | FD max error |
|---|---:|---:|---:|---:|---:|
| GDF | 653.41 / 603.43 | 639.02 / 589.11 | -29.4% | +0.5% | `7.05e-11` |
| RSDF | 129.64 / 120.76 | 116.58 / 107.72 | -29.4% | -0.1% | `6.92e-11` |

This exact change is retained because it removes duplicate work without a
regression. It does not close the 15% total-time or 25% peak-memory gate. The
8 GiB reciprocal block policy still dominates peak RSS, so the next experiment
must bound that workspace and report its wall-time cost separately.

### O4 increment 4: separate DF block ownership and bounded-memory profiles

The original `stress_host_reciprocal_block_size` affected both host PP and DF.
That made a nominal DF memory experiment change the low-level numerical PP
derivative as well. `stress_df_host_reciprocal_block_size` now independently
controls only the DF source and metric builders. If omitted, it inherits the
legacy global value for backward compatibility. Reports expose both requests.

Paired Water8 measurements with the fast PP route fixed:

| Backend/profile | Stress change | Source change | Peak RSS change | DF scalar parity | Selected FD |
|---|---:|---:|---:|---:|---:|
| GDF block4096 | +19.4% | +20.0% | -52.8% | target-J `6.75e-14 Eh` | `6.90e-11` |
| RSDF block4096 | +15.3% | +17.2% | -33.8% | target-J `1.01e-13 Eh`, adjoint-RHF `6.66e-15 Eh` | `3.60e-11` |
| GDF block8192 | +30.8% | +31.6% | -31.5% | target-J `1.23e-13 Eh` | `1.83e-9` |

Block4096 is retained as an explicit OOM-avoidance profile, not a speed default.
Block8192 is rejected as a general balanced profile because the larger GEMMs
used fewer CPU cores and were slower on this host. The relation between block
size and wall time is therefore non-monotonic and should not be auto-tuned from
memory size alone.

The fast PP route uses a low-level cell displacement and its raw PP derivative
varied between independent runs even when central energy and Z residual were
identical. For this reason the benchmark now records the six raw contracted DF
target-Coulomb and adjoint-RHF scalars. DF block parity is judged from those
owner-specific values rather than total stress contaminated by PP run-to-run
variation.

### O2 increment 3: device-contracted analytic nonlocal PP

The host analytic reference spent 539--558 seconds building full nonlocal PP
strain matrices for Water8. The new route derives AO Fourier first moments from
an augmented Cartesian shell (`l -> l+1`) and directly accumulates six symmetric
projector responses into the source and adjoint densities on GPU. It uses no
cell displacement and retains no PP AO derivative matrix. Small projector
value-gradient templates remain an explicit host boundary.

Same-revision component gates cover Gamma, complex sampled-k, and DZVP d-shell
polarization. Water1 GDF/RSDF full six-component FD errors are about `6.3e-12
Eh/Bohr^3`. Water8 selected-FD results are:

| Backend | Stress (s) | PP (s) | Nonlocal PP (s) | DF (s) | RSS (GiB) | VRAM delta (GiB) | FD max error |
|---|---:|---:|---:|---:|---:|---:|---:|
| GDF | 605.21 | 12.05 | 6.40 | 592.85 | 12.38 | 7.32 | `6.88e-11` |
| RSDF | 123.31 | 12.06 | 6.34 | 110.92 | 8.68 | 7.33 | `6.90e-11` |

Relative to the host analytic matrix, PP wall fell by 97.8%. Relative to the
numerical fast route, total stress differs by less than 0.5%, PP wall is
slightly lower, and sampled VRAM delta is about 36% lower. Both runs report an
empty numerical-component list and `production=true`.

These Water8 cases enabled detailed profiling and are single runs. A complete
sampled-k end-to-end FD gate, profiler-off three-repeat median, and Water20/64
user scaling gate remain open. Detailed evidence is in
`S7_GPU_Analytic_Nonlocal_PP_2026-07-23.md` and
`../validation/results/2026-07-23_o2_device_analytic_nonlocal_pp.json`.

The first device implementation still rebuilt the same AO and projector
responses for every oxygen atom. Grouping atoms by element/angular channel and
moving the component loop outside the atom loop reduced each overlap call count
from 960 to 120. The warm Water8 nonlocal-only wall fell from 6.42 to 1.96 s
(`-69.5%`). A final Water8 RSDF workflow retained the `6.90e-11 Eh/Bohr^3` FD
error while reducing PP from 12.06 to 7.58 s and total stress from 123.31 to
117.61 s. This is 4.2% faster than the numerical fast route, with no numerical
stress component.

The 3D local GTH reciprocal setup then moved its FFT-frequency G vectors,
$G^2$, damping, polynomial slope, and atom structure-factor sums from NumPy to
CuPy. It preserves PySCF FFT ordering through GPU4PySCF's `_get_Gv_with_base`.
Water8 local setup fell from 1.315 to 0.125 s (`-90.5%`). The final RSDF run
retained a `6.90e-11 Eh/Bohr^3` FD error, with PP 6.37 s and total stress
116.49 s. This is 5.2% faster than the numerical fast route and 35.6% lower in
sampled VRAM delta.

A clean rerun after restoring the accepted C source path gave 114.29 s total,
6.40 s device PP response, 1.90 s nonlocal PP, and 107.60 s DF response. The
selected-FD error remained `6.90e-11 Eh/Bohr^3`, peak RSS was 8.67 GiB, and the
sampled VRAM delta was 7.41 GiB. The small total-time change is ordinary
host-source timing variation; the PP work counts and memory contract are
unchanged.

### Rejected O4 experiment: fusing the AO coordinate moment into the C kernel

An experiment extended the host fused overlap/source kernel from seven outputs
(overlap plus left/right AO derivatives) to ten outputs by adding the AO
coordinate moment. This removed the separate `GTO_ft_rc` pass and preserved
source-component parity. On the paired Water8 RSDF run, however, host source
time only changed from about 109.15 s to 107.17 s (`-1.8%`), while peak RSS rose
from about 8.67 GiB to 9.53 GiB (`+10%`). It therefore failed both the stage
speed criterion and the no-memory-regression rule. The ten-output kernel was
rejected, the seven-output kernel and separate coordinate-moment pass were
restored, and the accepted route was rebuilt and rerun successfully.

### O4 increment 5: reverse adjoint audit and factorized AO-pair response

The established PBC gradient reverse-Cholesky algebra was connected to stress
and checked independently. For complex synthetic source/metric directions,
the forward contraction `Re(G_C dC)` and reverse contraction
`Re(W_B dB) + Re(W_M dM)` agreed within `1.8e-15`. H2 GDF/RSDF Gamma and
sampled-k factorized-source runs passed selected FD with errors between
`1.98e-11` and `2.83e-11 Eh/Bohr^3`.

The first host source-adjoint implementation was exact but rejected as a speed
route. On Water8 RSDF it changed stress from 116.70 s to 139.80 s and RSS from
8.66 GiB to 9.29 GiB. Its full-grid projection had shape approximately
`(2,nG,nao^2)`. Block 1024 reduced RSS to 3.56 GiB but increased stress to
174.18 s; block 2048 took 180.75 s. These remain development evidence only.

Exclusive profiling of the accepted forward source found:

| Water8 RSDF source phase | Wall (s) |
|---|---:|
| AO Fourier integrals | 21.25 |
| AO-pair response total | 83.62 |
| response image assembly | 60.95 |
| response matrix products | 22.67 |
| auxiliary/kernel response | 0.11 |

Moving only the six matrix products to CuPy did not help because the host image
assembly remained. It gave 118.77 s stress and 9.16 GiB RSS and was rejected.

The accepted factorized formulation projects the three left-center, three
right-center, and Fourier-moment factors first. Atomic-coordinate and momentum
weights are then applied to the much smaller `(nao^2,naux)` results. This is an
exact reassociation of the same contractions; no cutoff, mesh, precision, or
screening setting changes. On the paired Water8 RSDF revision:

| Route | Source (s) | Stress (s) | RSS (GiB) | FD max error |
|---|---:|---:|---:|---:|
| assembled | 109.24 | 116.70 | 8.66 | prior exact gate |
| factorized | 79.73 | 87.11 | 8.39 | `6.90e-11` |

The source and total stress changes are `-27.0%` and `-25.4%`. Small cells are
slower in factorized form, so the direct route uses assembled response below a
256 MiB one-image working size and factorized response at or above it. This
auto policy is exact and records both requested and resolved modes.

The same Water8 GDF gate reduced host source from 591.01 s to 396.20 s
(`-33.0%`) and total stress from 605.21 s to 404.83 s (`-33.1%`). Peak RSS
fell from 12.38 GiB to 11.95 GiB. Against the prior FD-passed assembled GDF
run, the raw target-J and adjoint-RHF components differ by at most
`7.1e-15 Ha` and `1.2e-15 Ha`, respectively.

### O4 increment 6: bounded GPU factor projection

The first full-device factorized projection accelerated Water8 but retained
large host-to-device staging buffers. It gave 59.85 s stress and 9.37 GiB RSS
for RSDF, and 288.23 s and 13.91 GiB for GDF. A second implementation now
accumulates over 512 MiB G-vector transfer blocks. Each raw center/moment slice
is transferred once per block; only the fifteen projected outputs remain live
across blocks. An explicit free-VRAM preflight covers those outputs, one block,
and contraction temporaries.

Against host factorized projection, bounded Water8 RSDF reduced source/stress
from 79.73/87.11 s to 32.62/40.00 s (`-59.1%`/`-54.1%`) with RSS changing from
8.39 to 8.87 GiB. Selected FD remained `6.90e-11 Eh/Bohr^3`. GDF reduced
396.20/404.83 s to 262.50/271.11 s (`-33.7%`/`-33.0%`) with RSS changing from
11.95 to 12.44 GiB. Stress parity was `2.81e-17` for RSDF and `1.66e-17
Eh/Bohr^3` for GDF. H2 Gamma and sampled-k GDF/RSDF FD errors remained below
`2.9e-11 Eh/Bohr^3`.

This exact route passes the stage speed and less-than-10% memory-regression
criteria and is retained as an explicit production-capable opt-in. It is not
yet the automatic default because large sampled-k and Water20/64 projected
output scaling remain unmeasured. Water8 Z-vector/AX is only about 0.25 s; the
next source target is the host AO Fourier integral (20.5 s RSDF, 112.6 s GDF)
and the remaining GDF projection (128.1 s).

### O4 increment 7: GPU FT reverse-source contraction

GPU4PySCF 1.7.4's Apache-2.0 PBC FT strain kernels were adapted at the
low-level contraction boundary. Reverse Cholesky produces the exact source
adjoint $W_B(ij,P)$, and a bounded auxiliary/G-vector contraction forms
$T(ij,G)=\sum_P W_B(ij,P)A(G,P)$ on device. The strain kernel contracts this
G-dependent AO weight directly, so neither AO Fourier derivative tensors nor
six projected source matrices are built. Short-range source, vbar, auxiliary
Fourier response, Coulomb-kernel response, and metric derivatives preserve
their established exact owners.

Gamma separable/general and sampled-k complex pair tests passed. The apparent
sampled-k pair gap at default `1e-8` precision decreased from `2.19e-9` to
`1.55e-11` and `1.78e-13` at `1e-10` and `1e-12`, confirming an integral
screening difference rather than a fitted phase or factor.

H2 RSDF Gamma/`2x1x1` FD errors are `2.83e-11`/`1.98e-11`; H2 native GDF
Gamma is `2.81e-11 Eh/Bohr^3`. Water8 RSDF Gamma/`2x1x1` errors are
`6.90e-11`/`6.74e-11`, and native GDF Gamma is `6.88e-11 Eh/Bohr^3`.

Profiler-off Water8 three-run medians are:

| Backend | Stress (s) | DF (s) | Source (s) | RSS (GiB) |
|---|---:|---:|---:|---:|
| RSDF Gamma GPU FT | 8.057 | 1.328 | 0.620 | 3.135 |
| native GDF Gamma GPU FT | 12.209 | 5.482 | 3.783 | 3.569 |

Against the bounded forward paths, stress fell by 79.9% for RSDF and 95.5%
for GDF; RSS fell by 64.7% and 71.3%. At this measurement point the route was
retained as an exact explicit opt-in while native GDF sampled-k and system
scaling were still open.

The native GDF sampled-k owner was then connected through GPU4PySCF's BvK
compressed-to-canonical mapping. H2 `2x1x1` matched selected FD at `1.98e-11
Eh/Bohr^3`. Water8 `2x1x1` matched at `6.73e-11 Eh/Bohr^3`, with 48.67 s
total stress, 12.48 s DF, 8.42 s source, 3.54 GiB RSS, and 14.19 GiB sampled
VRAM delta. This removes the mathematical/native sampled-k limitation;
larger-system scaling remains before Python API default promotion.

The H2 `DZVP-GTH` larger-basis gate then covered native GDF and RSDF at Gamma
and `2x1x1`. FD maxima were `2.94e-11`, `2.92e-11`, `2.21e-12`, and
`1.99e-12 Eh/Bohr^3`, respectively. Stress wall ranged from 0.459 to 1.374 s,
peak RSS from 1.847 to 1.885 GiB, and sampled VRAM delta from 2.19 GiB at Gamma
to 9.78 GiB at `2x1x1`. This closes the first polarized larger-basis
correctness gate without changing numerical parameters. Water20/64 scaling is
still user-run evidence.

### Release-audit preparation

The CPU and GPU stress wheels now build and import from a clean temporary
installation. The C extension links directly to PySCF `libcgto` and `libcint`
with a wheel-relative RPATH instead of relying on the benchmark runner to
preload unresolved symbols. Both wheels contain the Apache license, project
notice, original PySCF notice, and provenance summary; the GPU wheel declares
its direct `gpu4pyscf-pbc-dcdft` dependency.

The wheel-installed optimized H2 workflow completed energy, force, and stress
and wrote JSON/extxyz frames. Both formats retain tensile stress/virial units
and the exact PP/DF response owners, `production=true`, and source-HF/Z owner.
Final regression totals are 73 CPU, 18 GPU, and 10 benchmark tests.

`current_water20_sz_stress_scaling.json` contains only the two GDF/RSDF
stress-only runs needed for the remaining user scaling gate. The independent
`validate_scaling_gate.py` rejects stale failed reports and requires both
backends, at least 60 atoms, exact owners, no numerical component or silent
fallback, and complete wall/RSS/VRAM evidence.

### Water20 scaling attempt and GPU FT dispatch repair

The first user-run Water20/SZV-GTH/Gamma scaling attempt failed for both GDF
and RSDF in the common reciprocal source stage. GDF failed after 60.86 s and
RSDF after 58.72 s with `CUDA_ERROR_ILLEGAL_ADDRESS`. The failed stage was
`stress_df_host_primitives`; sampled global device-use deltas were 15.93 and
15.34 GiB, far below the 94.97 GiB device capacity. This was not a VRAM OOM.

The traceback established that a Gamma-only calculation entered
`_accumulate_sampled_k_general_strain`. The generic GPU FT option had treated
Gamma as a one-point sampled-k/BvK case. Those two paths have the same
mathematical limit but require different kernel-side density layouts. The
dispatcher now selects the dedicated Gamma layout whenever every k-point is
zero and retains the sampled-k/BvK layout only for nonzero k-points.

The failed run also exposed an overly permissive automatic G-block. Water20
has 120 AOs, 120 auxiliary functions and `mesh=[239,239,239]`. The old block
of 16384 made one complex AO-pair operand 3.516 GiB and the reported active
working estimate 21.328 GiB. The automatic planner now caps the two dominant
complex AO-pair work arrays at 2 GiB. It resolves Water20 to block 4656, one
operand to 0.999 GiB and the same active estimate to 6.061 GiB. This only
partitions the exact G-vector sum; explicit user block sizes remain explicit.
The Water8 throughput cap remains 16384.

Post-repair verification consists of 21 GPU unit tests and 10 benchmark tests.
The dispatch-specific H2 Gamma/`2x1x1` GDF/RSDF gates passed with FD maxima
below `2.83e-11 Eh/Bohr^3`. Water8 Gamma GDF/RSDF passed at
`6.91e-11`/`6.90e-11 Eh/Bohr^3`, with stress walls of 12.47/9.58 s. These
results verified the route and small/medium correctness but, at that point,
did not prove the Water20 failure closed. The following rerun supplied the
required scaling evidence.

### Water20 scaling gate passed

The repaired Water20 queue completed both required production cases and
`validate_scaling_gate.py` returned `pass=true` with no case errors.

| Backend | State (s) | Stress (s) | Total analytic (s) | DF (s) | RSS (GiB) | Global VRAM peak (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| GDF | 7.393 | 61.516 | 68.908 | 15.213 | 5.428 | 12.994 |
| RSDF | 7.567 | 52.506 | 60.073 | 6.741 | 5.201 | 13.621 |

Both reports resolve `stress_df_gpu_ft_dispatch="gamma"`, block 4656 and an
estimated 6.061 GiB active DF block. They report the exact GPU FT DF owner and
exact analytic nonlocal PP owner, `production=true`, no low-level numerical
component, and no silent CPU fallback. Source HF, target Fock and Z-vector are
GPU-owned; the matrix-free Z solve converged in 18 iterations with residuals
near `1.35e-13`.

The failed runs peaked in the DF stage at 16.47/15.88 GiB global device use.
After repair, the DF-stage peaks are 7.45/7.47 GiB, reductions of about 54.8%
and 53.0%. The overall new peaks occur in the common PP stage, not DF. The
remaining Water20 bottleneck is PP plus source XC at 45.87/45.37 s. RSDF is
17.2% faster in stress wall because its DF response is 6.74 s versus 15.21 s
for GDF.

GDF and RSDF stress agree to a maximum absolute component difference of
`3.26e-8 Eh/Bohr^3`; their target energies differ by `2.88e-5 Eh`. This is a
cross-DF representation comparison, not an FD accuracy gate. Analytic
correctness remains owned by the existing H2/Water8 FD matrix. The successful
Water20 result closes the required system-scaling gate without changing the
Python API default.
