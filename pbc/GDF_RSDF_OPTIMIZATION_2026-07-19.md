# GDF/RSDF Optimization Work Record

## Scope

This record covers three exact, algebra-preserving optimization targets:

1. GDF reciprocal zero-order/derivative memory and duplicate work;
2. RSDF reciprocal source derivative memory and work;
3. RSDF SCF J/K and matrix-free AX repetition cost.

No cutoff, mesh, screening threshold, density, or force component may be
dropped.  Water64/DZ is user-run scaling evidence, not a development test.

## 1. RSDF compressed J/K route

### Diagnosis

The exact GPU-source builder already produced GPU4PySCF compressed factors,
but production converted them immediately to canonical dense AO-pair blocks.
The inherited provider therefore used dense project-local J/K contractions in
source HF, target evaluation, and every matrix-free AX iteration.

This is especially expensive in the Z-vector solve: the same J/K operator is
applied many times, so a slower contraction is multiplied by the iteration
count.  Existing GDF/RSDF reports had similar iteration counts but a much
larger RSDF time per iteration, which is consistent with a J/K implementation
cost rather than poorer solver convergence.

### Implemented ownership

- `compress=True` factors and AO-pair indices remain authoritative through
  source HF, target J/K, and AX.
- `uses_native_jk` is independent of GDF integral/derivative ownership, so
  RSDF can use upstream compressed J/K without claiming native GDF H1 or PP
  ownership.
- Canonical derivative blocks are lazy.
- Canonical creation retires compressed storage immediately.
- Canonical eviction and reference-provider eviction have separate rebuild
  rules.

### Pending gates

- selected FD was not repeated; the existing small release fixture remains
  the absolute FD gate.

Completed runtime gates:

- ownership/cache tests: 8 passed;
- exact-central Gamma and sampled-k energy-gradient parity: 2 passed;
- reciprocal-memory planner/scheduler tests: 7 passed;
- Water/Ethanol-SZ energy difference: `4.55e-13 Ha`;
- Water/Ethanol-SZ gradient maximum difference: `1.51e-13 Ha/Bohr`;
- identical 22 Z iterations and residual near `1.17e-13`;
- Z time per iteration: 5.9669 to 3.0972 s (`-48.09%`);
- analytic total: 572.4784 to 495.1274 s (`-13.51%`).

## 2. RSDF reciprocal source derivative

### Current dataflow

The production reverse pass first forms a source adjoint with logical size
`nao**2 * naux`.  The native source derivative then transforms that adjoint
to the sorted AO/auxiliary basis and, for each reciprocal block, constructs
AO-pair Fourier values plus AO- and auxiliary-side contracted sources.

The present code already shortens the lifetime of reciprocal block tensors and
retries failed blocks transactionally.  The remaining large object is the
dense source adjoint itself.  Removing only a small temporary will not change
the dominant scaling.

### Next implementation unit

Use a packed symmetric AO-pair adjoint at Gamma and consume it directly in a
source-specific CUDA contraction.  The kernel must compute both center
derivative channels from the packed source without first reconstructing a
dense `(nao, nao, naux)` tensor.

Required order:

1. prove that only the AO-symmetric part contributes at Gamma;
2. add dense-versus-packed algebra tests on random real Hermitian inputs;
3. add a packed short-range kernel adapter;
4. add a packed reciprocal adapter;
5. preserve the existing dense path as a test reference until full-gradient
   parity passes;
6. consider sampled-k only after explicit conjugate-pair algebra is fixed.

This work is not yet implemented.  A reshape-only patch would not be enough,
because the current CUDA and Fourier interfaces consume dense sorted AO-pair
storage.

## 3. GDF reciprocal zero-order/derivative duplication

### Current dataflow

The one-pass Gamma route constructs the weighted states `D`, `D+T`, and
`D-T`.  It traverses short- and reciprocal-space integrals once to build
zero-order occupied-space intermediates and traverses reciprocal derivatives
again.  For large response densities, `D+T` and `D-T` can be effectively full
rank, so occupied-space workspaces become much larger than source-HF occupied
space.

### Next implementation unit

Replace the quadratic signed-state construction with the already established
reverse-adjoint algebra:

1. obtain the target-J/Z-bilinear adjoint of the existing provider CDERI;
2. push that adjoint through the source and metric factorization;
3. contract source, metric, and auxiliary-center derivatives directly;
4. keep Ewald and pseudopotential terms in their existing explicit owners;
5. compare against the current signed-state route component by component.

This removes the need to rebuild zero-order integral state solely for the
derivative and avoids full-rank `D+T`/`D-T` occupied buffers.  It is a larger
change than R6 because native GDF compressed factors do not currently expose
all reverse-factorization metadata through a stable project-local object.
Implementation should begin with Gamma and a canonical reference route, then
move to compressed provider factors after parity is established.

## Validation Policy

R6 was validated with small Gamma/sampled-k tests before the scaling run.
Water/Ethanol-SZ provides timing and memory evidence; no Water64 calculation
was started.  The packed RSDF source derivative and GDF reverse-adjoint
sections remain designs only and have no runtime performance claim yet.

## 4. RSDF/multigrid workspace lifetime

### Diagnosis

The R6 Water/Ethanol run reached 62.250 GiB in
`gradient_target_explicit`, after the 54.469 GiB RSDF response.  Separate
stage tracking showed that the `240^3` multigrid XC derivative itself peaked
at only about 7 GiB.  The larger value came from completed workspaces that
remained resident or allocator-reserved while the next stage began:

- the energy-owned multigrid tasks and sorted Gaussian pairs;
- unused CuPy response blocks retained by the memory pool;
- the 4.230 GiB canonical CDERI after its final analytic response use.

### Implemented lifetime boundaries

1. Evaluate the multigrid target XC derivative immediately after the Z-vector
   solve, before one-electron and RSDF derivative work.
2. Keep only the small `(natm, 3)` derivative, call `MultiGridNumInt.reset()`,
   and return unused CuPy/pinned/FFT caches.
3. After the contracted RSDF response returns, release unused CuPy blocks.
4. Retire the canonical CDERI with the provider's rebuild-on-next-request
   contract, then return the newly unused blocks to CUDA.

No grid point, reciprocal vector, AO pair, or response equation is removed.
The change affects allocation lifetime, not numerical algebra.

### Water/Ethanol-SZ measurements

All runs use Gamma sampling, SZV-GTH orbital/auxiliary bases, 500 Ha cutoff,
r2SCAN, `multigrid_xc`, and an explicit `240 x 240 x 240` target mesh.

| Route | Gradient time (s) | Analytic total (s) | Peak VRAM (GiB) |
| --- | ---: | ---: | ---: |
| R6 baseline | 393.437 | 495.127 | 62.250 |
| staged multigrid | 386.781 | 488.427 | 61.670 |
| plus pool release | 388.152 | 489.742 | 58.998 |
| plus final CDERI release | 386.840 | 488.453 | 57.414 |

The final route returns 11.121 GiB at the post-response ownership boundary,
including 4.230 GiB of canonical CDERI.  The whole-run peak decreases by
4.836 GiB (`7.77%`).

The old progress labels placed local-part-1 and nonlocal GTH pseudopotential
contractions under the preceding `z_nuclear_response` and
`gradient_target_explicit` stages.  The bounded nuclear-response tracker
itself observed only 0.641 GiB of device-use growth, while the surrounding
stage reached about 57 GiB.  Dedicated `gradient_pseudo_explicit` and
`gradient_pseudo_z_response` progress stages now expose this ownership.

An isolated Water/Ethanol probe then separated the two PP terms.  Upstream
`vppnl_nuc_grad` increased device use by only 0.254 GiB and took 0.440 s for a
representative density.  The actual high-water operation was the reciprocal
local-part-1 derivative, whose former free-memory policy selected very large
AO-pair Fourier blocks.

Relative to R6, the final energy difference is `4.55e-13 Ha`; the gradient
maximum and RMS differences are `4.89e-14` and `9.85e-15 Ha/Bohr`.  The
single-run timing change is `-1.68%` for the analytic gradient and `-1.35%`
for analytic total, but timing is secondary to the deterministic lifetime
and memory reduction established here.

## 5. Bounded local-part-1 reciprocal derivative

### Diagnosis and implementation

The old planner assigned 60% of currently free VRAM to one block.  On the
Water/Ethanol cell it selected 2,752 of 12,167 G-vectors per block.  Each
complex AO-pair Fourier tensor scales as
`block_size * nkpts * nao^2 * 16 bytes`; at `nao=828` the selected block alone
is about 28.1 GiB, with a second overlapping working tensor producing a
56.5 GiB observed increase.

The retained planner uses at most 10% of free VRAM, caps a block at 1,024
vectors, aligns ordinary blocks to 32 vectors, and permits smaller blocks
when even 32 vectors exceed the memory budget.  Every G-vector is still
processed exactly once.  Execution metadata records the resolved block size,
count, grid count, memory fraction, and estimated pair-Fourier bytes.

### Isolated kernel measurement

| Policy | Block size/count | Time (s) | Peak increase (GiB) |
| --- | ---: | ---: | ---: |
| former 60% policy | 2,752 / 5 | 13.581 | 56.527 |
| bounded 10% policy | 448 / 28 | 13.681 | 9.293 |

The output norm was identical.  The extra 23 kernel blocks added 0.100 s
(`0.74%`) while reducing the isolated peak by 47.234 GiB.

### Full Water/Ethanol result

Final result:

`benchmark/backup/results/water_ethanol_rsdf_r10_local_part1_bounded_mesh240_2026-07-19/000_water_ethanol_rsdf_r6_mesh240_gpu_rsdf_multigrid_xc/report.json`

| Quantity | R6 baseline | R10 bounded | Change |
| --- | ---: | ---: | ---: |
| PP explicit + Z-response time | 27.295 s | 27.548 s | +0.93% |
| analytic gradient | 393.437 s | 387.521 s | -1.50% |
| analytic total | 495.127 s | 489.080 s | -1.22% |
| sampled peak VRAM | 62.250 GiB | 52.242 GiB | -16.08% |
| host peak RSS | 29.385 GiB | 29.395 GiB | +0.04% |

The final peak now belongs to `gdf_response_tensors` at 52.242 GiB.  The two
new PP stages peak at 10.627 and 10.307 GiB.  Relative to R6, energy changes
by `4.55e-13 Ha`; the gradient maximum and RMS differences are `5.49e-14`
and `1.00e-14 Ha/Bohr`.

## 6. Early Gamma CDERI retirement and bounded source reciprocal blocks

### Ownership and block policy

For a single fused Gamma observable, all source and metric adjoints are
complete before source-basis sorting and reciprocal differentiation begin.
The canonical CDERI has no remaining consumer in that gradient.  It is now
retired at this exact boundary even when source sorting would fit without the
release.  A future canonical request rebuilds it through the existing
provider contract.

The source reciprocal policy changes from 25% of free VRAM with a 4,096-vector
cap to 15% with a 512-vector cap.  Every one of the 29,791 reciprocal vectors
is retained; only the partition changes.

### Water/Ethanol measurement

Final result:

`benchmark/backup/results/water_ethanol_rsdf_r11_early_cderi_block512_mesh240_2026-07-19/000_water_ethanol_rsdf_r6_mesh240_gpu_rsdf_multigrid_xc/report.json`

| Quantity | R10 | R11 | Change |
| --- | ---: | ---: | ---: |
| canonical CDERI during source reciprocal | 4.229 GiB | 0 GiB | -4.229 GiB |
| reciprocal block size/count | 960 / 32 | 512 / 59 | bounded |
| estimated reciprocal block working set | 19.674 GiB | 10.493 GiB | -46.67% |
| source reciprocal time | 221.204 s | 224.070 s | +1.30% |
| complete response time | 240.346 s | 243.308 s | +1.23% |
| analytic gradient | 387.521 s | 390.116 s | +0.67% |
| analytic total | 489.080 s | 491.707 s | +0.54% |
| sampled peak VRAM | 52.242 GiB | 39.192 GiB | -24.98% |

Relative to the original R6 result, R11 reduces peak VRAM by 23.059 GiB
(`37.04%`) and remains 3.420 s faster in analytic total.  Energy is identical
at the stored precision; the gradient maximum and RMS differences are
`1.25e-13` and `2.41e-14 Ha/Bohr`.  No OOM rebatch occurred.

## 7. Continuation plan after the 2026-07-20 profile review

Egg-box work is paused.  Mesh, cutoff, target-XC quadrature and rigid-phase
corrections are not part of this continuation unit.  The next work preserves
every reciprocal vector, AO pair, response term and convergence threshold and
targets only measured GDF/RSDF runtime and memory costs.

### 7.1 Current production profile

The current comparison uses GPU, Gamma sampling, SZV-GTH, r2SCAN and
`multigrid_xc`.  Water64 values are user-generated read-only scaling evidence;
development and repeated timing use Water8 and Water20.

| System | DF | Total (s) | HF SCF (s) | Gradient (s) | Dominant DF response (s) | One-electron (s) | AX/Z (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Water20 | GDF | 10.207 | 1.342 | 7.649 | 4.626 | 0.934 | 0.304 |
| Water20 | RSDF | 10.927 | 1.546 | 8.163 | 5.124 | 0.941 | 0.292 |
| Water64 | GDF | 73.725 | 8.034 | 61.924 | 41.516 | 9.159 | 4.317 |
| Water64 | RSDF | 61.869 | 11.721 | 46.381 | 25.740 | 9.344 | 4.330 |

The GDF fused target-J/Z response is about 56% of the Water64 total.  The RSDF
source/metric response is about 42%; RSDF HF SCF contributes another 19%.
Matrix-free AX/Z is only about 7% and is not the first optimization target.

### 7.2 Stage P0: reproducible timing baseline

1. Run Water8 and Water20 in independent processes at least three times.
2. Record medians, run-to-run spread, CUDA phase timing, host RSS and sampled
   device/CuPy high-water values.
3. Record Z iterations, residual, energy and gradient checksums.
4. Use the current GDF `one_pass_fused` and RSDF native fused response as the
   immutable numerical references.
5. Do not repeat full FD in every timing run; retain one selected FD release
   case per route.

No optimization claim is accepted from a single best timing.

### 7.3 Completed P0 baseline

The 2026-07-20 P0 queue used 64 host threads and three independent processes
per system/backend.  All 12 cases completed without fallback.  The complete
machine-readable record is
`benchmark/results/current_gdf_rsdf_optimization_baseline/optimization_baseline.json`.

| System | DF | Median total (s) | Median DF response (s) | Total relative spread | Maximum energy drift (Ha) | Maximum gradient drift (Ha/Bohr) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Water8 | GDF | 4.667 | 1.374 | 0.41% | 0 | 2.31e-14 |
| Water8 | RSDF | 5.191 | 1.900 | 0.32% | 0 | 2.09e-14 |
| Water20 | GDF | 10.732 | 5.284 | 0.30% | 1.14e-13 | 3.96e-14 |
| Water20 | RSDF | 10.793 | 5.111 | 0.35% | 1.14e-13 | 1.87e-14 |

Water8 used 15 Z iterations and Water20 used 18.  The maximum recorded Z
residuals were `2.58e-14` and `1.36e-13`, respectively.  These timing spreads
are sufficiently smaller than the 5% stop rule and the 10% Water20 promotion
gate.

## 8. GDF reverse-adjoint response

### 8.1 Objective

Replace the current signed-state `D`, `D+T`, `D-T` construction with direct
reverse propagation of the target-J/Z adjoint through the CDERI
factorization.  The response should contract source, metric and
auxiliary-center derivatives without rebuilding full-rank occupied-state
workspaces solely for the derivative.

### 8.2 Implementation order

1. Expose stable project-local factorization metadata from the native GDF
   provider.
2. Implement a canonical Gamma reverse-adjoint reference.
3. Compare source, metric and auxiliary-center components against the current
   signed-state route separately.
4. Consume compressed provider factors without canonical expansion.
5. Keep Ewald and pseudopotential terms under their existing explicit owners.
6. Preserve the signed-state route as a selectable validation reference until
   full release parity passes.
7. Extend to sampled k-points only after the Gamma route is complete.

### 8.3 Performance gate

- GDF response-stage median reduction: at least 30%;
- Water20 end-to-end median reduction: at least 10%;
- no increase in peak VRAM above the current bounded route;
- if two measured implementations remain below 5% end-to-end improvement,
  stop this direction and move to the next exclusive bottleneck.

### 8.4 Implemented Gamma reverse-adjoint route

The native provider now exposes cached project-local Gamma factorization
metadata: compressed central factors, AO-pair and diagonal maps, the
auxiliary whitening transform, build signature and exact split parameters.
The production prototype expands the polarization identity algebraically
instead of factorizing `D+T` and `D-T` as two full-rank states.

One metric-solved primitive tensor

$$
W_P = \sum_Q (M^{-1})_{PQ} B_Q
$$

builds the complete target-J/Z source adjoint.  The matching metric source is
the contraction of that primitive source with $W$, avoiding two additional
auxiliary-by-AO-square intermediates.  Existing CUDA SR, LR, metric,
auxiliary-center and Ewald derivative owners are reused.  The signed-state
reference remains selectable as `one_pass_fused`; `reverse_adjoint_fused`
selects the new Gamma route.

The exact SR/LR split was re-profiled because the direct LR contraction
changes its optimum.  Scale `1.9` balances SR and LR work better than the old
signed-state value `1.5`.  This changes only the exact range partition, not a
cutoff or retained vector.  Reverse reciprocal blocks have an 8 GiB
working-set cap and process every vector transactionally.

Water20 three-run medians are stored under
`benchmark/results/current_gdf_reverse_adjoint_release_bounded`.

| Quantity | Signed-state | Reverse adjoint | Change |
| --- | ---: | ---: | ---: |
| GDF response | 5.285 s | 3.469 s | -34.36% |
| Analytic total | 10.614 s | 8.781 s | -17.27% |
| Peak VRAM delta | 12.586 GiB | 10.566 GiB | -16.05% |
| Z iterations | 18 | 18 | unchanged |

The energy difference is about `1e-13 Ha`; the representative full-gradient
maximum difference is `8.37e-12 Ha/Bohr`.  Metric, reciprocal plus
auxiliary-center, short-range and Ewald components independently match the
signed route within `1e-9 Ha/Bohr`.  The Water8 selected-FD error is
`5.18e-9 Ha/Bohr`, with Z residual `2.57e-14`.

`optimized_fused` is the all-k-point opt-in dispatcher: it selects reverse
adjoint at Gamma and the validated complex one-pass fused route at sampled
k-points.  Sampled-k full-gradient parity passes.  A true sampled-k reverse
adjoint remains deferred until its momentum-transfer and conjugate-pair
algebra is implemented; no real Gamma storage or complex-to-real cast is used.

## 9. Reciprocal transform reuse

After the GDF reverse-adjoint route is correct, inspect duplicate construction
of G-vectors, Coulomb weights, auxiliary Fourier values, AO-pair Fourier
values and derivative transforms.  Reuse is bounded to one reciprocal block;
the complete AO-pair/grid tensor must not be cached.

The preferred dataflow is:

```text
one reciprocal block transform
  -> source derivative contraction
  -> metric derivative contraction
  -> auxiliary-center contraction
  -> release block workspace
```

A two-buffer CUDA-stream prototype may overlap transform generation for block
`n+1` with contraction of block `n`.  It is retained only if synchronization
auditing and three-run timing show at least 5% end-to-end improvement.

### 9.1 Dependency and cache decision

The reverse source is unavailable until all zero-order blocks complete, the
auxiliary metric is inverted and $W$ is assembled.  A zero-order Fourier
block therefore cannot be consumed by the derivative adjoint during the same
pass.  Retaining it until the reverse pass would cache the forbidden complete
AO-pair/grid tensor.  The implementation recomputes bounded reciprocal
transforms and caps each working set at 8 GiB.

Eliminating zero-order reconstruction requires the provider's compressed
central CDERI plus a conjugate-aware reverse of its native whitening
factorization.  Stage 8's stable metadata is the boundary for that future
work.  Since the current route passes the response and end-to-end gates, a
two-stream overlap prototype is not promoted without a synchronization audit
and at least 5% additional end-to-end gain.

## 10. RSDF packed symmetric source adjoint

### 10.1 Objective

At Gamma, replace the dense logical source adjoint
`(nao, nao, naux)` with a packed real-symmetric AO-pair representation of size

$$
N_{\mathrm{aux}}\frac{N_{\mathrm{AO}}(N_{\mathrm{AO}}+1)}{2}.
$$

The packed source must be consumed directly by short-range and reciprocal
CUDA contractions.  Reconstructing a dense tensor before either kernel does
not satisfy this stage.

### 10.2 Implementation order

1. Prove the Gamma symmetric-adjoint algebra on random real Hermitian inputs.
2. Add dense-versus-packed source and force component tests.
3. Implement the packed short-range adapter.
4. Implement the packed reciprocal adapter and both nuclear-center channels.
5. Retain transactional reciprocal blocking and current OOM rebatching.
6. Preserve the dense route as a reference until full-gradient parity passes.
7. Derive explicit complex conjugate-pair algebra before sampled-k support;
   do not cast sampled-k adjoints to real storage.

### 10.3 Performance gate

- source-adjoint resident bytes reduced by at least 40%;
- RSDF response-stage median reduction: at least 30%;
- Water20 end-to-end median reduction: at least 10%;
- no regression in exact-central compressed J/K or early CDERI retirement.

### 10.4 Implemented result and stop decision

The Gamma source adjoint is now kept as a packed lower triangle through both
the short-range and reciprocal CUDA adapters.  Water20 stores 6,969,600 bytes
instead of 13,824,000 bytes (`-49.58%`).  A derivative-only exact SR/LR split
with `rsdf_derivative_omega_scale=1.1` reduced the median response from
5.1106 to 4.6835 s (`-8.36%`) and the workflow from 10.7931 to 10.3901 s
(`-3.73%`).  The Water8 selected-FD error is `5.12e-9 Ha/Bohr`.

The 30% response target was not met.  An isolated Water20 short-range probe
showed that the current auxiliary-shell limit 80 is already the fastest tested
layout: 2.728 s versus 2.841, 2.894, 2.867, 3.080 and 4.125 s for limits 60,
48, 40, 32 and 20.  The remaining 2.7 s is therefore the exact
`PBCsr_ejk_int3c2e_ip1` integral workload, not Python launch or batch-shape
overhead.  Further reduction requires a new CUDA primitive; changing cutoff,
screening, mesh or force channels remains forbidden.

## 11. RSDF source-HF SCF J/K

The compressed J/K route is already active, so optimization starts with a
new timing split rather than another representation change:

1. exact-central CDERI construction;
2. first J/K contraction;
3. repeated SCF J/K contractions;
4. diagonalization and density update;
5. allocation, index conversion and host/device transfer.

Only the measured dominant substage is modified.  Candidate changes are
iteration-persistent contraction workspaces, one compressed-factor traversal
for simultaneous J/K, removal of repeated AO-pair index conversion and
prevention of canonical expansion before its first derivative consumer.

A 30% HF-SCF reduction changes the Water64 RSDF total by only about 6%, so
this stage follows the response work.

### 11.1 Measured source-HF split

Non-synchronizing CUDA events now profile logical J/K calls while separate
events cover diagonalization and density updates.  Three Water20 runs gave
the following medians:

| source-HF phase | median (s) | fraction of 1.3712 s SCF |
| --- | ---: | ---: |
| exact-central build | 0.7160 | 52.2% |
| 19 compressed J/K calls | 0.1517 | 11.1% |
| first J/K | 0.0156 | 1.1% |
| repeated J/K | 0.1361 | 9.9% |
| diagonalization | 0.0769 | 5.6% |
| density update | 0.0049 | 0.4% |
| Fock/DIIS/dispatch and other wall remainder | 0.4207 | 30.7% |

The upstream compressed Gamma factors are pinned mapped-host arrays, not
persistent device arrays.  A bounded Gamma device-cache candidate therefore
retains factors only below 512 MiB, 2% of free VRAM, and a 2 GiB reserve.
On Water20 it reduced J/K by 2.78% and AX by 3.31%, but the full workflow by
only 0.16%, below promotion threshold.  The default remains `host`; the
`device`/`auto` modes are explicit scaling probes.  Sampled-k factors are not
silently uploaded because the upstream sampled-k loop has a different buffer
ownership contract.

## 12. Shared one-electron bottleneck

The current Water64 one-electron section is about 9.3 seconds and AX/Z is
about 4.3 seconds for both GDF and RSDF.  These become active only after DF
response and RSDF SCF improvements move them to the top of the exclusive
profile.  Mixed precision, fitted force correction, looser screening,
reduced cutoff and dropped response channels are not allowed.

### 12.1 Coupled pseudopotential source

The explicit and adjoint pseudopotential terms used to traverse the same
local-part-1 and nonlocal projector derivatives independently:

$$
G_{\mathrm{PP}}[D] + 2G_{\mathrm{PP}}[T_Z].
$$

Because every PP derivative contraction is linear in its density source, the
production Gamma path now evaluates the exact coupled source once,

$$
G_{\mathrm{PP}}[D+2T_Z],
$$

and reports it as an unresolved `coupled_pseudopotential` contribution.
Sampled-k retains the existing resolved route until its complex phase algebra
is independently promoted.

Water20 PP time fell from 1.3797 to 0.6963 s for GDF and from 1.3787 to
0.6956 s for RSDF.  Relative to the immediately preceding optimized routes,
the workflow improved from 8.9233 to 8.2664 s (`-7.36%`) for GDF and from
10.3901 to 9.6962 s (`-6.68%`) for RSDF.  Peak VRAM was unchanged for GDF;
the RSDF sampled peak changed by 0.20 GiB, within the stage-sampling spread.
Gradient maximum differences from the resolved route were `6.22e-14` and
`6.44e-14 Ha/Bohr`.  Water8 selected-FD errors were `5.18e-9` and
`5.12e-9 Ha/Bohr`, and both sampled-k full-gradient gates passed.

Relative to the P0 baseline, the current Water20 workflow is 8.2664 s for GDF
(`-22.98%`) and 9.6962 s for RSDF (`-10.16%`).

The public GPU gradient default selects native GDF `optimized_fused`, but both
derivative omega scales remain `1.0`, preserving the PySCF-resolved split.
The measured GDF 1.9 and RSDF 1.1 partitions are explicit opt-in performance
options.  Reference GDF retains `two_pass`, and sampled-k keeps its validated
complex one-pass/resolved PP ownership.

### 12.2 Water64 omega opt-in gate

Water64/SZV-GTH confirms that the Water20 optimum is system-dependent and
must not become a default.  All baseline jobs explicitly used scale 1.0.

| backend | scale | workflow (s) | gradient (s) | DF response (s) |
| --- | ---: | ---: | ---: | ---: |
| GDF | 1.0 | 48.174 | 36.400 | 18.223 |
| GDF | 1.9 opt-in | 56.278 | 44.529 | 26.315 |
| RSDF | 1.0 | 54.650 | 39.164 | 20.923 |
| RSDF | 1.1 opt-in | 57.580 | 42.171 | 23.973 |

The opt-in split was 16.82% slower end-to-end for GDF and 5.36% slower for
RSDF.  For GDF, scale 1.9 changed the derivative reciprocal grid from 3,375
to 15,625 points: short-range derivative time fell from 10.371 to 2.746 s,
but long-range derivative plus reciprocal zero-order rose from 4.888 to
22.281 s.  For RSDF, the source mesh changed from $25^3$ to $27^3$; SR fell
from 5.126 to 4.464 s while reciprocal work rose from 14.090 to 17.710 s.

Despite the different partitions, tuned-versus-default gradient maximum
differences were only `1.07e-11 Ha/Bohr` (GDF) and `6.38e-12 Ha/Bohr`
(RSDF).  The scale-1.0 selected-FD errors were `7.32e-9` and
`6.39e-9 Ha/Bohr`.  Thus the partition remains algebraically accurate, but
the performance optimum does not transfer from Water20 to Water64.

### 12.3 Water64 paired optimization gate at omega scale 1.0

A fresh before/after gate isolates code optimization from both omega tuning
and target-grid changes.  Both sides use Water64/SZV-GTH, Gamma, r2SCAN,
multigrid-XC mesh $240^3$, kinetic cutoff 500 Ha and derivative omega scale
1.0.  The pre-optimization source is commit `a112d2c`; its only local change
is replacing the historical hard-coded GDF scale 1.5 by 1.0.  The current
source uses GDF `optimized_fused`, packed/bounded RSDF response, compressed
J/K/AX ownership and the coupled pseudopotential contraction.

| backend | quantity | before | after | change |
| --- | --- | ---: | ---: | ---: |
| GDF | workflow without FD | 62.023 s | 48.316 s | -22.10% |
| GDF | analytic gradient | 50.164 s | 36.485 s | -27.27% |
| GDF | sampled peak VRAM delta | 47.795 GiB | 11.131 GiB | -76.71% |
| RSDF | workflow without FD | 64.177 s | 54.671 s | -14.81% |
| RSDF | analytic gradient | 48.193 s | 39.204 s | -18.65% |
| RSDF | sampled peak VRAM delta | 24.576 GiB | 9.203 GiB | -62.55% |

For GDF, the target-J/Z bilinear response fell from 29.253 to 18.252 s and
the two resolved pseudopotential contractions were replaced by one coupled
2.905 s contraction.  For RSDF, the contracted response fell from 25.427 to
20.951 s, matrix-free AX/Z from 6.440 to 4.705 s and the pseudopotential work
was likewise halved.

The energy difference is exactly zero at report precision for both backends.
The maximum complete-gradient differences are `2.40e-13 Ha/Bohr` for GDF and
`1.45e-13 Ha/Bohr` for RSDF; RMS differences are `3.84e-14` and `2.82e-14
Ha/Bohr`.  The paired runs therefore measure an implementation improvement,
not a change in the DC-DFT result.

## 13. Validation and promotion gates

Every implementation unit proceeds in this order:

1. algebra tests and current-route component parity;
2. Water1/Water8 Gamma energy and complete gradient parity;
3. selected analytic/full-energy FD;
4. sampled-k energy and gradient parity;
5. Water20 three-run timing and memory medians;
6. package import, examples and benchmark regression tests;
7. user-run Water64/SZ final scaling validation.

Numerical gates are:

- new-versus-reference energy difference `<= 1e-9 Ha`;
- new-versus-reference gradient maximum difference
  `<= 1e-9 Ha/Bohr` during optimization;
- CPU/GPU gradient maximum difference `<= 1e-7 Ha/Bohr`;
- analytic-FD maximum difference `<= 1e-5 Ha/Bohr`;
- Z-vector residual `<= 1e-8`;
- no silent CPU fallback, dropped block, NaN or Inf;
- no Gamma-only completion claim before the sampled-k gate passes.

The implementation order is therefore P0 baseline, GDF reverse-adjoint,
GDF reciprocal reuse, RSDF packed adjoint, RSDF SCF J/K, shared bottleneck
review, Water20 release gate and user-run Water64 final validation.
