# S9 GPU Fourier-Source Design

## Motivation

After bounded device factor projection, Water8 still spends most of its stress
time in the long-range GDF/RSDF source:

| Backend | AO Fourier integral (s) | Remaining projection (s) |
|---|---:|---:|
| RSDF | 20.52 | 7.82 |
| GDF | 112.63 | 128.12 |

The Z-vector/AX solve is only about 0.25--0.26 s and is not the next target.

## Relevant established algorithms

GPU4PySCF 1.7.4 contains blocked PBC FT-AO strain kernels used by AFTDF and
RSJK, notably `PBC_ft_aopair_ej_strain_deriv` and
`PBC_ft_aopair_ek_strain_deriv`. They combine shell-pair screening, BvK image
phases, G-vector blocking, and direct density contraction on device. CP2K's GPW
implementation follows the same broad scheduling principle: create screened
Gaussian/grid tasks, process bounded grid batches, and contract before large
intermediates escape the task.

GPU4PySCF is Apache-2.0 and can be adapted with provenance. CP2K is GPL; only
its published algorithmic ideas may be used, not source copied into this
repository.

## Why the existing kernel is not a drop-in GDF/RSDF replacement

The AFT Coulomb kernel assumes a separable contraction between one AO density
$D_{ij}$ and one reciprocal weight $v(G)$. The reverse GDF/RSDF source
contraction has a general adjoint

$$
W_B(ij,P),
$$

and an auxiliary Fourier factor $A(G,P)$. Its AO/G weight is

$$
T(ij,G)=\sum_P W_B(ij,P)A(G,P).
$$

Therefore substituting AFT/RSJK stress would change the DF representation and
is not allowed. The reusable part is the lower-level strain FT kernel, after
forming the exact $T(ij,G)$ for the chosen GDF/RSDF source.

## Proposed exact pipeline

1. Build the fused target-J plus adjoint-RHF CDERI adjoint on device.
2. Push it backwards through the Cholesky source solve to obtain one fused
   $W_B(ij,P)$ rather than two observable arrays.
3. For one bounded G/auxiliary tile, form $T(ij,G)$ with device GEMM.
4. Transform $T$ to the FTOpt/BvK AO representation without a host round trip.
5. Feed $T$ to the GPU AO-pair strain kernel and accumulate six scalars.
6. Keep auxiliary-function and Coulomb-kernel strain derivatives in the current
   exact code initially; move them only after pair-source parity is closed.
7. Preserve the current host factorized route as the strict fallback/reference.

No SVD truncation, density fitting change, omega change, mesh change, mixed
precision, or screening relaxation belongs to this stage.

## Implementation gates

### S9.1 Gamma pair-source prototype

- H2 and Water1, GDF and RSDF.
- Compare only the reciprocal AO-pair source contribution against the current
  host analytic result.
- Required maximum difference: `1e-10 Eh/Bohr^3`.
- Record FTOpt permutation symmetry and every normalization factor explicitly.

### S9.2 Bounded memory ownership

- Bound both G and auxiliary tiles from current free VRAM.
- Do not retain full $T(ij,G)$, full $dB$, or six AO response matrices.
- Report tile sizes, estimated/observed device bytes, kernel calls, and host
  transfer bytes.

### S9.3 Sampled-k/BvK parity

- H2 `2x1x1` complex k-point gate before Water8.
- Validate canonical/reverse pair phases independently.
- Reject any path that casts complex BvK weights to real without a proven
  symmetry reduction.

### S9.4 End-to-end adoption

- H2 six-component FD and Water8 selected FD.
- Water8 stage reduction at least 25% or peak memory reduction at least 25%.
- Water20/64 remains a user-run scaling gate.

## Main risk

The GPU4PySCF exchange strain kernel accepts a G-dependent AO weight, but its
shell-pair symmetry, AO sorting, BvK layout, and final factors were designed for
exchange energy. A prototype must derive these conventions rather than infer
them from matching shapes. Until that parity test passes, this is a research
route and cannot replace the bounded factorized backend.

## Implemented result

The low-level parity gates now cover:

- separable Gamma $D_{ij}v(G)$ weights;
- general Gamma $T(ij,G)$ weights;
- complete Gamma long-range source adjoints;
- general complex sampled-k canonical-pair weights;
- complete sampled-k long-range source adjoints.

The sampled-k discrepancy follows the integral screening threshold: tightening
cell precision from `1e-8` to `1e-10` and `1e-12` reduced the arbitrary-weight
pair difference from `2.19e-9` to `1.55e-11` and `1.78e-13`. No phase or
normalization correction was fitted.

End-to-end selected-FD gates are:

| System/backend | Sampling | FD max error (Eh/Bohr^3) |
|---|---|---:|
| H2 RSDF | Gamma | `2.83e-11` |
| H2 RSDF | `2x1x1` | `1.98e-11` |
| H2 native GDF | Gamma | `2.81e-11` |
| H2 native GDF | `2x1x1` | `1.98e-11` |
| Water8 RSDF | Gamma | `6.90e-11` |
| Water8 RSDF | `2x1x1` | `6.74e-11` |
| Water8 native GDF | Gamma | `6.88e-11` |
| Water8 native GDF | `2x1x1` | `6.73e-11` |

The larger-basis H2 `DZVP-GTH` gate also passes without changing the mesh,
precision, screening, or DF representation:

| Backend | Sampling | FD max error (Eh/Bohr^3) | Stress (s) | Source (s) | RSS (GiB) |
|---|---|---:|---:|---:|---:|
| native GDF | Gamma | `2.94e-11` | 0.648 | 0.232 | 1.880 |
| RSDF | Gamma | `2.92e-11` | 0.459 | 0.156 | 1.847 |
| native GDF | `2x1x1` | `2.21e-12` | 1.374 | 0.361 | 1.885 |
| RSDF | `2x1x1` | `1.99e-12` | 1.258 | 0.416 | 1.880 |

Profiler-off Water8 three-run medians:

| Backend | Stress (s) | DF (s) | Source (s) | RSS (GiB) |
|---|---:|---:|---:|---:|
| RSDF Gamma GPU FT | 8.057 | 1.328 | 0.620 | 3.135 |
| native GDF Gamma GPU FT | 12.209 | 5.482 | 3.783 | 3.569 |

The prior bounded forward paths required 40.00 s/8.87 GiB for RSDF and
271.11 s/12.44 GiB for GDF. The new reverse-source GPU FT route therefore
reduces stress by 79.9% and 95.5%, respectively, while reducing RSS by 64.7%
and 71.3%. The GPU4PySCF FT kernel owns only the reciprocal AO-pair source;
short-range source, vbar, and metric derivatives remain explicit exact owners.

The production-capable development backend is:

```json
"stress_df_response_backend": "host_analytic_source_adjoint_gpu_ft"
```

Native sampled-k GDF now expands GPU4PySCF's compressed BvK factors through
the same canonical-pair convention used by the exact RSDF adapter. Water8
`2x1x1` takes 48.67 s total, including 12.48 s DF and 8.42 s source, with
3.54 GiB peak RSS. H2 `DZVP-GTH` closes the first larger-basis correctness
gate. Current GPU user queues select this exact route explicitly and record
that choice in each run input; Python API default selection remains a separate
compatibility decision.

## Scaling repair after the first Water20 attempt

The first Water20 Gamma run exposed an array-layout requirement that the small
pair tests did not exercise at the same scale. A generic `gpu_ft` request had
sent the single zero k-point through the sampled-k/BvK accumulator. Gamma is a
mathematical special case of sampled k, but GPU4PySCF's strain kernel expects a
different packed density layout for the dedicated Gamma FT context. Dispatch
therefore follows the actual k-point values:

- all k-points zero: dedicated Gamma FT context and Gamma density layout;
- any nonzero k-point: sampled-k FT context and complex BvK density layout;
- GPU FT disabled: established host analytic Fourier owner.

This is a data-layout dispatch, not a change to the reciprocal source formula.
H2 and Water8 Gamma/sampled-k FD gates pass after the split.

The automatic G-vector block planner also has a working-set guard. Let
$n_s$ be the sorted AO count. The two dominant complex AO-pair buffers require
approximately

$$
B_{\mathrm{pair}}(n_G)=2n_G n_s^2\,\mathrm{sizeof}(\mathrm{complex128}).
$$

Automatic blocks satisfy $B_{\mathrm{pair}}\le 2$ GiB, remain aligned to 16
G points, and retain the existing absolute cap of 16384. Water20 SZV-GTH has
$n_s=120$, so the automatic block changes from 16384 to 4656. The complete
active estimate changes from 21.328 to 6.061 GiB. Explicitly requested block
sizes remain user-controlled. Blocking changes kernel call count and peak
workspace only; every G point and every exact contraction is still evaluated.

The repaired Water20 user gate completed at block 4656. GDF/RSDF stress walls
were 61.52/52.51 s with peak RSS 5.43/5.20 GiB. DF-stage global device peaks
fell from the failed runs' 16.47/15.88 GiB to 7.45/7.47 GiB. Both reports carry
the Gamma GPU FT owner, `production=true`, no numerical component and no silent
fallback; the independent scaling validator passed both cases.
