# Water8 Egg-Box Stage 1/2 Study

## Scope

This study uses Water8, SZV-GTH, Gamma sampling, native GPU GDF and a fixed
source-HF density.  r2SCAN was scanned on explicit cubic meshes of
`123^3`, `173^3`, `211^3`, `240^3` and `271^3`.  PBE at `173^3` is the
semilocal control.  Each x/y/z scan samples eight equally spaced phases over
one uniform-grid interval.

The reusable runner is
`benchmark/diagnostics/run_water8_eggbox_stage12.py`.  Machine-readable results
are under `benchmark/results/current_water8_eggbox_stage12/`.

## Stage 1: Phase Scan

| XC | Mesh | Maximum axis energy peak-to-peak (Ha) | Maximum sampled slope (Ha/Bohr) | Maximum analytic-rigid-FD error (Ha/Bohr) |
| --- | ---: | ---: | ---: | ---: |
| PBE | 173 | 6.89138e-5 | 1.78441e-3 | 1.12489e-8 |
| r2SCAN | 123 | 1.66216e-4 | 3.34331e-3 | 2.01481e-8 |
| r2SCAN | 173 | 2.51813e-4 | 6.39595e-3 | 3.35670e-8 |
| r2SCAN | 211 | 1.22642e-4 | 3.86654e-3 | 3.42628e-8 |
| r2SCAN | 240 | 6.05443e-5 | 2.17389e-3 | 2.45727e-8 |
| r2SCAN | 271 | 1.93810e-5 | 7.86235e-4 | 1.20356e-8 |

The one-grid endpoint closure error is at roundoff for every case.  The total
translation sum differs from the target-XC component by at most
`2.77e-13 Ha/Bohr`; every individual non-XC component sum is below
`3.16e-13 Ha/Bohr`.

## Stage 2: Mesh Convergence

Convergence is not monotonic at a fixed geometry or at the worst sampled
phase.  Moving from `123^3` to `173^3` makes the r2SCAN maximum amplitude and
slope larger.  From `173^3` to `271^3`, however, the maximum energy amplitude
falls by 92.3% (13.0 times) and the worst sampled slope falls by 87.7%
(8.13 times).

At the same `173^3` mesh, r2SCAN is 3.65 times larger than PBE in energy
amplitude and 3.58 times larger in worst sampled slope.  This supports the
greater grid sensitivity of the r2SCAN MGGA channels.

`271^3` is not sufficient for a `1e-4 Ha/Bohr` worst-phase force criterion:
the observed value is `7.86e-4 Ha/Bohr`.  A single geometry's translation sum
must not be used as a convergence gate because phase changes can reverse the
sign and alter the apparent error non-monotonically.

## Decision

1. Keep `271^3` as the current high-mesh reference, not as a converged
   production setting.
2. Do not rigidly translate the physical structure or subtract the mean force
   as a correction.
3. Proceed to an energy/Fock/gradient-consistent interlaced-grid prototype on
   Water1 unit tests and Water8 phase scans.
4. Compare two-phase diagonal and eight-corner averaging against direct mesh
   refinement before any Water20 or Water64 promotion.

The complete 18-case study took 510.9 seconds on the recorded GPU environment.
