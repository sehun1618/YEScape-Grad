# Molecular GPU Gradient Optimization

Last updated: 2026-07-21

## Production Route

The public `RDCGradient` route now combines the validated restricted
density-fitted optimizations that were previously split between the base and
optional cuEST-hybrid implementations:

1. KS Fock and HF response reuse one DF/CDERI object.
2. CPHF and the adjoint response return only requested MO blocks.
3. The self-adjoint RHF identity replaces `3 * Natm` overlap-response builds
   with one response evaluation.
4. `Z:F1` is contracted by the DF derivative kernel without materializing
   `Hessian.make_h1`; known low-rank density factors bypass an SVD.
5. S1 is reduced immediately to `S1_oo` and `Z:(S1_vo*e_occ)` rather than
   retaining `(3*Natm, NMO, Nocc)`.
6. Shared CDERI storage is released at its last use before direct `Z:F1`.

The older AO-output, separate-DF, dense-density route remains available as a
regression reference through explicit constructor options.

## Unrestricted Route

The UHF route now follows the same bounded-memory design, with alpha and beta
channels kept coupled where the Coulomb response requires it:

1. UCPHF returns only the two virtual-occupied response blocks.
2. One response to the alpha/beta Z transition densities replaces all
   perturbation-indexed `Ax(S1)` builds.
3. S1 is reduced to two occupied-occupied tensors and one spin-summed
   `Z:(S1_vo*e_occ)` vector.
4. Direct DF differentiation evaluates
   $2 Z_\alpha:F^1_\alpha + 2 Z_\beta:F^1_\beta$. The exact
   derivative-energy factors are 2 for the
   total-density Coulomb pair and 4 for each spin exchange pair.
5. The shared CPHF CDERI is released before the direct derivative contraction.

On the OH/STO-3G regression case with PBE grid `(50, 194)`, the optimized and
full UHF gradients agreed within `3.0e-13` a.u. Across two alternating-order
runs, median gradient time was 1.159 s versus 1.437 s and sampled process VRAM
was 810 MiB versus 1,806 MiB. This small-system result verifies the route and
its allocation behavior; it is not a large-system scaling claim.

## Exact Memory Budget

`df_deriv_memory_gb` temporarily caps the memory visible to GPU4PySCF's direct
DF derivative kernel. It changes only auxiliary blocking. The option is not a
screening threshold and does not alter the integral expression.

The implementation serializes this temporary backend setting with a process
lock and restores the original GPU4PySCF function after the derivative call.
The default is `None`, which preserves upstream automatic sizing. A fixed
budget can be too small for the mandatory density-factor intermediates of a
larger calculation, so it remains opt-in.

## Development Measurements

Conditions: RTX PRO 6000 Blackwell, PBE, grid `(50, 194)`, SCF/CPHF tolerance
`1e-10`. Timings include the gradient only and use independent converged RHF
objects for each route.

| System | Route | Gradient | Peak process VRAM | Max difference |
|---|---|---:|---:|---:|
| water8/def2-SVP | optimized | 1.278 s median | 2,116 MiB | `2.85e-10` |
| water8/def2-SVP | AO/dense reference | 1.328 s median | 2,132 MiB | reference |
| water32/def2-SVP | optimized, upstream memory | 40.331 s | 57,106 MiB | `1.25e-10` |
| water32/def2-SVP | AO/dense reference | 44.129 s | 57,206 MiB | reference |
| water32/def2-SVP | optimized, 32 GiB budget | 40.284 s | 28,904 MiB | `1.25e-10` |

For water32, the optimized algebra reduced gradient time by 8.6%. The 32 GiB
budget reduced observed peak VRAM by 49.4% relative to the uncapped optimized
run without a measurable slowdown in this case. These figures are development
measurements rather than universal scaling guarantees.

Tracked evidence:

- `validation/results/2026-07-21_rhf_optimization.json`
- `validation/results/2026-07-21_uhf_optimization.json`

The full per-run reports are generated under
`../benchmark/development/results/` and are intentionally ignored by Git.

## Remaining Work

The next high-value targets are geometry-step Z-vector warm starts and a fully
owned auxiliary-streaming derivative kernel that does not rely on temporarily
capping a private GPU4PySCF helper.
