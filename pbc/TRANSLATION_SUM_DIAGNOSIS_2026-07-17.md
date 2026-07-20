# Translation-Sum Diagnosis

## Conclusion

The nonzero translation sum is not a missing GDF/RSDF response term. It is
the derivative of the finite uniform-grid egg-box error in the target-DFA XC
energy, with r2SCAN substantially more sensitive than PBE.

This conclusion follows from three independent checks:

1. At a nonstationary grid phase, the analytic translation sum and the central
   finite difference of the total energy under a rigid translation agree to
   about $3.3\times10^{-8}$ Ha/Bohr.
2. The component decomposition assigns essentially the entire translation sum
   to `target_xc`; one-electron, Coulomb, nuclear, Pulay and response sums are
   at numerical zero.
3. GDF and RSDF give the same egg-box amplitude and slope, so the effect is not
   owned by either DF response implementation.

## Reproducible Probe

The probe uses Water8, SZV-GTH, Gamma sampling and a rigid displacement along
$x$. The analytic gradient and a $10^{-4}$ Bohr rigid-energy central FD are
evaluated around one quarter of a uniform-grid step, where the egg-box energy
is not stationary.

| Backend | XC | Cutoff (Ha) | Mesh | Egg-box peak-to-peak (Ha) | Analytic sum | Rigid FD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GDF | r2SCAN | 200 | $123^3$ | $1.6622\times10^{-4}$ | $3.99814\times10^{-3}$ | $3.99812\times10^{-3}$ |
| GDF | r2SCAN | 400 | $173^3$ | $1.9222\times10^{-4}$ | $5.60125\times10^{-3}$ | $5.60121\times10^{-3}$ |
| GDF | r2SCAN | 600 | $211^3$ | $1.2264\times10^{-4}$ | $4.29307\times10^{-3}$ | $4.29304\times10^{-3}$ |
| GDF | r2SCAN | 1000 | $271^3$ | $1.9381\times10^{-5}$ | $-8.72621\times10^{-4}$ | $-8.72610\times10^{-4}$ |
| GDF | PBE | 400 | $173^3$ | $1.5826\times10^{-5}$ | $-4.54249\times10^{-4}$ | $-4.54247\times10^{-4}$ |
| RSDF | r2SCAN | 400 | $173^3$ | $1.9222\times10^{-4}$ | $5.60125\times10^{-3}$ | $5.60122\times10^{-3}$ |

The detailed values are stored in
`benchmark/backup/evidence/translation_sum_diagnosis_water8.json`. The executable probe is
`benchmark/diagnostics/diagnose_rigid_translation_eggbox.py`.

## Why Cutoff Convergence Is Not Monotonic Point by Point

Increasing the cutoff changes both the grid spacing and the phase of every
atom relative to that grid. The egg-box amplitude converges downward, but the
slope at one particular geometry can change sign or temporarily increase.
Consequently, one geometry's translation sum is not a sufficient convergence
criterion. A phase scan or a comparison of complete gradients at successive
meshes is required.

At 1000 Ha the Water8 r2SCAN egg-box amplitude is about 90% smaller than at
400 Ha, but the worst-phase translation slope is still about
$8.7\times10^{-4}$ Ha/Bohr. A universal claim that 500 or 1000 Ha is converged
would therefore be unsupported.

## Recommended Solution

### Production policy now

1. Keep the mesh fixed for a fixed-cell optimization or trajectory.
2. Run the rigid-phase probe along all three lattice directions at candidate
   meshes.
3. Compare the complete gradient at successive meshes, not only the total
   translation sum at one geometry.
4. Set `net_gradient_tolerance_ha_per_bohr` in validation queues so an
   unconverged result cannot silently pass as release evidence.
5. Choose the mesh from the intended force tolerance; do not use one universal
   cutoff for all functionals, bases and cells.

### Algorithmic solution

For a lower-cost finite-grid correction, energy and gradient must use the same
phase-averaged XC quadrature. A practical candidate is an interlaced set of
uniform grids shifted by half a grid step in each lattice direction. Averaging
all $2^3$ shifts cancels the leading odd aliasing modes, but it can cost up to
eight XC evaluations. This route requires a separate implementation and
energy/gradient FD gate before production use.

An atom-adaptive or validated multigrid target-XC implementation is the longer
term solution. It must preserve the same target density, r2SCAN $\tau$ terms,
and analytic grid derivative contract.

### What not to do

Do not subtract the center-of-mass force from every atom in the production
gradient. That makes the sum zero but is not the derivative of the computed
finite-grid energy, so it can hide the numerical error and break energy-force
consistency in optimization or molecular dynamics.
