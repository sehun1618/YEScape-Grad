# Target multigrid with GDF and RSDF source HF

## Scope

The source calculation is unchanged:

```text
source RHF/KRHF hcore, J, K       selected GDF or RSDF
source Hessian and Z-vector AX    selected GDF or RSDF
source density                    D_HF
```

Only the fixed-density target DFA evaluation is selectable.  The public GPU
driver accepts:

```python
mf.set_target_integrator("uniform")
mf.set_target_integrator("multigrid_xc")
mf.set_target_integrator("multigrid_j_xc")
```

The CPU package accepts `uniform` only and raises an explicit GPU-only error
for the two multigrid modes.  There is no silent fallback.

## Energy and Fock ownership

For M1 (`multigrid_xc`),

$$
V_{\mathrm{target}} = J_{\mathrm{GDF/RSDF}}[D_{\mathrm{HF}}]
                       + V_{\mathrm{xc}}^{\mathrm{MG}}[D_{\mathrm{HF}}].
$$

The adapter calls `MultiGridNumInt.nr_rks(..., with_j=False)`, ignores any
Coulomb tag produced as an intermediate by upstream code, and obtains the
target J matrix and Coulomb energy from the selected GDF/RSDF provider.

For M2 (`multigrid_j_xc`),

$$
V_{\mathrm{target}} = J_{\mathrm{MG}}[D_{\mathrm{HF}}]
                       + V_{\mathrm{xc}}^{\mathrm{MG}}[D_{\mathrm{HF}}].
$$

The selected provider still owns source HF, exchange and response, but it no
longer contributes the explicit target $dJ/dR$.  M2 therefore changes the
finite numerical representation of target Coulomb relative to finite-auxbasis
GDF/RSDF and is not expected to reproduce M1 energy at a fixed auxiliary basis.

Target hcore and GTH pseudopotential always remain selected-provider terms.
Calling the upstream high-level `multigrid_numint()` modifier is deliberately
avoided because it can change J and pseudopotential ownership together.

## Gradient ownership

The explicit target terms are

$$
G_A^{\mathrm{M1}} =
  G_A^{J,\mathrm{GDF/RSDF}} + G_A^{\mathrm{xc,MG}},
$$

and

$$
G_A^{\mathrm{M2}} = G_A^{J+\mathrm{xc,MG}}.
$$

Both use `multigrid_v2.get_veff_ip1` with the matching `with_j` flag and
`with_pseudo_vloc_orbital_derivative=False`.  GPU4PySCF currently returns this
primitive multiplied by the number of sampled k-points, so the package divides
by $N_k$ exactly once.  Gamma and $2\times1\times1$ central-FD gates confirm the
normalization.

M2 disables the selected-DF target-J derivative.  Native GDF changes a
requested target-J/Z fused contraction to source-only one-pass response.  RSDF
uses its resolved response observable and retains only the source-HF bilinear
term.  The source-HF $d(J-K)$ response is never removed.

## Workspace and metadata

Energy stores the target Fock, density, multigrid object and generated task
cache in `TargetEvaluationWorkspace`.  Gradient reuses the same object and does
not rebuild target XC.  The input signature includes target mode and optional
mesh, so changing either invalidates the converged state.

Reports record:

- J, XC, hcore and PP owners;
- resolved mesh, task count and per-level meshes;
- energy and gradient multigrid wall time;
- whether gradient reused the energy task cache;
- source-HF energy and density/MO identities;
- host RSS, sampled device use and CuPy pool high-water marks.

## Validation on 2026-07-17

H2 PBE Gamma and H2 r2SCAN sampled-k tests cover GDF/RSDF M1/M2.  All selected
analytic-FD errors are below $3.5\times10^{-7}$ Ha/Bohr; Z-vector residuals are
below $6.2\times10^{-13}$.  Within one process, RSDF U0/M1/M2 source-HF values
satisfy

$$
\Delta E_{\mathrm{HF}}=0,\qquad
\max|\Delta D_{\mathrm{HF}}|=3.3\times10^{-16}.
$$

Water8 r2SCAN three-run medians are:

| Source | Target | target energy (s) | gradient (s) | target explicit (s) |
| --- | --- | ---: | ---: | ---: |
| GDF | uniform | 1.191 | 3.659 | 1.379 |
| GDF | multigrid XC | 0.752 | 2.656 | 0.399 |
| RSDF | uniform | 1.090 | 4.054 | 1.397 |
| RSDF | multigrid XC | 0.638 | 3.063 | 0.405 |

M1 reduces end-to-end gradient time by about 27% for GDF and 24% for RSDF.
Host peak RSS falls by about 23%, while the CuPy pool high-water increases by
about 0.51 GiB for GDF and 0.28 GiB for RSDF because multigrid task data remain
cached for gradient reuse.

Water8 rigid phase scans show similar, not systematically smaller, egg-box
amplitudes: about $1.92\times10^{-4}$ Ha along x/y and
$2.52\times10^{-4}$ Ha along z for both uniform and multigrid XC.  M1 is
therefore kept as explicit opt-in rather than promoted to default.

M2 is also explicit opt-in.  With the small `SZV-GTH` auxiliary basis, Water8
M2 energy differs from selected-GDF/RSDF M1 by about 3.10 Ha because target J
uses the reciprocal multigrid Coulomb representation instead of finite
auxiliary fitting.  Its analytic derivative is internally correct, but energy
comparison requires separate target-J mesh/auxbasis convergence.

## PP-only decision

Water20 M1 profiles leave the complete one-electron section at about 1.8 s,
versus total gradients of 12.8 s (GDF) and 16.7 s (RSDF).  Only part of that
1.8 s is pseudopotential work.  A PP-only multigrid route would also require a
new hcore, explicit derivative and nuclear-CPHF representation and is unlikely
to deliver the required 5% end-to-end gain.  It remains deferred; no hidden PP
ownership change is present in M1 or M2.

## Evidence

- `benchmark/multigrid_correctness_gate.json`
- `benchmark/multigrid_performance_gate.json`
- `benchmark/results/multigrid_correctness_gate/`
- `benchmark/results/multigrid_performance_gate/`
- `benchmark/results/multigrid_translation/`
