# Molecular structures

All coordinates are in Angstrom.  XYZ files do not encode electronic charge or
spin, so callers must pass the values in the table explicitly to
`pyscf.gto.M`.  Here `spin` follows PySCF's convention, `spin = N_alpha -
N_beta = 2S`.

The stress cases are deliberately non-equilibrium geometries.  They are inputs
for numerical and algorithmic validation, not recommended equilibrium
structures or reference geometries for chemical conclusions.

| File | Charge | Spin | Role |
|---|---:|---:|---|
| `water1.xyz` | 0 | 0 | Regular closed-shell RHF control |
| `oh.xyz` | 0 | 1 | Small UHF case that has exposed a relatively large Z-vector residual |
| `o2_triplet.xyz` | 0 | 2 | Better-conditioned regular UHF control |
| `no_cation.xyz` | +1 | 0 | Charged, closed-shell, linear RHF response case |
| `nh2_doublet.xyz` | 0 | 1 | Bent UHF case with coupled alpha/beta response |
| `ch2_triplet_distorted.xyz` | 0 | 2 | Asymmetric triplet UHF response case |
| `ch3_doublet_distorted.xyz` | 0 | 1 | Non-symmetric open-shell UHF case |
| `h2_stretched.xyz` | 0 | 0 | Near-dissociation RHF case with a small orbital gap and strong static-correlation character |
| `water1_asymmetric_stretch.xyz` | 0 | 0 | Large, asymmetric force and grid-response tail |
| `water_dimer_shared_proton.xyz` | 0 | 0 | Strong hydrogen-bond/proton-sharing tail with a midpoint proton |

## Provenance and intended use

`no_cation.xyz`, `h2_stretched.xyz`, `nh2_doublet.xyz`, and
`ch2_triplet_distorted.xyz` reproduce the geometries used by the earlier
DC-Forge molecular RHF/UHF gradient-completeness audit.  That audit used them
to test orbital-Hessian, moving-gauge cancellation, analytic-gradient, and
finite-difference behavior outside the single equilibrium-water case.

`ch3_doublet_distorted.xyz` reproduces a non-symmetric open-shell case from the
earlier molecular open-shell survey.  `o2_triplet.xyz` reproduces the regular
UHF checkpoint example used to distinguish a normal open-shell calculation
from the more difficult OH response case.

`water1_asymmetric_stretch.xyz` is the first water molecule extracted and
translated from the earlier `water8_oh_stretch.xyz` tail structure.  Its two
O-H distances are approximately 0.9924 and 1.2500 Angstrom.

`water_dimer_shared_proton.xyz` is the relevant neutral H4O2 fragment extracted
and translated from the earlier `water8_proton_shared.xyz` tail structure.  It
has an O--O distance of approximately 2.6900 Angstrom and one hydrogen at the
O--O midpoint, approximately 1.3450 Angstrom from both oxygen atoms.

These files should not all be placed in one undifferentiated pass/fail test.
Use the regular controls for smoke tests and the stress cases for explicit
convergence, residual, finite-difference, and failure-reporting tests.
