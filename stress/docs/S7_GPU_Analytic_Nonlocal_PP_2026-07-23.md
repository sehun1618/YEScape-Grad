# GPU analytic nonlocal GTH PP strain

## Purpose

The exact reference built the full nonlocal pseudopotential strain matrix on
the host and uploaded it before contracting two densities.  For Water8 this
single primitive took about 539--558 seconds.  The new opt-in backend

```json
"stress_pp_response_backend":
  "gpu4pyscf_native_contracted_analytic_nonlocal_device"
```

computes the same analytic derivative without constructing a
`(6,nk,nao,nao)` derivative matrix and without strained cells.

## Fourier first-moment recurrence

For a Cartesian Gaussian AO

$$
\chi_{l_xl_yl_z}(\mathbf r-\mathbf R)
=x^{l_x}y^{l_y}z^{l_z}e^{-\alpha|\mathbf r-\mathbf R|^2},
$$

multiplication by one relative coordinate raises one Cartesian exponent.  For
example,

$$
(r_y-R_y)\chi_{l_xl_yl_z}=\chi_{l_x,l_y+1,l_z}.
$$

The implementation makes an internal cell copy with every shell angular
momentum raised from $l$ to $l+1$.  One additional GPU FT-AO call then supplies
the three selected first moments.  The exact AO strain is

$$
\frac{\partial F_p(\mathbf G+\mathbf k)}
 {\partial\epsilon_{xy}}
=i(G+k)_x M_{yp}(\mathbf G+\mathbf k),
$$

with the $-F_p/2$ normalization derivative on diagonal strain components.
GPU4PySCF's internal s/p Cartesian normalization is corrected by the exact
shell-wise ratio between the original and augmented shells.

The nonlocal projector derivative is evaluated from

$$
\frac{\partial p}{\partial\epsilon_{xy}}
=-(G+k)_x\frac{\partial p}{\partial(G+k)_y}.
$$

Only the six symmetric Voigt combinations are accumulated.  Projector overlap
and response tensors are contracted directly with the source and adjoint
densities, so no AO derivative matrix is retained.

Atoms sharing an element and angular projector channel are processed as one
GPU batch. AO strain is built once per reciprocal block and Voigt component;
the projector response is built once per channel rather than once per atom.
For eight oxygen atoms this reduces each response-overlap call count from 960
to 120.

## Ownership

- GPU: central and moment FT-AO, structure factors, six Voigt projector/AO
  contractions, source/adjoint density traces, local GTH reciprocal value and
  slope, and local structure-factor sums.
- Host: small element/channel radial-angular projector value and gradient
  templates for each reciprocal block and k point.
- No displaced cell, extra SCF, target DFA rebuild, or extra Z-vector solve is
  used.

The host boundary is reported as
`small_projector_value_gradient_template`; this route is not described as a
pure-GPU integral engine.

## Correctness

- Random two-density SZV Gamma response matches the host analytic matrix.
- Complex Hermitian `kmesh=[2,1,1]` response matches the host analytic matrix.
- DZVP-MOLOPT-SR-GTH d polarization, whose moment uses an f shell, matches the
  host analytic matrix.
- Water1 GDF/RSDF six-component FD errors are about
  $6.3\times10^{-12}$ Eh/Bohr$^3$.
- Water8 GDF/RSDF selected-FD errors are about
  $6.9\times10^{-11}$ Eh/Bohr$^3$.

All reported runs have `production=true` and an empty
`low_level_numerical_components` list.

## Water8 performance

| Backend | Route | Stress (s) | PP (s) | Nonlocal PP (s) | RSS (GiB) | VRAM delta (GiB) |
|---|---|---:|---:|---:|---:|---:|
| GDF | host analytic matrix | 1141.76 | 546.07 | 538.92 | 12.72 | 7.43 |
| GDF | numerical fast | 603.43 | 12.13 | 4.98 | 12.73 | 11.48 |
| GDF | device analytic, before atom batching | 605.21 | 12.05 | 6.40 | 12.38 | 7.32 |
| RSDF | host analytic matrix | 674.76 | 565.57 | 558.35 | 9.00 | 6.46 |
| RSDF | numerical fast | 122.81 | 12.18 | 5.02 | 9.01 | 11.50 |
| RSDF | device analytic, atom batched + GPU local | 114.29 | 6.40 | 1.90 | 8.67 | 7.41 |

Before atom batching, the device analytic route removed 97.8% of PP wall
relative to the host matrix route while matching numerical-fast total wall.
The final atom-batched RSDF route with GPU-native local GTH setup removes 98.9%
of PP wall relative to the host matrix route and is 5.2% faster end-to-end than
the numerical fast route. Sampled VRAM delta is about 35.6% lower than the
numerical route. A Water8
nonlocal-only profile measured 6.42 to 1.96 seconds after batching, a 69.5%
reduction. Moving local $G^2$, damping, polynomial, and structure-factor work
from NumPy to CuPy reduced local setup from 1.315 to 0.125 seconds (`-90.5%`).
The dominant remaining wall time is the exact host DF source
primitive: about 593 seconds for GDF and 110 seconds for RSDF.

These are profiled single runs, not the required three-repeat release median.
The final atom-batched GDF full workflow was not rerun because its unchanged
exact DF source costs about ten minutes. Its final correctness is covered by
same-kernel component tests and the Water1 GDF six-component FD gate; the
Water8 GDF row above is the immediately preceding unbatched profile.

## Limits

GPU4PySCF FT-AO supports shells through `LMAX=4`.  The first-moment recurrence
therefore accepts original AO angular momentum through f (`l<=3`) and raises a
clear `NotImplementedError` for an original g shell.  A complete sampled-k
end-to-end energy-FD gate and Water20/64 user scaling evidence remain open.
