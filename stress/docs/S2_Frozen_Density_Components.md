# S2 Frozen-Density Cell Derivatives

## Purpose

S2 separates the explicit target-energy cell derivative from orbital response.
The spin-summed AO density matrix from the central source-HF calculation is
held fixed while the lattice, fractional atom positions, basis images,
reciprocal vectors, and integration weights respond to strain.

This is different from the S1 oracle. S1 reruns HF after every strain and
therefore differentiates the complete relaxed DC-DFT energy. S2 does not rerun
HF and is designed to expose errors one physical component at a time.

## Current decomposition

The fixed-density target energy is recorded as

$$
E_{\mathrm{DFA}}[D]
= E_{\mathrm{kin}}[D]
+ E_{\mathrm{PP}}[D]
+ E_J[D]
+ E_{\mathrm{xc}}[D]
+ E_{\mathrm{Ewald}}.
$$

`GammaGDFFrozenDensityEvaluator` rebuilds only the target GDF representation
for a strained cell and evaluates every term with the unchanged central AO
density. `component_finite_difference` differentiates these components with a
fixed integer mesh and fixed fractional atomic coordinates.

The initial analytic primitives were:

- kinetic integral strain response;
- uniform-grid PBE XC density, AO, coordinate, and volume-weight response;
- Ewald ion-ion response.

For a distorted H2 cell with GTH-SZV, PBE, Gamma/GDF, and mesh `(15,15,15)`,
their maximum differences from component FD at strain step $2.5\times10^{-4}$
are respectively

| component | max error (Eh/Bohr^3) |
| --- | ---: |
| kinetic | $1.89\times10^{-10}$ |
| XC | $5.31\times10^{-10}$ |
| Ewald | $9.96\times10^{-11}$ |

## Why GDF PP is not borrowed from PySCF FFT stress

`pyscf.pbc.grad.rks_stress.get_vxc(..., with_nuc=True)` differentiates the
pseudopotential in an FFT representation. The DC-DFT target hcore used here is
built by `GDF.get_pp()`, whose local part is assembled from range-separated
Gaussian and reciprocal pieces. At finite numerical settings they are
different representations.

On the same distorted H2 test, treating the FFT PP derivative as if it were
the GDF PP derivative gives a maximum error of
$2.92\times10^{-5}$ Eh/Bohr$^3$. Therefore it is retained only under the name
`fft_pseudopotential_not_gdf` in diagnostics. The small public
`frozen_density_analytic_components` object remains intentionally partial,
while the complete GDF explicit assembly uses package-owned GDF PP kernels.

## Completed GDF-specific kernels

The complete PBE/Gamma/GDF route now includes homogeneous-strain derivatives of:

1. GDF local and nonlocal GTH pseudopotential assembly;
2. the Gaussian auxiliary Coulomb metric;
3. the three-center source tensor;
4. reciprocal vectors, quadrature weights, cell volume, and lattice images;
5. the Cholesky/metric solve that maps the source to CDERI.

Existing nuclear-gradient tensors could not simply be summed with atomic
coordinates. They differentiate basis centers at a fixed lattice and do not
contain the lattice-image and reciprocal-vector response. A low-level
finite-difference reconstruction remains an independent development oracle and
is not called by `gamma_gdf_analytic_stress`.
