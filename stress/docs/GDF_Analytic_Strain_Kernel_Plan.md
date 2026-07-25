# GDF Analytic Homogeneous-Strain Kernel Plan

## Boundary established by the reference

The DC Lagrangian, target XC, overlap, and Z-vector assembly are already
validated. The remaining numerical section is the derivative of GDF-owned AO
matrices. Replacing it does not require changing the DC response equation.

The replacement must produce six engineering-Voigt derivatives for:

$$
h^{(1)}_{\mathrm{GDF\ PP}},\qquad
J^{(1)}[D],\qquad K^{(1)}[D].
$$

It must use the exact central GDF representation, including its auxiliary
basis, source mesh, metric mesh, source range split, PP range split, and
linear-dependence convention.

## A. GDF pseudopotential

`GDF.get_pp()` has three owners that must remain separate.

1. Local part 1, short range
   - Add a periodic three-center strain driver that retains lattice-image
     displacement weights.
   - Ordinary atom-center `ip1/ip2` derivatives are insufficient because cell
     strain also moves every translated image.
2. Local part 1, reciprocal tail
   - Differentiate reciprocal vectors, $G$-space quadrature weights, Coulomb
     kernel, fake-nuclear Fourier factors, and AO-pair Fourier factors.
   - Hold the central integer mesh fixed.
3. Local part 2 and nonlocal projectors
   - Differentiate AO images and projector centers in the same GDF value
     representation.
   - Keep PySCF's low-level finite-difference projector derivative as an
     independent parity oracle until an image-aware analytic kernel passes.

Completion for PP is matrix parity against `gdf_matrix_finite_difference`,
not only density-contracted energy parity. The initial gate is
`max_abs(dVpp_analytic-dVpp_matrix_fd) <= 1e-7 Ha` on small cells.

## B. GDF Coulomb metric and three-center source

For one Gamma CDERI block,

$$
C=L^{-1}B,\qquad M=LL^\dagger,
$$

and

$$
C^{(1)}=L^{-1}B^{(1)}-L^{-1}L^{(1)}C.
$$

The existing nuclear-gradient Cholesky response can be reused unchanged once
strain derivatives $B^{(1)}$ and $M^{(1)}$ are available.

1. Metric short range
   - Implement image-weighted `int2c2e_ip1/ip2` strain contractions.
2. Metric reciprocal tail
   - Differentiate auxiliary Fourier factors, weighted Coulomb kernel, and
     reciprocal quadrature weights.
3. Metric $G=0$
   - Include the explicit $1/\Omega$ derivative.
4. Three-center short range
   - Implement image-weighted primary/auxiliary derivative contractions.
5. Three-center reciprocal tail
   - Differentiate AO-pair FT, auxiliary FT, Coulomb kernel, and weights.
6. `vbar`
   - Include overlap and volume response in the same gauge as the central
     stored CDERI.

Each source and metric subterm is compared with a low-level finite difference
of the corresponding pre-Cholesky value before `dC` is assembled. The final
`dC` must reproduce both `dJ` and `dK` matrix references.

## C. Numerical settings

GDF CDERI and GDF PP use independent automatically selected range splits. The
reference records both for each strain pair. Development rules are:

- integer source, metric, and PP meshes must match between `+/-` pairs;
- the central range split is used for analytic decomposition;
- changing `omega` is not a physical strain term because the exact SR+LR sum
  is independent of the split;
- residual dependence on `omega` is reported as numerical convergence error;
- no system-dependent fitted correction is allowed.

## D. Implementation order

1. analytic reciprocal metric and its component FD tests;
2. analytic metric $G=0$ term;
3. short-range metric image kernel;
4. reciprocal three-center source;
5. short-range three-center image kernel;
6. Cholesky response and J/K matrix parity;
7. GDF PP reciprocal part;
8. GDF PP short-range/local/nonlocal parts;
9. replace matrix-FD calls in the development stress;
10. only then set `production=true` and mark S2/S3 complete.

The first five steps work on small Gamma cells only. Sampled k-points and GPU
contractions remain later stages and do not enter these kernels prematurely.

## E. Completed implementation boundary (2026-07-21)

Completed analytic matrix kernels:

- full Gamma auxiliary metric: short range, reciprocal tail, and G=0;
- full Gamma three-center source: short range, reciprocal tail, and vbar;
- Cholesky/CDERI response and fixed-density J/K contraction;
- Ewald exchange-divergence response, including the Madelung scalar;
- GTH GDF local-part1, including its representation-specific pseudo G=0 term;
- GTH local-part2 Gaussian-polynomial reciprocal response;
- GTH nonlocal reciprocal projector response;
- 3D ion-ion Ewald real/self/reciprocal response and the probe-charge
  Madelung response.

The short-range source is evaluated by `c_ext/sr3c_strain.c`. It reuses
PySCF's screened shell traversal and accumulates

$$
B^{(1)}_{xy}=I^{ip1}_x(R_j-R_i)_y+I^{ip2}_x(R_j-R_P)_y
$$

inside the image loop. Translational invariance therefore removes the need
for a third derivative integral on the second primary AO center.

The production PBE/Gamma/GDF path no longer calls the matrix-FD oracle or
PySCF's internally displaced-cell PP/Ewald helpers. Those routines remain
independent tests only. Reciprocal AO/projector arrays and Ewald vectors are
blocked to bound temporary memory, but the current image-resolved CPU kernels
prioritize transparent equation parity over speed.
