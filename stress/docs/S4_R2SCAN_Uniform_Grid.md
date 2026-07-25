# S4 DC-r2SCAN Uniform-Grid Stress

## MGGA strain variables

For r2SCAN the XC energy depends on density, density gradient, and positive
kinetic-energy density,

$$
E_{\mathrm{xc}}=\int_\Omega
 f(\rho,\nabla\rho,\tau)\,d\mathbf r,
\qquad
\tau=\frac12\sum_{\mu\nu}D_{\mu\nu}
\nabla\phi_\mu\cdot\nabla\phi_\nu.
$$

The strain derivative includes AO values, first and second AO derivatives,
grid-coordinate motion, cell-volume weight response, and the explicit
$\tau^{(1)}$ channel. The version-guarded PySCF 2.13/2.14 uniform-grid primitive
provides these density channels; stress-owned GDF, PP, Ewald, overlap, and
kinetic kernels provide the remaining terms without constructing strained
cells.

## Validation

- Distorted H2 fixed-density r2SCAN XC agrees with component FD within
  $5.58\times10^{-10}$ Eh/Bohr$^3$ over the full tensor.
- Cubic H2 complete analytic stress agrees with full relaxed DC-r2SCAN FD
  within $8.86\times10^{-11}$ Eh/Bohr$^3$.
- Distorted H2 complete analytic stress agrees within
  $2.20\times10^{-10}$ Eh/Bohr$^3$.

Evidence is stored in
`validation/results/2026-07-21_s4_gamma_gdf_r2scan_analytic.json` and
`validation/results/2026-07-21_s4_gamma_gdf_r2scan_distorted_h2.json`.

## Mesh and phase error

Uniform-grid r2SCAN is not exactly invariant to a rigid translation at finite
mesh. `benchmark_mgga_mesh_phase.py` measures this error at fixed physical
geometry for increasing integer meshes. The result is reported as numerical
grid convergence and is never removed by post-processing the stress tensor.

For a rigid shift `(0.23, 0.17, 0.11)` Bohr, the maximum stress difference is
$1.05\times10^{-4}$, $2.18\times10^{-6}$, and
$1.28\times10^{-8}$ Eh/Bohr$^3$ at meshes 9, 15, and 21, respectively. The
compact report is `validation/results/2026-07-21_s4_r2scan_mesh_phase.json`.
