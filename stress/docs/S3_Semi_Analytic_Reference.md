# S3 Gamma/GDF Analytic Stress

## Why this route exists

The original development route tested the complete DC-DFT stress Lagrangian
before every GDF homogeneous-strain kernel was available. The current route
combines:

1. one source-HF calculation and one target DFA state;
2. the same adjoint Z-vector used by the established nuclear-gradient code;
3. analytic uniform-grid target XC strain response;
4. analytic GDF metric/source/CDERI/J/K strain response;
5. analytic GTH local-part1, local-part2, and nonlocal pseudopotential response;
6. overlap/Pulay and Ewald terms in the same tensile-positive convention.

No strained cell is constructed in the production stress call. The supported
scope is deliberately narrow: restricted DC-PBE, Gamma, 3D GTH PP, GDF, and
uniform target integration. r2SCAN, RSDF, sampled k-points, and GPU remain
later stages.

## Assembly

With the central HF density $D$, target energy-weighted density
$D^E_{\mathrm{target}}$, and source Z-vector $Z$, one strain component is

$$
\frac{dE_{\mathrm{DC}}}{d\epsilon}
= \operatorname{Tr}[D h^{(1)}]
+ \frac{1}{2}\operatorname{Tr}[D J^{(1)}]
+ E_{\mathrm{xc}}^{(1)}
+ E_{\mathrm{Ewald}}^{(1)}
- \operatorname{Tr}[D^E_{\mathrm{target}} S^{(1)}]
+ 4\langle Z, B^{(1)}_{\mathrm{HF}}\rangle.
$$

The source response $B^{(1)}_{\mathrm{HF}}$ contains the complete
$h^{(1)}+J^{(1)}-K^{(1)}/2$ matrix, occupied-energy overlap response, and HF
metric-density response. The same converged Z-vector is contracted with all
six strain components; no CPHF equation is solved six times.

## Validation

For distorted H2/GTH-DZVP/GTH-HF/PBE/Gamma-GDF, mesh `(15,15,15)`, and
matrix-FD step $2.5\times10^{-4}$:

- AO dimension: 10;
- real-linear Z dimension: 18;
- Z residual: $1.73\times10^{-18}$;
- maximum orbital-response stress: $1.51\times10^{-4}$ Eh/Bohr$^3$;
- maximum difference from full strained-HF energy FD:
  $1.62\times10^{-10}$ Eh/Bohr$^3$;
- stress-reference wall time after the central energy: 1.56 s;
- full stress-FD wall time: 2.51 s.

Those numbers describe the historical matrix-FD reference. The completed H2
analytic evidence is stored in
`validation/results/2026-07-21_s3_gamma_gdf_analytic.json`; its maximum error is
$8.47\times10^{-11}$ Eh/Bohr$^3$. The independent H2O/GTH-PBE evidence in
`validation/results/2026-07-21_s3_gamma_gdf_analytic_h2o.json` has maximum
error $5.10\times10^{-10}$ Eh/Bohr$^3$ and Z residual $6.94\times10^{-18}$.

The report also stores GDF's resolved numerical decomposition for every
strain pair. CDERI and pseudopotential builders choose independent range
separations and meshes. A `+/-` source, metric, or PP mesh mismatch is rejected
rather than differentiated through a discrete mesh change.

## Capability and performance boundary

`low_level_numerical_components` is empty and the canonical API is
`gamma_gdf_analytic_stress`. The CPU implementation is not yet optimized:
H2 took 6.69 s versus 1.32 s for full FD, and H2O took 13.85 s versus 2.79 s.
The dominant costs are the transparent image-resolved PP and GDF source/CDERI
kernels. Correctness promotion therefore does not imply a speed claim.
