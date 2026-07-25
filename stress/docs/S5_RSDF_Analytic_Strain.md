# S5 Gamma RSDF Analytic Stress

## Representation ownership

RSDF uses two independently resolved range splits:

$$
B=B_{\mathrm{SR}}(\omega_B)+B_{\mathrm{LR}}(\omega_B,\mathcal G_B)+B_{G=0},
$$

$$
M=M_{\mathrm{SR}}(\omega_M)+M_{\mathrm{LR}}(\omega_M,\mathcal G_M)+M_{G=0}.
$$

The source therefore uses `source_omega`, `source_mesh`, and `precision_R`,
whereas the metric uses `metric_omega`, `metric_mesh`, and `precision_j2c`.
These parameters are copied from the built central provider and are not
re-optimized during differentiation.

The RSDF short-range metric must reproduce PySCF's screened numerical
representation. A stress-owned C callback uses the same shell-pair bounds and
image list as `PBCsr2c`, accumulating each integral and its image-weighted
strain response in one traversal. The reciprocal and zero-mode responses then
reuse the equations already validated for GDF.

## Validation

| system / target | max analytic-FD error (Eh/Bohr$^3$) |
| --- | ---: |
| H2 / PBE | $5.90\times10^{-11}$ |
| H2 / r2SCAN | $5.56\times10^{-11}$ |
| H2O / PBE | $4.74\times10^{-10}$ |

The reconstructed central CDERI differs from actual RSDF HDF5 storage by at
most $7.52\times10^{-10}$ in these gates. The analytic route performs no
strained-cell calculation and reports source/metric parameters separately.

Changing source/metric omega is a convergence diagnostic, not a new default.
The package follows the values resolved by PySCF unless the caller explicitly
configures the RSDF provider.
