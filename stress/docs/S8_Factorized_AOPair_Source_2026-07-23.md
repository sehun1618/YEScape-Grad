# S8 Factorized AO-Pair Source Response

## Problem

The analytic long-range source derivative used to build each Voigt response
as a full grid tensor,

$$
I^{(1)}_{xy}(G,ij)
=-I^i_x(G,ij)R_{i,y}-I^j_x(G,ij)R_{j,y}
+i(G+q)_xM_y(G,ij),
$$

and then multiply it by the weighted auxiliary Fourier values. Water8 profiling
showed that constructing these six large tensors cost more than the Fourier
integrals or the matrix products themselves.

## Exact reassociation

Coordinates that depend only on the AO-pair index can be applied after the
matrix product,

$$
[I^i_x R_{i,y}]^T W
=R_{i,y}[I^i_x]^T W.
$$

Likewise the reciprocal momentum can be applied to the auxiliary weight before
the moment projection,

$$
[k_xM_y]^T W=M_y^T[k_xW].
$$

The implementation therefore projects the three center derivatives and the
moment factors first, then assembles only six `(nao^2,naux)` Voigt outputs.
The formula, integer mesh, range-separation parameter, precision, and screening
are unchanged.

## Auto policy

The factorized route performs more matrix products and is slower for small
systems. The production direct-contracted builder estimates one assembled
image as

$$
n_G n_{AO}^2\,\mathrm{sizeof}(\mathrm{complex128}).
$$

It uses the original assembled route below 256 MiB and the factorized route at
or above 256 MiB. Reports expose the requested mode, resolved mode, and image
size. Users do not need to tune a numerical parameter.

## Evidence

- H2 GDF/RSDF, Gamma and `2x1x1`: selected FD below
  `2.9e-11 Eh/Bohr^3`.
- Water8 RSDF: selected FD `6.90e-11 Eh/Bohr^3`.
- Water8 RSDF source: 109.24 s to 79.73 s (`-27.0%`).
- Water8 RSDF total stress: 116.70 s to 87.11 s (`-25.4%`).
- Peak RSS: 8.66 GiB to 8.39 GiB (`-3.1%`).
- Water8 GDF source: 591.01 s to 396.20 s (`-33.0%`).
- Water8 GDF total stress: 605.21 s to 404.83 s (`-33.1%`).
- Water8 GDF RSS: 12.38 GiB to 11.95 GiB (`-3.5%`).
- GDF target-J/adjoint raw parity against the earlier FD-passed run:
  `7.1e-15 Ha` / `1.2e-15 Ha`.

The source-adjoint and device-only GEMM experiments are retained only as
development evidence because they were slower than this route.

## Bounded device projection

Uploading each full center or moment tensor to the GPU made the projection
faster, but it also created multi-GiB transfer staging buffers. The bounded
device route accumulates the same sums over 512 MiB G-vector blocks. Within
each block, every center/moment slice is uploaded once and the fifteen small
projected outputs remain on device. This changes neither the integral nor the
floating-point precision.

| Water8 route | Source (s) | Stress (s) | RSS (GiB) | DF-stage device use (GiB) |
|---|---:|---:|---:|---:|
| RSDF host factorized | 79.73 | 87.11 | 8.39 | 2.59 |
| RSDF full device projection | 52.42 | 59.85 | 9.37 | 4.50 |
| RSDF bounded device projection | 32.62 | 40.00 | 8.87 | 3.23 |
| GDF host factorized | 396.20 | 404.83 | 11.95 | 2.58 |
| GDF full device projection | 279.59 | 288.23 | 13.91 | 5.53 |
| GDF bounded device projection | 262.50 | 271.11 | 12.44 | 3.13 |

The bounded RSDF selected-FD error is `6.90e-11 Eh/Bohr^3`. Its stress agrees
with the host factorized route within `2.81e-17 Eh/Bohr^3`; the GDF difference
is `1.66e-17 Eh/Bohr^3`. A device-memory preflight accounts for the projected
outputs, one transfer block, and contraction temporaries before allocation.

This backend is accepted as an exact explicit opt-in:

```json
"stress_df_response_backend":
  "host_analytic_direct_contracted_factorized_device_bounded"
```

The default remains the host assembled/factorized auto route until large
sampled-k and Water20/64 evidence closes the device-memory scaling gate.
