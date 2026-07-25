# S6 Sampled-k Analytic Strain

## Scope

S6 extends the restricted CPU GDF/RSDF DC-PBE and DC-r2SCAN analytic stress
from Gamma to sampled k points. Fractional reciprocal coordinates and equal
k-point weights are held fixed while the absolute Cartesian k vectors follow
the strained reciprocal lattice.

## Complex three-center source

For an ordered pair $(k_i,k_j)$, let $q=k_j-k_i$. The reciprocal source uses
the physical Fourier momentum $G+q$. A translated ket AO carries

$$
e^{+i k_j\cdot L},
$$

while a real-space short-range image pair carries

$$
e^{-i k_i\cdot L_i+i k_j\cdot L_j}.
$$

These phases are invariant under homogeneous strain when scaled k points are
fixed. The strain kernel therefore differentiates Gaussian centers,
reciprocal momenta, and volume factors without differentiating the Bloch
phase separately.

The stress-owned short-range C callback uses PySCF's ME screening data and
accumulates the value plus nine ordered strain components before the image
indices are discarded. It never applies Gamma triangular-storage symmetry to
a complex pair.

## Metric and CDERI

The GDF metric is evaluated at $q$ as

$$
M(q)=M_{\mathrm{SR}}(q)+M_{\mathrm{LR}}(q)
     +\delta_{q0}M_{G=0}.
$$

RSDF uses its own exact screened two-center image list. Its complex callback
multiplies each retained image by $e^{iq\cdot L}$, preserving central PySCF
metric parity. For both backends,

$$
C(k_i,k_j)=L(q)^{-1}B(k_i,k_j),\qquad M(q)=L(q)L(q)^\dagger
$$

is differentiated in the lower-triangular complex Cholesky gauge. Reverse
pairs are reconstructed as conjugate AO transposes.

## J, K, and k-point normalization

The periodic matrices retain PySCF's explicit average:

$$
J_k=\frac{1}{N_k}\sum_{k'}J_{k,k'},\qquad
K_k=\frac{1}{N_k}\sum_{k'}K_{k,k'}.
$$

The later energy contraction contributes a second $1/N_k$. The implementation
keeps complex matrices until the final Brillouin-zone energy sum and does not
silently cast away imaginary parts.

## Pseudopotential and Madelung terms

- Local part 1 uses one diagonal source and overlap response per k point.
- Local part 2 evaluates the common reciprocal potential against k-dependent
  Bloch AOs on the uniform real-space grid.
- Nonlocal projectors use $G+k$ while the ionic structure factor remains on
  $G$.
- The Ewald exchange scalar is differentiated on the Monkhorst-Pack probe
  supercell used by `pyscf.pbc.tools.madelung`.

## Validation

The full CPU suite contains 63 tests. On the distorted H2/GTH-SZV cell with a
2x1x1 k mesh, selected relaxed-energy FD differences are:

| Backend | Target XC | Components | Max error (Eh/Bohr^3) |
| --- | --- | --- | ---: |
| GDF | PBE | xx, xy | 3.62e-10 |
| RSDF | PBE | xx, xy | 1.91e-10 |
| GDF | r2SCAN | xx | 3.71e-10 |

Evidence is stored in `stress/validation/results/2026-07-21_s6_*.json`.
The correctness-first CPU implementation remains slower than selected FD on
this tiny system; S6 does not claim a speedup.
