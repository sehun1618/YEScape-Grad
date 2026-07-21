# D2C-D4 dispersion

## Scope

The optional `dispersion="d2c"` route evaluates

\[
E_{\mathrm{DC+D2C}}
= E_{\mathrm{DFA}}[n_{\mathrm{HF}}] + E_{\mathrm{D2C-D4}}(\mathbf R,\mathbf A).
\]

The functional-specific rational-damping parameters are stored in
`pyscf_pbc_dcdft/dispersion/parameters/d2c_d4.json`. They reproduce Table S1
of the Supporting Information for Lee et al., JCTC 2024,
DOI `10.1021/acs.jctc.4c00689`. The published choices use
$s_6=s_9=1$ and fit $s_8$, $a_1$, and $a_2$ for each target DFA.

## Gradient ownership

D4 depends on elements, coordinates, charge, lattice, and periodicity. It does
not depend on the HF orbital coefficients used by the DC-DFT Lagrangian.
Therefore,

\[
\frac{dE_{\mathrm{DC+D2C}}}{d\mathbf R_A}
= \frac{dE_{\mathrm{DFA}}[n_{\mathrm{HF}}]}{d\mathbf R_A}
+ \frac{\partial E_{\mathrm{D2C-D4}}}{\partial\mathbf R_A},
\]

and the D2C term enters the explicit channel without changing the Z-vector
equation. The `dftd4` library returns energy, analytic nuclear gradient, and
virial in atomic units. Energy and gradient are active here; the virial is
retained for future stress support.

## CPU and GPU boundary

Both public packages use the same parameter registry and periodic evaluator.
On a GPU run, HF, target DFA, and DC response retain their GPU providers. D4
is a small host-library call and is reported as such; it is not an electronic
structure CPU fallback.

## Limitations

The parameters were fitted to molecular benchmark data. Their use in periodic
condensed systems is an extrapolation that requires convergence and reference
validation. Only functionals listed in the bundled registry are accepted.
The option is disabled by default and requires `dftd4>=4.2`.
