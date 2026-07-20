# Coupled pseudopotential response

## Why one source is enough

The final DC-DFT gradient contains the GTH pseudopotential derivative
contracted with the source-HF density $D$ and with the Z-vector transition
density $T_Z$:

$$
G_{\mathrm{PP}}[D] + 2G_{\mathrm{PP}}[T_Z].
$$

Both the local-part-1 SR/LR kernels and the nonlocal projector contraction are
linear maps of the density.  The Gamma production route therefore forms
$D+2T_Z$ and traverses each PP derivative primitive once.  This is an
algebraic reassociation, not a fitted force or screening approximation.

## Code ownership

- `grad/gdf.py` forms the coupled density and keeps the resulting force in
  `GDFGradientResult.coupled_pseudopotential`.
- `df/gdf/local_part1_derivative.py` provides batched/shared Fourier geometry
  support and the one-source coupled contraction.
- `grad/nonlocal_pseudo.py` builds the Gamma nonlocal projector half-integrals
  once and contracts any leading density-set axis.
- sampled-k calculations keep the prior resolved explicit/Z route.

The coupled term is added to the explicit device assembly only once.  It is
not also added to the Z-response array.  Metadata sets
`pseudopotential_component_resolved=false` so downstream analysis does not
mistake this exact combined channel for separately available components.

## Validation and performance

Separate-versus-batched local and nonlocal contractions agree within
`1e-9 Ha/Bohr`; complete Water20 gradients agree within `6.5e-14 Ha/Bohr`.
Water8 selected central FD errors are below `5.2e-9 Ha/Bohr`.  On Water20,
the PP section is approximately halved and the complete GDF/RSDF workflows
improve by 7.36% and 6.68%, respectively.
