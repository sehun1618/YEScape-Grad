# PySCF 2.14.0 and GPU4PySCF 1.8.0 compatibility

## Validated environment

The repository was audited with PySCF 2.14.0 and GPU4PySCF 1.8.0.  The
periodic stress adapter accepts both PySCF 2.13.x and 2.14.x, but keeps an
upper bound below 2.15 because it intentionally calls private stress
primitives.  Runtime signature checks remain active for every private entry
point.

## Improvements inherited without a local algorithm change

- PySCF 2.14 chunks periodic CCGDF k-point pairs while building out-of-core
  three-center tensors.  CPU sampled-k GDF construction therefore gains its
  lower-memory path automatically.
- GPU4PySCF 1.8 improves periodic GDF construction, CDERI loading and
  decompression, `get_j`, one-electron derivatives, multigrid density kernels,
  and `eval_xc_eff` memory use.  The DC-DFT GPU routes call these upstream
  primitives through their existing providers and inherit the improvements.
- GPU4PySCF 1.8 improves auxiliary linear-dependency handling and changes the
  central periodic GDF builder default from Cholesky to eigen-decomposition.
  The local DC derivative and exact-central contracts require the same
  Cholesky CDERI representation as CPU PySCF, so only those process-local build
  scopes restore Cholesky and then restore the upstream global default.  Other
  GPU4PySCF 1.8 construction and contraction improvements remain active.
- GPU4PySCF 1.8 also sends small-mesh PBC hcore/PP evaluation through
  `multigrid_v2`, even when GDF or RSDF was selected.  The DC source and target
  hcore are now evaluated explicitly through the selected DF provider because
  their analytic one-electron derivative must have the same owner.
- GPU4PySCF 1.8 removes the private stress helper
  `pbctools._get_Gv_with_base`.  The local PP strain implementation now uses
  `gpu4pyscf.pbc.gto.cell.get_Gv_weights`, preserving device-resident
  reciprocal vectors without depending on the removed symbol.
- The new multigrid stress implementation is useful as an independent
  self-consistent-DFT validation oracle.  It cannot replace the full DC-DFT
  stress because the latter also differentiates the frozen HF-density
  constraint and its orbital response.

## Local kernels that remain necessary

GPU4PySCF 1.8 implements sampled-k `get_j_e1_kpts`, but it returns an
atom-contracted J energy derivative.  `get_k_e1_kpts` remains unimplemented.
The DC response requires the full AO matrices

$$
\frac{\partial}{\partial R_{Ax}}\left(J-\frac12K\right),
$$

so the local FFTDF J/K derivative implementation cannot be replaced by the
new upstream J-only interface.

Likewise, upstream self-consistent stress and second-order SCF do not include
the DC-DFT Lagrangian.  They are candidates for validation and optional source
HF convergence acceleration, respectively, rather than default replacements.

The molecular GPU package also used the private MO-projected DF response
kernel.  GPU4PySCF 1.8 moved it from `df.hessian.jk.get_jk` to
`df.hessian.rhf._get_jk` and changed unrestricted inputs to padded spin
arrays.  A version-compatible adapter now normalizes both layouts and removes
the temporary UHF padding from the returned response.

## Deferred opt-in opportunities

- Benchmark periodic second-order SCF for difficult source-HF convergence.
  It should remain opt-in because ordinary converged SCF is already equivalent
  and easy systems may not recover the setup cost.
- Exploit k-point time-reversal symmetry in DC derivative contractions only
  after complex phase, odd derivative, and non-TR-symmetric mesh tests exist.
- Compare upstream self-consistent multigrid stress with the corresponding
  non-DC limit as an additional regression oracle.

No production default is changed by these deferred items.
