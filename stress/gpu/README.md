# GPU PBC DC-DFT Stress

This namespace is isolated from the existing GPU gradient package. S7 provides
production restricted GDF/RSDF analytic stress through `gpu_analytic_stress`.

The source HF state, target Fock/XC stress, HF AX/Z-vector, complex Cholesky
response, component-streamed J/K response, and final DC Lagrangian contraction
run on the active CuPy device. PySCF/GPU4PySCF 2c/3c homogeneous-strain
integrals and the current one-electron/GTH strain primitives are still
generated analytically on the host and transferred as primitives. This is a
declared ownership boundary, not a silent full CPU-stress fallback.

The release contract explicitly accepts this integral-engine boundary, as a
GPU electronic-structure route need not imply that the host-side integral
engine itself is a CUDA kernel. The boundary, worker count, memory estimate,
and observed peak VRAM remain visible in every result. This production label
does not mean that every operation runs on GPU.

For an MLIP-ready same-state record, run the source calculation once and use:

```python
from gpu4pyscf_pbc_dcdft_stress import gpu_energy_force_stress
from pyscf_pbc_dcdft_stress import write_mlip_extxyz, write_mlip_json

result = gpu_energy_force_stress(converged_dc_method)
write_mlip_json(result.record, "frame.json")
write_mlip_extxyz(result.record, "frame.xyz")
```

The workflow reuses the gradient Z-vector for stress and does not rerun the
source HF SCF.

## Experimental contracted GPU primitives

The default remains the fully analytic host-integral/device-response route.
Two explicit benchmark options avoid materializing large PP and DF strain
matrices:

```json
"stress_pp_response_backend": "gpu4pyscf_native_contracted",
"stress_df_response_backend": "gpu_fixed_density_fd_contracted"
```

The PP route shares GPU AO/grid work across the source and adjoint densities.
The DF route differentiates only the target Coulomb and adjoint RHF scalar
contractions, rebuilding fixed-density J/K at 12 displaced cells. Neither
route reruns source SCF, target DFA, or the Z-vector. Because both contain a
low-level centered cell displacement, reports mark them as experimental with
`production=false`; they do not silently replace the analytic defaults.

The PP route is validated for Water8 GTH-SCAN and H2 GTH-HF. The original H2
probe used `mesh=[9,9,9]`, which was insufficient for the full local GTH
potential. At common `ke_cutoff_ha=400` (`mesh=[75,75,75]`), exact/contracted
H2 stress differs by at most `2.9e-13 Eh/Bohr^3`. The PP route remains opt-in
because its nonlocal factor derivative still uses centered cell displacement.

## Analytic direct-contracted DF

The following opt-in route keeps analytic host source/metric derivatives but
contracts each canonical k-pair immediately into target-J and adjoint-RHF
scalars:

```json
"stress_pp_response_backend": "host_analytic_matrix",
"stress_df_response_backend": "host_analytic_direct_contracted"
```

It does not retain pair derivatives, central factors, or six AO J/K response
matrices. H2 Gamma/k211 GDF/RSDF agrees with the matrix-producing exact route to
`1e-18 Eh/Bohr^3`, passes six-component FD, and reports `production=true`.
The host integral engine remains an explicit owner; this is a memory/response
contraction optimization, not a claim that 2c/3c integrals are GPU kernels.

The direct route now chooses between two exact AO-pair response layouts. It
keeps the original assembled image below 256 MiB and uses center/moment-first
factorization for larger images. This avoids repeated multi-GiB response
assembly without changing mesh, cutoff, omega, precision, or screening.
Reports expose `stress_df_pair_response_mode_requested`,
`stress_df_pair_response_mode_resolved`, and
`stress_df_pair_response_image_bytes`. The explicit development override is:

```json
"stress_df_response_backend": "host_analytic_direct_contracted_factorized"
```

Large exact factor projections can instead use a bounded GPU transfer route:

```json
"stress_df_response_backend":
  "host_analytic_direct_contracted_factorized_device_bounded"
```

It accumulates over 512 MiB G-vector blocks and performs a free-VRAM preflight.
It is an explicit opt-in until larger sampled-k memory gates are complete.

The faster reverse-source FT route contracts the GDF/RSDF source adjoint
directly with GPU4PySCF's blocked AO-pair strain kernel:

```json
"stress_df_response_backend": "host_analytic_source_adjoint_gpu_ft"
```

It is FD-validated for RSDF and native GDF at Gamma/`2x1x1`, including an H2
`DZVP-GTH` gate. The repaired Water20 GDF/RSDF scaling gate also passes at
61.52/52.51 s stress and 5.43/5.20 GiB peak RSS. Current GPU user queues opt in
explicitly; every `run_input.json` and report records the selected backend.

Against the prior bounded-factor projection, Water8 RSDF/GDF stress fell by
79.9%/95.5% and peak RSS by 64.7%/71.3%; selected FD remained below
`6.9e-11 Eh/Bohr^3`. See
`../docs/S9_GPU_FT_Source_Design_2026-07-23.md`.

## Device-contracted analytic nonlocal PP

The following opt-in backend replaces the numerical nonlocal PP factor
derivative with an exact augmented-shell Fourier first moment:

```json
"stress_pp_response_backend":
  "gpu4pyscf_native_contracted_analytic_nonlocal_device"
```

Central and first-moment FT-AO values, six symmetric Voigt responses, and the
two density traces run on the GPU. Small radial/angular projector templates are
built on the host and this boundary is explicit in metadata. No PP derivative
AO matrix or displaced cell is used. Gamma, complex sampled-k, and d-polarized
DZVP component tests pass; Water1 and Water8 GDF/RSDF FD gates report
`production=true`. See `../docs/S7_GPU_Analytic_Nonlocal_PP_2026-07-23.md`.

The Python API default remains `host_analytic_matrix` as a compatibility
choice. The exact optimized route has passed the Water20 scaling gate and is
available through explicit production opt-in in current user queues.

Detailed CUDA stream timing is opt-in:

```json
"stress_detailed_profile": true
```

The report separates stream-event intervals from Python wall time and states
that stream elapsed is not a hardware kernel activity trace.
