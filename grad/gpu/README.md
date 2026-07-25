# Molecular GPU DC-DFT Gradients

This package provides molecular DC-DFT analytic gradients for GPU4PySCF.

- `gpu4pyscf_dcdft_grad.grad.rhf` is the default restricted adjoint route.
- `gpu4pyscf_dcdft_grad.grad.uhf` contains the unrestricted route.
- `gpu4pyscf_dcdft_grad.grad.backends` contains optional cuEST SCF and hybrid
  execution paths.

The restricted and unrestricted implementations avoid constructing full
perturbation-indexed response tensors by using adjoint HF responses and direct
contracted DF derivative terms. Both public routes use shared DF storage,
MO-direct CPHF/UCPHF and adjoint responses, factorized direct `Z:F1`, and
streamed S1 contractions by default.

```python
from gpu4pyscf_dcdft_grad import RDCGradient

gradient = RDCGradient(hf, xc="PBE").kernel()
```

The unrestricted route accepts the same memory budget and response controls:

```python
from gpu4pyscf_dcdft_grad import UDCGradient

gradient = UDCGradient(
    uhf, xc="PBE", response_backend="mo_direct", share_df=True,
    direct_z_f1=True, df_deriv_memory_gb=32.0,
).kernel()
```

Large direct-derivative jobs can use an exact memory budget. This changes DF
batch sizes but not the computed expression.

```python
gradient = RDCGradient(
    hf, xc="PBE", df_deriv_memory_gb=32.0
).kernel()
```

Set `response_backend="ao"`, `share_df=False`, and
`factorized_z_f1=False` only for regression comparison with the older route.
For UHF, the corresponding full-reference switch is `direct_z_f1=False`.
See [OPTIMIZATION.md](OPTIMIZATION.md) for the implementation and benchmark
record.

The cuEST backends require the optional `cuest` dependency and are not imported
by the base package.

GPU4PySCF 1.8 moved the private MO-projected DF response kernel from
`df.hessian.jk` to `df.hessian.rhf`.  The package compatibility adapter accepts
both the 1.7 and 1.8 layouts and removes the occupied-space padding required by
the new unrestricted kernel before returning alpha/beta response blocks.
