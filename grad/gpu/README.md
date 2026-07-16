# Molecular GPU DC-DFT Gradients

This package provides molecular DC-DFT analytic gradients for GPU4PySCF.

- `gpu4pyscf_dcdft_grad.grad.rhf` is the default restricted adjoint route.
- `gpu4pyscf_dcdft_grad.grad.uhf` contains the unrestricted route.
- `gpu4pyscf_dcdft_grad.grad.backends` contains optional cuEST SCF and hybrid
  execution paths.

The restricted implementation avoids constructing the full perturbation-
indexed response tensor by using an adjoint HF response and direct contracted
DF derivative terms.

```python
from gpu4pyscf_dcdft_grad import RDCGradient

gradient = RDCGradient(hf, xc="PBE").kernel()
```

The cuEST backends require the optional `cuest` dependency and are not imported
by the base package.
