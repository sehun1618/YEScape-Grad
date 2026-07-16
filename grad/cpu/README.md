# Molecular CPU DC-DFT Gradients

This package provides PySCF-style molecular DC-DFT energies and analytic
nuclear gradients.

Public modules are named by reference type:

- `pyscf_dcdft_grad.grad.rhf`: restricted closed-shell
- `pyscf_dcdft_grad.grad.uhf`: unrestricted open-shell
- `pyscf_dcdft_grad.grad.rohf`: restricted open-shell
- `pyscf_dcdft_grad.grad.api`: automatic RHF/UHF dispatch

The public interface is:

```python
from pyscf_dcdft_grad import dc_grad

gradient = dc_grad(hf, xc="PBE").kernel()
```

ROHF support is exposed explicitly as `ROGradients`; automatic dispatch is
currently limited to RHF and UHF references.
