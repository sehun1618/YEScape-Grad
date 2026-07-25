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

## Response And Memory Strategy

RHF, UHF, and the UHF representation used by ROHF share the same optimized
non-variational structure:

1. Solve the CPHF/UCPHF equation for the Z vector.
2. Apply the self-adjoint HF response once to the Z transition density.
3. Build overlap derivative blocks for one atom and Cartesian direction and
   immediately contract them with the adjoint response.
4. For density-fitted references, consume PySCF's DF Hessian `_gen_jk`
   generator one atom at a time instead of retaining the full `make_h1` list.

The last item uses the same PySCF integral implementation and equations as
`make_h1`; only the lifetime of its returned atom blocks changes. Non-DF
references continue to use the public `make_h1` fallback.
