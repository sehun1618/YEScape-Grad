# Periodic DC-DFT Gradients

The periodic implementation is split into a CPU base package and a GPU
extension, following the PySCF/GPU4PySCF package boundary:

- `cpu`: `pyscf-pbc-dcdft`, the CPU energy, response, and gradient package
- `gpu`: `gpu4pyscf-pbc-dcdft`, the CuPy/GPU4PySCF extension
- `benchmark`: queue-based timing and finite-difference validation tools
- `docs`: theory notes and equation-to-code implementation guides

The active scope is restricted, closed-shell, fixed-cell DC-DFT. FFTDF, GDF,
and RSDF routes support Gamma-point and sampled-k-point calculations; exact
capabilities and numerical gates are documented in the package READMEs and
validation scripts.

Install the CPU package first and then the GPU extension when needed:

```bash
python -m pip install -e ./pbc/cpu
python -m pip install -e ./pbc/gpu
```

Generated benchmark results are intentionally excluded from version control.
