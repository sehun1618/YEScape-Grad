# DC-DFT Analytic Gradients

This repository develops analytic nuclear gradients for density-corrected
density functional theory (DC-DFT) with PySCF and GPU4PySCF.

The repository is organized as a small monorepo:

- `grad/cpu`: molecular CPU implementation
- `grad/gpu`: molecular GPU implementation
- `pbc/cpu`: periodic CPU package (`pyscf-pbc-dcdft`)
- `pbc/gpu`: periodic GPU package (`gpu4pyscf-pbc-dcdft`)
- `pbc/benchmark`: reproducible periodic benchmark driver and small inputs
- `pbc/docs`: periodic theory and implementation documentation

Only source code, tests, examples, and reproducible benchmark inputs are
tracked. Generated wheels, compiled extensions, caches, checkpoints, and raw
benchmark results are intentionally excluded.

## Project Status

The periodic packages currently target restricted closed-shell fixed-cell
energy and analytic nuclear gradients. Supported periodic routes and their
validation status are documented under `pbc/docs` and in each package README.

The molecular CPU/GPU code is being consolidated from research prototypes
into stable PySCF-style package interfaces. See the package READMEs for the
public API and known limitations.

## Development

Each package has its own `pyproject.toml`, tests, examples, and optional
dependencies. Install a package in editable mode from its directory, for
example:

```bash
python -m pip install -e 'pbc/cpu[test]'
```

GPU tests require a compatible CUDA, CuPy, and GPU4PySCF environment.

## License

This project is distributed under the Apache License 2.0, matching the
licenses used by PySCF and GPU4PySCF. See `LICENSE`, `NOTICE`, and
`THIRD_PARTY_NOTICES/` for attribution and source-provenance details.
