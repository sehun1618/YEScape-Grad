# Third-Party Notices and Provenance

This repository is licensed as a whole under the Apache License, Version 2.0.
It interoperates closely with, and in selected files adapts code from, the
following upstream projects.

## PySCF

- Project: https://github.com/pyscf/pyscf
- License: Apache License 2.0
- Copyright: The PySCF Developers
- Preserved notice: `PYSCF_NOTICE`

The CPU packages depend on PySCF for molecular and periodic SCF, Gaussian
integrals, density fitting, numerical integration, and gradient primitives.
The periodic Fourier AO derivative loader and C callbacks adapt PySCF Fourier
integral conventions and retain file-level attribution.

## GPU4PySCF

- Project: https://github.com/pyscf/gpu4pyscf
- License: Apache License 2.0
- Copyright: The PySCF Developers

The GPU packages depend on GPU4PySCF for CUDA SCF, numerical integration,
periodic density fitting, integral kernels, and gradient primitives. The
one-pass native GDF derivative modules adapt GPU4PySCF periodic GDF derivative
implementations and retain file-level attribution and modification notices.

## Redistribution

The repository root and each independently buildable Python package include:

- a copy of the Apache License 2.0;
- this project's `NOTICE` file; and
- the original PySCF `NOTICE` file.

Dependency names are used only to identify compatibility and provenance. They
do not imply endorsement by the upstream projects.
