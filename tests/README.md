# Tests

Run the commands below from the repository root, with DC-Forge and the test
dependencies installed.
Tests that calculate D2C dispersion are skipped when DFT-D4 is not installed;
parameter lookup and validation tests still run.

## Analytic gradients versus finite differences

Compare analytic gradients against central finite differences for all Cartesian
components: RHF (water) / UHF (NH2), conventional / DF, CPU / GPU (eight cases).
Each displacement reruns HF; these checks use tighter settings than the examples
to resolve finite differences (`step=1e-4 Bohr`, PBE/def2-SVP, grid level 3).

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -m pytest \
  tests/cpu/test_gradient_fd.py tests/gpu/test_gradient_fd.py \
  -m 'not slow' -v --capture=tee-sys \
  --junitxml=tests/results/gradient-fd.xml
```

CUDA dependencies and an available GPU are required for the GPU cases.
For CPU only, omit `tests/gpu/test_gradient_fd.py`.
The output reports maximum errors in Ha/Bohr and eV/Angstrom.

## CPU and GPU suites

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -m pytest tests/cpu tests/unit -m 'not gpu and not slow' --junitxml=tests/results/cpu-unit.xml
python -m pytest tests/gpu
```

GPU tests require an available CUDA device. Finite-difference tests rerun HF
at displaced geometries; they do not hold the source density fixed.
Check SCF convergence and the reported Z-vector residual, not only whether a
calculation returned a gradient.

## Scanners

`tests/unit/test_scanner.py` checks CPU/GPU RHF/UHF conventional/DF scanner
parity, default density reuse, explicit/fresh guesses, geometry resets, and
failure recovery. It also checks D2C updates and geomeTRIC optimization when
the optional packages are installed.

```bash
python -m pytest tests/unit/test_scanner.py -v
```
