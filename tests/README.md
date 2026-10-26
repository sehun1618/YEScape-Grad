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

## Release artifacts and installed-package checks

Build an sdist, unpack it, and build the wheel from that unpacked source.
Inspect both artifacts before distribution:

```bash
python tests/check_distribution.py dist/*.tar.gz dist/*.whl
```

The check rejects local development/results directories, checkpoints, logs,
and the local-only theory note. It also checks required package/source files.
Install the wheel in a separate environment and run outside the repository,
with PYTHONPATH unset:

```bash
python /path/to/tests/check_installed.py --backend cpu --prefix /path/to/environment
```

The installed-package check covers HF/PBE sources, R/U, conventional/DF,
public API routing, and finite energy/gradient outputs. CPU also compares
the built OpenMP grid backend with pyscf_default. Use --backend gpu in a
GPU-configured environment; that check requires native cuTENSOR explicitly.
This requirement is for the release check, not for ordinary package installation.
These smoke checks do not replace finite-difference validation.

## Scanners

`tests/unit/test_scanner.py` checks CPU/GPU RHF/UHF conventional/DF scanner
parity, default density reuse, explicit/fresh guesses, geometry resets, and
failure recovery. It also checks D2C updates and geomeTRIC optimization when
the optional packages are installed.

```bash
python -m pytest tests/unit/test_scanner.py -v
```

## DFA-source development branch

Run `python tests/check_dfa_source.py --full-fd` from the repository root in
the configured environment. It compares relaxed-source FD energies with the
analytic gradient for LDA, PBE, TPSS, r2SCAN, B3LYP and CAM-B3LYP sources:
CPU/GPU, restricted/unrestricted, conventional/DF. The default target is
r2SCAN. Select `--target CAM-B3LYP` for a hybrid/RSH target or `--basis def2-svp`
for a larger orbital basis. `--backend cpu` needs no GPU installation.

The matrix writes per-case energies, gradients, term ledgers, Z residuals,
FD errors in Ha/Bohr and eV/Angstrom, and elapsed times under
`tests/results/dfa_source/run_*`. Each invocation creates a new directory and
prints its location; previous runs are never reused or overwritten. The
`run.json` file records command options, Git commit and worktree status
(not a complete snapshot of uncommitted source). Timings in this accuracy matrix are not controlled
performance benchmarks. Unit checks in `unit/test_dfa_source.py` cover the
native SC-DFT limit, source reuse/invalidation, checkpoint restoration, and
scanner reconvergence. They complement, not replace, the original HF tests.

### DFA-source validation, 2026-10-21

Environment: conda `gpu`, PySCF 2.14.0, GPU4PySCF 1.8.1, CuPy 14.0.1,
NumPy 2.4.6, native cuTENSOR. Water (R) and NH2 doublet (U); both
conventional and DF (def2-universal-jkfit), native pruned atom grids level 2,
source SCF tolerances 1e-12/1e-9, default CPHF tolerance 1e-9, FD step
2e-4 Bohr. CPU FD calculations used 48 threads. Jobs overlapped during
validation: elapsed times are not a speed comparison.

| Matrix | Cases / FD components | Max error (Ha/Bohr) | Max error (eV/Angstrom) |
|---|---:|---:|---:|
| Six source functionals to r2SCAN, STO-3G | 48 / 432 | 8.379e-9 | 4.310e-7 |
| PBE to CAM-B3LYP, STO-3G | 8 / 72 | 6.531e-9 | 3.359e-7 |
| PBE/r2SCAN to r2SCAN, def2-SVP | 16 / 144 | 5.750e-9 | 2.957e-7 |
| LC_WPBE to r2SCAN, STO-3G, selected component | 8 / 8 | 6.435e-10 | 3.309e-8 |

Raw files are in `results/dfa_source/{full_matrix,rsh_target,def2svp,lc_source}`.
All 80 cases passed the stated FD gates. This is representative small-molecule
validation, not an exhaustive functional or large-molecule certification.
The largest CPU/GPU analytic-gradient difference was 6.852e-10 Ha/Bohr
(3.523e-8 eV/Angstrom). Z residual reached 9.981e-8 in the LC_WPBE matrix;
warnings above 1e-8 remain visible. A good FD result does not mean that every
Z residual met 1e-8.

The final CPU suite passed 177 tests. The broader GPU run passed 61 existing
tests; four new source-lifecycle tests initially failed because the test tried
to reset native omega via the unsupported `omega=None` setter. This test was
corrected to restore the saved NumInt value. The final focused GPU source and
solver run passed 22 tests, with two existing native nonzero-level-shift CPHF
cases marked xfail. They are not newly introduced gradient failures. Initial
failure artifacts are retained. `cpu_release_check.xml` and
`gpu_release_check.xml` are the final regression records.

Executable output examples: `results/dfa_source/example_cpu.log` and
`example_gpu.log`, generated by `examples/08-dfa-source.py` and
`examples/gpu/05-dfa-source.py`. Method is `PBE-r2SCAN`, with separate source
and target functional/grid settings. No production commit or push was made.

### Extended publication checks

Run in conda `gpu`, from the repository root:

```bash
PYTHONPATH=. python tests/check_publication.py --phase limit
PYTHONPATH=. python tests/check_publication.py --phase diversity
PYTHONPATH=. python -m pytest tests -m slow
```

`limit` compares identical source/target functionals with native SC-DFT
gradients (PBE, r2SCAN, B3LYP, CAM-B3LYP; R/U, conventional/DF, CPU/GPU).
`diversity` checks benzene, diffuse-basis OH anion, distorted CH3 doublet,
and stretched H2 against relaxed-source FD. Benzene uses three selected
components; the other molecules use all components. Displaced SCFs start
from the central density and reconverge. This follows a solution branch;
it does not establish that the branch is the global ground state.

For an installed wheel, run `check_installed.py --backend gpu --prefix ENV
--fd` outside the checkout with `PYTHONPATH` unset. Use a venv without
`--system-site-packages` and install GPU dependencies into that environment.
The test requires native cuTENSOR and checks one FD component for each of
eight HF/KS-source, R/U, conventional/DF cases. This is an installation
check, not a replacement for the full numerical suite. Detailed local
reports and logs are stored under `results/publication_check/`.


### GPU4PySCF 1.9.0 compatibility validation

The current dependency target is PySCF 2.14.0 / GPU4PySCF 1.9.0.
The 1.8.1 results above are historical and were not relabeled as 1.9 results.
The 1.9 upgrade checks used CuPy 14.0.1 and native cuTENSOR 2.6.0 on CUDA 13:

- CPHF, source J/K and DF memory tests: 92 passed, 2 expected failures.
- The expected failures are native restricted CPHF nonzero-level-shift
  broadcasting cases; the default zero-level-shift path is tested separately.
- 28 small-molecule cases covered HF/PBE/r2SCAN/CAM-B3LYP/LC-WPBE/HSE06
  sources with restricted/unrestricted and conventional/DF paths.
- Maximum gradient change when switching to native 1.9 CPHF was
  7.35e-14 Ha/Bohr; maximum sampled-coordinate FD error was 2.15e-7 Ha/Bohr.

This is focused compatibility validation, not a fresh-install certification
or a repeat of the full publication validation matrix above. Run the focused
checks from the repository root in the selected GPU environment:

```bash
python -m pytest tests/unit/test_cphf.py tests/unit/test_source_jk.py tests/unit/test_gpu_memory.py -q
```
