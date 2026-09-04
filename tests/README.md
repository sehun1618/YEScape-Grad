# DC-Forge tests

The suite follows package ownership and check cost:

- `unit/`: imports, public contracts, structures, D2C parameters, and the GPU
  memory planner. These do not run an HF calculation.
- `cpu/`: molecular CPU RHF/UHF, conventional/DF energy, analytic gradient,
  term ledger, and finite differences.
- `gpu/`: corresponding GPU calculations plus CPU/GPU parity. These tests skip
  when GPU4PySCF or a CUDA device is unavailable.

Finite differences are not taken by moving atoms while keeping one AO density
fixed. Each `+h` and `-h` point creates a new molecule, reconverges source HF,
and evaluates the target DFA on that source density. The resulting derivative
checks the full nonvariational HF-DFT gradient, including Z-vector response.

The regular release controls are water for RHF and triplet O2 for UHF. The
remaining files in `structures/` are stress cases and belong to the `slow`
lane; a difficult structure is not allowed to weaken regular tolerances.

## Validation map

| Capability | Path | Independent check |
|---|---|---|
| Public imports and rejected scope | `unit/test_contracts.py` | Fresh Python process and expected exceptions |
| XYZ charge/spin metadata | `unit/test_structures.py` | Atom count and electron/spin parity |
| D2C parameters and derivative | `unit/test_d2c.py` | Published parameter values and central FD |
| Adaptive GPU DF memory planner | `unit/test_gpu_memory.py` | Deterministic synthetic size cases |
| CPU RHF/UHF fixed-density energy | `cpu/test_energy.py` | Independent PySCF RKS/UKS supplied-density evaluation |
| CPU analytic gradient and term ledger | `cpu/test_gradient.py` | Term recomposition, `direct`/`make_h1`, conventional/DF |
| CPU complete gradient | `cpu/test_gradient_fd.py` | Relaxed-source central FD |
| Difficult molecular tails | `cpu/test_stress_cases.py` | Selected FD without weakening regular gates |
| GPU energy | `gpu/test_energy.py` | CPU/GPU parity |
| GPU analytic gradient routes | `gpu/test_gradient.py` | CPU parity, `direct`/`make_h1`, DF AO/MO-direct parity |
| GPU complete gradient | `gpu/test_gradient_fd.py` | Relaxed-source central FD |

The regular FD gate uses `def2-SVP`, an unpruned level-3 molecular grid,
`conv_tol=1e-11`, `conv_tol_grad=1e-9`, and a `1e-4 Bohr` central step. The
analytic and displaced calculations use identical settings. FD thresholds are
`2e-6 Ha/Bohr` for RHF CPU, `3e-6 Ha/Bohr` for UHF CPU and RHF GPU, and
`4e-6 Ha/Bohr` for UHF GPU. The observed errors should be inspected before
loosening any gate; a test threshold is not an estimate of method accuracy.

## Commands

Run from the repository root. GPU commands must use the `gpu` Conda environment
because it owns PySCF, GPU4PySCF, DFT-D4, CuPy, and cuTENSOR.

```bash
source /home/ksh/miniconda3/etc/profile.d/conda.sh
conda activate gpu

# Fast API and CPU checks
python -m pytest -q tests/unit tests/cpu -m "not fd and not slow"

# CPU finite differences
python -m pytest -q tests/cpu -m fd

# Fast GPU checks and CPU/GPU parity
python -m pytest -q tests/gpu -m "gpu and not fd and not slow"

# GPU finite differences
python -m pytest -q tests/gpu -m "gpu and fd"

# Complete suite, including slow route and stress-structure checks
python -m pytest -q
```

Useful focused paths are:

```bash
python -m pytest -q tests/cpu/test_gradient_fd.py -k rhf
python -m pytest -q tests/gpu/test_gradient_fd.py -k density_fit
python -m pytest -q tests/unit/test_contracts.py
```

CPU conventional gradients permit `grid_response=False`. The current CPU DF
path requires both grid and auxiliary-basis response; all GPU paths require
grid response, GPU DF additionally requires auxiliary response, and GPU
gradients require all atoms. Negative tests preserve these current boundaries
so an unsupported partial derivative cannot look like a valid total gradient.
