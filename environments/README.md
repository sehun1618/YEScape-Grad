# Environment examples

`pyproject.toml` defines DC-Forge's installation requirements.
These files pin dependency snapshots; they do not define minimum supported
versions or lock every transitive dependency.

Run from the repository root:

```bash
conda env create -f environments/cpu.yml
conda activate dcforge-cpu
python -m pip install .
python -m pip check
```

For CUDA 13, replace `cpu.yml` with `gpu-cu13.yml` and activate
`dcforge-gpu-cu13`. Install a compatible NVIDIA driver separately.
For other CUDA families, follow the
[GPU4PySCF installation guide](https://github.com/pyscf/gpu4pyscf#installation);
do not combine CUDA variants.

## Validation status

Snapshot date: 2026-10-20.

| File | Evidence |
|---|---|
| `cpu.yml` | Python/NumPy/SciPy/PySCF versions used for 16 molecular CPU hybrid/RSH analytic-gradient versus finite-difference checks |
| `gpu-cu13.yml` | Installed GPU package versions only; a fresh environment and GPU numerical tests have not been validated for this snapshot |

The CPU checks covered water and NH2, conventional/DF, B3LYP, PBE0,
CAM-B3LYP and LC_WPBE. They do not certify every functional.
Neither environment file has been recreated in a clean environment yet.
Older minimum versions have not been established by testing.

The default CPU `becke_omp` backend needs a C compiler with OpenMP.
Alternatively, select `mf_grad.grid_response_backend = 'pyscf_default'`.

## Optional packages

Neither YAML file explicitly installs DFT-D4 or cuTENSOR.
DC-Forge's own installation requirements do not include either package;
third-party GPU packages may have their own dependency rules.

- D2C dispersion: install `dftd4` separately. The local installed version is
  `4.2.0`; this snapshot's hybrid/RSH checks did not include dispersion.
- cuTENSOR: install a version compatible with the selected GPU4PySCF/CuPy
  combination. The local installed version is `cutensor-cu13==2.6.0`.
  Installation alone does not ensure activation: the inspected environment
  reported a CuPy contraction fallback on a plain GPU4PySCF import.

After installation, run `python -m pip check` and the relevant tests described
in the main README. Environment creation is not numerical validation.
