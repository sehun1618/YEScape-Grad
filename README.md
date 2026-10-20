# DC-Forge

Molecular HF-DFT energies and analytic nuclear gradients built on PySCF
and, optionally, GPU4PySCF.

## Installation

Requires Python 3.11+. Use a separate environment.
Run the commands below from the repository root.

For pinned dependency examples, see [environments](environments/README.md).
`pyproject.toml` defines installation requirements; the environment files
record tested or installed versions, not unverified minimum versions.

### CPU

```bash
python -m pip install .
# Optional: editable installation with test dependencies
python -m pip install -e '.[test]'
```

Installs NumPy, SciPy, and **PySCF 2.14.0**; an existing PySCF version may change.
No GPU packages are installed.

The default CPU grid backend, `becke_omp`, requires a C compiler with OpenMP.
If it cannot be built, set
`mf_grad.grid_response_backend = 'pyscf_default'`.

### GPU

Install **GPU4PySCF 1.8.1** separately for your CUDA environment, then install
DC-Forge. Follow the [upstream installation guide](https://github.com/pyscf/gpu4pyscf#installation).

| CUDA family | GPU4PySCF package | Optional cuTENSOR package |
|---|---|---|
| 11.x | `gpu4pyscf-cuda11x` | `cutensor-cu11` |
| 12.x | `gpu4pyscf-cuda12x` | `cutensor-cu12` |
| 13.x | `gpu4pyscf-cuda13x` | `cutensor-cu13` |

Example for a compatible CUDA 12 environment:

```bash
python -m pip install 'gpu4pyscf-cuda12x==1.8.1'
python -m pip install .
python -m pip check
```

- Match the driver, CUDA, and upstream CuPy/cuTENSOR versions.
- Do not mix CUDA variants of GPU4PySCF or CuPy in one environment.
- `nvcc --version` reports the Toolkit; `nvidia-smi` reports driver CUDA support.
- DC-Forge does not install GPU packages, drivers, or the Toolkit.
- The table lists installation choices, not validated configurations.
  Run the GPU tests on your environment.

### Optional libraries

**D2C:** install DFT-D4 separately. It is unnecessary without dispersion.

```bash
python -m pip install 'dftd4>=4.2'
```

**cuTENSOR:** optional GPU acceleration. Install a compatible version separately
using the upstream guide, and verify that GPU4PySCF loads it.

Neither library is installed by DC-Forge. Both retain their own licenses.


## Features

- Restricted and unrestricted HF source orbitals (RHF/UHF).
- Conventional and density-fitted molecular calculations on CPU/GPU.
- Fixed-HF-density DFT energies and analytic gradients, including orbital
  response, moving-grid response and DF auxiliary-basis response.
- Optional D2C dispersion corrections for supported functional parameters.
- Restoring converged HF orbitals from checkpoints.

PBC, stress, ROHF, and experimental DFA-source gradients are not included.
This is a molecular extraction, not a claim of new numerical validation for
every functional or hardware platform.


## Examples

### HF-DFT energy and gradient

```python
from pyscf import gto
from dcforge.cpu.energy import RHFDFT

mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
            basis='def2-svp')
mf = RHFDFT(mol, xc='PBE').density_fit()
mf.grids.level = 4
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()  # Ha/Bohr
force = -gradient
print('HF energy [Ha]:', mf.e_hf)
print('HF-DFT energy [Ha]:', energy)
print('Force [Ha/Bohr]:', force)
```

`kernel()` converges HF and then evaluates the target functional without
target DFT SCF. Energy is in Hartree; forces are minus the energy gradient.
DC-Forge prints total energy and forces at every verbose level. Level 3 adds
result summaries and wall times; level 4 adds calculation settings and detailed
diagnostics. PySCF internal logs follow the same `verbose` setting.
The printed forces are `-gradient`; the gradient API still returns `dE/dR`.

### Conventional and density-fitted calculations

Using the `mol` above, choose one of the following:

```python
# Conventional integrals
mf = RHFDFT(mol, xc='PBE')

# Automatically selected auxiliary basis
mf = RHFDFT(mol, xc='PBE').density_fit()

# Explicit JK-fit auxiliary basis for def2-svp
mf = RHFDFT(mol, xc='PBE').density_fit(auxbasis='def2-svp-jkfit')
```

Then call `kernel()` and `nuc_grad_method().kernel()` as above.

### Open-shell calculations

Use `UHFDFT` and set `spin = Nalpha - Nbeta`. For example, an OH doublet:

```python
from pyscf import gto
from dcforge.cpu.energy import UHFDFT

mol = gto.M(atom='O 0 0 0; H 0 0 .97', basis='def2-svp', spin=1)
mf = UHFDFT(mol, xc='PBE').density_fit()
mf.grids.level = 4
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()
force = -gradient
```

### GPU calculations

Use the GPU classes with the same calculation interface. This example uses
closed-shell water; use `UHFDFT` for an open-shell molecule.

```python
from pyscf import gto
from dcforge.gpu.energy import RHFDFT

mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
            basis='def2-svp')
mf = RHFDFT(mol, xc='PBE').density_fit()
mf.grids.level = 4
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()
force = -gradient
```

Install the GPU dependencies described above before using this example.

### D2C dispersion

After installing DFT-D4, enable D2C for a supported target functional:

```python
from dcforge.cpu.energy import RHFDFT

# Use the water molecule defined above.
mf = RHFDFT(mol, xc='PBE').density_fit().set_dispersion('d2c')
mf.grids.level = 4
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()
force = -gradient

print('HF-DFT energy without dispersion [Ha]:', mf.e_dft)
print('D2C dispersion energy [Ha]:', mf.e_d2c)
print('Total energy [Ha]:', energy)
```

Both the returned energy and gradient include D2C dispersion.
Parameters are selected for the target functional; unsupported functionals
raise an error rather than falling back to ordinary D4.

### Example files

Run examples from the repository checkout:

- `examples/01-rhf_dft.py`: restricted conventional/DF calculations.
- `examples/02-uhf_dft.py`: unrestricted calculations.
- `examples/03-xc.py`: target functional selection.
- `examples/04-checkpoint.py`: conventional/DF RHF checkpoint restoration.
- `examples/05-grid.py`: grid levels, explicit radial/angular grids
  (shared or element-specific), and CPU grid-response backends.
- `examples/06-d2c.py`: dispersion.
- `examples/gpu/`: corresponding molecular GPU workflows.

## Tests

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python -m pytest tests/cpu tests/unit -m 'not gpu and not slow' --junitxml=tests/results/cpu-unit.xml
python -m pytest tests/gpu
```

GPU tests require an available CUDA device. Finite-difference tests rerun HF
at displaced geometries; they do not hold the source density fixed.
Check SCF convergence and the reported Z-vector residual, not only whether a
calculation returned a gradient.

## Attribution and distribution

Developed by Sehun Kim <sehun1618@yonsei.ac.kr>,
Theoretical and Computational Chemistry Laboratory (TCCL), Yonsei University.

Original source comments, scientific citations and upstream attribution are
preserved. Upstream license notices are in `third_party/`. These notices do
not themselves license DC-Forge's original contributions. A project-wide
license must be chosen by the rights holders before public distribution.
