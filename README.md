# YEScape-Grad: Yonsei Electronic Structure Calculations and Property Evaluation

Molecular non-self-consistent energies and analytic nuclear gradients built on PySCF
and, optionally, GPU4PySCF.

## Installation

Requires Python 3.11+. Use a separate environment.
Run the commands below from the repository root.

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

Install **GPU4PySCF 1.9.0** separately for your CUDA environment, then install
YEScape-Grad. Follow the [GPU4PySCF installation guide](https://github.com/pyscf/gpu4pyscf#installation).

| CUDA family | GPU4PySCF package | Optional cuTENSOR package |
|---|---|---|
| 11.x | `gpu4pyscf-cuda11x` | `cutensor-cu11` |
| 12.x | `gpu4pyscf-cuda12x` | `cutensor-cu12` |
| 13.x | `gpu4pyscf-cuda13x` | `cutensor-cu13` |

Example for a compatible CUDA 12 environment:

```bash
python -m pip install 'gpu4pyscf-cuda12x==1.9.0'
python -m pip install .
python -m pip check
```

- Match the driver, CUDA, and upstream CuPy/cuTENSOR versions.
- Do not mix CUDA variants of GPU4PySCF or CuPy in one environment.
- `nvcc --version` reports the Toolkit; `nvidia-smi` reports driver CUDA support.
- YEScape-Grad does not install GPU packages, drivers, or the Toolkit.
- The table lists installation choices, not validated configurations.
  Run the GPU tests on your environment.

YEScape-Grad targets **PySCF 2.14.0 and GPU4PySCF 1.9.0**.
GPU calculations have been validated with CUDA 13 and cuTENSOR.
See `environments/gpu-cu13.yml` for the tested dependency versions.
The CUDA 12 installation example above has not been validated.

### Optional libraries

**D2C:** install DFT-D4 separately. It is unnecessary without dispersion.

```bash
python -m pip install 'dftd4'
```

**cuTENSOR:** optional GPU acceleration. Install a compatible version separately
using the upstream guide, and verify that GPU4PySCF loads it.

Neither library is installed by YEScape-Grad. Both retain their own licenses.


## Features

- Molecular HF-DFT and DFA-source non-SCF energies and analytic gradients.
- Restricted and unrestricted HF or Kohn–Sham references.
- Conventional and density-fitted calculations.
- CPU and GPU implementations based on PySCF and GPU4PySCF.
- LDA, GGA, tau-meta-GGA, hybrid and RSH source and target functionals.
- Optional D2C dispersion corrections for supported functionals.
- Molecular energy/gradient scanners for geometry optimization.

Nonlocal correlation functionals, including VV10, are not supported.

### Analytic-gradient support

| Calculation | CPU | GPU |
|---|---|---|
| Restricted/unrestricted HF or KS source | Supported | Supported |
| Conventional / density-fitted J and K | Supported | Supported |
| LDA, GGA, tau-meta-GGA, hybrid, RSH | Supported families; representative functionals tested | Supported families; representative functionals tested |
| Full grid and auxiliary-basis response | Supported; enabled by default | Supported; enabled by default |
| Atom-subset gradient | Supported | Full molecule only |
| `only_dfj=True` gradient | Not supported | Not supported |
| D2C correction | HF source and available target parameters; requires DFT-D4 | Same |
| Scanner / geometry optimization | Supported | Supported |
| PBC, ROHF, fractional occupations, complex orbitals | Outside current scope | Outside current scope |
| NLC/VV10, Laplacian-dependent meta-GGA, Hessian | Not supported | Not supported |

Complete energy derivatives require grid response and, for DF, auxiliary-basis
response. CPU conventional calculations also expose a frozen-grid diagnostic;
DF and GPU gradients require full response. KS-source Fock derivatives use
`direct`; GPU KS-source Z response uses the AO backend. Support for a functional
family does not establish accuracy for every functional, molecule or SCF state.

Geometry optimization uses PySCF's optimizer interface. Install `geometric`
separately if needed. Scanners reuse the previous source density as the initial
guess and reconverge source SCF at every geometry; use `scanner(mol, dm0=None)` for a
fresh guess. Scanner views share source/DF state with their original object.


## Examples

### HF-DFT energy and gradient

```python
from pyscf import gto
from yescape_grad.cpu.energy import RHFDFT

mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
            basis='def2-svp')
mf = RHFDFT(mol, xc='PBE').density_fit()
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()  # Ha/Bohr
force = -gradient
print('HF energy [Ha]:', mf.e_hf)
print('HF-DFT energy [Ha]:', energy)
print('Force [Ha/Bohr]:', force)
```

`kernel()` converges HF and then evaluates the target functional without
target DFT SCF. Energy is in Hartree; forces are minus the energy gradient.
YEScape-Grad prints total energy and forces at every verbose level. Level 3 adds
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
from yescape_grad.cpu.energy import UHFDFT

mol = gto.M(atom='O 0 0 0; H 0 0 .97', basis='def2-svp', spin=1)
mf = UHFDFT(mol, xc='PBE').density_fit()
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()
force = -gradient
```

### GPU calculations

Use the GPU classes with the same calculation interface. This example uses
closed-shell water; use `UHFDFT` for an open-shell molecule.

```python
from pyscf import gto
from yescape_grad.gpu.energy import RHFDFT

mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
            basis='def2-svp')
mf = RHFDFT(mol, xc='PBE').density_fit()
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()
force = -gradient
```

Install the GPU dependencies described above before using this example.

### Geometry optimization

With the optional `geometric` package installed:

```python
from pyscf.geomopt.geometric_solver import optimize

scanner = mf.nuc_grad_method().as_scanner()
optimized_mol = optimize(scanner)
# Independent fresh-guess evaluation, if needed:
energy, gradient = scanner(optimized_mol, dm0=None)
```

The same interface works for CPU/GPU and conventional/DF calculations.

### D2C dispersion

After installing DFT-D4, enable D2C for a supported target functional:

```python
from yescape_grad.cpu.energy import RHFDFT

# Use the water molecule defined above.
mf = RHFDFT(mol, xc='PBE').density_fit().set_dispersion('d2c')
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

## Attribution and distribution

Developed by Sehun Kim <sehun1618@yonsei.ac.kr>,
Theoretical and Computational Chemistry Laboratory (TCCL), Yonsei University.

Original source comments, scientific citations and upstream attribution are
preserved. YEScape-Grad is distributed under the Apache License 2.0; see
[LICENSE](LICENSE). PySCF and GPU4PySCF retain their own licenses.
