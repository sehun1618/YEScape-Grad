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


## Examples

### NSC energy and gradient

Evaluate r²SCAN on stationary PBE orbitals (`PBE → r²SCAN`):

```python
from pyscf import gto
from yescape_grad.cpu.energy import RNSC

mol = gto.M(
    atom='''
        O  0.000   0.000   0.000
        H  0.000  -0.757   0.587
        H  0.000   0.757   0.587
    ''',
    basis='def2-svp',
)
mf = RNSC(mol, source_xc='PBE', xc='r2SCAN').density_fit()
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-6
mf.mf_source.grids.level = 4
mf.grids.level = 4

energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()
force = -gradient

print('Source energy [Ha]:', mf.e_source)
print('Target energy [Ha]:', energy)
print('Force [Ha/Bohr]:', force)
```

`source_xc` determines the stationary orbitals and their response kernel.
`xc` selects the target functional. `kernel()` converges the source SCF and
then evaluates the target energy without target SCF. Source and target grids
are configured independently. The gradient is `dE/dR`; force is its negative.

### Source and target choices

For the same molecule, HF-DFT is selected explicitly with `source_xc='HF'`:

```python
# PBE orbitals -> r2SCAN
pbe_source = RNSC(mol, source_xc='PBE', xc='r2SCAN')

# HF orbitals -> r2SCAN
hf_source = RNSC(mol, source_xc='HF', xc='r2SCAN')
```

See [source/target examples](examples/03-source-target.py) for meta-GGA,
global-hybrid and range-separated-hybrid sources, and the identical-functional
limit. These combinations illustrate the interface, not an accuracy ranking.

### Conventional and density-fitted calculations

```python
# Conventional integrals
conventional = RNSC(mol, source_xc='PBE', xc='r2SCAN')

# Automatically selected JK-fit auxiliary basis
fitted = RNSC(mol, source_xc='PBE', xc='r2SCAN').density_fit()

# Explicit auxiliary basis
fitted = RNSC(mol, source_xc='PBE', xc='r2SCAN').density_fit(
    auxbasis='def2-universal-jkfit',
)
```

Configure and run either object as in the first example. Full grid response
and DF auxiliary-basis response are enabled by default.
See the [restricted example](examples/01-restricted.py) for both complete runs.

### Unrestricted calculations

Use `UNSC` with `spin = Nalpha - Nbeta`. This NH₂ doublet example evaluates
PBE on PBE0 orbitals:

```python
from pyscf import gto
from yescape_grad.cpu.energy import UNSC

open_shell_mol = gto.M(
    atom='''
        N  0.000   0.000   0.000
        H  0.000  -0.780   0.600
        H  0.000   0.780   0.600
    ''',
    basis='def2-svp',
    spin=1,
)
umf = UNSC(open_shell_mol, source_xc='PBE0', xc='PBE').density_fit()
umf.conv_tol = 1e-8
umf.conv_tol_grad = 1e-6
umf.mf_source.grids.level = 4
umf.grids.level = 4

energy = umf.kernel()
gradient = umf.nuc_grad_method().kernel()
force = -gradient
```

See the [unrestricted example](examples/02-unrestricted.py) for conventional
and DF calculations.

### GPU calculations

Use the same interface with the GPU imports in place of the CPU imports:

```python
from yescape_grad.gpu.energy import RNSC, UNSC
```

Install the GPU dependencies above first. Complete GPU examples use the same
systems and settings as their CPU counterparts:
[restricted](examples/gpu/01-restricted.py),
[unrestricted](examples/gpu/02-unrestricted.py), and
[source/target choices](examples/gpu/03-source-target.py).

### Geometry optimization

For the PBE → r²SCAN workflow `mf` from the first example, install `geometric`
and pass its gradient scanner to PySCF's optimizer interface:

```python
from pyscf.geomopt.geometric_solver import optimize

scanner = mf.nuc_grad_method().as_scanner()
optimized_mol = optimize(scanner, maxsteps=50)
```

The scanner uses the previous source density as an initial guess and
reconverges source SCF at each geometry. See the complete
[CPU](examples/07-geom_opt.py) and [GPU](examples/gpu/07-geom_opt.py) examples.

### D2C dispersion

D2C is parameterized for HF sources and selected target functionals; it is
not a general correction for arbitrary KS sources. With DFT-D4 installed:

```python
from yescape_grad.cpu.energy import RNSC

# Use the water molecule from the first example.
d2c = RNSC(mol, source_xc='HF', xc='r2SCAN').density_fit()
d2c.set_dispersion('d2c')
d2c.conv_tol = 1e-8
d2c.conv_tol_grad = 1e-6
d2c.grids.level = 4

energy = d2c.kernel()
gradient = d2c.nuc_grad_method().kernel()
```

Both results include the dispersion correction. See the
[D2C example](examples/06-d2c.py) for its separate energy and gradient terms.

Additional examples cover [source checkpoints](examples/04-checkpoint.py)
and [source/target grids](examples/05-grid.py).

## Attribution and distribution

Developed by Sehun Kim <sehun1618@yonsei.ac.kr>,
Theoretical and Computational Chemistry Laboratory (TCCL), Yonsei University.

Original source comments, scientific citations and upstream attribution are
preserved. YEScape-Grad is distributed under the Apache License 2.0; see
[LICENSE](LICENSE). PySCF and GPU4PySCF retain their own licenses.
