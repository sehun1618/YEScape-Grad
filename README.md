# DC-Forge

Molecular non-self-consistent energies and analytic nuclear gradients built on PySCF
and, optionally, GPU4PySCF.

This development branch extends HF sources to restricted/unrestricted KS
sources: LDA, GGA, tau-dependent meta-GGA, global hybrids and RSH. Target SCF
is never performed. A density alone is insufficient: its stationary source
SCF, orbital energies and response kernel define the gradient.

```python
from pyscf import gto
from dcforge.cpu.energy import RNSC  # GPU: dcforge.gpu.energy

mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587', basis='def2-svp')
mf = RNSC(mol, xc='r2SCAN', source_xc='PBE').density_fit()
mf.mf_source.grids.level = 4  # source SCF and its response
mf.grids.level = 4            # fixed-density target evaluation
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()
```

Use `UNSC` for unrestricted sources. Energies are in Ha and gradients in
Ha/Bohr; force is minus gradient. Source and target have independent grids.
Explicit `kernel(run_source=False)` reuses supplied orbitals: the caller
is responsible for stationarity, and an unconverged flag produces a warning.

## CPU response backend

For a **density-fitted HF source**, both restricted and unrestricted CPU gradients
now default to `response_backend='mo_direct'`. Conventional and DFA-source gradients
use `'ao'`; the HF-only MO provider must not be used for a DFA source's XC kernel.
The target functional does not select the response backend.

```python
mf = RNSC(mol, xc='PBE').density_fit()  # UNSC works the same way
mf.kernel()
grad = mf.nuc_grad_method()
g = grad.kernel()                       # DF-HF: MO-direct by default
print(grad.response_backend_used)       # actual route, including memory fallback
# Reference route: mf.nuc_grad_method().set(response_backend='ao').kernel()
```

CPU MO-direct transforms and caches DF OO/VO/VV factors once per gradient call,
including spin-coupled Coulomb and spin-local exchange for UHF. It has an additional
`O(naux*nmo^2)` cache/setup cost; it is not always faster or smaller than AO.
The cache is local to the call and rebuilt after geometry/orbital changes.
If estimated cache plus workspace exceeds 80% of the remaining `max_memory` budget,
the driver logs a warning and uses AO. `response_mo_cache_bytes` and
`response_mo_setup_s` expose cache size and setup wall time. No persistent MO cache
is retained between calls. The final physical Z residual is checked with the AO
operator even when the solve uses MO-direct.

This response choice is independent of `fock_deriv_method='direct'`, which controls
source-Fock nuclear derivative contractions. GPU paths are unchanged.

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

Install **GPU4PySCF 1.8.1** separately for your CUDA environment, then install
DC-Forge. Follow the [GPU4PySCF installation guide](https://github.com/pyscf/gpu4pyscf#installation).

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
python -m pip install 'dftd4'
```

**cuTENSOR:** optional GPU acceleration. Install a compatible version separately
using the upstream guide, and verify that GPU4PySCF loads it.

Neither library is installed by DC-Forge. Both retain their own licenses.


## Features

- Molecular HF-DFT and DFA-source non-SCF energies and analytic gradients.
- Restricted and unrestricted HF or Kohn–Sham references.
- Conventional and density-fitted calculations.
- CPU and GPU implementations based on PySCF and GPU4PySCF.
- LDA, GGA, tau-meta-GGA, hybrid and RSH source and target functionals.
- Optional D2C dispersion corrections for supported functionals.
- Molecular energy/gradient scanners for geometry optimization.

Nonlocal correlation functionals, including VV10, are not supported.

Geometry optimization uses PySCF's optimizer interface. Install `geometric`
separately if needed. Scanners reuse the previous source density as the initial
guess and reconverge source SCF at every geometry; use `scanner(mol, dm0=None)` for a
fresh guess. Scanner views share source/DF state with their original object.


## Examples

### HF-DFT energy and gradient

```python
from pyscf import gto
from dcforge.cpu.energy import RHFDFT

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
from dcforge.cpu.energy import RHFDFT

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

## Developer notes

`UNSC` is the unrestricted counterpart. Both are actual source-neutral
classes; `RHFDFT`/`UHFDFT` remain compatibility aliases, with HF as the default
source. Pass `mf_source=` to attach a native
conventional HF/KS object; use `.density_fit()` on the workflow afterward.
`kernel(run_source=False)` explicitly reuses supplied orbitals; the caller
is responsible for source stationarity. An unconverged flag emits a warning,
not a rejection. See [theory and code flow](THEORY.md). Legacy
`mf_hf`, `run_hf`, `e_hf`/`e_uhf` names are retained for old callers; use
`mf_source`, `e_source` and `dm_source` for source-neutral code.
Only the source-neutral attributes store state; old attribute names are
property views of the same state. Use `kernel_source()`, `run_source()` and
`make_source_rdm1()` for explicit source operations. The old HF method names
forward to these methods and do not run a separate calculation path.

General energy/gradient drivers live in `rnsc.py` and `unsc.py`;
common nonvariational and Z-vector routines use `r_nv.py`/`u_nv.py` and
`r_zvec.py`/`u_zvec.py`. Direct imports of the former module paths must be
updated; the public `RHFDFT`/`UHFDFT` class imports remain available.
The GPU DF `rhf_nv.py`/`uhf_nv.py` and `rhf_zvec.py`/`uhf_zvec.py`
retain their names because these optimized kernels are HF-specific.

Each CPU/GPU backend has a flat `source/` package:

```text
source/
  rhf.py   # restricted HF Fock nuclear derivative
  uhf.py   # unrestricted HF Fock nuclear derivative
  rks.py   # restricted KS Fock nuclear derivative
  uks.py   # unrestricted KS Fock nuclear derivative
```

RKS and UKS own their occupation factors, density pairs and Fock contractions;
UKS does not dispatch through RKS. Each backend's `source/xc.py` implements
the same XC derivative, including tau and moving-grid response, using explicit
NumPy/CPU or CuPy/GPU arrays. GPU use does not invoke CPU SCF. The old
`grad/source_fock.py` and `grad/source_xc.py` paths have been removed.
Common overlap/Pulay assembly and Z solvers stay in `grad/`. Source SCF and
its AO response operator still come from PySCF/GPU4PySCF; no duplicate SCF
engine is introduced. GPU DF HF-only MO-direct kernels remain in `df/grad/`.

Source grids must use native atom-centered quadrature without density pruning
(`small_rho_cutoff=0`, set automatically for newly constructed KS sources).
Source/target grid settings are independent. Explicit `kernel(run_source=False)`
reuses supplied source orbitals without comparing array hashes or past settings.
The caller is responsible for their suitability; an unconverged flag emits a
warning. Nonstationary sources need not yield gradients consistent with
reoptimized-source finite differences. Missing/malformed orbitals and incompatible
DF ownership remain errors. Normal SCF and scanner calls still require convergence.
No NLC/VV10, Laplacian
MGGA, fractional occupations, complex orbitals, ROHF, PBC or solvent wrappers.
Published D2C parameters are allowed only with an HF source.

DFA-source gradients use `fock_deriv_method='direct'`; native Hessian
`make_h1` does not contain the complete moving-grid derivative. GPU DFA
sources use `response_backend='ao'`, including source fxc. HF's MO-direct
acceleration remains unchanged. The source-XC derivative is a streamed,
atom-resolved reference contraction, not yet a large-molecule optimization.


## Attribution and distribution

Developed by Sehun Kim <sehun1618@yonsei.ac.kr>,
Theoretical and Computational Chemistry Laboratory (TCCL), Yonsei University.

Original source comments, scientific citations and upstream attribution are
preserved. Upstream license notices are in `third_party/`. These notices do
not themselves license DC-Forge's original contributions. A project-wide
license must be chosen by the rights holders before public distribution.
