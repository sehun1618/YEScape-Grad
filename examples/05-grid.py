'''Choose molecular XC grids by level or explicit atomic grid sizes.'''

from pathlib import Path

import numpy as np
from pyscf import gto, scf

from dcforge.cpu.energy import RHFDFT

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'tests' / 'structures' / 'water1.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    verbose=0,
)
mf_hf = scf.RHF(mol).set(
    conv_tol=1e-8,
    conv_tol_grad=1e-6,
    max_cycle=100,
).run()

# Reuse one HF density; set grid options before kernel().

# Case 1: PySCF level presets (element-dependent radial/angular grid sizes)
print('\nr2SCAN grid-level convergence on the same RHF density')
for level in (2, 4, 6):
    mf = RHFDFT(mol, xc='r2SCAN', mf_hf=mf_hf)
    mf.grids.level = level
    energy = mf.kernel(run_hf=False)
    print(f'level {level}: {mf.grids.coords.shape[0]} points, '
          f'E = {energy:.15f} Ha')

# Case 2: one explicit (radial points, angular points) pair for all atoms
mf = RHFDFT(mol, xc='r2SCAN', mf_hf=mf_hf)
mf.grids.atom_grid = (99, 590)
energy = mf.kernel(run_hf=False)
print(f'\nUniform atomic sizes: {mf.grids.coords.shape[0]} points, '
      f'E = {energy:.15f} Ha')

# Case 3: different atomic grid sizes for each element
mf = RHFDFT(mol, xc='r2SCAN', mf_hf=mf_hf)
mf.grids.atom_grid = {'H': (75, 302), 'O': (99, 590)}
energy = mf.kernel(run_hf=False)
print(f'\nElement-specific sizes: {mf.grids.coords.shape[0]} points, '
      f'E = {energy:.15f} Ha')

# atom_grid overrides level for specified elements; others use level presets.

# Case 4: choose the CPU grid-response backend on the same grids
# Same grids and response equations; only the derivative backend changes.
grad_default = mf.nuc_grad_method().set(
    grid_response=True,
    grid_response_backend='pyscf_default',
    cphf_max_cycle=200,
)
reference = grad_default.kernel()

# becke_omp is the default and requires the OpenMP extension.
grad_omp = mf.nuc_grad_method().set(
    grid_response=True,
    grid_response_backend='becke_omp',
    cphf_max_cycle=200,
)
optimized = grad_omp.kernel()
print('\nMax grid-backend gradient difference [Ha/Bohr]:',
      np.max(np.abs(optimized - reference)))
np.testing.assert_allclose(optimized, reference, atol=1e-9, rtol=0)
