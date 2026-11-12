"""Restore a stationary PBE source for r2SCAN energy and gradient (CPU)."""

from pathlib import Path

import numpy as np
from pyscf import gto
from pyscf.scf import chkfile

from yescape_grad.cpu.energy import RNSC

# Files in the current working directory are overwritten on rerun.

#
# 1. Conventional integrals
#
mol = gto.M(
    atom='''
        O  0.000   0.000   0.000
        H  0.000  -0.757   0.587
        H  0.000   0.757   0.587
    ''',
    basis='def2-svp',
    verbose=3,
)
mf = RNSC(mol, source_xc='PBE', xc='r2SCAN')
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-6
mf.max_cycle = 100
mf.mf_source.grids.level = 4
mf.grids.level = 4
path = str(Path('pbe_conv.chk').resolve())
mf.chkfile = path
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()
assert mf.mf_source.converged

# Reload the molecule and source orbitals from the checkpoint.
mol2, state = chkfile.load_scf(path)
restored = RNSC(mol2, source_xc='PBE', xc='r2SCAN')
restored.mf_source.grids.level = 4
restored.grids.level = 4

# Recreate the original functional/grid/DF configuration: orbitals alone
# do not specify the source Hamiltonian. DF integrals are rebuilt.
mf_source = restored.mf_source
mf_source.e_tot = state['e_tot']
mf_source.mo_coeff = state['mo_coeff']
mf_source.mo_energy = state['mo_energy']
mf_source.mo_occ = state['mo_occ']

# Justified only because this example checked convergence before saving.
restored.mf_source.converged = True

energy2 = restored.kernel(run_source=False)
gradient2 = restored.nuc_grad_method().kernel()

np.testing.assert_allclose(energy2, energy, atol=1e-9, rtol=0)
np.testing.assert_allclose(gradient2, gradient, atol=1e-8, rtol=0)
print('Restored checkpoint:', path)
print('Max gradient difference [Ha/Bohr]:',
      np.max(np.abs(gradient2 - gradient)))


#
# 2. Density fitting
#
mol = gto.M(
    atom='''
        O  0.000   0.000   0.000
        H  0.000  -0.757   0.587
        H  0.000   0.757   0.587
    ''',
    basis='def2-svp',
    verbose=3,
)
mf = RNSC(mol, source_xc='PBE', xc='r2SCAN')
mf = mf.density_fit(auxbasis='def2-universal-jkfit')
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-6
mf.max_cycle = 100
mf.mf_source.grids.level = 4
mf.grids.level = 4
path = str(Path('pbe_df.chk').resolve())
mf.chkfile = path
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()
assert mf.mf_source.converged

# Reload the molecule and source orbitals from the checkpoint.
mol2, state = chkfile.load_scf(path)
restored = RNSC(mol2, source_xc='PBE', xc='r2SCAN')
restored = restored.density_fit(auxbasis='def2-universal-jkfit')
restored.mf_source.grids.level = 4
restored.grids.level = 4

# Recreate the original functional/grid/DF configuration: orbitals alone
# do not specify the source Hamiltonian. DF integrals are rebuilt.
mf_source = restored.mf_source
mf_source.e_tot = state['e_tot']
mf_source.mo_coeff = state['mo_coeff']
mf_source.mo_energy = state['mo_energy']
mf_source.mo_occ = state['mo_occ']

# Justified only because this example checked convergence before saving.
restored.mf_source.converged = True

energy2 = restored.kernel(run_source=False)
gradient2 = restored.nuc_grad_method().kernel()

np.testing.assert_allclose(energy2, energy, atol=1e-9, rtol=0)
np.testing.assert_allclose(gradient2, gradient, atol=1e-8, rtol=0)
print('Restored checkpoint:', path)
print('Max gradient difference [Ha/Bohr]:',
      np.max(np.abs(gradient2 - gradient)))
