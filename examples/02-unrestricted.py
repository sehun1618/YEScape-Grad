"""PBE0 -> PBE: conventional and DF NSC energies and gradients."""

import numpy as np
from pyscf import gto

from yescape_grad.cpu.energy import UNSC

mol = gto.M(
    atom='''
        N  0.000   0.000   0.000
        H  0.000  -0.780   0.600
        H  0.000   0.780   0.600
    ''',
    basis='def2-svp',
    spin=1,
    verbose=3,
)

#
# 1. Conventional integrals
#
mf = UNSC(mol, source_xc='PBE0', xc='PBE')
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-6
mf.max_cycle = 100
mf.mf_source.grids.level = 4
mf.grids.level = 4

# Converge the source SCF and evaluate the target energy on its density.
energy = mf.kernel()

# Include source orbital response and moving-grid response.
mf_grad = mf.nuc_grad_method()
gradient = mf_grad.kernel()

assert mf.converged
assert np.isfinite(energy)
assert np.isfinite(gradient).all()

print('Conventional', 'energy [Ha]:', energy)
print('Gradient [Ha/Bohr]:', gradient)
print('Force [Ha/Bohr]:', -gradient)


#
# 2. Density fitting
#
mf = UNSC(mol, source_xc='PBE0', xc='PBE')
mf = mf.density_fit()  # Default exchange-capable auxiliary basis.
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-6
mf.max_cycle = 100
mf.mf_source.grids.level = 4
mf.grids.level = 4

# Converge the source SCF and evaluate the target energy on its density.
energy = mf.kernel()

# Include source orbital response and moving-grid response.
# Include auxiliary-basis response for the density-fitted calculation.
mf_grad = mf.nuc_grad_method()
gradient = mf_grad.kernel()

assert mf.converged
assert np.isfinite(energy)
assert np.isfinite(gradient).all()

print('DF', 'energy [Ha]:', energy)
print('Gradient [Ha/Bohr]:', gradient)
print('Force [Ha/Bohr]:', -gradient)
