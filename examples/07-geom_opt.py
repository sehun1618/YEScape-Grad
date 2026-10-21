'''Molecular D2C-r2SCAN geometry optimization with optional geomeTRIC.'''

from pyscf import gto
from pyscf.geomopt.geometric_solver import optimize
from dcforge.cpu.energy import RHFDFT

mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
            basis='def2-svp', verbose=3)
mf = RHFDFT(mol, xc='r2SCAN').density_fit()
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-6
mf.grids.level = 4
scanner = mf.nuc_grad_method().as_scanner()

# Reuses the previous HF density as a guess, but reconverges HF every step.
# For an independent single point: energy, gradient = scanner(mol, dm0=None).
optimized = optimize(scanner, maxsteps=50)
optimized.tofile('optimized.xyz', format='xyz')
