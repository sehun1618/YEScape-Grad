"""PBE -> r2SCAN geometry optimization; requires geomeTRIC."""

from pyscf import gto
from pyscf.geomopt.geometric_solver import optimize

from yescape_grad.gpu.energy import RNSC

mol = gto.M(
    atom='''
        O  0.000   0.000   0.000
        H  0.000  -0.757   0.587
        H  0.000   0.757   0.587
    ''',
    basis='def2-svp',
    verbose=3,
)
mf = RNSC(mol, source_xc='PBE', xc='r2SCAN').density_fit()
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-6
mf.max_cycle = 100
mf.mf_source.grids.level = 4
mf.grids.level = 4

# The gradient scanner returns (energy, gradient) for each new geometry.
scanner = mf.nuc_grad_method().as_scanner()
# Source density is an initial guess only: reconverge PBE at every geometry.
# Target r2SCAN is evaluated on that density without target SCF.
optimized = optimize(scanner, maxsteps=50)

# Overwrites optimized.xyz in the current working directory.
optimized.tofile('optimized.xyz', format='xyz')
