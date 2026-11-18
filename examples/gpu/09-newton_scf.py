"""PBE0 -> PBE with source Newton SCF and geometry optimization."""

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
# Remove density_fit() for conventional integrals.
mf = RNSC(mol, source_xc='PBE0', xc='PBE').density_fit()
mf.source_solver = 'newton'
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-6
mf.max_cycle = 100
mf.mf_source.grids.level = 4
mf.grids.level = 4

# Newton optimizes the source orbitals, not the non-SCF target energy.
# Use UNSC with the appropriate mol.spin/charge for an unrestricted source.
scanner = mf.nuc_grad_method().set(grid_response=True).as_scanner()
energy, gradient = scanner(mol)
print('Target energy (Ha):', energy)
print('Gradient (Ha/Bohr):')
print(gradient)

# At each new geometry, reuse the previous source density as the initial
# guess and reconverge it with Newton. Requires geomeTRIC.
optimized = optimize(scanner, maxsteps=50)

# Overwrites optimized_newton.xyz in the current working directory.
optimized.tofile('optimized_newton.xyz', format='xyz')
