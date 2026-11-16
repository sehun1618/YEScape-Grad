"""HF -> PBE with a Sn effective core potential and density fitting."""

import numpy as np
from pyscf import gto

from yescape_grad.cpu.energy import RNSC

# SnH4 at an illustrative tetrahedral geometry (coordinates in Angstrom).
# The orbital basis and ECP are separate molecular inputs.
mol = gto.M(
    atom='''
        Sn  0.000   0.000   0.000
        H   0.980   0.980   0.980
        H   0.980  -0.980  -0.980
        H  -0.980   0.980  -0.980
        H  -0.980  -0.980   0.980
    ''',
    basis='def2-svp',
    ecp={'Sn': 'def2-svp'},
    charge=0,
    spin=0,
    verbose=3,
)

# The Sn ECP replaces 28 core electrons; 26 electrons remain in SnH4.
assert mol.atom_nelec_core(0) == 28
assert mol.nelectron == 26

mf = RNSC(mol, source_xc='HF', xc='PBE').density_fit()
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-6
mf.max_cycle = 100
mf.grids.level = 4

energy = mf.kernel()
# ECP nuclear derivatives are included through the upstream gradient routines,
# in both the target explicit term and the source-response correction.
gradient = mf.nuc_grad_method().kernel()

assert mf.converged
assert np.isfinite(energy)
assert np.isfinite(gradient).all()

print('DF with Sn ECP', 'energy [Ha]:', energy)
print('Gradient [Ha/Bohr]:', gradient)
print('Force [Ha/Bohr]:', -gradient)
