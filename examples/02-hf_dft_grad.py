'''Evaluate a conventional molecular PBE@HF analytic nuclear gradient.'''

from pathlib import Path

import numpy as np
from pyscf import gto

from dcforge.cpu.energy import RHFDFT

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'structures' / 'water1.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    verbose=0,
)

mf = RHFDFT(mol, xc='PBE').set(
    conv_tol=1e-12,
    conv_tol_grad=1e-10,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()

# The target F^KS_vo is contracted with the source-RHF adjoint Z-vector.
# grid_response=True includes the moving numerical-grid contribution.
mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    cphf_conv_tol=1e-12,
    cphf_max_cycle=200,
)
gradient = mf_grad.kernel()

np.set_printoptions(precision=12, suppress=True)
print(f'PBE@HF energy = {energy:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(gradient)
print('Translation sum [Ha/Bohr]')
print(gradient.sum(axis=0))
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print(f'Term recomposition = {np.max(np.abs(gradient - mf_grad.de_skel - mf_grad.de_nvar)):.6e}')
