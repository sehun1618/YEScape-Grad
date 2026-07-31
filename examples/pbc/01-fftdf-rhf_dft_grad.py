'''Evaluate a Gamma-point FFTDF PBE@RHF energy and nuclear gradient.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import RHFDFT

cell = gto.Cell()
cell.atom = '''
O   0.10  -0.20   0.30
H   1.55   0.15   0.05
H  -0.35   1.25  -0.25
'''
cell.a = np.eye(3) * 7.0
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-hf'
cell.ke_cutoff = 15
cell.verbose = 0
cell.build()

# One FFTDF provider owns the finite reciprocal representation used by both
# source RHF and target PBE.  The target density is never self-consistently
# optimized: kernel() evaluates E_PBE[P_RHF] after converging source RHF.
mf = RHFDFT(cell, xc='PBE').run()

# The analytic derivative is dE/dR.  It contains fixed-density target terms,
# one source-HF adjoint Z-vector correction, and the Ewald ionic derivative.
mf_grad = mf.nuc_grad_method().run()
gradient = mf_grad.de
force = -gradient

np.set_printoptions(precision=12, suppress=True)
print(f'FFT mesh = {mf.with_df.mesh.tolist()}')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'PBE@RHF energy = {mf.e_dft:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(gradient)
print('Force -dE/dR [Ha/Bohr]')
print(force)
print(f'Z solver = {mf_grad.z_solver}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print(
    'Term recomposition = '
    f'{np.max(np.abs(gradient - mf_grad.de_terms)):.6e} Ha/Bohr'
)
