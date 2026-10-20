'''Molecular RHF-DFT energy and analytic nuclear gradient.'''

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
np.set_printoptions(precision=12, suppress=True)


# Case 1: conventional HF-PBE energy and gradient
# The target PBE energy uses the converged HF density, without PBE SCF.
mf = RHFDFT(mol, xc='PBE').set(
    conv_tol=1e-8,
    conv_tol_grad=1e-6,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()

mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    cphf_conv_tol=1e-7,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
).run()

print('\n[conventional]')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target HF-PBE energy = {energy:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print('Force -dE/dR [Ha/Bohr]')
print(-mf_grad.de)
print(f'Z solver = {mf_grad.z_solver}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print('Term recomposition max abs = '
      f'{np.max(np.abs(mf_grad.de - mf_grad.de_skel - mf_grad.de_nvar)):.3e}')


# Case 2: density-fitted HF-PBE energy and complete DF gradient
# Omitting auxbasis lets PySCF select the JK-fit basis.
mf = RHFDFT(mol, xc='PBE').density_fit().set(
    conv_tol=1e-8,
    conv_tol_grad=1e-6,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()

# Include moving-grid and auxiliary-basis response for the complete gradient.
mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    auxbasis_response=True,
    cphf_conv_tol=1e-7,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
).run()

print('\n[DF]')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target HF-PBE energy = {energy:.15f} Ha')
print(f'Auxiliary basis = {mf.auxbasis}')
print(f'Source/target share with_df = {mf.with_df is mf.mf_hf.with_df}')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print('Force -dE/dR [Ha/Bohr]')
print(-mf_grad.de)
print(f'Z solver = {mf_grad.z_solver}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print('Term recomposition max abs = '
      f'{np.max(np.abs(mf_grad.de - mf_grad.de_skel - mf_grad.de_nvar)):.3e}')
