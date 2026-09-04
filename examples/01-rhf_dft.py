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


#
# Case 1: conventional PBE@RHF energy and gradient
#
# RHFDFT.kernel() converges source RHF, constructs dm_hf, and evaluates the
# inherited RKS energy once.  It never runs a target PBE SCF calculation.
#
mf = RHFDFT(mol, xc='PBE').set(
    conv_tol=1e-12,
    conv_tol_grad=1e-10,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()

mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    cphf_conv_tol=1e-12,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
).run()

print('\n[conventional]')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target PBE@RHF energy = {energy:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print('Force -dE/dR [Ha/Bohr]')
print(-mf_grad.de)
print(f'Z solver = {mf_grad.z_solver}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print('Term recomposition max abs = '
      f'{np.max(np.abs(mf_grad.de - mf_grad.de_skel - mf_grad.de_nvar)):.3e}')


#
# Case 2: density-fitted PBE@RHF energy and complete DF gradient
#
# density_fit() returns a copied PySCF-style object.  Omitting auxbasis asks
# PySCF to select its matching exchange-capable JK-fit basis.
#
mf = RHFDFT(mol, xc='PBE').density_fit().set(
    conv_tol=1e-12,
    conv_tol_grad=1e-10,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()

# grid_response and auxbasis_response are completeness requirements, not
# performance switches.  The latter differentiates the moving DF basis.
mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    auxbasis_response=True,
    cphf_conv_tol=1e-12,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
).run()

print('\n[DF]')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target PBE@RHF energy = {energy:.15f} Ha')
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
