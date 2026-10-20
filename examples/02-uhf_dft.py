'''Molecular UHF-DFT energy and analytic nuclear gradient.'''

from pathlib import Path

import numpy as np
from pyscf import gto

from dcforge.cpu.energy import UHFDFT

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'structures' / 'oh.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    spin=1,
    verbose=0,
)
np.set_printoptions(precision=12, suppress=True)


# Case 1: conventional HF-PBE energy and gradient
mf = UHFDFT(mol, xc='PBE').set(
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
print(f'Source UHF energy = {mf.e_uhf:.15f} Ha')
print(f'Target HF-PBE energy = {energy:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print('Force -dE/dR [Ha/Bohr]')
print(-mf_grad.de)
print(f'UCPHF solver = {mf_grad.z_solver}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print(f'Overlap-response RHS count = {mf_grad.overlap_response_nrhs}')


# Case 2: density-fitted HF-PBE energy and complete DF gradient
mf = UHFDFT(mol, xc='PBE').density_fit().set(
    conv_tol=1e-8,
    conv_tol_grad=1e-6,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()
mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    auxbasis_response=True,
    cphf_conv_tol=1e-7,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
).run()

print('\n[DF]')
print(f'Source UHF energy = {mf.e_uhf:.15f} Ha')
print(f'Target HF-PBE energy = {energy:.15f} Ha')
print(f'Auxiliary basis = {mf.auxbasis}')
print(f'Source/target share with_df = {mf.with_df is mf.mf_hf.with_df}')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print('Force -dE/dR [Ha/Bohr]')
print(-mf_grad.de)
print(f'UCPHF solver = {mf_grad.z_solver}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print(f'Overlap-response RHS count = {mf_grad.overlap_response_nrhs}')
