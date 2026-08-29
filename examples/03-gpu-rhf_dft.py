'''GPU molecular RHF-DFT energy and analytic nuclear gradient.'''

from pathlib import Path

import numpy as np
from pyscf import gto

from dcforge.gpu.energy import RHFDFT

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'structures' / 'water1.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    verbose=0,
)
np.set_printoptions(precision=12, suppress=True)


#
# Case 1: GPU conventional PBE@RHF
#
mf = RHFDFT(mol, xc='PBE').set(
    conv_tol=1e-10,
    conv_tol_grad=1e-9,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()
mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    cphf_conv_tol=1e-9,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
).run()

print('\n[gpu conventional]')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target PBE@RHF energy = {energy:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print(f'Source-Fock derivative = {mf_grad.fock_deriv_method_used}')


#
# Case 2: GPU density-fitted PBE@RHF
#
mf = RHFDFT(mol, xc='PBE').density_fit().set(
    conv_tol=1e-10,
    conv_tol_grad=1e-9,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()
mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    auxbasis_response=True,
    cphf_conv_tol=1e-9,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
    fock_deriv_planner_limit_bytes=4 * 1024**3,
).run()

print('\n[gpu DF]')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target PBE@RHF energy = {energy:.15f} Ha')
print(f'Auxiliary basis = {mf.auxbasis}')
print(f'Source/target share with_df = {mf.with_df is mf.mf_hf.with_df}')
print(f'Response backend = {mf_grad.response_backend_used}')
print(f'DF derivative planner = {mf_grad.fock_deriv_planner["mode"]}')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print(f'Source-Fock derivative = {mf_grad.fock_deriv_method_used}')
