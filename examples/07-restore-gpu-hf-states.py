'''Restore standalone RHF/UHF states for GPU HF-DFT energy and gradient.

Run ``06-save-gpu-hf-states.py`` first.  This file reconstructs each final DF
representation from checkpoint metadata, validates the saved canonical state
with the active GPU Fock, and deliberately calls ``kernel(run_hf=False)``.
'''

from pathlib import Path

import numpy as np

from dcforge.gpu.energy import RHFDFT, UHFDFT

OUTPUT = Path.cwd() / 'hf_state_examples'
np.set_printoptions(precision=12, suppress=True)


#
# Case 1: PBE@DF-RHF from the saved water source
#
rhf_path = OUTPUT / 'water_df_rhf.chk'
mf_rhf = RHFDFT.from_hf_state(rhf_path, xc='PBE')
mf_rhf.grids.level = 4
energy_rhf = mf_rhf.kernel(run_hf=False)
grad_rhf = mf_rhf.nuc_grad_method().set(
    grid_response=True,
    auxbasis_response=True,
    cphf_conv_tol=1e-10,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
    response_backend='mo_direct',
    fock_deriv_planner_limit_bytes=4 * 1024**3,
).run()

receipt_rhf = mf_rhf.source_validation
print('[restored RHF]')
print('HF kernel rerun = False')
print(f'Target PBE@RHF energy = {energy_rhf:.15f} Ha')
print(f'Source validation max|F_vo| = {receipt_rhf.fock_vo_max_abs:.6e}')
print('Source generalized-eigen residual = '
      f'{receipt_rhf.generalized_eigen_residual_relative:.6e}')
print(f'Z residual max abs = {grad_rhf.z_residual_max_abs:.6e}')
print('Gradient dE/dR [Ha/Bohr]')
print(grad_rhf.de)


#
# Case 2: PBE@DF-UHF from the saved triplet O2 source
#
uhf_path = OUTPUT / 'o2_df_uhf.chk'
mf_uhf = UHFDFT.from_hf_state(uhf_path, xc='PBE')
mf_uhf.grids.level = 4
energy_uhf = mf_uhf.kernel(run_hf=False)
grad_uhf = mf_uhf.nuc_grad_method().set(
    grid_response=True,
    auxbasis_response=True,
    cphf_conv_tol=1e-10,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
    response_backend='mo_direct',
    fock_deriv_planner_limit_bytes=4 * 1024**3,
).run()

receipt_uhf = mf_uhf.source_validation
print('\n[restored UHF]')
print('HF kernel rerun = False')
print(f'Target PBE@UHF energy = {energy_uhf:.15f} Ha')
print(f'Source validation max|F_vo| = {receipt_uhf.fock_vo_max_abs:.6e}')
print('Source generalized-eigen residual = '
      f'{receipt_uhf.generalized_eigen_residual_relative:.6e}')
print(f'Z residual max abs = {grad_uhf.z_residual_max_abs:.6e}')
print('Gradient dE/dR [Ha/Bohr]')
print(grad_uhf.de)
