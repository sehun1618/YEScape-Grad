'''GPU periodic RHF-DFT energies and analytic nuclear gradients.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import RHFDFT

cell = gto.Cell()
cell.atom = '''
    O  0.10 -0.20  0.30
    H  1.55  0.15  0.05
    H -0.35  1.25 -0.25
'''
cell.a = np.eye(3) * 7.
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-hf'
cell.ke_cutoff = 100
cell.precision = 1e-8
cell.verbose = 0
cell.build()

np.set_printoptions(precision=12, suppress=True)


#
# Case 1: GPU FFTDF PBE@RHF
#
mf = RHFDFT(cell, xc='PBE').set(
    conv_tol=1e-10,
    conv_tol_grad=1e-8,
    max_cycle=80,
).run()
mf_grad = mf.nuc_grad_method().set(
    cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8,
    cphf_max_cycle=200,
    cphf_refinement_solver='minres',
).run()

print('\n[GPU FFTDF]')
print(f'Provider = {type(mf.with_df).__name__}')
print(f'Cell mesh = {cell.mesh.tolist()}')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target PBE@RHF energy = {mf.e_dft:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print('Force -dE/dR [Ha/Bohr]')
print(-mf_grad.de)
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print('Term recomposition max abs = '
      f'{np.max(np.abs(mf_grad.de - mf_grad.de_terms)):.3e} Ha/Bohr')


#
# Case 2: GPU GDF PBE@RHF
#
mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv').set(
    conv_tol=1e-10,
    conv_tol_grad=1e-8,
    max_cycle=80,
).run()
mf_grad = mf.nuc_grad_method().set(
    cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8,
    cphf_max_cycle=200,
    cphf_refinement_solver='minres',
    gdf_response_mode='auto',
).run()

print('\n[GPU GDF]')
print(f'Provider = {type(mf.with_df).__name__}')
print(f'Auxiliary basis = {mf.with_df.auxbasis}')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target PBE@RHF energy = {mf.e_dft:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print(f'GDF response partition = {mf_grad.gdf_term_partition}')
print(f'Hcore backend = {mf_grad.hcore_backend}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print('Term recomposition max abs = '
      f'{np.max(np.abs(mf_grad.de - mf_grad.de_terms)):.3e} Ha/Bohr')


#
# Case 3: GPU RSDF PBE@RHF
#
# Replace 'PBE' by 'R2SCAN' in any case above to use the supported meta-GGA
# target on the same source-HF/provider contract.
#
mf = RHFDFT(cell, xc='PBE').rs_density_fit(auxbasis='gth-szv').set(
    conv_tol=1e-10,
    conv_tol_grad=1e-8,
    max_cycle=80,
).run()
mf_grad = mf.nuc_grad_method().set(
    cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8,
    cphf_max_cycle=200,
    cphf_refinement_solver='minres',
).run()

print('\n[GPU RSDF]')
print(f'Provider = {type(mf.with_df).__name__}')
print(f'RSDF settings = {mf.with_df.response_settings()}')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target PBE@RHF energy = {mf.e_dft:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print(f'Hcore backend = {mf_grad.hcore_backend}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print('Term recomposition max abs = '
      f'{np.max(np.abs(mf_grad.de - mf_grad.de_terms)):.3e} Ha/Bohr')
