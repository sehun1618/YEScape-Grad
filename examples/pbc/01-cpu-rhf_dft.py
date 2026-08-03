'''CPU periodic RHF-DFT energies and analytic nuclear gradients.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import RHFDFT

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
# Case 1: FFTDF PBE@RHF
#
# RHFDFT uses the Cell's ordinary FFTDF provider when no DF transformation is
# requested.  Replace 'PBE' by 'R2SCAN' to run the supported meta-GGA target.
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
).run()

print('\n[CPU FFTDF]')
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
# Case 2: GDF PBE@RHF
#
# density_fit() returns a copied object.  The explicit auxiliary basis makes
# the small GTH-SZV example reproducible; auxbasis=None delegates selection to
# PySCF's periodic matching/even-tempered policy.
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
    cderi_response_storage='contracted',
).run()

print('\n[CPU GDF]')
print(f'Provider = {type(mf.with_df).__name__}')
print(f'Auxiliary basis = {mf.with_df.auxbasis}')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target PBE@RHF energy = {mf.e_dft:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print(f'CDERI response storage = {mf_grad.cderi_response_storage_used}')
print(f'CDERI response block size = {mf_grad.cderi_response_blksize_used}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print('Term recomposition max abs = '
      f'{np.max(np.abs(mf_grad.de - mf_grad.de_terms)):.3e} Ha/Bohr')


#
# Case 3: RSDF PBE@RHF
#
# rs_density_fit() changes the Gaussian provider but not the fixed-density
# energy definition.  Source and metric range separations are resolved
# independently by PySCF and can be inspected after build/kernel.
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
    cderi_response_storage='contracted',
).run()

print('\n[CPU RSDF]')
print(f'Provider = {type(mf.with_df).__name__}')
print(f'RSDF settings = {mf.with_df.response_settings()}')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target PBE@RHF energy = {mf.e_dft:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print(f'CDERI response storage = {mf_grad.cderi_response_storage_used}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
print('Term recomposition max abs = '
      f'{np.max(np.abs(mf_grad.de - mf_grad.de_terms)):.3e} Ha/Bohr')
