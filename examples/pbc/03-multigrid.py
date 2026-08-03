'''GDF/RSDF Hartree with target-XC-only periodic multigrid.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import RHFDFT as CPURHFDFT
from dcforge.gpu.pbc.energy import RHFDFT as GPURHFDFT

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
# The Cell-owned cutoff determines the master mesh used by target XC.
cell.ke_cutoff = 300
cell.precision = 1e-8
cell.verbose = 0
cell.build()

np.set_printoptions(precision=12, suppress=True)


#
# Case 1: CPU GDF Hartree + PBE multigrid XC
#
# To use RSDF, replace density_fit by rs_density_fit.  The Gaussian provider
# still owns source RHF, target Hartree, CPHF, Hcore, and Gaussian response.
#
mf = CPURHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
# mf = CPURHFDFT(cell, xc='PBE').rs_density_fit(auxbasis='gth-szv')
mf = mf.multigrid_numint(
    ntasks=3,
    ke_ratio=3.,
    rel_cutoff=20.,
).set(
    conv_tol=1e-10,
    conv_tol_grad=1e-8,
    max_cycle=80,
).run()
mf_grad = mf.nuc_grad_method().set(
    cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8,
    cphf_max_cycle=200,
    multigrid_xc_deriv_method='single_pass',
).run()

print('\n[CPU GDF Hartree + PBE multigrid XC]')
print(f'HF/Hartree provider = {type(mf.with_df).__name__}')
print(f'Target XC integrator = {type(mf._numint).__name__}')
print(f'Cell mesh = {cell.mesh.tolist()}')
print('CPU MG hierarchy = '
      f'ntasks={mf._numint.ntasks}, '
      f'ke_ratio={mf._numint.ke_ratio:g}, '
      f'rel_cutoff={mf._numint.rel_cutoff:g}')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target PBE@RHF energy = {mf.e_dft:.15f} Ha')
print(f'GDF Hartree contribution = {mf.e_coul:.15f} Ha')
print(f'Multigrid XC contribution = {mf.e_xc:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print(f'XC derivative backend = {mf_grad.xc_deriv_backend}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')


#
# Case 2: GPU GDF Hartree + R2SCAN multigrid XC
#
# GPU4PySCF multigrid-v2 uses cell.mesh directly and does not expose the CPU
# ntasks/ke_ratio/rel_cutoff hierarchy.  RSDF is selected by the commented
# replacement line exactly as in the CPU case.
#
mf = GPURHFDFT(cell, xc='R2SCAN').density_fit(auxbasis='gth-szv')
# mf = GPURHFDFT(cell, xc='R2SCAN').rs_density_fit(auxbasis='gth-szv')
mf = mf.multigrid_numint().set(
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

print('\n[GPU GDF Hartree + R2SCAN multigrid XC]')
print(f'HF/Hartree provider = {type(mf.with_df).__name__}')
print(f'Target XC integrator = {type(mf._numint).__name__}')
print(f'Cell mesh = {cell.mesh.tolist()}')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target R2SCAN@RHF energy = {mf.e_dft:.15f} Ha')
print(f'GDF Hartree contribution = {mf.e_coul:.15f} Ha')
print(f'Multigrid XC contribution = {mf.e_xc:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print(f'XC derivative backend = {mf_grad.xc_deriv_backend}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
