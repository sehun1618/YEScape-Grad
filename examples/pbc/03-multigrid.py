'''Choose provider-J + MG-XC or fused target MG-J+XC.'''

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
# Case 1: CPU FFTDF Hartree + PBE multigrid XC
#
# The active provider owns source RHF, target Hartree, CPHF, Hcore and their
# derivatives.  Only target XC moves to multigrid.  Use density_fit or
# rs_density_fit on the commented lines to select GDF or RSDF instead.
#
mf = CPURHFDFT(cell, xc='PBE').fft_density_fit()
# mf = CPURHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
# mf = CPURHFDFT(cell, xc='PBE').rs_density_fit(auxbasis='gth-szv')
mf = mf.multigrid_numint(
    ntasks=3,
    ke_ratio=3.,
    rel_cutoff=20.,
    with_j=False,
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

print('\n[CPU FFTDF Hartree + PBE multigrid XC]')
print(f'HF/Hartree provider = {type(mf.with_df).__name__}')
print(f'Target XC integrator = {type(mf._numint).__name__}')
print(f'Cell mesh = {cell.mesh.tolist()}')
print('CPU MG hierarchy = '
      f'ntasks={mf._numint.ntasks}, '
      f'ke_ratio={mf._numint.ke_ratio:g}, '
      f'rel_cutoff={mf._numint.rel_cutoff:g}')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target PBE@RHF energy = {mf.e_dft:.15f} Ha')
print(f'Provider Hartree contribution = {mf.e_coul:.15f} Ha')
print(f'Multigrid XC contribution = {mf.e_xc:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print(f'XC derivative backend = {mf_grad.xc_deriv_backend}')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')


#
# Case 2: GPU RSDF source/response + R2SCAN multigrid J+XC
#
# GPU4PySCF multigrid-v2 uses cell.mesh directly and does not expose the CPU
# ntasks/ke_ratio/rel_cutoff hierarchy.  RSDF is selected by the commented
# replacement line exactly as in the CPU case.  ``with_j=True`` changes only
# the fixed-density target energy/Fock/explicit derivative: source RHF J/K and
# the CPHF/Z-vector operator remain RSDF.
#
mf = GPURHFDFT(cell, xc='R2SCAN').rs_density_fit(auxbasis='gth-szv')
# mf = GPURHFDFT(cell, xc='R2SCAN').density_fit(auxbasis='gth-szv')
# mf = GPURHFDFT(cell, xc='R2SCAN').fft_density_fit()
mf = mf.multigrid_numint(with_j=True).set(
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

print('\n[GPU RSDF source/response + R2SCAN multigrid J+XC]')
print(f'Source HF/CPHF provider = {type(mf.with_df).__name__}')
print(f'Target J+XC integrator = {type(mf._numint).__name__}')
print(f'Cell mesh = {cell.mesh.tolist()}')
print('GPU multigrid large-IFFT real-view copies = '
      f'{mf._numint.ifft_real_stats["materialized_calls"]}')
print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target R2SCAN@RHF energy = {mf.e_dft:.15f} Ha')
print(f'Multigrid Hartree contribution = {mf.e_coul:.15f} Ha')
print(f'Multigrid XC contribution = {mf.e_xc:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(mf_grad.de)
print(f'XC derivative backend = {mf_grad.xc_deriv_backend}')
print('Combined MG J+XC gradient term is stored in '
      'mf_grad.de_j_xc_multigrid')
print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
