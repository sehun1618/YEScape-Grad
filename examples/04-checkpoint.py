'''Restore conventional and density-fitted RHF checkpoints for HF-DFT gradients.'''

import numpy as np
from pyscf import gto, scf
from pyscf.scf import chkfile

from dcforge.cpu.energy import RHFDFT

np.set_printoptions(precision=12, suppress=True)

# Checkpoints remain in the working directory and are overwritten on rerun.

# Case 1: conventional RHF water
mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
            basis='def2-svp', verbose=0)
mf_hf = scf.RHF(mol)
mf_hf.chkfile = 'rhf.chk'
mf_hf.conv_tol = 1e-8
mf_hf.conv_tol_grad = 1e-6
mf_hf.max_cycle = 100
mf_hf.kernel()

# Restore the saved HF orbitals, without running HF again.
mol, state = chkfile.load_scf('rhf.chk')
mf_hf = scf.RHF(mol)
mf_hf.e_tot = state['e_tot']
mf_hf.mo_coeff = state['mo_coeff']
mf_hf.mo_energy = state['mo_energy']
mf_hf.mo_occ = state['mo_occ']
# Assumes a converged checkpoint; this flag does not verify convergence.
mf_hf.converged = True

mf = RHFDFT(mol, xc='r2SCAN', mf_hf=mf_hf)
mf.grids.level = 4
energy = mf.kernel(run_hf=False)
mf_grad = mf.nuc_grad_method().set(cphf_max_cycle=200)
gradient = mf_grad.kernel()

print('\n[RHF checkpoint]')
print(f'HF-r2SCAN energy [Ha]: {energy:.15f}')
print('Gradient [Ha/Bohr]:\n', gradient)
print('Force [Ha/Bohr]:\n', -gradient)





# Case 2: density-fitted RHF water
mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
            basis='def2-svp', verbose=0)
mf_hf = scf.RHF(mol).density_fit(auxbasis='def2-svp-jkfit')
mf_hf.chkfile = 'rhf_df.chk'
mf_hf.conv_tol = 1e-8
mf_hf.conv_tol_grad = 1e-6
mf_hf.max_cycle = 100
mf_hf.kernel()

mol, state = chkfile.load_scf('rhf_df.chk')
# Recreate the same DF settings; they are not stored in the SCF checkpoint.
mf = RHFDFT(mol, xc='r2SCAN').density_fit(auxbasis='def2-svp-jkfit')
mf_hf = mf.mf_hf
mf_hf.e_tot = state['e_tot']
mf_hf.mo_coeff = state['mo_coeff']
mf_hf.mo_energy = state['mo_energy']
mf_hf.mo_occ = state['mo_occ']
mf_hf.converged = True

mf.grids.level = 4
energy = mf.kernel(run_hf=False)
mf_grad = mf.nuc_grad_method().set(cphf_max_cycle=200,
                                   auxbasis_response=True)
gradient = mf_grad.kernel()

print('\n[DF-RHF checkpoint]')
print(f'HF-r2SCAN energy [Ha]: {energy:.15f}')
print('Gradient [Ha/Bohr]:\n', gradient)
print('Force [Ha/Bohr]:\n', -gradient)

# The checkpoint restores orbitals, not CDERI; DF integrals are rebuilt as needed.
