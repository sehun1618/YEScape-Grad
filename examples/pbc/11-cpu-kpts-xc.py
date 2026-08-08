'''CPU sampled-k LDA, GGA and R2SCAN fixed-density examples.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import KRHFDFT

cell = gto.Cell()
cell.atom = 'H 0 0 0; H 0 0 1.4'
cell.a = np.eye(3) * 6.0
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pbe'
cell.precision = 1e-8
cell.mesh = [15, 15, 15]
cell.verbose = 0
cell.build()

kpts = cell.make_kpts([2, 1, 1], wrap_around=True)


#
# Case 1: converge the source KRHF once and evaluate sampled-k PBE@KRHF
#
# KRHFDFT.kernel() converges only the source KRHF.  The target PBE energy is
# then evaluated at the fixed source density; target KRKS orbitals are never
# optimized.
#
mf_pbe = KRHFDFT(
    cell, kpts=kpts, xc='PBE', kpts_wrap_around=True,
).set(conv_tol=1e-12, max_cycle=100).run()
grad_pbe = mf_pbe.nuc_grad_method().kernel()

print('\n[Case 1: PBE@KRHF]')
print(f'Energy = {mf_pbe.e_dft:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(grad_pbe)


#
# Case 2: LDA and GGA targets on the same converged KRHF reference
#
# Passing mf_hf explicitly shares the same source object and DF provider.
# run_hf=False prevents an unnecessary repeat SCF while still rebuilding the
# target Fock and target energy for each functional.
#
for xc in ('LDA,VWN', 'BLYP'):
    mf = KRHFDFT(
        cell, kpts=kpts, xc=xc, mf_hf=mf_pbe.mf_hf,
        kpts_wrap_around=True,
    )
    mf.kernel(run_hf=False)
    gradient = mf.nuc_grad_method().kernel()

    print(f'\n[Case 2: {xc}@KRHF]')
    print(f'Energy = {mf.e_dft:.15f} Ha')
    print('Gradient dE/dR [Ha/Bohr]')
    print(gradient)


#
# Case 3: R2SCAN with the explicit k-resolved kinetic-energy-density channel
#
# The CPU uniform-grid derivative includes
# tau = 1/2 sum_k P_k grad(phi_k*) . grad(phi_k).  TPSS and other sampled-k
# MGGAs remain outside the validated capability even if libxc can evaluate
# their zero-order energy.
#
mf_r2scan = KRHFDFT(
    cell, kpts=kpts, xc='R2SCAN', mf_hf=mf_pbe.mf_hf,
    kpts_wrap_around=True,
)
mf_r2scan.kernel(run_hf=False)
mf_r2scan_grad = mf_r2scan.nuc_grad_method()
grad_r2scan = mf_r2scan_grad.kernel()

print('\n[Case 3: R2SCAN@KRHF]')
print(f'Energy = {mf_r2scan.e_dft:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(grad_r2scan)
print(f'XC derivative backend = {mf_r2scan_grad.xc_deriv_backend}')


#
# Case 4: CPU sampled-k multigrid is an explicit capability boundary
#
# PySCF 2.14 pair-multigrid collocation and AO pass-2 kernels accept Gamma
# only.  DC-Forge therefore rejects this request immediately; it does not
# silently replace multigrid by the uniform-grid result.
#
mf_gdf = KRHFDFT(
    cell, kpts=kpts, xc='PBE', kpts_wrap_around=True,
).density_fit(auxbasis='gth-szv')
try:
    mf_gdf.multigrid_numint()
except NotImplementedError as error:
    print('\n[Case 4: sampled-k CPU multigrid boundary]')
    print(error)
