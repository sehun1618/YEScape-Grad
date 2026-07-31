'''Use GDF/RSDF for Hartree and multigrid for target XC only.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import RHFDFT

cell = gto.Cell()
cell.atom = '''
O   0.10  -0.20   0.30
H   1.55   0.15   0.05
H  -0.35   1.25  -0.25
'''
cell.a = np.eye(3) * 7
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-hf'
cell.ke_cutoff = 20
cell.precision = 1e-8
cell.verbose = 0
cell.build()

np.set_printoptions(precision=12, suppress=True)
for backend in ('gdf', 'rsdf'):
    mf = RHFDFT(cell, xc='PBE')
    if backend == 'gdf':
        mf.density_fit(auxbasis='gth-szv')
    else:
        mf.rs_density_fit(auxbasis='gth-szv')

    # ``multigrid_numint`` follows the PySCF copy convention.  The returned
    # object keeps source RHF, CPHF, target Hartree J, hcore and all Gaussian
    # response terms with the selected GDF/RSDF.  Only target XC uses MG.
    #
    # CPU Multigrid controls:
    #   cell.ke_cutoff : finest FFT/task resolution (set before cell.build)
    #   ntasks        : number of reciprocal grid levels (DC-Forge default 3)
    #   ke_ratio      : adjacent level cutoff ratio
    #   rel_cutoff    : real-space AO-pair/task cutoff used by the C backend
    # The same settings are used by energy and gradient because they belong to
    # the copied target object's MultiGridNumInt2 instance.
    mf = mf.multigrid_numint(
        ntasks=3,
        ke_ratio=3.0,
        rel_cutoff=20.0,
    )
    # PySCF's four-level partition remains available as a reference:
    # mf = mf.multigrid_numint(ntasks=4)
    mf.conv_tol = 1e-11
    mf.run()
    mf_grad = mf.nuc_grad_method()
    # XC derivative algorithm: single_pass is production; double_pass is
    # retained as a slower numerical oracle for regression comparisons.
    mf_grad.multigrid_xc_deriv_method = 'single_pass'
    # mf_grad.multigrid_xc_deriv_method = 'double_pass'
    mf_grad.run()

    print(f'\n[{backend.upper()} Hartree + multigrid XC]')
    print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
    print(f'PBE@RHF energy = {mf.e_dft:.15f} Ha')
    print(f'Hartree energy ({backend.upper()}) = {mf.e_coul:.15f} Ha')
    print(f'XC energy (multigrid) = {mf.e_xc:.15f} Ha')
    print(f'Multigrid mesh = {np.asarray(mf._numint.mesh)}')
    print(
        'Multigrid levels = '
        f'{mf._numint.ntasks} '
        f'(ke_ratio={mf._numint.ke_ratio:g}, '
        f'rel_cutoff={mf._numint.rel_cutoff:g})')
    print('Gradient dE/dR [Ha/Bohr]')
    print(mf_grad.de)
    print(f'XC derivative backend = {mf_grad.xc_deriv_backend}')
    print(
        'XC derivative method = '
        f'{mf_grad.multigrid_xc_deriv_method_used}')
    print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
