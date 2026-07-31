'''Compare Gamma-point GDF and RSDF PBE@RHF analytic gradients.'''

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
    # Shared-provider Gaussian response uses this PySCF-style MB budget for
    # short-range integral and long-range reciprocal-grid blocking.
    mf.with_df.max_memory = 1000
    mf.conv_tol = 1e-11
    mf.run()
    mf_grad = mf.nuc_grad_method().run()

    print(f'\n[{backend.upper()}]')
    print(f'Source RHF energy = {mf.e_hf:.15f} Ha')
    print(f'PBE@RHF energy = {mf.e_dft:.15f} Ha')
    print('Gradient dE/dR [Ha/Bohr]')
    print(mf_grad.de)
    parity = mf.with_df.cderi_value_parity_max_abs
    print(
        'Reconstructed/stored CDERI max abs = '
        + ('not evaluated' if parity is None else f'{parity:.6e}'))
    print(
        'CDERI response storage / block size = '
        f'{mf_grad.cderi_response_storage_used} / '
        f'{mf_grad.cderi_response_blksize_used}')
    print(
        'Materialized full CDERI response = '
        f'{mf_grad.cderi1 is not None}')
    print(
        'Fourier auxiliary response storage = '
        f'{mf.with_df.fourier_response_aux_storage}')
    print(
        'Unblocked AO-pair Fourier bytes = '
        f'{mf.with_df.fourier_response_ao_pair_bytes}')
    print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.6e}')
    if backend == 'rsdf':
        print(f'RSDF settings = {mf.with_df.response_settings()}')
