'''Reuse a GDF provider factor independently from source RHF orbitals.'''

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pyscf.pbc import gto

from dcforge.gpu.pbc.df.gdf import (
    load_provider_factor_checkpoint,
    save_provider_factor_checkpoint,
)
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

with TemporaryDirectory() as directory:
    checkpoint = Path(directory) / 'water1-gdf-factor.h5'

    # The first source calculation builds both CDERI and RHF orbitals.  Only
    # the provider factor and its layout/checksums are written here.
    pbe = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv').run()
    saved = save_provider_factor_checkpoint(checkpoint, pbe.with_df)

    # Changing target XC does not invalidate the source-authoritative GDF
    # operator.  This new object restores CDERI but still performs a fresh
    # source RHF because no orbital state was accepted from the checkpoint.
    r2scan = RHFDFT(
        cell, xc='R2SCAN').density_fit(auxbasis='gth-szv')
    restored = load_provider_factor_checkpoint(
        checkpoint, r2scan.with_df, storage='mapped_host')
    r2scan.run()

    print(f'factor checkpoint = {checkpoint}')
    print(f'factor SHA256 = {saved["factor_fingerprint"]}')
    print(f'restored same factor = '
          f'{saved["factor_fingerprint"] == restored["factor_fingerprint"]}')
    print(f'fresh source RHF converged = {r2scan.mf_hf.converged}')
    print(f'E_R2SCAN@HF = {r2scan.e_dft:.15f} Ha')

# A changed lattice, atomic coordinate, basis, pseudopotential, auxiliary
# basis, metric threshold, k-point, AO ordering, package ABI, or checksum is
# rejected before the factor is installed.  Interrupted ``.tmp`` files do not
# carry the complete marker and are never accepted.
