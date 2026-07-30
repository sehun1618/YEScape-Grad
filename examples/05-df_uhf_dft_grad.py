'''Evaluate open-shell DF-PBE@DF-UHF energy and analytic gradient.'''

from pathlib import Path

import numpy as np
from pyscf import gto

from dcforge.cpu.energy import UHFDFT

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'structures' / 'oh.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    spin=1,
    verbose=0,
)

mf = UHFDFT(mol, xc='PBE').density_fit().set(
    conv_tol=1e-12,
    conv_tol_grad=1e-10,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()

mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    auxbasis_response=True,
    fock_deriv_method='direct',
    cphf_conv_tol=1e-12,
    cphf_max_cycle=200,
)
gradient = mf_grad.kernel()

np.set_printoptions(precision=12, suppress=True)
print(f'Auxiliary basis = {mf.auxbasis}')
print(f'Shared with_df = {mf.with_df is mf.mf_hf.with_df}')
print(f'DF-PBE@DF-UHF energy = {energy:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(gradient)
print('Target auxiliary-basis response [Ha/Bohr]')
print(mf_grad.de_aux)
print(f'Source-Fock derivative = {mf_grad.fock_deriv_method_used}')
print(f'Overlap-response RHS count = {mf_grad.overlap_response_nrhs}')
