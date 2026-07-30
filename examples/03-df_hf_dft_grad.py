'''Evaluate molecular DF-PBE@DF-HF energy and analytic nuclear gradient.'''

from pathlib import Path

import numpy as np
from pyscf import gto

from dcforge.cpu.energy import RHFDFT

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'structures' / 'water1.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    verbose=0,
)

# density_fit() applies one exchange-capable molecular DF provider to both the
# source RHF and target supplied-density DFT evaluator.  An explicit auxbasis
# or with_df object can be passed with the same PySCF argument names.
mf = RHFDFT(mol, xc='PBE').density_fit().set(
    conv_tol=1e-12,
    conv_tol_grad=1e-10,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()

# A complete DF derivative needs both target-grid motion and auxiliary-basis
# response.  Source orbital response comes from DF-RHF gen_response.  The
# default source-Fock derivative contracts Z directly; set
# fock_deriv_method='make_h1' to run the PySCF DF-Hessian reference instead.
mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    auxbasis_response=True,
    cphf_conv_tol=1e-12,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
)
gradient = mf_grad.kernel()

np.set_printoptions(precision=12, suppress=True)
print(f'Auxiliary basis = {mf.auxbasis}')
print(f'Shared with_df  = {mf.with_df is mf.mf_hf.with_df}')
print(f'Source-Fock derivative = {mf_grad.fock_deriv_method_used}')
print(f'DF-PBE@DF-HF energy = {energy:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(gradient)
print('Target auxiliary-basis response [Ha/Bohr]')
print(mf_grad.de_aux)
print(f'Term recomposition = {np.max(np.abs(gradient - mf_grad.de_skel - mf_grad.de_nvar)):.6e}')
