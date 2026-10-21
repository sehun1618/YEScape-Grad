'''Molecular UHF-DFT energy and analytic nuclear gradient.'''

from pathlib import Path

from pyscf import gto

from dcforge.cpu.energy import UHFDFT

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'tests' / 'structures' / 'oh.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    spin=1,
    verbose=3,
)

# Case 1: conventional HF-r2SCAN energy and gradient
mf = UHFDFT(mol, xc='r2SCAN').set(
    conv_tol=1e-8,
    conv_tol_grad=1e-6,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()
mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
).run()
force = -mf_grad.de

# Case 2: density-fitted HF-r2SCAN energy and complete DF gradient
mf = UHFDFT(mol, xc='r2SCAN').density_fit().set(
    conv_tol=1e-8,
    conv_tol_grad=1e-6,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()
mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    auxbasis_response=True,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
).run()
force = -mf_grad.de
