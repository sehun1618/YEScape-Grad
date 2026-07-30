'''Evaluate molecular DF-PBE@DF-RHF analytic gradient with GPU4PySCF.'''

from pathlib import Path

from pyscf import gto

from dcforge.gpu.energy import RHFDFT

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'structures' / 'water1.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    verbose=0,
)

mf = RHFDFT(mol, xc='PBE').density_fit().set(
    conv_tol=1e-12,
    conv_tol_grad=1e-10,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()

mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    auxbasis_response=True,
    cphf_conv_tol=1e-9,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
)
gradient = mf_grad.kernel()

print(f'Backend = {mf.device}')
print(f'Auxiliary basis = {mf.auxbasis}')
print(f'Source-Fock derivative = {mf_grad.fock_deriv_method_used}')
print(f'Target DF-PBE@DF-RHF energy = {energy:.15f} Ha')
print('Analytic gradient (Ha/Bohr):')
print(gradient)
print(f'Z physical residual max abs = {mf_grad.z_residual_max_abs:.6e} Ha')
