'''Evaluate conventional water-cation PBE@UHF gradient with GPU4PySCF.'''

from pathlib import Path

from pyscf import gto

from dcforge.gpu.energy import UHFDFT

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'structures' / 'water1.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    charge=1,
    spin=1,
    verbose=0,
)

mf = UHFDFT(mol, xc='PBE').set(
    conv_tol=1e-10,
    conv_tol_grad=1e-9,
    max_cycle=100,
)
mf.grids.level = 4
energy = mf.kernel()

mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    cphf_conv_tol=1e-9,
    cphf_max_cycle=200,
    fock_deriv_method='direct',
)
gradient = mf_grad.kernel()

print(f'Backend = {mf.device}')
print('Integral representation = conventional')
print(f'Source-Fock derivative = {mf_grad.fock_deriv_method_used}')
print(f'Target PBE@UHF energy = {energy:.15f} Ha')
print('Analytic gradient (Ha/Bohr):')
print(gradient)
print(f'UCPHF solver = {mf_grad.z_solver}')
print(f'Z physical residual max abs = {mf_grad.z_residual_max_abs:.6e} Ha')
