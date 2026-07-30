'''Evaluate molecular DF-PBE@DF-RHF energy with GPU4PySCF.'''

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

print(f'Backend = {mf.device}')
print(f'Auxiliary basis = {mf.auxbasis}')
print(f'Shared with_df = {mf.with_df is mf.mf_hf.with_df}')
print(f'Source DF-RHF energy = {mf.e_hf:.15f} Ha')
print(f'Target DF-PBE@DF-RHF energy = {energy:.15f} Ha')
