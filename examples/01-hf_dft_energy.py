'''Evaluate a molecular PBE@HF energy without running a target DFT SCF.'''

from pathlib import Path

from pyscf import gto

from dcforge.cpu.energy import RHFDFT

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'structures' / 'water1.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    verbose=0,
)

# RHFDFT.kernel() first converges the source RHF orbitals.  It then evaluates
# the inherited RKS energy with dm=mf.mf_hf.make_rdm1(); no target RKS SCF is
# performed.
mf = RHFDFT(mol, xc='PBE').set(
    conv_tol=1e-12,
    conv_tol_grad=1e-10,
    max_cycle=100,
)
mf.grids.level = 4
e_dft = mf.kernel()

print(f'Source RHF energy       = {mf.e_hf:.15f} Ha')
print(f'Target PBE@HF energy    = {e_dft:.15f} Ha')
print(f'HF density electron no. = {mf.dm_hf.dot(mol.intor("int1e_ovlp")).trace():.12f}')
