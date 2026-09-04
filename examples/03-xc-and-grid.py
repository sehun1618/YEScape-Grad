'''Reuse one converged RHF density across target XC and grid choices.

``xc`` changes the target energy model.  ``grids.level`` changes only the
numerical quadrature used to evaluate that model.  The source RHF orbitals are
converged once and are never optimized for any target functional below.
NLC/VV10 and empirical dispersion are outside the DC-Forge molecular scope.
'''

from pathlib import Path

from pyscf import gto, scf

from dcforge.cpu.energy import RHFDFT

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'structures' / 'water1.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    verbose=0,
)
mf_hf = scf.RHF(mol).set(
    conv_tol=1e-12,
    conv_tol_grad=1e-10,
    max_cycle=100,
).run()


#
# Case 1: change the target functional while keeping one source RHF density
#
print(f'Source RHF energy = {mf_hf.e_tot:.15f} Ha')
print('\nTarget functional families at grids.level = 4')
for xc in ('LDA,VWN', 'PBE', 'TPSS', 'B3LYP', 'CAM-B3LYP'):
    mf = RHFDFT(mol, xc=xc, mf_hf=mf_hf)
    mf.grids.level = 4
    energy = mf.kernel(run_hf=False)
    print(f'{xc:12s}  E_DFA[HF] = {energy:.15f} Ha')


#
# Case 2: converge the numerical grid for one fixed target functional
#
print('\nPBE quadrature convergence on the same RHF density')
reference = None
records = []
for level in (2, 4, 6):
    mf = RHFDFT(mol, xc='PBE', mf_hf=mf_hf)
    mf.grids.level = level
    energy = mf.kernel(run_hf=False)
    records.append((level, mf.grids.coords.shape[0], energy))
    reference = energy

print('level  grid points  E_PBE[HF] (Ha)      delta from level 6 (uHa)')
for level, ngrids, energy in records:
    print(f'{level:5d}  {ngrids:11d}  {energy:18.12f}  '
          f'{(energy - reference) * 1e6:23.6f}')
