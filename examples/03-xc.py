'''Reuse one converged RHF density across target functional choices.

``xc`` changes the target energy model. The source RHF orbitals are
converged once and are never optimized for any target functional below.
NLC/VV10 is unsupported. D2C dispersion is enabled separately; see 06-d2c.py.
'''

from pathlib import Path

from pyscf import gto, scf

from yescape_grad.cpu.energy import RNSC

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'tests' / 'structures' / 'water1.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    verbose=0,
)
mf_source = scf.RHF(mol).set(
    conv_tol=1e-8,
    conv_tol_grad=1e-6,
    max_cycle=100,
).run()


# Case 1: change the target functional while keeping one source RHF density
print(f'Source RHF energy = {mf_source.e_tot:.15f} Ha')
print('\nTarget functional families at grids.level = 4')
for xc in ('LDA,VWN', 'r2SCAN', 'TPSS', 'B3LYP', 'CAM-B3LYP'):
    mf = RNSC(mol, xc=xc, mf_source=mf_source)
    mf.grids.level = 4
    energy = mf.kernel(run_source=False)
    print(f'{xc:12s}  E_HF-XC = {energy:.15f} Ha')
