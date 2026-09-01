'''Save standalone GPU RHF/UHF states for later HF-DFT gradients.

Run this source-HF job first.  ``07-restore-gpu-hf-states.py`` is a separate
target HF-DFT job that reads the two generated checkpoints without rerunning
either HF calculation.
'''

from pathlib import Path

from gpu4pyscf.scf import hf, uhf
from pyscf import gto

from dcforge.hf_state import save_hf_state

OUTPUT = Path.cwd() / 'hf_state_examples'
OUTPUT.mkdir(parents=True, exist_ok=True)


#
# Case 1: closed-shell DF-RHF source
#
mol_rhf = gto.M(
    atom='''
        O  0.000000  0.000000  0.000000
        H  0.000000 -0.757000  0.587000
        H  0.000000  0.757000  0.587000
    ''',
    unit='Angstrom',
    basis='def2-svp',
    verbose=0,
)
source_rhf = hf.RHF(mol_rhf).density_fit().set(
    conv_tol=1e-12,
    conv_tol_grad=1e-10,
    max_cycle=100,
)
source_rhf.kernel()
if not source_rhf.converged:
    raise RuntimeError('standalone GPU DF-RHF did not converge')

rhf_path = OUTPUT / 'water_df_rhf.chk'
rhf_manifest = save_hf_state(
    rhf_path,
    source_rhf,
    extra_provenance={'example': 'standalone GPU DF-RHF'},
)
print('[RHF source]')
print(f'Checkpoint = {rhf_path}')
print(f'Energy = {source_rhf.e_tot:.15f} Ha')
print(f'Representation = {rhf_manifest["representation"]}')
print(f'State fingerprint = {rhf_manifest["state_fingerprint"]}')


#
# Case 2: open-shell DF-UHF source
#
mol_uhf = gto.M(
    atom='''
        O  0.000000  0.000000 -0.605000
        O  0.000000  0.000000  0.605000
    ''',
    unit='Angstrom',
    basis='def2-svp',
    charge=0,
    spin=2,
    verbose=0,
)
source_uhf = uhf.UHF(mol_uhf).density_fit().set(
    conv_tol=1e-12,
    conv_tol_grad=1e-10,
    max_cycle=100,
)
source_uhf.kernel()
if not source_uhf.converged:
    raise RuntimeError('standalone GPU DF-UHF did not converge')

uhf_path = OUTPUT / 'o2_df_uhf.chk'
uhf_manifest = save_hf_state(
    uhf_path,
    source_uhf,
    extra_provenance={'example': 'standalone GPU DF-UHF triplet O2'},
)
print('\n[UHF source]')
print(f'Checkpoint = {uhf_path}')
print(f'Energy = {source_uhf.e_tot:.15f} Ha')
print(f'Representation = {uhf_manifest["representation"]}')
print(f'State fingerprint = {uhf_manifest["state_fingerprint"]}')
