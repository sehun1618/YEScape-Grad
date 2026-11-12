'''Representative DFA-source diversity and identical-functional checks.

Use --phase diversity or limit. Run in conda gpu, with PYTHONPATH at the repo.
Each invocation preserves its logs/results in a fresh output directory.
'''

import argparse
from importlib import import_module
import json
from pathlib import Path
import tempfile
import traceback
import numpy as np
from pyscf import gto, lib
from tests.check_dfa_source import check
from tests.common import as_numpy, make_mol


def molecule(name):
    if name == 'benzene':
        atoms = []
        for symbol, radius in [('C', 1.397), ('H', 2.48)]:
            for angle in np.arange(6) * np.pi / 3:
                atoms.append((symbol, (radius*np.cos(angle), radius*np.sin(angle), 0.)))
        return gto.M(atom=atoms, basis='def2-svp', verbose=0)
    if name == 'hydroxide':
        return gto.M(atom='O 0 0 0; H 0 0 .97', charge=-1,
                     basis='aug-cc-pvdz', verbose=0)
    if name == 'stretched_h2':
        return gto.M(atom='H 0 0 0; H 0 0 2.0', basis='def2-svp', verbose=0)
    return make_mol('ch3_doublet_distorted.xyz', basis='def2-svp')


def scf_limit(backend, unrestricted, use_df, xc):
    api = import_module('yescape_grad.' + backend + '.energy')
    mol = make_mol('nh2_doublet.xyz' if unrestricted else 'water1.xyz', basis='sto-3g')
    mf = (api.UNSC if unrestricted else api.RNSC)(mol, source_xc=xc, xc=xc)
    if use_df:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level = mf.mf_source.grids.level = 2
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    mf.max_cycle = 200
    mf.kernel()
    grad = mf.nuc_grad_method()
    actual = as_numpy(grad.kernel())
    native = mf.mf_source.nuc_grad_method()
    native.grid_response = True
    if use_df:
        native.auxbasis_response = True
    reference = as_numpy(native.kernel())
    error = float(np.max(np.abs(actual-reference)))
    energy_error = abs(float(mf.e_tot-mf.mf_source.e_tot))
    return dict(passed=error < 1e-7 and energy_error < 1e-9,
                energy_error_ha=energy_error, error_ha_bohr=error,
                error_ev_ang=error*51.4220674763, residual=grad.z_residual_max_abs,
                target_rhs=grad.z_rhs_max_abs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['diversity', 'limit'], required=True)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], nargs='+', default=['cpu', 'gpu'])
    parser.add_argument('--output', type=Path, default=Path('tests/results/publication_check'))
    args = parser.parse_args()
    lib.num_threads(48)
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=args.phase+'_', dir=args.output))
    print('RESULTS', output.resolve(), flush=True)
    results = []
    cases = [('benzene', 'PBE', False), ('hydroxide', 'PBE0', False),
             ('ch3', 'B3LYP', True), ('stretched_h2', 'PBE', False)]
    for backend in args.backend:
        if backend == 'gpu':
            from gpu4pyscf.lib import cutensor
            import cupy as cp
            assert cutensor.cutensor is not None and cutensor.contract_engine is None
            cp.testing.assert_allclose(cutensor.contract(
                'ij,jk->ik', cp.ones((2,3)), cp.ones((3,2))), 3)
        jobs = cases if args.phase == 'diversity' else [
            ('same_xc', xc, u) for xc in ('PBE', 'r2SCAN', 'B3LYP', 'CAM-B3LYP')
            for u in (False, True)]
        for use_df in (False, True):
            for name, xc, unrestricted in jobs:
                key = f'{backend}_{name}_{xc}_{"u" if unrestricted else "r"}_{"df" if use_df else "conv"}'
                print('START', key, flush=True)
                try:
                    if args.phase == 'limit':
                        result = scf_limit(backend, unrestricted, use_df, xc)
                    else:
                        mol = molecule(name)
                        # Benzene: representative bond/ring directions; smaller systems: all 3N.
                        indices = [(0,0), (1,1), (6,0)] if name == 'benzene' else None
                        result = check(backend, unrestricted, use_df, xc, True,
                                       mol.basis, mol=mol, indices=indices, grid_level=3)
                        result['atoms_angstrom'] = [(mol.atom_symbol(i), r.tolist())
                            for i, r in enumerate(mol.atom_coords(unit='Angstrom'))]
                except Exception:
                    result = dict(passed=False, traceback=traceback.format_exc())
                result.update(name=key, phase=args.phase)
                (output/(key+'.json')).write_text(json.dumps(result, indent=2)+'\n')
                results.append(result)
                (output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
                print('PASS' if result['passed'] else 'FAIL', key,
                      result.get('error_ha_bohr', result.get('traceback')), flush=True)
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)
