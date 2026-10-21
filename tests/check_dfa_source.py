'''Standalone relaxed-source FD matrix for molecular non-SCF gradients.

Run in conda gpu with PYTHONPATH pointing to this checkout. Each displaced
energy reconverges source SCF; target SCF is never run. Outputs include both
force-error units and a complete term ledger. A failed case leaves its error
record and does not erase earlier results.
'''

import argparse
import importlib
import json
from pathlib import Path
import time
import traceback
import subprocess
import tempfile

import numpy as np
from pyscf import gto, lib


def check(backend, unrestricted, use_df, xc, full_fd, basis, target='r2SCAN'):
    energy = importlib.import_module('dcforge.'+backend+'.energy')
    cls = energy.UNSC if unrestricted else energy.RNSC
    atom = ('N 0 0 0; H 0 -.8 .6; H 0 .8 .6' if unrestricted else
            'O 0 0 0; H 0 -.757 .587; H 0 .757 .587')
    mol = gto.M(atom=atom, basis=basis, spin=int(unrestricted), verbose=0)

    def build(m):
        mf = cls(m, xc=target, source_xc=xc)
        if use_df:
            mf = mf.density_fit(auxbasis='def2-universal-jkfit')
        mf.mf_source.grids.level = mf.grids.level = 2
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        mf.max_cycle = 200
        mf.kernel()
        return mf

    start = time.perf_counter()
    mf = build(mol)
    grad = mf.nuc_grad_method().run()
    indices = [(a, x) for a in range(mol.natm) for x in range(3)] if full_fd else [(1, 2)]
    records = []
    for a, x in indices:
        energies = []
        for sign in (-1, 1):
            coords = mol.atom_coords()
            coords[a, x] += sign*2e-4
            displaced = mol.set_geom_(coords, unit='Bohr', inplace=False)
            energies.append(build(displaced).e_tot)
        fd = (energies[1]-energies[0])/4e-4
        records.append(dict(atom=a, axis=x, analytic=float(grad.de[a,x]), fd=fd,
                            error=float(grad.de[a,x]-fd)))
    error = max(abs(r['error']) for r in records)
    result = dict(backend=backend, unrestricted=unrestricted, df=use_df, source=xc,
                  target=target, basis=basis, energy=mf.e_tot, gradient=grad.de.tolist(),
                  residual=grad.z_residual_max_abs, fd=records,
                  error_ha_bohr=error, error_ev_ang=error*51.4220674763,
                  elapsed_s=time.perf_counter()-start,
                  terms={k:np.asarray(v).tolist() for k,v in vars(grad).items()
                         if k.startswith('de_') and isinstance(v,np.ndarray)})
    result['passed'] = error < 2e-6 and grad.z_residual_max_abs < 1e-7
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', nargs='+', default=['cpu', 'gpu'])
    parser.add_argument('--source', nargs='+', default=['LDA,VWN', 'PBE', 'TPSS', 'r2SCAN', 'B3LYP', 'CAM-B3LYP'])
    parser.add_argument('--full-fd', action='store_true')
    parser.add_argument('--basis', default='sto-3g')
    parser.add_argument('--target', default='r2SCAN')
    parser.add_argument('--output', type=Path, default=Path('tests/results/dfa_source'))
    args = parser.parse_args()
    lib.num_threads(48)
    args.output.mkdir(parents=True, exist_ok=True)
    # A fresh directory preserves previous runs, including failed/partial runs.
    args.output = Path(tempfile.mkdtemp(prefix='run_', dir=args.output))
    print('Results:', args.output.resolve(), flush=True)
    repo = Path(__file__).resolve().parents[1]
    metadata = dict(vars(args), output=str(args.output))
    metadata['commit'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    metadata['worktree_status'] = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=repo, text=True)
    (args.output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    results = []
    for backend in args.backend:
        if backend == 'gpu':
            from gpu4pyscf.lib import cutensor as ct
            import cupy as cp
            assert ct.cutensor is not None and ct.contract_engine is None
            cp.testing.assert_allclose(ct.contract('ij,jk->ik',cp.ones((2,3)),cp.ones((3,2))),3)
        for unrestricted in (False, True):
            for use_df in (False, True):
                for xc in args.source:
                    name = f'{backend}_{"u" if unrestricted else "r"}_{"df" if use_df else "conv"}_{xc}'
                    print('START', name, flush=True)
                    try:
                        result = check(backend, unrestricted, use_df, xc, args.full_fd, args.basis, args.target)
                    except Exception:
                        result = dict(passed=False, traceback=traceback.format_exc())
                    result['name'] = name
                    (args.output/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
                    results.append(result)
                    print('PASS' if result['passed'] else 'FAIL', name,
                          result.get('error_ha_bohr',result.get('traceback')),flush=True)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    assert all(r['passed'] for r in results), 'See per-case results'
