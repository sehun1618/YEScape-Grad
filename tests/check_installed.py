'''Run outside the checkout: check_installed.py --backend cpu|gpu --prefix ENV.'''

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np
from pyscf import gto, lib
import yescape_grad

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--backend', choices=['cpu', 'gpu'], required=True)
parser.add_argument('--prefix', type=Path, required=True)
parser.add_argument('--fd', action='store_true', help='Check one relaxed-source FD component per case')
args = parser.parse_args()
assert Path(yescape_grad.__file__).resolve().is_relative_to(args.prefix.resolve()), yescape_grad.__file__
print('Installed package:', yescape_grad.__file__, flush=True)
lib.num_threads(48)
if args.backend == 'gpu':
    import cupy as cp
    from gpu4pyscf.lib import cutensor
    assert cutensor.cutensor is not None and cutensor.contract_engine is None
    cp.testing.assert_allclose(cutensor.contract(
        'ij,jk->ik', cp.ones((2, 3)), cp.ones((3, 2))), 3)
api = import_module('yescape_grad.' + args.backend + '.energy')
for unrestricted in (False, True):
    for use_df in (False, True):
        for source in ('HF', 'PBE'):
            mol = gto.M(atom='N 0 0 0; H 0 -.8 .6; H 0 .8 .6' if unrestricted
                        else 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                        spin=int(unrestricted), basis='sto-3g', verbose=0)
            mf = (api.UNSC if unrestricted else api.RNSC)(mol, xc='r2SCAN', source_xc=source)
            if use_df:
                mf = mf.density_fit()
            mf.grids.level = 0
            if args.fd:
                mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
            if source != 'HF':
                mf.mf_source.grids.level = 0
            energy = mf.scf()
            assert mf.converged and np.isfinite(energy)
            grad = mf.Gradients()
            result = grad.kernel()
            if args.backend == 'gpu':
                result = cp.asnumpy(result)
            assert np.isfinite(result).all()
            if args.backend == 'cpu':
                assert grad.grid_response_backend == 'becke_omp'
                other = mf.nuc_grad_method().set(grid_response_backend='pyscf_default').kernel()
                np.testing.assert_allclose(result, other, atol=1e-9, rtol=0)
            try:
                mf.Hessian()
            except NotImplementedError:
                pass
            else:
                raise AssertionError('Native target Hessian was exposed')
            if args.fd:
                scanner = mf.as_scanner()
                energies = []
                for sign in (-1, 1):
                    coords = mol.atom_coords()
                    coords[1, 2] += sign * 2e-4
                    energies.append(scanner(mol.set_geom_(coords, unit='Bohr', inplace=False)))
                    assert scanner.converged
                error = abs((energies[1] - energies[0]) / 4e-4 - result[1, 2])
                assert error < 2e-6, error
                print('FD error:', error, 'Ha/Bohr;', error*51.4220674763, 'eV/Angstrom', flush=True)
            print('PASS', args.backend, 'U' if unrestricted else 'R',
                  'DF' if use_df else 'conv', source, flush=True)
