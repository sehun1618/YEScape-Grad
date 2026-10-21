'''Public aliases must not invoke target SCF or expose stale success state.'''

from importlib import import_module
from unittest import mock
from argparse import Namespace
from pathlib import Path
import json
import runpy

import pytest
from pyscf import gto, lib


@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('use_df', [False, True])
def test_public_aliases_and_failed_rerun(backend, unrestricted, use_df):
    lib.num_threads(48)
    api = import_module('dcforge.' + backend + '.energy')
    mol = gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g', verbose=0)
    mf = (api.UNSC if unrestricted else api.RNSC)(
        mol, xc='r2SCAN', source_xc='PBE')
    if use_df:
        mf = mf.density_fit()
    mf.grids.level = mf.mf_source.grids.level = 0
    assert type(mf.Gradients()) is type(mf.nuc_grad_method())
    assert type(mf.Gradients()).__module__.startswith('dcforge.')
    with pytest.raises(NotImplementedError):
        mf.Hessian()
    with mock.patch.object(mf.mf_source, 'kernel', wraps=mf.mf_source.kernel) as run:
        energy = mf.scf()
        assert run.call_count == 1
    assert mf.converged and mf.e_tot == energy
    assert mf.kernel(run_source=False) == pytest.approx(energy, abs=1e-10)

    for failure in (False, RuntimeError('source failed'), KeyboardInterrupt()):
        # Seed an earlier successful result, then fail the next SCF attempt.
        mf.converged = mf.mf_source.converged = True
        mf.e_tot = mf.e_source = mf.e_dft = mf.e_d2c = energy
        mf.dm_source = object()
        mf.d2c_result = object()
        with mock.patch.object(mf.mf_source, 'kernel',
                               side_effect=failure if failure else None,
                               return_value=0.):
            with pytest.raises((RuntimeError, KeyboardInterrupt)):
                mf.kernel()
        assert not mf.converged and not mf.mf_source.converged
        assert all(getattr(mf, name) is None for name in
                   ('e_tot', 'e_source', 'e_dft', 'e_d2c', 'dm_source', 'd2c_result'))


def test_validation_runs_do_not_overwrite(tmp_path):
    script = Path(__file__).parents[1] / 'check_dfa_source.py'
    for target in ('r2SCAN', 'CAM-B3LYP'):
        # Empty matrix exercises the real CLI output setup without an SCF.
        args = Namespace(backend=[], source=['PBE'], full_fd=False,
                         basis='sto-3g', target=target, output=tmp_path)
        with mock.patch('argparse.ArgumentParser.parse_args', return_value=args):
            runpy.run_path(str(script), run_name='__main__')
    runs = list(tmp_path.glob('run_*'))
    assert len(runs) == 2
    assert {json.loads((p / 'run.json').read_text())['target'] for p in runs} == {
        'r2SCAN', 'CAM-B3LYP'}
    assert all((p / 'summary.json').exists() for p in runs)
