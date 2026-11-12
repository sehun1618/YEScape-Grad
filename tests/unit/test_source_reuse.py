'''Explicit checkpoint reuse is user-controlled; dimensional checks remain.'''

from importlib import import_module
from unittest import mock
import numpy as np
import pytest
from pyscf import gto, lib


@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('unrestricted', [False, True])
def test_explicit_reuse_without_array_fingerprints(backend, unrestricted):
    lib.num_threads(48)
    api = import_module('yescape_grad.'+backend+'.energy')
    mol = gto.M(atom='N 0 0 0; H 0 -.8 .6; H 0 .8 .6' if unrestricted
                else 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                spin=int(unrestricted), basis='sto-3g', verbose=0)
    mf = (api.UNSC if unrestricted else api.RNSC)(mol, xc='PBE')
    mf.grids.level = 0
    e0 = mf.kernel()
    source = mf.mf_source
    # A MO phase change alters bytes but not the physical density.
    source.mo_coeff *= -1
    source.converged = False
    with mock.patch.object(source, 'kernel', side_effect=AssertionError('source reran')):
        with mock.patch('pyscf.lib.logger.warn') as warn:
            assert mf.kernel(run_source=False) == pytest.approx(e0, abs=1e-10)
            assert warn.called
        assert not source.converged
        assert np.all(np.isfinite(mf.nuc_grad_method().kernel()))
        # Still reject malformed checkpoint dimensions before contraction.
        source.mo_energy = source.mo_energy[..., :-1]
        with pytest.raises(ValueError, match='orbital shapes'):
            mf.kernel(run_source=False)


def test_normal_scf_failure_is_not_silently_accepted():
    from yescape_grad.cpu.energy import RNSC
    mf = RNSC(gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g', verbose=0), xc='PBE')
    with mock.patch.object(mf.mf_source, 'kernel', return_value=0.):
        with pytest.raises(RuntimeError, match='did not converge'):
            mf.kernel()
