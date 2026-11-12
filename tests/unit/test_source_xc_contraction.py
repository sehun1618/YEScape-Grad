'''Backend-native XC contractions agree with the array-einsum reference.'''

from importlib import import_module
from unittest import mock

import numpy as np
import pytest
from pyscf import gto, lib


@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('xc', ['LDA,VWN', 'PBE', 'r2SCAN'])
def test_xc_contraction_parity(backend, unrestricted, xc):
    lib.num_threads(48)
    xc_module = import_module('yescape_grad.' + backend + '.source.xc')
    grad_xc_fock = xc_module.grad_xc_fock
    dft = import_module('pyscf.dft' if backend == 'cpu' else 'gpu4pyscf.dft')
    mol = gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g', verbose=0)
    source = (dft.UKS if unrestricted else dft.RKS)(mol, xc=xc)
    source.grids.level = 0
    source.small_rho_cutoff = 0
    source.kernel()
    dm = source.make_rdm1()
    q = np.asarray([[.2, -.1], [-.1, -.3]])
    if unrestricted:
        q = np.stack((q, -.7*q))
    reference_contract = np.einsum
    if backend == 'gpu':
        import cupy as cp
        q = cp.asarray(q)
        reference_contract = cp.einsum
    actual = grad_xc_fock(source, dm, q, [0, 1], block_size=128)
    owner = lib if backend == 'cpu' else xc_module
    name = 'einsum' if backend == 'cpu' else 'contract'
    with mock.patch.object(owner, name, reference_contract):
        reference = grad_xc_fock(source, dm, q, [0, 1], block_size=128)
    if backend == 'gpu':
        actual, reference = actual.get(), reference.get()
    np.testing.assert_allclose(actual, reference, atol=1e-11, rtol=1e-11)
