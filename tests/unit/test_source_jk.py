'''Check the integral calls, not just cancellation of zero-weighted outputs.'''

from importlib import import_module
import numpy as np
import pytest
from pyscf import gto, lib


@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('density_fit', [False, True])
@pytest.mark.parametrize('xc', ['PBE', 'CAM-B3LYP', 'LC_WPBE'])
def test_source_derivative_skips_unused_integrals(backend, unrestricted, density_fit, xc, monkeypatch):
    lib.num_threads(48)
    xp = np if backend == 'cpu' else import_module('cupy')
    energy = import_module('yescape_grad.'+backend+'.energy')
    mol = gto.M(atom='N 0 0 0; H 0 -.8 .6; H 0 .8 .6' if unrestricted
                else 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                spin=int(unrestricted), basis='sto-3g', verbose=0)
    mf = (energy.UNSC if unrestricted else energy.RNSC)(mol, xc='r2SCAN', source_xc=xc)
    if density_fit:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    mf.mf_source.grids.level = 0
    mf.kernel_source()
    source = mf.mf_source
    c, occ = source.mo_coeff, source.mo_occ
    if unrestricted:
        co = [cs[:, os > 0] for cs, os in zip(c, occ)]
        cv = [cs[:, os == 0] for cs, os in zip(c, occ)]
        z = [xp.full((v.shape[1], o.shape[1]), .01) for o, v in zip(co, cv)]
    else:
        co, cv = c[:, occ > 0], c[:, occ == 0]
        z = xp.full((cv.shape[1], co.shape[1]), .01)
    records = []
    if backend == 'cpu':
        native = source.nuc_grad_method()
        for name in ('get_j', 'get_k', 'get_jk'):
            original = getattr(native, name)
            def trace(mol, dm, *args, _name=name, _original=original, **kwargs):
                records.append((_name, len(dm), kwargs.copy()))
                return _original(mol, dm, *args, **kwargs)
            monkeypatch.setattr(native, name, trace)
        monkeypatch.setattr(source, 'nuc_grad_method', lambda: native)
    else:
        module = import_module('yescape_grad.gpu.source.'+('uks' if unrestricted else 'rks'))
        owner = module if density_fit else module.tdrhf
        name = 'jk_energies_per_atom_bounded' if density_fit else '_jk_energies_per_atom'
        original = getattr(owner, name)
        def trace(*args, **kwargs):
            pairs = args[2] if density_fit else args[1]
            js = kwargs['j_factor']
            assert js is not None
            records.append((len(pairs), js, kwargs['k_factor']))
            return original(*args, **kwargs)
        monkeypatch.setattr(owner, name, trace)
    module = import_module('yescape_grad.'+backend+'.source.'+('uks' if unrestricted else 'rks'))
    result = module.grad_fock(mf.nuc_grad_method(), co, cv, z,
                              source.make_rdm1(), list(range(mol.natm)))
    assert bool(xp.isfinite(result).all())
    if backend == 'cpu':
        if xc == 'PBE':
            assert len(records) == 1 and records[0][:2] == ('get_j', 2)
        else:
            lr = [r for r in records if r[2].get('omega') is not None]
            assert len(lr) == 1
            assert lr[0][0] == 'get_jk'
            assert lr[0][2].get('with_j', True)
            if unrestricted:
                assert all(n == (2 if name == 'get_j' else 4) for name, n, kw in records)
    else:
        assert all(j != 0 or k != 0 for n, js, ks in records for j, k in zip(js, ks))
        if xc == 'PBE':
            assert len(records) == 1 and records[0][0] == 1 and records[0][2] == [0.]
        else:
            assert len(records) == 2
            assert all(j == 0 for j in records[-1][1])
            assert records[-1][0] == (2 if unrestricted else 1)
