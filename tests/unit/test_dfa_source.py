'''Source-neutral lifecycle and the identical-functional SCF limit.'''

from importlib import import_module
from io import StringIO
import numpy as np
import pytest
from pyscf import gto, lib


@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
def test_source_module_layout(backend):
    for name in ('rhf', 'uhf', 'rks', 'uks'):
        module_name = 'dcforge.' + backend + '.source.' + name
        module = import_module(module_name)
        function = module.grad_fock_direct if name.endswith('hf') else module.grad_fock
        assert function.__module__ == module_name
    for spin in ('r', 'u'):
        common = import_module('dcforge.' + backend + '.grad.' + spin + '_nv')
        assert not hasattr(common, 'grad_fock_direct')
        assert hasattr(common, 'nonvariational_correction')


@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('df', [False, True])
def test_ks_source(backend, unrestricted, df):
    lib.num_threads(48)
    module = import_module('dcforge.'+backend+'.energy')
    cls = module.UNSC if unrestricted else module.RNSC
    assert cls.__name__ == ('UNSC' if unrestricted else 'RNSC')
    assert (module.UHFDFT if unrestricted else module.RHFDFT) is cls
    mol = gto.M(atom='N 0 0 0; H 0 -.8 .6; H 0 .8 .6' if unrestricted
                else 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                spin=int(unrestricted), basis='sto-3g', verbose=0)
    mf = cls(mol, xc='PBE', source_xc='PBE')
    legacy = cls(mol, mf_hf=mf.mf_source)
    assert legacy.mf_source is mf.mf_source
    with pytest.raises(ValueError, match='not both'):
        cls(mol, mf_hf=mf.mf_source, mf_source=mf.mf_source)
    if df:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    mf.mf_source.grids.level = mf.grids.level = 1
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    mf.stdout = stream = StringIO()
    mf.verbose = 4
    mf.kernel()
    assert mf.mf_hf is mf.mf_source
    assert mf.e_hf == mf.e_uhf == mf.e_source
    assert mf.dm_hf is mf.dm_source
    mf.mf_hf, mf.dm_hf, mf.e_hf = mf.mf_source, mf.dm_source, mf.e_source
    assert not {'mf_hf', 'e_hf', 'e_uhf', 'dm_hf'}.intersection(mf.__dict__)
    assert 'PBE-PBE' in stream.getvalue()
    assert 'Source grid' in stream.getvalue()
    assert 'Source XC' in stream.getvalue()
    assert 'Target XC' in stream.getvalue()
    assert 'Source functional' not in stream.getvalue()
    assert 'Target functional' not in stream.getvalue()
    grad = mf.nuc_grad_method()
    result = grad.kernel()
    native = mf.mf_source.nuc_grad_method()
    native.grid_response = True
    if df:
        native.auxbasis_response = True
    np.testing.assert_allclose(result, native.kernel(), atol=1e-7, rtol=0)
    e = mf.e_tot
    assert mf.kernel(run_source=False) == pytest.approx(e, abs=1e-10)
    assert mf.kernel(run_hf=False) == pytest.approx(e, abs=1e-10)
    with pytest.raises(ValueError, match='Conflicting'):
        mf.kernel(run_source=False, run_hf=True)
    mf.mf_source.xc = 'B3LYP'
    assert np.isfinite(mf.kernel(run_source=False))
    mf.mf_source.xc = 'PBE'
    old_omega = mf.mf_source._numint.omega
    mf.mf_source.omega = .2
    assert np.isfinite(mf.kernel(run_source=False))
    mf.mf_source._numint.omega = old_omega
    mf.mf_source.grids.level = 2
    assert np.isfinite(mf.kernel(run_source=False))
    mf.mf_source.grids.level = 1
    scanner = grad.as_scanner()
    coords = mol.atom_coords()
    coords[1, 2] += .002
    e, g = scanner(mol.set_geom_(coords, unit='Bohr', inplace=False))
    assert np.isfinite(e) and np.all(np.isfinite(g))
    with pytest.raises(ValueError, match='reconverge'):
        mf.as_scanner()(mol, run_source=False)


def test_changed_source_grid_rebuilds():
    from dcforge.cpu.energy import RNSC
    mol = gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g', verbose=0)
    mf = RNSC(mol, xc='PBE', source_xc='PBE')
    mf.mf_source.grids.level = mf.grids.level = 0
    mf.kernel()
    old_size = len(mf.mf_source.grids.coords)
    mf.mf_source.grids.level = 2
    mf.kernel()
    assert len(mf.mf_source.grids.coords) > old_size
    mf.kernel(run_source=False)


def test_source_options():
    from dcforge.cpu.energy import RNSC
    from pyscf import dft, scf
    mol = gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g',verbose=0)
    source = dft.RKS(mol, xc='PBE')
    source.small_rho_cutoff = 0
    mf = RNSC(mol, xc='r2SCAN', mf_source=source)
    assert mf.mf_source is source
    with pytest.raises(ValueError, match='not both'):
        RNSC(mol, mf_source=source, source_xc='PBE')
    with pytest.raises(TypeError):
        RNSC(mol, mf_source=scf.UHF(mol))
    mf.set_dispersion('d2c')
    with pytest.raises(NotImplementedError, match='HF sources'):
        mf.kernel()
    mf.set_dispersion(None)
    source.small_rho_cutoff = 1e-7
    with pytest.raises(ValueError, match='small_rho_cutoff'):
        mf.kernel()


def test_ks_checkpoint(tmp_path):
    from pyscf import dft, scf
    from dcforge.cpu.energy import RNSC
    lib.num_threads(48)
    mol = gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g',verbose=0)
    mf = RNSC(mol, xc='r2SCAN', source_xc='PBE')
    mf.mf_source.grids.level = mf.grids.level = 1
    mf.chkfile = str(tmp_path/'source.chk')
    mf.conv_tol = 1e-12
    mf.kernel()
    expected = mf.nuc_grad_method().kernel()
    restored_mol, state = scf.chkfile.load_scf(mf.chkfile)
    source = dft.RKS(restored_mol, xc='PBE')
    source.grids.level = 1
    source.small_rho_cutoff = 0
    source.__dict__.update(state)
    source.converged = True
    restored = RNSC(restored_mol, xc='r2SCAN', mf_source=source)
    restored.grids.level = 1
    restored.kernel(run_source=False)
    np.testing.assert_allclose(restored.nuc_grad_method().kernel(), expected,
                               atol=1e-9, rtol=0)
    restored.kernel(run_source=False)


@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('df', [False, True])
def test_empty_rotation_space(backend, df):
    mol = gto.M(atom='He 0 0 0', basis='sto-3g', verbose=0)
    cls = import_module('dcforge.'+backend+'.energy').RNSC
    mf = cls(mol, xc='r2SCAN', source_xc='PBE')
    if df:
        mf = mf.density_fit()
    mf.mf_source.grids.level = mf.grids.level = 1
    mf.kernel()
    grad = mf.nuc_grad_method().run()
    assert grad.z_solver == 'zero_dimensional'
    np.testing.assert_allclose(grad.de, 0., atol=1e-10, rtol=0)
