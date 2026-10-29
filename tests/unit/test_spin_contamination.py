"""UHF spin warning at the shared CPU/GPU source-state boundary."""
import io
import numpy as np
import pytest
from pyscf import gto
from dcforge.cpu.energy import UNSC, RNSC
from dcforge.cpu.energy.rnsc import _check_source_state


@pytest.mark.parametrize('spin,s2,warn', [(0,0.,False),(0,.099,False),(0,.1,True),
    (1,.849,False),(1,.85,True),(1,.65,True),(2,2.,False),(2,2.1,True)])
def test_spin_warning_boundary(spin, s2, warn):
    atom = 'H 0 0 0' if spin == 1 else 'H 0 0 0; H 0 0 1'
    mol = gto.M(atom=atom, spin=spin, basis='sto-3g', verbose=0)
    mf = UNSC(mol, xc='PBE')
    source = mf.mf_source
    n = mol.nao
    source.mo_coeff = np.stack([np.eye(n)]*2)
    source.mo_energy = np.zeros((2,n))
    source.mo_occ = np.zeros((2,n))
    source.converged = True
    source.spin_square = lambda: (s2, 0.)
    mf.stdout = io.StringIO(); mf.verbose = 3
    _check_source_state(mf)
    message = mf.stdout.getvalue()
    assert ('spin contaminated' in message) == warn
    if warn:
        assert 'Be very cautious when using the HF-DFT energy and gradient' in message
        assert 'consider a different source' in message
        assert 'Results are returned unchanged' not in message
        assert f'target S(S+1)={spin/2*(spin/2+1):.6f}' in message


@pytest.mark.parametrize('restricted,source_xc', [(True,None),(False,'PBE')])
def test_non_uhf_sources_skip_spin_check(restricted, source_xc):
    mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
    mf = (RNSC if restricted else UNSC)(mol, source_xc=source_xc)
    source = mf.mf_source
    n=mol.nao
    source.mo_coeff = np.eye(n) if restricted else np.stack([np.eye(n)]*2)
    source.mo_energy = np.zeros(n) if restricted else np.zeros((2,n))
    source.mo_occ = np.zeros_like(source.mo_energy)
    source.converged = True
    if not restricted:source.grids.level=0
    def unexpected():raise AssertionError('Only UHF sources should be checked')
    source.spin_square=unexpected
    _check_source_state(mf)


@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('use_df', [False, True])
def test_broken_symmetry_energy_and_gradient_warn_without_changing_results(backend, use_df):
    from importlib import import_module
    from pyscf import lib
    lib.num_threads(2)
    xp = np if backend == 'cpu' else pytest.importorskip('cupy')
    cls = import_module('dcforge.'+backend+'.energy').UNSC
    mol = gto.M(atom='H 0 0 0; H 0 0 3', basis='sto-3g', verbose=0)
    mf = cls(mol, xc='PBE')
    if use_df:mf=mf.density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level=0;mf.conv_tol=1e-10
    mf.stdout=io.StringIO();mf.verbose=3
    dm=xp.asarray([[[1.,0.],[0.,0.]],[[0.,0.],[0.,1.]]])
    energy=mf.kernel(dm0=dm)
    assert float(mf.mf_source.spin_square()[0]) >= .1
    assert 'spin contaminated' in mf.stdout.getvalue()
    mf.stdout.seek(0);mf.stdout.truncate(0)
    grad=mf.nuc_grad_method();grad.stdout=mf.stdout;grad.verbose=3
    gradient=grad.kernel()
    assert 'spin contaminated' in mf.stdout.getvalue()
    # Alter only the diagnostic accessor; the electronic state is unchanged.
    original=mf.mf_source.spin_square
    try:
        mf.mf_source.spin_square=lambda: (0.,1.)
        mf.stdout.seek(0);mf.stdout.truncate(0)
        energy_without_warning=mf.kernel(run_source=False)
        grad=mf.nuc_grad_method();grad.stdout=mf.stdout;grad.verbose=3
        gradient_without_warning=grad.kernel()
        assert 'spin contaminated' not in mf.stdout.getvalue()
    finally:mf.mf_source.spin_square=original
    assert abs(energy-energy_without_warning) < 1e-10
    np.testing.assert_allclose(gradient,gradient_without_warning,rtol=0,atol=1e-10)
