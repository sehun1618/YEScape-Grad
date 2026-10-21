"""Molecular scanner state, density reuse, and CPU/GPU geometry parity."""

import importlib

import numpy as np
import pytest
from pyscf import lib

from tests.common import as_numpy, configure_workflow, displaced_mol, make_mol


@pytest.mark.parametrize('backend', [
    pytest.param('cpu', marks=pytest.mark.cpu),
    pytest.param('gpu', marks=pytest.mark.gpu),
])
@pytest.mark.parametrize('unrestricted', [False, True], ids=['rhf', 'uhf'])
@pytest.mark.parametrize('df', [False, True], ids=['conventional', 'df'])
def test_scanner_density_reuse_and_geometry_parity(backend, unrestricted, df, monkeypatch):
    if backend == 'gpu':
        cp = pytest.importorskip('cupy')
        pytest.importorskip('gpu4pyscf')
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA device required')
    module = importlib.import_module(f'dcforge.{backend}.energy')
    workflow = module.UNSC if unrestricted else module.RNSC
    mol = make_mol('nh2_doublet.xyz' if unrestricted else 'water1.xyz')

    def make(mol):
        mf = workflow(mol, xc='PBE')
        if df:
            mf = mf.density_fit(auxbasis='def2-svp-jkfit')
        mf.max_cycle = 200
        return configure_workflow(mf)

    scanner = make(mol).nuc_grad_method().as_scanner()
    assert isinstance(scanner, lib.GradScanner)
    assert scanner.as_scanner() is scanner
    assert scanner.base.as_scanner() is scanner.base
    passed_guesses = []
    original = scanner.base.mf_source.kernel

    def capture(dm0=None, **kwargs):
        passed_guesses.append(None if dm0 is None else as_numpy(dm0).copy())
        return original(dm0=dm0, **kwargs)

    monkeypatch.setattr(scanner.base.mf_source, 'kernel', capture)
    scanner(mol)
    assert passed_guesses[-1] is None
    previous_dm = as_numpy(scanner.base.mf_source.make_rdm1()).copy()
    moved = displaced_mol(mol, 1, 2, .02)
    energy, gradient = scanner(moved)
    np.testing.assert_allclose(passed_guesses[-1], previous_dm, atol=1e-13)
    assert scanner.converged
    assert scanner.base.mol is scanner.base.mf_source.mol is moved
    if df:
        assert scanner.base.with_df is scanner.base.mf_source.with_df
        assert scanner.base.with_df.mol is moved
    reference = make(moved)
    np.testing.assert_allclose(energy, reference.kernel(), atol=1e-9, rtol=0)
    np.testing.assert_allclose(as_numpy(gradient),
        as_numpy(reference.nuc_grad_method().kernel()), atol=1e-7, rtol=0)
    # Return to the original geometry, explicitly suppressing automatic reuse.
    energy, gradient = scanner(mol, dm0=None)
    assert passed_guesses[-1] is None
    reference = make(mol)
    np.testing.assert_allclose(energy, reference.kernel(), atol=1e-9, rtol=0)
    np.testing.assert_allclose(as_numpy(gradient),
        as_numpy(reference.nuc_grad_method().kernel()), atol=1e-7, rtol=0)
    # Explicit guesses and raw geometry input also pass through correctly.
    dm = scanner.base.mf_source.make_rdm1()
    scanner(mol.atom_coords(unit=mol.unit), dm0=dm)
    np.testing.assert_allclose(passed_guesses[-1], as_numpy(dm), atol=1e-13)

    def fail(**kwargs):
        raise RuntimeError('deliberate source failure')

    monkeypatch.setattr(scanner.base.mf_source, 'kernel', fail)
    with pytest.raises(RuntimeError, match='deliberate source failure'):
        scanner(mol)
    assert not scanner.converged and scanner.de is None
    assert scanner.base.e_tot is None
    monkeypatch.setattr(scanner.base.mf_source, 'kernel', capture)
    scanner(mol)
    assert scanner.converged and passed_guesses[-1] is None

    # Gradient failure must not inherit the successful energy convergence flag.
    kernel = scanner.kernel
    monkeypatch.setattr(scanner, 'kernel', fail)
    with pytest.raises(RuntimeError, match='deliberate source failure'):
        scanner(mol)
    assert not scanner.converged and scanner.de is None
    monkeypatch.setattr(scanner, 'kernel', kernel)
    scanner(mol)
    assert scanner.converged


@pytest.mark.cpu
def test_energy_scanner_incompatible_guess_and_frozen_source_rejected(monkeypatch):
    from dcforge.cpu.energy import RNSC
    scanner = configure_workflow(RNSC(make_mol('water1.xyz'), xc='PBE')).as_scanner()
    scanner(scanner.mol)
    changed = scanner.mol.copy()
    changed.basis = 'sto-3g'
    changed.build(False, False)
    original = scanner.mf_source.kernel
    guesses = []

    def capture(dm0=None, **kwargs):
        guesses.append(dm0)
        return original(dm0=dm0, **kwargs)

    monkeypatch.setattr(scanner.mf_source, 'kernel', capture)
    scanner(changed)
    assert guesses[-1] is None and scanner.converged
    with pytest.raises(ValueError, match='reconverge'):
        scanner(changed, run_source=False)


@pytest.mark.cpu
def test_scanner_keyboard_interrupt_invalidates_gradient_state(monkeypatch):
    from dcforge.cpu.energy import RNSC
    mol = make_mol('water1.xyz', basis='sto-3g')
    scanner = configure_workflow(RNSC(mol, xc='PBE')).nuc_grad_method().as_scanner()
    scanner(mol)

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(scanner, 'kernel', interrupt)
    with pytest.raises(KeyboardInterrupt):
        scanner(mol)
    assert not scanner.converged and scanner.de is None


@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
def test_scanner_d2c_geometry_update(backend):
    pytest.importorskip('dftd4')
    if backend == 'gpu':
        pytest.importorskip('cupy')
        pytest.importorskip('gpu4pyscf')
    from dcforge.cpu.disp.d2c import evaluate_d2c
    cls = importlib.import_module(f'dcforge.{backend}.energy').RNSC
    mol = make_mol('water1.xyz')
    mf = configure_workflow(cls(mol, xc='r2SCAN').density_fit(auxbasis='def2-svp-jkfit'))
    scanner = mf.set_dispersion('d2c').nuc_grad_method().as_scanner()
    scanner(mol)
    mol = displaced_mol(mol, 1, 2, .03)
    energy, gradient = scanner(mol)
    correction = evaluate_d2c(mol, 'r2SCAN', gradient=True)
    scanner.base.set_dispersion(None)
    plain_energy, plain_gradient = scanner(mol, dm0=None)
    np.testing.assert_allclose(energy-plain_energy, correction.energy, atol=1e-9, rtol=0)
    np.testing.assert_allclose(as_numpy(gradient)-as_numpy(plain_gradient),
                               correction.gradient, atol=1e-7, rtol=0)


@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
def test_scanner_geometry_optimization(backend):
    pytest.importorskip('geometric')
    if backend == 'gpu':
        pytest.importorskip('gpu4pyscf')
    from pyscf.geomopt.geometric_solver import kernel
    cls = importlib.import_module(f'dcforge.{backend}.energy').RNSC
    mf = configure_workflow(cls(make_mol('water1.xyz'), xc='PBE').density_fit(
        auxbasis='def2-svp-jkfit'))
    converged, mol = kernel(mf, maxsteps=15)
    assert converged
    fresh = configure_workflow(cls(mol, xc='PBE').density_fit(auxbasis='def2-svp-jkfit'))
    fresh.kernel()
    assert np.max(np.abs(as_numpy(fresh.nuc_grad_method().kernel()))) < 4.5e-4
