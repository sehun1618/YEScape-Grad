'''Result summaries must not perform extra electronic-structure calculations.'''
from io import StringIO
from types import SimpleNamespace
from importlib import import_module

import numpy as np
import pytest
from pyscf import gto, lib

from dcforge.cpu.energy import RHFDFT, UHFDFT
from dcforge.tools.logger import energy_summary, gradient_summary
from dcforge.tools import logger as result_logger


@pytest.mark.parametrize('workflow,atom,spin', [
    (RHFDFT, 'H 0 0 0; H 0 0 1', 0),
    (UHFDFT, 'O 0 0 0; H 0 0 1', 1),
    (UHFDFT, 'O 0 0 0; H 0 0 1; H 0 1 0', 0)])
@pytest.mark.parametrize('df', [False, True])
@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
def test_kernel_output(workflow, atom, spin, df, backend):
    lib.num_threads(2)
    unrestricted = workflow is UHFDFT
    if backend == 'gpu':
        workflow = getattr(import_module('dcforge.gpu.energy'), workflow.__name__)
    mol = gto.M(atom=atom, basis='sto-3g', spin=spin, verbose=0)
    mf = workflow(mol, xc='PBE')
    if df:
        mf = mf.density_fit()
    mf.grids.level = 0
    mf.verbose = 4
    mf.stdout = stream = StringIO()
    energy = mf.kernel()
    text = stream.getvalue()
    assert text.count('DC-Forge Energy') == 1
    assert text.count('Energy workflow wall time =') == 1
    assert 'HF-PBE' in text
    assert ('Unrestricted' if unrestricted else 'Restricted') in text
    assert backend.upper() in text
    assert f'{energy:.15f} Ha' in text
    stream.seek(0)
    stream.truncate()
    assert mf.energy_tot() == pytest.approx(energy, abs=1e-10, rel=0)
    assert 'DC-Forge Energy' not in stream.getvalue()
    assert 'Energy workflow wall time =' not in stream.getvalue()
    grad = mf.nuc_grad_method()
    values = grad.kernel()
    text = stream.getvalue()
    assert text.count('DC-Forge Gradient') == 1
    assert text.count('Gradient workflow wall time =') == 1
    assert text.count('Total wall time =') == 1
    assert 'gradients ---------------' not in text
    assert 'Z-vector residual (max abs)' in text
    assert 'Ha/Bohr' in text
    mf.verbose = grad.verbose = 0
    stream.seek(0)
    stream.truncate()
    energy_summary(mf, unrestricted=unrestricted)
    gradient_summary(grad, unrestricted=unrestricted)
    assert stream.getvalue() == ''
    np.testing.assert_allclose(grad.kernel(), values, atol=1e-10, rtol=0)
    assert stream.getvalue() == ''


@pytest.mark.parametrize('residual,limit,verdict', [
    (1e-9, 1e-8, 'PASS'), (1e-5, 1e-8, 'WARNING'),
    (np.nan, 1e-8, 'WARNING'), (None, 1e-8, 'NOT CHECKED'),
    (1e-9, None, 'NOT CHECKED')])
def test_diagnostics(residual, limit, verdict):
    mol = gto.M(atom='H 0 0 0; H 0 0 1', verbose=0)
    mf = SimpleNamespace(xc='r2SCAN', dispersion='d2c',
                         mf_hf=SimpleNamespace(converged=True))
    grad = SimpleNamespace(base=mf, mol=mol, de=np.ones((1, 3)),
                           atmlst=[1], verbose=3, stdout=StringIO(),
                           z_residual_max_abs=residual,
                           cphf_residual_warn_tol=limit)
    gradient_summary(grad)
    text = grad.stdout.getvalue()
    assert 'HF-r2SCAN' in text and 'D2C' in text
    assert 'Selected-atom gradient sum' in text
    assert text.split('Residual check ')[-1].lstrip().startswith(verdict)


def test_file_output_and_missing_energy(tmp_path):
    with (tmp_path / 'calculation.log').open('w+') as stream:
        mf = SimpleNamespace(xc='r2SCAN', dispersion='d2c', verbose=3,
                             stdout=stream, mf_hf=SimpleNamespace(converged=True),
                             e_hf=-10., e_dft=-11., e_d2c=-.1, e_tot=-11.1)
        energy_summary(mf)
        stream.seek(0)
        text = stream.read()
        assert '-11.100000000000000 Ha' in text
        assert '-0.100000000000000 Ha' in text
    mf.stdout = StringIO()
    mf.e_dft = None
    energy_summary(mf)
    assert 'not available' in mf.stdout.getvalue()


@pytest.mark.pbc
@pytest.mark.slow
@pytest.mark.parametrize('backend', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('nk', [1, 2])
def test_periodic_output(backend, nk):
    from pyscf.pbc import gto as pbcgto
    lib.num_threads(2)
    cell = pbcgto.M(atom='H 0 0 0; H 0 0 1.4', a=np.eye(3)*6., unit='Bohr',
                    basis='gth-szv', pseudo='gth-pbe', mesh=[15]*3, verbose=0)
    module = import_module('dcforge.' + backend + '.pbc.energy')
    if nk == 1:
        mf = module.RHFDFT(cell, xc='PBE')
    else:
        mf = module.KRHFDFT(cell, kpts=cell.make_kpts([2, 1, 1]), xc='PBE')
    mf.stdout = StringIO()
    mf.verbose = 4
    mf.kernel()
    grad = mf.nuc_grad_method()
    grad.kernel()
    text = mf.stdout.getvalue()
    assert text.count('DC-Forge Energy') == 1
    assert text.count('DC-Forge Gradient') == 1
    assert text.count('Energy workflow wall time =') == 1
    assert text.count('Gradient workflow wall time =') == 1
    assert 'Z-vector residual (max abs)' in text
    assert 'Gradient table unavailable' not in text


@pytest.mark.parametrize('verbose', [0, 1, 2, 3, 4])
def test_time_verbosity_and_repeated_calls(monkeypatch, verbose):
    clock = iter([1., 1.125, 10., 10.5])
    monkeypatch.setattr(result_logger, 'perf_counter', lambda: next(clock))
    obj = SimpleNamespace(verbose=verbose, stdout=StringIO())
    token = object()

    @result_logger.log_kernel_time('Energy workflow')
    def kernel(self, value):
        return value

    for _ in range(2):
        assert kernel(obj, token) is token
    text = obj.stdout.getvalue()
    if verbose >= 3:
        assert text.splitlines() == ['Energy workflow wall time = 0.125 s',
                                    'Energy workflow wall time = 0.500 s']
    else:
        assert text == ''


def test_failed_kernel_has_no_success_timing():
    obj = SimpleNamespace(verbose=3, stdout=StringIO())

    @result_logger.log_kernel_time('Gradient workflow')
    def kernel(self):
        raise RuntimeError('expected failure')

    with pytest.raises(RuntimeError, match='expected failure'):
        kernel(obj)
    assert obj.stdout.getvalue() == ''


def test_total_time_repeats_and_failure(monkeypatch):
    clock = iter([0., 2., 100., 103., 200., 204., 300., 400., 405.])
    monkeypatch.setattr(result_logger, 'perf_counter', lambda: next(clock))
    mf = SimpleNamespace(verbose=3, stdout=StringIO())
    grad = SimpleNamespace(base=mf, verbose=3, stdout=mf.stdout)

    @result_logger.log_kernel_time('Energy workflow')
    def energy(self, fail=False):
        if fail:
            raise RuntimeError('failed energy')

    @result_logger.log_kernel_time('Gradient workflow')
    def gradient(self):
        pass

    energy(mf)
    gradient(grad)
    gradient(grad)
    with pytest.raises(RuntimeError, match='failed energy'):
        energy(mf, fail=True)
    gradient(grad)
    totals = [s for s in mf.stdout.getvalue().splitlines() if s.startswith('Total')]
    assert totals == ['Total wall time = 103.000 s',
                      'Total wall time = 204.000 s',
                      'Total wall time = 5.000 s']


def test_nested_energy_time_is_not_counted_twice(monkeypatch):
    clock = iter([0., 1., 3., 5.])
    monkeypatch.setattr(result_logger, 'perf_counter', lambda: next(clock))
    mf = SimpleNamespace(verbose=3, stdout=StringIO())
    grad = SimpleNamespace(base=mf, verbose=3, stdout=mf.stdout)

    @result_logger.log_kernel_time('Energy workflow')
    def energy(self):
        pass

    @result_logger.log_kernel_time('Gradient workflow')
    def gradient(self):
        energy(self.base)

    gradient(grad)
    assert 'Total wall time = 5.000 s' in mf.stdout.getvalue()
