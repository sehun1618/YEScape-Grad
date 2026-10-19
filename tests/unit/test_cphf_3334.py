'''Regression for PySCF #3334 without changing installed solver modules.

Run CPU checks: python -m pytest -q tests/unit/test_cphf_3334.py -m 'not gpu'
Run GPU checks in conda gpu: python -m pytest -q tests/unit/test_cphf_3334.py -m gpu
'''

import importlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from pyscf import lib
from pyscf.scf import cphf

from dcforge.tools.cphf import solve


@pytest.mark.parametrize('backend', [
    'pyscf', pytest.param('gpu4pyscf', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('overlap', [False, True])
@pytest.mark.parametrize('nset', [1, 2])
@pytest.mark.parametrize('level_shift', [0., .2])
def test_cphf_against_diagonal_solution(backend, unrestricted, overlap, nset,
                                      level_shift):
    xp = np if backend == 'pyscf' else pytest.importorskip('cupy')
    module = importlib.import_module(backend + '.scf.' +
                                    ('ucphf' if unrestricted else 'cphf'))
    # Two almost parallel RHS from upstream #3334, or a small single RHS
    # that the installed 1e-13 squared-norm cutoff drops altogether.
    rhs = xp.asarray([[1., 0.], [1., 3e-8]] if nset == 2 else [[3e-8, 0.]])
    if overlap:
        # Test overlap/sign/level-shift semantics with well-resolved RHS.
        # The upstream eps floor still permits residuals above tol for tiny
        # components; #3334 does not promise arbitrary physical accuracy.
        rhs = xp.asarray([[1., .3], [.2, .8]][:nset])
    diag = xp.asarray([.1, .2])
    expected_vo = -rhs / (1. + diag)
    if unrestricted:
        energy = (xp.asarray([-.5, .5]), xp.asarray([-.5, .5]))
        occ = (xp.asarray([1., 0.]), xp.asarray([1., 0.]))
        if overlap:
            h1 = tuple(xp.zeros((nset, 2, 1)) for _ in range(2))
            s1 = tuple(xp.full_like(h, .01) for h in h1)
            for spin in range(2):
                h1[spin][:, 1, 0] = rhs[:, spin] - .005
            diagonal = xp.asarray([0., .1, 0., .2])
        else:
            h1 = (rhs[:, :1, None], rhs[:, 1:, None])
            s1 = None
            diagonal = diag
    else:
        energy = xp.asarray([-.5, .5, .5])
        occ = xp.asarray([2., 0., 0.])
        if overlap:
            h1 = xp.zeros((nset, 3, 1))
            h1[:, 1:, 0] = rhs - .005
            s1 = xp.full_like(h1, .01)
            diagonal = xp.asarray([0., .1, .2])
        else:
            h1 = rhs[:, :, None]
            s1 = None
            diagonal = diag

    def vind(x):
        return (x.reshape(-1, diagonal.size) * diagonal).reshape(x.shape)

    upstream_functions = (module.solve, module.solve_nos1, module.solve_withs1)
    if backend == 'gpu4pyscf' and not unrestricted and not overlap and level_shift:
        # GPU4PySCF 1.8.1 has a separate 2D/3D level-shift broadcasting bug.
        # Prove it also occurs natively before marking this unsupported path.
        # DC-Forge's Z callers use the default zero level shift.
        try:
            module.solve(vind, energy, occ, h1 * 1e8, tol=1e-11,
                         level_shift=level_shift, verbose=0)
        except ValueError as exc:
            if 'Out shape is mismatched' not in str(exc):
                raise
            with pytest.raises(ValueError, match='Out shape is mismatched'):
                solve(module, vind, energy, occ, h1, tol=1e-11,
                      level_shift=level_shift, verbose=0)
            pytest.xfail('Native GPU CPHF nonzero level_shift broadcasting bug')
    answer, _ = solve(module, vind, energy, occ, h1, s1=s1,
                      tol=1e-11, max_cycle=20, level_shift=level_shift, verbose=0)
    if unrestricted:
        vo = xp.stack([x[:, -1, 0] for x in answer], axis=1)
        if overlap:
            for x in answer:
                assert float(xp.max(xp.abs(x[:, 0, 0] + .005))) < 1e-14
    else:
        vo = answer[:, -2:, 0]
        if overlap:
            assert float(xp.max(xp.abs(answer[:, 0, 0] + .005))) < 1e-14
    assert float(xp.max(xp.abs(vo - expected_vo))) < 1e-10
    assert float(xp.max(xp.abs((1 + diag) * vo + rhs))) < 1e-10
    assert upstream_functions == (module.solve, module.solve_nos1, module.solve_withs1)


def test_threshold_reduces_small_rhs_physical_residual():
    energy = np.array([-.5, .5])
    occ = np.array([2., 0.])
    rhs = np.array([[3e-8]])
    native, _ = cphf.solve(lambda x: .1*x, energy, occ, rhs, tol=1e-11)
    fixed, _ = solve(cphf, lambda x: .1*x, energy, occ, rhs, tol=1e-11)
    assert abs(1.1*fixed.item() + rhs.item()) < 1e-20
    # Native already-fixed future versions are allowed to match, not regress.
    assert abs(1.1*fixed.item() + rhs.item()) <= abs(1.1*native.item() + rhs.item())


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize('tol', [1e-3, 1e-9, 1e-12])
def test_dtype_floor_and_no_global_patch(dtype, tol, monkeypatch):
    native_krylov = lib.krylov
    seen = []

    def capture(aop, b, **kwargs):
        seen.append(kwargs['lindep'])
        return np.zeros_like(b)

    # The shim binds _qr from the original solver globals even with a spy.
    capture.__globals__['_qr'] = native_krylov.__globals__['_qr']
    monkeypatch.setattr(lib, 'krylov', capture)
    energy = np.array([-.5, .5], dtype=np.float32 if dtype in
                      (np.float32, np.complex64) else np.float64)
    solve(cphf, lambda x: x, energy, np.array([2., 0.]),
          np.ones((1, 1), dtype=dtype), tol=tol)
    assert seen == [max(tol**2, np.finfo(np.empty(0, dtype=dtype).real.dtype).eps)]
    assert lib.krylov is capture


def test_concurrent_solves_keep_independent_thresholds():
    def run(tol):
        return solve(cphf, lambda x: .1*x, np.array([-.5, .5]),
                     np.array([2., 0.]), np.array([[3e-8]]), tol=tol)[0].item()
    with ThreadPoolExecutor(2) as executor:
        answers = list(executor.map(run, [1e-3, 1e-11] * 8))
    np.testing.assert_allclose(answers[::2], 0, atol=0)
    np.testing.assert_allclose(answers[1::2], -3e-8/1.1, atol=1e-20)


@pytest.mark.parametrize('tol', [0, -1, float('nan'), float('inf')])
def test_invalid_tolerance(tol):
    with pytest.raises(ValueError, match='finite and positive'):
        solve(cphf, lambda x: x, np.array([-.5, .5]), np.array([2., 0.]),
              np.ones((1, 1)), tol=tol)


@pytest.mark.parametrize('device', [
    'cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('active_spin', [0, 1])
def test_uhf_single_active_spin_uses_fixed_cphf(device, active_spin):
    xp = np if device == 'cpu' else pytest.importorskip('cupy')
    module = importlib.import_module(f'dcforge.{device}.grad.uhf_zvec')
    energy = (xp.asarray([-.5, .5]), xp.asarray([-.5, .5]))
    occ = [xp.zeros(2), xp.zeros(2)]
    occ[active_spin][0] = 1
    Co = tuple(xp.eye(2)[:, o > 0] for o in occ)
    Cv = tuple(xp.eye(2)[:, o == 0] for o in occ)
    fvo = tuple(xp.full((v.shape[1], o.shape[1]), 3e-8) for v, o in zip(Cv, Co))
    fvind = xp.zeros_like
    if device == 'cpu':
        zvec, info = module.solve_z_vector(energy, occ, Co, Cv, fvo, fvind,
                                          conv_tol=1e-11)
    else:
        vind = module.gen_vind(Co, Cv, fvind)
        zvec, info = module.solve_z_vector(energy, occ, fvo, vind, conv_tol=1e-11)
    assert info['residual_max_abs'] < 1e-20
    assert float(xp.max(xp.abs(zvec[active_spin] - fvo[active_spin]))) < 1e-20
