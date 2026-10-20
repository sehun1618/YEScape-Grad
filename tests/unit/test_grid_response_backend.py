"""CPU molecular grid backend parity; run in conda gpu, no GPU required."""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pytest
from pyscf import gto, dft, lib
from pyscf.dft import radi, gen_grid
from pyscf.grad import rks, uks
from dcforge.cpu.energy import RHFDFT, UHFDFT
from dcforge.cpu.grad import grid_response as gr


@pytest.mark.parametrize('adjust', [None, radi.treutler_atomic_radii_adjust,
                                   radi.becke_atomic_radii_adjust])
def test_weights(adjust):
    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.atom_grid = (20, 50)
    grids.radii_adjust = adjust
    reference = list(rks.grids_response_becke(grids))
    for threads in (1, 2):
        for actual, expected in zip(gr.grids_response_becke(grids, threads), reference):
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, atol=3e-12, rtol=1e-11)
    # Moving grid: displace both the nucleus and its atom-centred grid.
    owner, atom, axis, h = 1, 0, 2, 1e-5
    coords, _, w1 = reference[owner]
    vol = grids.gen_atomic_grids(mol, grids.atom_grid, grids.radi_method,
                                grids.level, grids.prune)[mol.atom_symbol(owner)][1]
    weights = []
    for sign in (-1, 1):
        xyz = mol.atom_coords()
        xyz[atom, axis] += sign*h
        displaced = mol.set_geom_(xyz, unit='Bohr', inplace=False)
        shifted_grids = dft.gen_grid.Grids(displaced)
        shifted_grids.radii_adjust = adjust
        points = coords.copy()
        if atom == owner:
            points[:, axis] += sign*h
        weights.append(gr.partition(points, vol, owner, gr.parameters(shifted_grids), 2)[0])
    np.testing.assert_allclose((weights[1]-weights[0])/(2*h), w1[atom, axis],
                               atol=5e-9, rtol=1e-7)


@pytest.mark.parametrize('df', [False, True])
@pytest.mark.parametrize('unrestricted', [False, True])
@pytest.mark.parametrize('xc', ['PBE', 'r2SCAN'])
def test_gradient(df, unrestricted, xc):
    lib.num_threads(2)
    mol = gto.M(atom=('O 0 0 0; H 0 0 1' if unrestricted else
                      'O 0 0 0; H 0 0 1; H 0 1 0'),
                spin=int(unrestricted), basis='sto-3g', verbose=0)
    mf = (UHFDFT if unrestricted else RHFDFT)(mol, xc=xc)
    if df:
        mf = mf.density_fit()
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-8
    mf.grids.level = 1
    mf.kernel()
    original = (rks.get_veff, uks.get_veff, rks.grids_response_cc,
                rks.get_vxc_full_response, uks.get_vxc_full_response)
    drivers = [mf.nuc_grad_method() for _ in range(2)]
    assert all(g.grid_response_backend == 'becke_omp' for g in drivers)
    drivers[0].grid_response_backend = 'pyscf_default'
    values = [g.kernel() for g in drivers]
    error = np.max(np.abs(values[0]-values[1]))
    print(f'{xc} UHF={unrestricted} DF={df}: max gradient delta {error:.3e} Ha/Bohr')
    assert error < 1e-9
    for name in ('de_skel', 'de_nvar'):
        np.testing.assert_allclose(getattr(drivers[0], name), getattr(drivers[1], name),
                                   atol=1e-9, rtol=0)
    dm = mf.mf_hf.make_rdm1()
    a, b = [g.get_veff(dm=dm) for g in drivers]
    for tag in ('exc1_grid', 'aux') if df else ('exc1_grid',):
        np.testing.assert_allclose(getattr(a, tag), getattr(b, tag), atol=1e-10, rtol=0)
    np.testing.assert_allclose(drivers[1].kernel(atmlst=[1]), values[0][[1]],
                               atol=1e-9, rtol=0)
    assert original == (rks.get_veff, uks.get_veff, rks.grids_response_cc,
                        rks.get_vxc_full_response, uks.get_vxc_full_response)
    assert drivers[0].grid_response_backend == 'pyscf_default'
    drivers[1].grid_response_backend = 'invalid'
    with pytest.raises(ValueError, match='grid_response_backend'):
        drivers[1].get_veff(dm=dm)


def test_unsupported_partition():
    mol = gto.M(atom='He 0 0 0', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.becke_scheme = gen_grid.stratmann
    with pytest.raises(NotImplementedError, match='original_becke'):
        list(gr.grids_response_becke(grids))


def test_concurrent_grid_calls():
    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.atom_grid = (20, 50)
    reference = list(rks.grids_response_becke(grids))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: list(gr.grids_response_becke(grids, 2)), range(2)))
    for result in results:
        for a, b in zip(result, reference):
            np.testing.assert_allclose(a[2], b[2], atol=3e-12, rtol=1e-11)


def test_fd_and_exception_isolation():
    lib.num_threads(2)
    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto-3g', verbose=0)

    def target(mol):
        mf = RHFDFT(mol, xc='PBE').density_fit()
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        mf.grids.level = 3
        mf.kernel()
        return mf

    mf = target(mol)
    g = mf.nuc_grad_method()
    g.cphf_conv_tol = 1e-11
    g.grid_response_backend = 'becke_omp'
    analytic = g.kernel()[1, 2]
    h = 1e-4
    energies = []
    for sign in (-1, 1):
        xyz = mol.atom_coords()
        xyz[1, 2] += sign*h
        energies.append(target(mol.set_geom_(xyz, unit='Bohr', inplace=False)).e_tot)
    error = abs((energies[1]-energies[0])/(2*h)-analytic)
    print(f'DF water PBE selected FD error {error:.3e} Ha/Bohr')
    assert error < 1e-7
    original = rks.grids_response_cc
    mf.grids.becke_scheme = gen_grid.stratmann
    with pytest.raises(NotImplementedError, match='original_becke'):
        g.get_veff(dm=mf.mf_hf.make_rdm1())
    assert rks.grids_response_cc is original
    mf.grids.becke_scheme = gen_grid.original_becke
    g.grid_response_backend = 'pyscf_default'
    assert np.isfinite(g.get_veff(dm=mf.mf_hf.make_rdm1())).all()


def test_disabled_and_missing_native(monkeypatch):
    mol = gto.M(atom='He 0 0 0', verbose=0)
    g = RHFDFT(mol, xc='PBE').nuc_grad_method()
    calls = []

    def unavailable():
        raise RuntimeError('extension unavailable')

    def provider(*args):
        calls.append(1)
        return 'native'

    monkeypatch.setattr(gr, 'native', unavailable)
    assert g.grid_response_backend == 'becke_omp'
    g.grid_response_backend = 'pyscf_default'
    assert gr.get_veff(g, provider) == 'native'
    g.grid_response_backend = 'becke_omp'
    with pytest.raises(RuntimeError, match='extension unavailable'):
        gr.get_veff(g, provider)
    g.grid_response = False
    assert gr.get_veff(g, provider) == 'native'
    assert g.grid_response_backend_used == 'inactive'
    assert len(calls) == 2
