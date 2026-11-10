"""Check AO reduction against independent XC-potential finite differences."""
import numpy as np
import pytest
from pyscf import dft, gto, lib
from dcforge.cpu.source.xc import grad_xc_fock


@pytest.mark.parametrize('xc', ['LDA,VWN', 'PBE', 'r2SCAN'])
@pytest.mark.parametrize('unrestricted', [False, True])
def test_atomic_reduction(xc, unrestricted):
    lib.num_threads(4)
    mol = gto.M(
        atom=('N 0 0 0; H 0 -.78 .58; H 0 .82 .63' if unrestricted
              else 'O 0 0 0; H 0 -.78 .58; H 0 .82 .63'),
        spin=int(unrestricted), basis='sto-3g', verbose=0)
    source = (dft.UKS if unrestricted else dft.RKS)(mol, xc=xc)
    source.grids.atom_grid = (20, 50)
    source.grids.prune = None
    source.small_rho_cutoff = 0
    source.grids.build()
    p = source.get_init_guess()
    q = np.random.default_rng(4).normal(size=p.shape) * .01
    q = q + q.swapaxes(-1, -2)
    ni = source._numint
    potential = ni.nr_uks if unrestricted else ni.nr_rks

    def scalar(displacement, moving_grid):
        coords = mol.atom_coords()
        coords[2, 2] += displacement
        moved = mol.set_geom_(coords, unit='Bohr', inplace=False)
        grids = source.grids
        if moving_grid:
            grids = dft.gen_grid.Grids(moved)
            grids.atom_grid = (20, 50)
            grids.prune = None
            grids.build()
        vxc = potential(moved, grids, xc, p)[2]
        return np.sum(vxc * q.swapaxes(-1, -2))

    for moving_grid in (False, True):
        h = .0005
        values = {k: scalar(k*h, moving_grid) for k in (-2, -1, 1, 2)}
        fd = (values[-2] - 8*values[-1] + 8*values[1] - values[2]) / (12*h)
        results = []
        for backend in ('pyscf_default', 'becke_omp'):
            timings = {}
            kwargs = dict(block_size=128, grid_response=moving_grid,
                          grid_response_backend=backend)
            subset = grad_xc_fock(source, p, q, [2, 0], timings=timings, **kwargs)
            full = grad_xc_fock(source, p, q, [0, 1, 2], **kwargs)
            np.testing.assert_allclose(subset, full[[2, 0]], atol=1e-10, rtol=0)
            np.testing.assert_allclose(subset[0, 2], fd, atol=1e-7, rtol=0)
            assert timings['total'] > 0 and timings['cpu_total'] > 0
            results.append(subset)
        np.testing.assert_allclose(*results, atol=1e-10, rtol=0)
