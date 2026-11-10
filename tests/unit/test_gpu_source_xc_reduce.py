"""Check AO reduction against independent XC-potential finite differences."""
import numpy as np
import pytest
from pyscf import gto, lib


@pytest.mark.gpu
@pytest.mark.parametrize('xc', ['LDA,VWN', 'PBE', 'r2SCAN'])
@pytest.mark.parametrize('unrestricted', [False, True])
def test_atomic_reduction(xc, unrestricted):
    from gpu4pyscf import dft
    from gpu4pyscf.dft.numint import NumInt
    import cupy as cp
    from dcforge.gpu.source.xc import grad_xc_fock

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
    p = cp.asarray(source.get_init_guess())
    q = np.random.default_rng(4).normal(size=p.shape) * .01
    q = cp.asarray(q + q.swapaxes(-1, -2))

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
        ni = NumInt()
        potential = ni.nr_uks if unrestricted else ni.nr_rks
        vxc = potential(moved, grids, xc, p)[2]
        return float(cp.sum(vxc * q.swapaxes(-1, -2)).get())

    for moving_grid in (False, True):
        h = .0005
        values = {k: scalar(k*h, moving_grid) for k in (-2, -1, 1, 2)}
        fd = (values[-2] - 8*values[-1] + 8*values[1] - values[2]) / (12*h)
        subset = grad_xc_fock(source, p, q, [2, 0], block_size=128, grid_response=moving_grid).get()
        full = grad_xc_fock(source, p, q, [0, 1, 2], block_size=128, grid_response=moving_grid).get()
        np.testing.assert_allclose(subset, full[[2, 0]], atol=1e-10, rtol=0)
        np.testing.assert_allclose(subset[0, 2], fd, atol=1e-7, rtol=0)
