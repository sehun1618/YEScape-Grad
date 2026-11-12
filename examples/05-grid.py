"""Separate source/target quadratures and CPU grid-derivative backends."""

import numpy as np
from pyscf import gto

from yescape_grad.cpu.energy import RNSC

mol = gto.M(
    atom='''
        O  0.000   0.000   0.000
        H  0.000  -0.757   0.587
        H  0.000   0.757   0.587
    ''',
    basis='def2-svp',
    verbose=3,
)
mf = RNSC(mol, source_xc='PBE', xc='r2SCAN').density_fit()
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-6
mf.max_cycle = 100

#
# 1. Independent source and target grids
#
# Source grid defines both the stationary source and its response.
mf.mf_source.grids.level = 4
# Target quadrature can be selected independently, before evaluation.
mf.grids.atom_grid = (99, 590)
mf.grids.prune = None
energy = mf.kernel()
# To change the source grid later, reset/rebuild it and rerun source SCF.
# Do not reuse source orbitals as stationary for a different source grid.

#
# 2. Same quadratures, different CPU partition-derivative backends
#
grad_default = mf.nuc_grad_method().set(
    grid_response_backend='pyscf_default',
)
reference = grad_default.kernel()

grad_omp = mf.nuc_grad_method().set(
    grid_response_backend='becke_omp',
)
optimized = grad_omp.kernel()

np.testing.assert_allclose(optimized, reference, atol=1e-9, rtol=0)
print('Max backend difference [Ha/Bohr]:',
      np.max(np.abs(optimized - reference)))
