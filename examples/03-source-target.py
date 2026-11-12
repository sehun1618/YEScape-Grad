"""Independent source/target choices; examples, not an accuracy ranking."""

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
# source_xc selects the stationary orbitals and their response kernel.
# xc selects the functional evaluated on those orbitals.
# These pairs illustrate the interface, not a ranking of force accuracy.
source_target_pairs = [
    ('HF', 'r2SCAN'),          # HF source
    ('PBE', 'r2SCAN'),         # GGA source
    ('r2SCAN', 'PBE'),         # Meta-GGA source
    ('PBE0', 'PBE'),           # Global-hybrid source
    ('CAM-B3LYP', 'B3LYP'),    # Range-separated-hybrid source
    ('PBE', 'PBE'),            # Self-consistent limit
]

for source, target in source_target_pairs:
    mf = RNSC(mol, source_xc=source, xc=target).density_fit()
    mf.conv_tol = 1e-8
    mf.conv_tol_grad = 1e-6
    mf.max_cycle = 100
    # HF has no source XC grid. KS sources have their own grid.
    if hasattr(mf.mf_source, 'grids'):
        mf.mf_source.grids.level = 4
    mf.grids.level = 4

    energy = mf.kernel()
    gradient = mf.nuc_grad_method().kernel()

    assert mf.converged and np.isfinite(gradient).all()
    print(f'{source} -> {target}: E={energy:.12f} Ha')
    print('Gradient [Ha/Bohr]:', gradient)
