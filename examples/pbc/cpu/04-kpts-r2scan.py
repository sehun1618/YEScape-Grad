'''CPU full-BZ sampled-k uniform-grid tau-MGGA@KRHF.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import KRHFDFT

cell = gto.Cell()
cell.atom = 'H 0 0 0; H 0 0 1.4'
cell.a = np.eye(3) * 6.0
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pbe'
cell.precision = 1e-8
cell.mesh = [15, 15, 15]
cell.verbose = 0
cell.build()

kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
mf = KRHFDFT(
    cell,
    kpts=kpts,
    kpts_wrap_around=True,
    xc='R2SCAN',
).run()
mf_grad = mf.nuc_grad_method()
gradient = mf_grad.kernel()

print('target r2SCAN@KRHF energy [Ha]:', mf.e_dft)
print('target r2SCAN@KRHF gradient [Ha/Bohr]:\n', gradient)
print('XC derivative backend:', mf_grad.xc_deriv_backend)

# Other pure LDA/GGA/tau-MGGA targets use the same API, for example
# ``xc='LDA,VWN'``, ``xc='BLYP'``, or ``xc='TPSS'``.  Hybrid/RSH,
# Laplacian-dependent MGGA, and NLC/VV10 targets remain outside this route.
# CPU sampled-k MultiGrid is deliberately unavailable in PySCF 2.14; this
# calculation uses its k-resolved uniform-grid integrator.
