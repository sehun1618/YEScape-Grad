"""Choose the CPU molecular grid derivative; both routes keep full response."""
import numpy as np
from pyscf import gto, lib
from dcforge.cpu.energy import RHFDFT

lib.num_threads(32)
mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
            basis='def2-svp', verbose=4)
mf = RHFDFT(mol, xc='r2SCAN').density_fit()
mf.grids.level = 4
mf.kernel()

# Case 1: original PySCF grid-response implementation (default).
grad_default = mf.nuc_grad_method()
grad_default.grid_response_backend = 'pyscf_default'
reference = grad_default.kernel()

# Case 2: same weights/derivatives, OpenMP atom-pair contraction.
# Build/install the optional extension first; see docs/cpu_grid_response.md.
# The same option is supported by UHFDFT and by conventional (non-DF) drivers.
grad_omp = mf.nuc_grad_method()
grad_omp.grid_response_backend = 'becke_omp'
optimized = grad_omp.kernel()
print('max gradient difference [Ha/Bohr]:', np.max(np.abs(optimized-reference)))
