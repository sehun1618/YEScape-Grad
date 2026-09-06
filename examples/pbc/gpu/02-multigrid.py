'''GPU GDF source/response with r2SCAN MultiGrid target XC.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import RHFDFT

cell = gto.Cell()
cell.atom = 'O 0 0 0; H 0 1.43 1.10; H 0 -1.43 1.10'
cell.a = np.eye(3) * 7.0
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-hf'
cell.ke_cutoff = 100
cell.precision = 1e-8
cell.build()


# GDF owns source RHF J/K, target J, and CPHF; MultiGrid owns target XC.
mf = RHFDFT(cell, xc='R2SCAN').density_fit()
mf = mf.multigrid_numint(with_j=False).run()
gradient = mf.nuc_grad_method().run().de
print('GPU GDF + MultiGrid-XC r2SCAN@RHF energy [Ha]:', mf.e_dft)
print('Gradient [Ha/Bohr]:\n', gradient)


# Set with_j=True to move only fixed-density target J to MultiGrid as well.
# Exact exchange K is still part of source RHF/CPHF and remains on GDF.
