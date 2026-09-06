'''CPU GDF source/response with MultiGrid target XC.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import RHFDFT

cell = gto.Cell()
cell.atom = 'O 0 0 0; H 0 1.43 1.10; H 0 -1.43 1.10'
cell.a = np.eye(3) * 7.0
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-hf'
cell.ke_cutoff = 100
cell.precision = 1e-8
cell.build()


# with_j=False: GDF owns target J; MultiGrid owns target XC only.
mf = RHFDFT(cell, xc='PBE').density_fit()
mf = mf.multigrid_numint(with_j=False).run()
gradient = mf.nuc_grad_method().run().de
print('GDF + MultiGrid-XC PBE@RHF energy [Ha]:', mf.e_dft)
print('Gradient [Ha/Bohr]:\n', gradient)


# with_j=True instead delegates fixed-density target J+XC to MultiGrid.
# Source RHF J/K and the CPHF operator still use GDF.
mf = RHFDFT(cell, xc='PBE').density_fit()
mf = mf.multigrid_numint(with_j=True).run()
gradient = mf.nuc_grad_method().run().de
print('GDF + MultiGrid-J+XC PBE@RHF energy [Ha]:', mf.e_dft)
print('Gradient [Ha/Bohr]:\n', gradient)
