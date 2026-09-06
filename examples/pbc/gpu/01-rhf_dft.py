'''GPU real-Gamma FFTDF and GDF HF-DFT calculations.'''

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


mf = RHFDFT(cell, xc='PBE').fft_density_fit().run()
gradient = mf.nuc_grad_method().run().de
print('GPU FFTDF PBE@RHF energy [Ha]:', mf.e_dft)
print('Gradient [Ha/Bohr]:\n', gradient)


mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis=None).run()
gradient = mf.nuc_grad_method().run().de
print('GPU GDF PBE@RHF energy [Ha]:', mf.e_dft)
print('Gradient [Ha/Bohr]:\n', gradient)
