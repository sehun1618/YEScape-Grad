'''CPU real-Gamma FFTDF and GDF HF-DFT calculations.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import RHFDFT

cell = gto.Cell()
cell.atom = '''
O  0.00  0.00  0.00
H  0.00  1.43  1.10
H  0.00 -1.43  1.10
'''
cell.a = np.eye(3) * 7.0
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-hf'
cell.ke_cutoff = 100
cell.precision = 1e-8
cell.build()


# FFTDF owns source RHF J/K, target J, CPHF, and their derivatives.
mf = RHFDFT(cell, xc='PBE').fft_density_fit().run()
gradient = mf.nuc_grad_method().run().de
print('FFTDF PBE@RHF energy [Ha]:', mf.e_dft)
print('FFTDF gradient [Ha/Bohr]:\n', gradient)


# GDF uses one shared provider for the same parts of the equation.
# auxbasis=None delegates auxiliary-basis selection to PySCF.
mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis=None).run()
gradient = mf.nuc_grad_method().run().de
print('GDF PBE@RHF energy [Ha]:', mf.e_dft)
print('GDF gradient [Ha/Bohr]:\n', gradient)
