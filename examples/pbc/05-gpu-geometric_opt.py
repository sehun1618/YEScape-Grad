'''Pass a GPU PBC gradient scanner to PySCF's geomeTRIC adapter.'''

import numpy as np
from pyscf.geomopt import geometric_solver
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import RHFDFT

cell = gto.Cell()
cell.atom = '''
    O  0.000  0.000  0.000
    H  0.000 -0.810  0.650
    H  0.000  0.735  0.550
'''
cell.a = np.eye(3) * 7.
cell.unit = 'Angstrom'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-hf'
cell.ke_cutoff = 100
cell.precision = 1e-8
cell.verbose = 3
cell.build()


#
# Case 1: build the fixed-cell nuclear-gradient scanner
#
# Replace density_fit by rs_density_fit to use RSDF.  To move only target XC
# to multigrid, add ``mf = mf.multigrid_numint()`` before creating the scanner.
#
mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
scanner = mf.nuc_grad_method().set(
    cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8,
    cphf_max_cycle=200,
    cphf_refinement_solver='minres',
).as_scanner()


#
# Case 2: optimize nuclear coordinates through PySCF's geomeTRIC adapter
#
# The lattice remains fixed.  This example does not calculate stress or a cell
# derivative.  The optional ``geometric`` package must be installed.
#
history = []


def record_step(environment):
    '''Record one geomeTRIC engine evaluation.'''
    gradient = np.asarray(environment['gradients'])
    history.append((
        environment['self'].cycle,
        float(environment['energy']),
        float(np.max(np.abs(gradient))),
        float(scanner.z_residual_max_abs),
    ))


converged, optimized_cell = geometric_solver.kernel(
    scanner,
    assert_convergence=False,
    callback=record_step,
    maxsteps=20,
)

print(f'converged = {converged}')
print('cycle  E_PBE@HF(Ha)      max|dE/dR|(Ha/Bohr)  Z residual')
for cycle, energy, gradient_max_abs, residual in history:
    print(f'{cycle:5d}  {energy:16.10f}  '
          f'{gradient_max_abs:20.6e}  {residual:.3e}')
print('optimized coordinates (Angstrom)')
for symbol, coordinate in zip(
        optimized_cell.elements,
        optimized_cell.atom_coords(unit='Angstrom')):
    print(f'{symbol:2s} {coordinate[0]:14.8f} '
          f'{coordinate[1]:14.8f} {coordinate[2]:14.8f}')
