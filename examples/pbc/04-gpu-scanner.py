'''GPU PBC energy/gradient scanner, curve, and selected FD check.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import RHFDFT

_HOH_ANGLE_DEGREES = 104.52


def water_geometry(bond_length):
    '''Return symmetric water coordinates in Angstrom.'''
    half_angle = np.deg2rad(_HOH_ANGLE_DEGREES / 2.)
    y = bond_length * np.sin(half_angle)
    z = bond_length * np.cos(half_angle)
    return [
        ('O', (0., 0., 0.)),
        ('H', (0., -y, z)),
        ('H', (0., y, z)),
    ]


def displaced_cell(cell, atom, axis, displacement):
    '''Return a copied Cell displaced by one Cartesian amount in Bohr.'''
    coordinates = cell.atom_coords(unit='Bohr')
    coordinates[atom, axis] += displacement
    displaced = cell.copy()
    displaced.set_geom_(coordinates, unit='Bohr')
    return displaced


cell = gto.Cell()
cell.atom = water_geometry(.96)
cell.a = np.eye(3) * 7.
cell.unit = 'Angstrom'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-hf'
# The analytic/FD lesson uses a tighter grid than the basic API examples so
# finite-mesh error does not obscure the derivative identity.
cell.ke_cutoff = 300
cell.precision = 1e-8
cell.verbose = 0
cell.build()


#
# Case 1: construct one GDF gradient scanner
#
# Each call reconverges source RHF.  The preceding density is only the next
# SCF initial guess.  The following two replacements select RSDF or XC-only MG.
#
mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
# mf = RHFDFT(cell, xc='PBE').rs_density_fit(auxbasis='gth-szv')
# mf = mf.multigrid_numint()
mf.set(conv_tol=1e-12, conv_tol_grad=1e-10, max_cycle=100)

scanner = mf.nuc_grad_method().set(
    cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8,
    cphf_max_cycle=200,
    cphf_refinement_solver='minres',
    # For a large matrix-free Z solve, a previous response can be transported
    # with the old/new AO cross-overlap.  MO phases and occupied/virtual
    # rotations are therefore handled explicitly.  The current physical
    # residual must still improve on the diagonal-gap guess.  This small-water
    # example uses the dense/CPHF oracle and reports a deliberate fallback.
    z_warm_start=True,
    z_warm_start_min_subspace_overlap=.5,
).as_scanner()


#
# Case 2: use the scanner to build an energy/gradient curve
#
records = []
for bond_length in (.90, .96, 1.02):
    energy, gradient = scanner(water_geometry(bond_length))
    records.append((
        bond_length,
        energy,
        gradient[1, 2],
        scanner.base.cycles,
        scanner.z_residual_max_abs,
        scanner.z_warm_start_metadata['used'],
    ))

minimum = min(record[1] for record in records)
print('R_OH(A)  E_PBE@HF(Ha)     Delta(mEh)  g(H1,z)(Ha/Bohr)  cycles  '
      'Z residual  warm')
for bond_length, energy, component, cycles, residual, warm in records:
    print(f'{bond_length:7.3f}  {energy:16.10f}  '
          f'{(energy - minimum) * 1e3:10.4f}  '
          f'{component:17.9e}  {cycles:6d}  {residual:.3e}  {warm}')


#
# Case 3: compare one analytic component with relaxed-source central FD
#
# Plus and minus geometries each reconverge source RHF.  This is not a frozen-
# density derivative and does not run target RKS SCF.
#
step = 1e-4  # Bohr
reference = cell.copy()
reference.set_geom_(water_geometry(.96), unit='Angstrom')
plus = displaced_cell(reference, atom=1, axis=2, displacement=step)
minus = displaced_cell(reference, atom=1, axis=2, displacement=-step)

energy_scanner = scanner.base
energy_plus = energy_scanner(plus)
energy_minus = energy_scanner(minus)
energy, gradient = scanner(reference)
finite_difference = (energy_plus - energy_minus) / (2. * step)
analytic = gradient[1, 2]

print('\nRelaxed-source central finite difference')
print(f'reference energy = {energy:.12f} Ha')
print(f'analytic dE/dR  = {analytic:.12e} Ha/Bohr')
print(f'central FD      = {finite_difference:.12e} Ha/Bohr')
print(f'absolute error  = {abs(analytic - finite_difference):.3e} Ha/Bohr')
print('term-ledger residual = '
      f'{np.max(np.abs(scanner.de - scanner.de_terms)):.3e} Ha/Bohr')
