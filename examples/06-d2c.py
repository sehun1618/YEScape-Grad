'''Molecular D2C-r2SCAN energy and gradient with separate dispersion contributions.'''

# Ref.
# M. Lee et al., "Correcting Dispersion Corrections with Density-Corrected
# DFT", J. Chem. Theory Comput. 20, 7155-7167 (2024).
# https://doi.org/10.1021/acs.jctc.4c00689
# D2C damping parameters: Supporting Information, Table S1 (D2C-DFT column).

from pathlib import Path

import numpy as np
from pyscf import gto

from dcforge.cpu.disp.d2c import evaluate_d2c, get_d2c_parameters
from dcforge.cpu.energy import RNSC

ROOT = Path(__file__).resolve().parents[1]

mol = gto.M(
    atom=str(ROOT / 'tests' / 'structures' / 'water1.xyz'),
    unit='Angstrom',
    basis='def2-svp',
    verbose=0,
)
np.set_printoptions(precision=12, suppress=True)


# D2C uses published target-functional parameters; unsupported XC raises an error.
mf = RNSC(mol, xc='r2SCAN').set_dispersion('d2c').set(
    conv_tol=1e-8,
    conv_tol_grad=1e-6,
)
mf.grids.level = 4
energy = mf.kernel()
mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    cphf_max_cycle=200,
).run()

print('Published D2C parameters')
print(get_d2c_parameters(mf.xc))
print(f'HF-r2SCAN energy       = {mf.e_dft:.15f} Ha')
print(f'D2C dispersion energy        = {mf.e_d2c:.15f} Ha')
print(f'D2C-r2SCAN energy     = {energy:.15f} Ha')
print('HF-r2SCAN gradient [Ha/Bohr]')
print(mf_grad.de - mf_grad.de_d2c)
print('D2C dispersion gradient [Ha/Bohr]')
print(mf_grad.de_d2c)
print('D2C-r2SCAN gradient [Ha/Bohr]')
print(mf_grad.de)


# Check the D2C-only derivative by FD; this geometry-only term needs no HF rerun.
step = 1e-4
coords = mol.atom_coords()
energies = []
for sign in (-1.0, 1.0):
    displaced = mol.copy()
    displaced_coords = coords.copy()
    displaced_coords[1, 2] += sign * step
    displaced.set_geom_(displaced_coords, unit='Bohr')
    energies.append(evaluate_d2c(displaced, mf.xc).energy)

finite_difference = (energies[1] - energies[0]) / (2.0 * step)
assert abs(energy - mf.e_dft - mf.e_d2c) < 1e-14
assert abs(mf_grad.de_d2c[1, 2] - finite_difference) < 1e-9
assert np.max(np.abs(mf_grad.de_d2c.sum(axis=0))) < 1e-12

print(f'D2C analytic dE/dR   = {mf_grad.de_d2c[1, 2]:.12e} Ha/Bohr')
print(f'D2C finite difference = {finite_difference:.12e} Ha/Bohr')
