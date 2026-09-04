'''Add published D2C-D4 energy and gradient to GPU molecular PBE@RHF.'''

from pathlib import Path

import numpy as np
from pyscf import gto

from dcforge.cpu.disp.d2c import evaluate_d2c, get_d2c_parameters
from dcforge.gpu.energy import RHFDFT

ROOT = Path(__file__).resolve().parents[2]

mol = gto.M(
    atom=str(ROOT / 'structures' / 'water1.xyz'),
    unit='Angstrom',
    basis='def2-svp',
    verbose=0,
)
np.set_printoptions(precision=12, suppress=True)


#
# D2C is off by default.  set_dispersion('d2c') asks DC-Forge to look up the
# published parameters for the selected target functional.  An unsupported
# target functional raises instead of silently falling back to ordinary D4.
#
mf = RHFDFT(mol, xc='PBE').set_dispersion('d2c').set(
    conv_tol=1e-12,
    conv_tol_grad=1e-10,
)
mf.grids.level = 4
energy = mf.kernel()
mf_grad = mf.nuc_grad_method().set(
    grid_response=True,
    cphf_conv_tol=1e-9,
    cphf_max_cycle=200,
).run()

print('Published D2C parameters')
print(get_d2c_parameters(mf.xc))
print(f'PBE@RHF energy       = {mf.e_dft:.15f} Ha')
print(f'D2C-D4 energy        = {mf.e_d2c:.15f} Ha')
print(f'PBE@RHF+D2C energy   = {energy:.15f} Ha')
print('PBE@RHF gradient [Ha/Bohr]')
print(mf_grad.de - mf_grad.de_d2c)
print('D2C-D4 gradient [Ha/Bohr]')
print(mf_grad.de_d2c)
print('PBE@RHF+D2C gradient [Ha/Bohr]')
print(mf_grad.de)


#
# Small self-check: the D2C analytic derivative for one O-H coordinate must
# match a central finite difference of the D2C energy.  This does not rerun HF
# because D2C contains no density or orbital response.
#
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
