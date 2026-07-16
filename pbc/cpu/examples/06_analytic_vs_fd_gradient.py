"""Opt-in central-FD check of one analytic DC-PBE gradient component.

This example reruns HF at both displaced geometries.  It is a validation
workflow, not the production gradient algorithm.
"""

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


STEP_BOHR = 5.0e-4
ATOM_ID = 1
AXIS = 2
COORDINATES = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])


def calculate(coordinates, *, with_gradient=False):
    cell = gto.Cell(
        atom=[("H", coordinates[0]), ("H", coordinates[1])],
        a=np.eye(3) * 6.0,
        unit="Bohr",
        basis="gth-dzvp",
        pseudo="gth-hf",
        ke_cutoff=20.0,
        verbose=0,
    ).build()
    mf = dft.DCRKS(cell, xc="PBE")
    energy = mf.kernel()
    if with_gradient:
        return energy, mf.nuc_grad_method().kernel()
    return energy


energy, analytic = calculate(COORDINATES, with_gradient=True)
plus = COORDINATES.copy()
minus = COORDINATES.copy()
plus[ATOM_ID, AXIS] += STEP_BOHR
minus[ATOM_ID, AXIS] -= STEP_BOHR
finite_difference = (calculate(plus) - calculate(minus)) / (2.0 * STEP_BOHR)

print(f"Energy: {energy:.12f} Ha")
print(f"Analytic: {analytic[ATOM_ID, AXIS]: .12e} Ha/Bohr")
print(f"FD      : {finite_difference: .12e} Ha/Bohr")
print(
    "Difference: "
    f"{analytic[ATOM_ID, AXIS] - finite_difference: .3e} Ha/Bohr"
)
