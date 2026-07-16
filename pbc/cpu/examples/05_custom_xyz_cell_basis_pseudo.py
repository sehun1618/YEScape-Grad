"""Read an XYZ file and choose cell, basis, pseudo, cutoff, and k-mesh.

XYZ coordinates are interpreted as Angstrom.  ``--cell-bohr`` defines three
cubic lattice lengths in Bohr.  The initial completion scope is restricted
closed-shell; charge and spin are fixed to zero in this compact example.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pyscf.lib import param
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


def read_xyz(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    natm = int(lines[0].strip())
    atoms = []
    for line in lines[2 : 2 + natm]:
        symbol, x, y, z = line.split()[:4]
        # XYZ convention is Angstrom. Convert explicitly because the cell and
        # lattice below use Bohr.
        coordinate_bohr = np.asarray((float(x), float(y), float(z))) / param.BOHR
        atoms.append((symbol, coordinate_bohr))
    if len(atoms) != natm:
        raise ValueError(f"{path} contains {len(atoms)} atoms; expected {natm}")
    return atoms


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("xyz", type=Path)
parser.add_argument("--cell-bohr", type=float, nargs=3, default=(8.0, 8.0, 8.0))
parser.add_argument("--basis", default="gth-szv")
parser.add_argument("--pseudo", default="gth-hf")
parser.add_argument("--cutoff-ha", type=float, default=30.0)
parser.add_argument("--kmesh", type=int, nargs=3, default=(1, 1, 1))
args = parser.parse_args()

cell = gto.Cell()
cell.atom = read_xyz(args.xyz)
cell.a = np.diag(args.cell_bohr)
cell.unit = "Bohr"
cell.basis = args.basis
cell.pseudo = args.pseudo
cell.ke_cutoff = args.cutoff_ha
cell.charge = 0
cell.spin = 0
cell.verbose = 3
cell.build()

if tuple(args.kmesh) == (1, 1, 1):
    mf = dft.DCRKS(cell, xc="PBE")
else:
    mf = dft.DCKRKS(cell, kpts=cell.make_kpts(args.kmesh), xc="PBE")
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()

print(f"Energy: {energy:.12f} Ha")
print("Gradient (Ha/Bohr):")
print(gradient)
