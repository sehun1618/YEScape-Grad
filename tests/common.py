"""Small shared helpers for molecular energy and gradient validation."""

from pathlib import Path

import numpy as np
from pyscf import gto


ROOT = Path(__file__).resolve().parents[1]
STRUCTURES = ROOT / "structures"

STRUCTURE_STATES = {
    "water1.xyz": (0, 0),
    "oh.xyz": (0, 1),
    "o2_triplet.xyz": (0, 2),
    "no_cation.xyz": (1, 0),
    "nh2_doublet.xyz": (0, 1),
    "ch2_triplet_distorted.xyz": (0, 2),
    "ch3_doublet_distorted.xyz": (0, 1),
    "h2_stretched.xyz": (0, 0),
    "water1_asymmetric_stretch.xyz": (0, 0),
    "water_dimer_shared_proton.xyz": (0, 0),
}


def read_xyz(path):
    """Return PySCF atom records from an XYZ file without its comment line."""
    lines = Path(path).read_text().splitlines()
    natm = int(lines[0])
    atoms = []
    for line in lines[2:2 + natm]:
        symbol, x, y, z = line.split()[:4]
        atoms.append((symbol, (float(x), float(y), float(z))))
    if len(atoms) != natm:
        raise ValueError(f"{path} declares {natm} atoms but contains {len(atoms)}")
    return atoms


def make_mol(name, basis="def2-svp", *, charge=None, spin=None):
    """Build one repository structure with its explicit electronic state."""
    if charge is None or spin is None:
        default_charge, default_spin = STRUCTURE_STATES[name]
        charge = default_charge if charge is None else charge
        spin = default_spin if spin is None else spin
    return gto.M(
        atom=read_xyz(STRUCTURES / name),
        basis=basis,
        charge=charge,
        spin=spin,
        unit="Angstrom",
        verbose=0,
    )


def displaced_mol(mol, atom, axis, displacement):
    """Return a copied molecule displaced in one Bohr Cartesian coordinate."""
    coords = mol.atom_coords().copy()
    coords[atom, axis] += displacement
    return mol.set_geom_(coords, unit="Bohr", inplace=False)


def configure_workflow(mf, *, grid_level=3):
    """Apply the common deterministic accuracy settings used by the tests."""
    mf.verbose = 0
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-9
    mf.grids.level = grid_level
    mf.grids.prune = None
    return mf


def run_workflow(workflow_cls, mol, *, xc="PBE", density_fit=False,
                 auxbasis=None, grid_level=3, dispersion=None):
    """Run one complete source-HF then fixed-density target-energy workflow."""
    mf = workflow_cls(mol, xc=xc)
    if density_fit:
        mf = mf.density_fit(auxbasis=auxbasis)
    configure_workflow(mf, grid_level=grid_level)
    if dispersion is not None:
        mf.set_dispersion(dispersion)
    energy = float(mf.kernel())
    return mf, energy


def finite_difference(workflow_cls, mol, atom, axis, *, step=1e-4,
                      xc="PBE", density_fit=False, auxbasis=None,
                      grid_level=3, dispersion=None):
    """Central difference of the complete relaxed-source HF-DFT energy.

    A fresh workflow is built at both displaced geometries. Thus the source HF
    orbitals are relaxed at each point, matching the analytic Z-vector
    gradient rather than differentiating only a frozen AO density matrix.
    """
    energies = []
    for sign in (1.0, -1.0):
        mol_d = displaced_mol(mol, atom, axis, sign * step)
        _, energy = run_workflow(
            workflow_cls, mol_d, xc=xc, density_fit=density_fit,
            auxbasis=auxbasis, grid_level=grid_level,
            dispersion=dispersion,
        )
        energies.append(energy)
    return (energies[0] - energies[1]) / (2 * step)


def finite_difference_gradient(workflow_cls, mol, **kwargs):
    """Return the complete ``(natm, 3)`` relaxed-source central difference."""
    gradient = np.empty((mol.natm, 3))
    for atom in range(mol.natm):
        for axis in range(3):
            gradient[atom, axis] = finite_difference(
                workflow_cls, mol, atom, axis, **kwargs)
    return gradient


def as_numpy(value):
    """Copy NumPy or CuPy output to a host NumPy array."""
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)
