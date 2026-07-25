"""Homogeneous strain transformations with fixed fractional coordinates."""

from __future__ import annotations

import numpy as np

from .conventions import component_index, strain_basis


def strained_lattice(lattice, component: int | str, amplitude: float) -> np.ndarray:
    """Return a PySCF row-vector lattice under one Voigt strain coordinate."""

    vectors = np.asarray(lattice, dtype=float)
    if vectors.shape != (3, 3):
        raise ValueError(f"lattice must have shape (3, 3), got {vectors.shape}")
    deformation = np.eye(3) + float(amplitude) * strain_basis(component)
    transformed = vectors @ deformation.T
    if np.linalg.det(transformed) <= 0:
        raise ValueError("strain produced a non-right-handed or singular cell")
    return transformed


def deform_cell(
    cell, component: int | str, amplitude: float, *, mesh=None,
    preserve_mesh=True,
):
    """Copy and strain a 3D PySCF cell while holding fractional atoms fixed."""

    if int(getattr(cell, "dimension", 3)) != 3:
        raise NotImplementedError("stress S1 supports three-dimensional cells only")
    if bool(getattr(cell, "space_group_symmetry", False)):
        raise NotImplementedError(
            "disable space_group_symmetry before finite-difference strain"
        )
    if not bool(getattr(cell, "_built", False)):
        raise RuntimeError("build the PySCF cell before applying strain")

    old_lattice = np.asarray(cell.lattice_vectors(), dtype=float)
    new_lattice = strained_lattice(old_lattice, component, amplitude)
    fractional = np.asarray(cell.get_scaled_atom_coords(), dtype=float)
    cartesian = fractional @ new_lattice
    labels = [entry[0] for entry in cell._atom]
    fixed_mesh = None
    if preserve_mesh or mesh is not None:
        fixed_mesh = np.asarray(cell.mesh if mesh is None else mesh, dtype=int)
        if fixed_mesh.shape != (3,) or np.any(fixed_mesh <= 0):
            raise ValueError(f"mesh must contain three positive integers: {fixed_mesh}")

    strained = cell.copy(deep=True)
    strained.a = new_lattice
    strained.atom = [(label, coordinate) for label, coordinate in zip(labels, cartesian)]
    strained.unit = "Bohr"
    strained.fractional = False
    strained.mesh = None if fixed_mesh is None else fixed_mesh.copy()
    strained.build(dump_input=False, parse_arg=False)

    observed_fractional = np.asarray(strained.get_scaled_atom_coords(), dtype=float)
    if not np.allclose(observed_fractional, fractional, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("strained cell did not preserve fractional coordinates")
    if fixed_mesh is not None and not np.array_equal(
        np.asarray(strained.mesh, dtype=int), fixed_mesh
    ):
        raise RuntimeError("strained cell did not preserve the requested integer mesh")
    return strained


def deform_kpoints(cell, strained_cell, kpts) -> np.ndarray:
    """Keep fractional reciprocal coordinates fixed under cell strain."""

    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    scaled = np.asarray(cell.get_scaled_kpts(points), dtype=float)
    transformed = np.asarray(strained_cell.get_abs_kpts(scaled), dtype=float)
    check = np.asarray(strained_cell.get_scaled_kpts(transformed), dtype=float)
    if not np.allclose(check, scaled, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("strained k-points did not preserve fractional coordinates")
    return transformed


def component_label(component: int | str) -> str:
    from .conventions import VOIGT_LABELS

    return VOIGT_LABELS[component_index(component)]
