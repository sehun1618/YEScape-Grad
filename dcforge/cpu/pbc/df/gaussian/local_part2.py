"""Analytic derivative of the Gaussian-DF GTH local-potential part 2.

PySCF splits a GTH local pseudopotential into a Coulomb-like ``part1`` and
four short-range Gaussian-polynomial terms, ``part2``.  GDF and RSGDF evaluate
part2 with :class:`pyscf.pbc.df.aft._IntPPBuilder`, not with the uniform FFT grid
used by :mod:`pyscf.pbc.grad.krhf`.  The derivative therefore has to follow
the same three-center integral representation as the Gaussian-DF energy.

For one nucleus ``A`` the derivative contains two conceptually different
pieces:

* an AO-center derivative, obtained from ``int3c1e_ip1``-family integrals;
* a pseudopotential-center derivative, obtained by replacing each fake
  s shell with its normalized p-shell derivative.

Both pieces are genuine analytic integral derivatives.  No displaced
geometry or finite-difference quantity is used in this module.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto, lib
from pyscf.pbc.df import aft, ft_ao
from pyscf.pbc.df.aft import RCUT_THRESHOLD
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.tools import k2gamma

_VALUE_INTORS = {
    1: "int3c1e",
    2: "int3c1e_r2_origk",
    3: "int3c1e_r4_origk",
    4: "int3c1e_r6_origk",
}
_LOWER_INTORS = {
    1: None,
    2: "int3c1e",
    3: "int3c1e_r2_origk",
    4: "int3c1e_r4_origk",
}
_AO_DERIVATIVE_INTORS = {
    1: "int3c1e_ip1",
    2: "int3c1e_ip1_r2_origk",
    3: "int3c1e_ip1_r4_origk",
    4: "int3c1e_ip1_r6_origk",
}


def contract_vpploc_part2_deriv(cell, densities, kpts=None, atmlst=None):
    r'''Contract GTH local-part-2 derivatives with several Gamma densities.

    PySCF's :func:`pyscf.pbc.gto.pseudo.pp_int.vpploc_part2_nuc_grad`
    contracts the ``int3c1e_ip1`` family directly in its periodic C driver.
    Calling that routine once for each density avoids materializing the
    atom-by-AO-square derivative tensors retained by
    :class:`LocalPart2Derivative`.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built periodic cell with a GTH pseudopotential.
        densities : ndarray of shape (nset,nkpts,nao,nao)
            Hermitian AO density sources.

    Kwargs:
        kpts : ndarray of shape (nkpts,3) or None
            Explicit k-points.  PySCF 2.14.0 exposes this contracted
            derivative at Gamma only.
        atmlst : sequence of int or None
            Requested nuclear centers.

    Returns:
        ndarray of shape (nset,len(atmlst),3)
            Density-contracted local-part-2 derivatives in Ha/Bohr.

    Notes:
        This is the production scalar-contraction route.  The class below is
        retained as the independent full-matrix reference needed to inspect
        individual AO-center and pseudopotential-center terms.
    '''
    kpts = np.zeros((1, 3)) if kpts is None else np.asarray(
        kpts, dtype=float).reshape(-1, 3)
    if len(kpts) != 1 or not gamma_point(kpts):
        raise NotImplementedError(
            'contracted local-part-2 derivative requires Gamma point')

    nao = cell.nao_nr()
    batch = np.asarray(densities)
    if batch.ndim == 2:
        batch = batch[None, None]
    elif batch.ndim == 3:
        batch = batch[:, None]
    if batch.shape[1:] != (1, nao, nao):
        raise ValueError(
            'local-part-2 density batch must have shape '
            f'(nset,1,{nao},{nao}), got {batch.shape}')

    if atmlst is None:
        atmlst = range(cell.natm)
    atmlst = [int(atom_id) for atom_id in atmlst]
    if any(atom_id < 0 or atom_id >= cell.natm for atom_id in atmlst):
        raise ValueError('atom index is outside the periodic cell')

    output = np.asarray([
        pp_int.vpploc_part2_nuc_grad(cell, density[0], kpts)
        for density in batch
    ])
    return output[:, atmlst]


class LocalPart2Derivative:
    """Reusable Gaussian-DF ``pp_loc_part2`` derivative evaluator.

    Parameters
    ----------
    cell
        Built PySCF periodic cell with a GTH pseudopotential.
    kpts
        Explicit sampled k-points in reciprocal Bohr units.

    Notes
    -----
    The expensive three-center integral tensors are built once.  Calling
    :meth:`matrix` for all atoms and Cartesian directions only performs AO
    slicing and small tensor contractions.  Production density contractions
    use :func:`contract_vpploc_part2_deriv`; this class remains the explicit
    AO-matrix reference.
    """

    def __init__(self, cell, kpts):
        self.cell = cell
        self.kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
        self.nkpts = len(self.kpts)
        self.nao = int(cell.nao_nr())
        self._builder = _build_intpp_builder(cell, self.kpts)
        self._components = tuple(self._build_component(index) for index in range(1, 5))

    def matrix(self, atom_id: int, axis: int) -> np.ndarray:
        """Return ``d V_loc,2(k) / d R[A,axis]`` for every sampled k-point."""

        atom_id = _validated_index("atom_id", atom_id, self.cell.natm)
        axis = _validated_index("axis", axis, 3)
        output = np.zeros(
            (self.nkpts, self.nao, self.nao), dtype=np.complex128
        )
        for component in self._components:
            output += self._ao_center_term(component, atom_id, axis)
            output += self._potential_center_term(component, atom_id, axis)
        return output

    def contract_batch(self, densities, atmlst=None) -> np.ndarray:
        '''Contract cached local-part-2 primitives with several densities.'''
        batch = np.asarray(densities)
        if batch.ndim == 2:
            batch = batch[None, None]
        elif batch.ndim == 3:
            batch = batch[:, None] if self.nkpts == 1 else batch[None]
        expected = (self.nkpts, self.nao, self.nao)
        if batch.shape[1:] != expected:
            raise ValueError(
                f'density batch must end in {expected}, got {batch.shape}')
        if atmlst is None:
            atmlst = range(self.cell.natm)
        atmlst = [int(atom_id) for atom_id in atmlst]
        output = np.zeros((len(batch), len(atmlst), 3), dtype=float)
        for position, atom_id in enumerate(atmlst):
            for axis in range(3):
                output[:, position, axis] = lib.einsum(
                    'kij,skji->s', self.matrix(atom_id, axis), batch).real
        return output

    def _build_component(self, index):
        s_fake = pp_int.fake_cell_vloc(self.cell, index)
        s_fake.precision = pp_int.EPS_PPL
        if s_fake.nbas == 0:
            zeros = np.zeros(
                (self.nkpts, 3, self.nao, self.nao), dtype=np.complex128
            )
            return {
                'power_index': index - 1,
                'ao_derivative': zeros,
                'center_value': np.zeros(
                    (self.nkpts, self.nao, self.nao, 0), dtype=np.complex128
                ),
                'center_lower': None,
                'shell_atoms': np.zeros(0, dtype=int),
                'shell_exponents': np.zeros(0),
                'auxiliary_locations': np.zeros(1, dtype=int),
            }

        ao_derivative = _sum_auxiliary(
            self._builder,
            _AO_DERIVATIVE_INTORS[index],
            s_fake,
            components=3,
        )
        p_fake, shell_atoms, shell_exponents = _derivative_fake_cell(
            self.cell, index
        )
        center_value = _integrals_by_auxiliary(
            self._builder,
            _VALUE_INTORS[index],
            p_fake,
            components=1,
        )
        lower_name = _LOWER_INTORS[index]
        center_lower = (
            None
            if lower_name is None
            else _integrals_by_auxiliary(
                self._builder, lower_name, p_fake, components=1
            )
        )
        return {
            'power_index': index - 1,
            'ao_derivative': ao_derivative,
            'center_value': center_value,
            'center_lower': center_lower,
            'shell_atoms': shell_atoms,
            'shell_exponents': shell_exponents,
            'auxiliary_locations': np.asarray(
                p_fake.ao_loc_nr(), dtype=int),
        }

    def _ao_center_term(
        self, component, atom_id, axis):
        p0, p1 = np.asarray(self.cell.aoslice_by_atom(), dtype=int)[atom_id, 2:]
        row = component['ao_derivative'][:, axis]
        output = np.zeros_like(row)

        # ``ip1`` differentiates the bra AO.  Nuclear translation carries the
        # opposite sign; Hermiticity supplies the corresponding ket term.
        output[:, p0:p1, :] -= row[:, p0:p1, :]
        output[:, :, p0:p1] -= row[:, p0:p1, :].transpose(0, 2, 1).conj()
        return output

    def _potential_center_term(
        self, component, atom_id, axis):
        output = np.zeros(
            (self.nkpts, self.nao, self.nao), dtype=np.complex128
        )
        for shell, parent_atom in enumerate(component['shell_atoms']):
            if int(parent_atom) != atom_id:
                continue
            auxiliary = int(component['auxiliary_locations'][shell]) + axis
            block = (
                2.0
                * float(component['shell_exponents'][shell])
                * component['center_value'][:, :, :, auxiliary]
            )
            if component['center_lower'] is not None:
                block -= (
                    2.0
                    * component['power_index']
                    * component['center_lower'][:, :, :, auxiliary]
                )
            output += block
        return output


def _build_intpp_builder(cell, kpts):
    """Initialize PySCF's integral PP builder without evaluating a value."""

    builder = aft._IntPPBuilder(cell, kpts)
    builder.bvk_kmesh = k2gamma.kpts_to_kmesh(cell, builder.kpts)
    builder.rs_cell = ft_ao._RangeSeparatedCell.from_cell(
        cell, builder.ke_cutoff, RCUT_THRESHOLD, verbose=0
    )

    fake_cells = {}
    for index in range(1, 5):
        fake = pp_int.fake_cell_vloc(cell, index)
        fake.precision = pp_int.EPS_PPL
        if fake.nbas:
            fake_cells[index] = fake
    if fake_cells:
        cutoff = builder._estimate_rcut_3c1e(builder.rs_cell, fake_cells)
        supercell = ft_ao.ExtendedMole.from_cell(
            builder.rs_cell, builder.bvk_kmesh, cutoff.max(), 0
        )
        builder.supmol = supercell.strip_basis(cutoff + 1.0)
    return builder


def _derivative_fake_cell(cell, index: int):
    """Replace part2 fake s shells by normalized Cartesian-vector p shells."""

    s_fake = pp_int.fake_cell_vloc(cell, index)
    s_fake.precision = pp_int.EPS_PPL
    coordinate_size = int(s_fake.natm) * 3
    environment = [np.asarray(s_fake._env[:coordinate_size]).copy()]
    basis = []
    shell_atoms = []
    shell_exponents = []
    pointer = coordinate_size

    for shell in np.asarray(s_fake._bas, dtype=np.int32):
        copied = shell.copy()
        atom_id = int(copied[gto.ATOM_OF])
        exponent = float(s_fake._env[int(copied[gto.PTR_EXP])])
        s_coefficient = float(s_fake._env[int(copied[gto.PTR_COEFF])])
        copied[gto.ANG_OF] = 1
        copied[gto.NPRIM_OF] = 1
        copied[gto.NCTR_OF] = 1
        copied[gto.KAPPA_OF] = 0
        copied[gto.PTR_EXP] = pointer
        copied[gto.PTR_COEFF] = pointer + 1
        basis.append(copied)
        environment.append(np.asarray([exponent, s_coefficient / np.sqrt(3.0)]))
        shell_atoms.append(atom_id)
        shell_exponents.append(exponent)
        pointer += 2

    fake = cell.copy(deep=False)
    fake._atm = s_fake._atm.copy()
    fake._bas = np.asarray(basis, dtype=np.int32).reshape(-1, gto.BAS_SLOTS)
    fake._env = np.hstack(environment).astype(float, copy=False)
    fake.precision = pp_int.EPS_PPL
    return fake, np.asarray(shell_atoms, dtype=int), np.asarray(shell_exponents)


def _integrals_by_auxiliary(builder, intor, auxcell, *, components: int):
    nao = int(builder.cell.nao_nr())
    nkpts = len(builder.kpts)
    naux = int(auxcell.nao_nr())
    if naux == 0:
        return np.zeros((nkpts, nao, nao, 0), dtype=np.complex128)
    kernel = builder.gen_int3c_kernel(
        intor, "s1", comp=components, j_only=True, auxcell=auxcell
    )
    real, imaginary = kernel()
    values = np.asarray(real) + (
        0.0 if imaginary is None else 1j * np.asarray(imaginary)
    )
    if components == 1:
        return values.reshape(nkpts, nao, nao, naux)
    return values.reshape(nkpts, components, nao, nao, naux)


def _sum_auxiliary(builder, intor, auxcell, *, components: int):
    values = _integrals_by_auxiliary(
        builder, intor, auxcell, components=components
    )
    return np.sum(values, axis=-1)


def _validated_index(name: str, value: int, upper_bound: int) -> int:
    result = int(value)
    if not 0 <= result < int(upper_bound):
        raise IndexError(f"{name}={result} is outside [0, {upper_bound})")
    return result
