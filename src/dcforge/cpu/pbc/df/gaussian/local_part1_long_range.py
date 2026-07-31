"""Fourier long-range derivative of Gaussian-DF GTH local part 1."""

from __future__ import annotations

import numpy as np
from pyscf import lib
from pyscf.pbc.df import aft, ft_ao
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools import pbc as pbc_tools

from .ft import gen_ft_center_derivative_kernel


class LongRangeLocalPart1:
    """Differentiate the reciprocal-space tail of ``get_pp_loc_part1``.

    The central Gaussian-DF matrix is a contraction

    ``sum_G V_nuc^LR(G) <mu_k|exp(iG.r)|nu_k>``.

    Nuclear motion changes both the AO-pair Fourier transform and the phase of
    the fake nuclear Gaussian.  The two contributions are evaluated
    separately below so that center ownership and signs remain inspectable.
    """

    def __init__(self, builder):
        self.builder = builder
        self.cell = builder.cell
        self.kpts = np.asarray(builder.kpts, dtype=float).reshape(-1, 3)
        self.nkpts = len(self.kpts)
        self.nao = int(self.cell.nao_nr())
        self.nao_pair = self.nao * (self.nao + 1) // 2
        self.fakenuc = aft._fake_nuc(self.cell, with_pseudo=True)
        self.charges = -np.asarray(self.cell.atom_charges(), dtype=float)
        self._ao_slices = np.asarray(self.cell.aoslice_by_atom(), dtype=int)

        self.mesh = np.asarray(builder.mesh, dtype=int)
        self.gv, self.gvbase, self.weights = self.cell.get_Gv_weights(self.mesh)
        self.gxyz = lib.cartesian_prod(
            [np.arange(len(values)) for values in self.gvbase]
        )
        reciprocal = self.cell.reciprocal_vectors()
        self.auxiliary_ft = ft_ao.ft_ao(
            self.fakenuc,
            self.gv,
            None,
            reciprocal,
            self.gxyz,
            self.gvbase,
        )
        self.weighted_potential = self._build_weighted_potential()
        self._basis_i = gen_ft_center_derivative_kernel(
            builder.supmol_ft,
            center="i",
            aosym="s1",
            return_complex=False,
            kpts=self.kpts,
        )
        self._basis_j = gen_ft_center_derivative_kernel(
            builder.supmol_ft,
            center="j",
            aosym="s1",
            return_complex=False,
            kpts=self.kpts,
        )
        self._pair_kernel = builder.supmol_ft.gen_ft_kernel(
            "s2", return_complex=False, kpts=self.kpts, verbose=0
        )

    def matrix(self, atom_id: int, axis: int) -> np.ndarray:
        """Return AO-basis plus pseudopotential-center LR derivatives."""

        atom_id, axis = self._indices(atom_id, axis)
        return self.basis_matrix(atom_id, axis) + self.center_matrix(
            atom_id, axis
        )

    def basis_matrix(self, atom_id: int, axis: int) -> np.ndarray:
        """Differentiate the Fourier AO pair on the translated atom."""

        atom_id, axis = self._indices(atom_id, axis)
        output = self._zeros()
        p0, p1 = self._ao_slices[atom_id, 2:]
        # Two real/imaginary buffers, three Cartesian components, and i/j
        # center kernels coexist: 2 * 2 * 3 float64 values per AO pair.
        block_size = self._block_size(
            96 * self.nao * self.nao * self.nkpts)
        buffer_i = np.empty((2, self.nkpts, 3, block_size, self.nao, self.nao))
        buffer_j = np.empty_like(buffer_i)
        potential_real = self.weighted_potential.real
        potential_imaginary = self.weighted_potential.imag

        for g0, g1 in lib.prange(0, len(self.gv), block_size):
            derivative_i = self._basis_i(
                self.gv[g0:g1],
                self.gxyz[g0:g1],
                self.gvbase,
                np.zeros(3),
                out=buffer_i,
            )
            derivative_j = self._basis_j(
                self.gv[g0:g1],
                self.gxyz[g0:g1],
                self.gvbase,
                np.zeros(3),
                out=buffer_j,
            )
            for kpoint in range(self.nkpts):
                row_real = np.zeros_like(derivative_i[0, kpoint, axis])
                row_imaginary = np.zeros_like(derivative_i[1, kpoint, axis])
                col_real = np.zeros_like(row_real)
                col_imaginary = np.zeros_like(row_imaginary)
                row_real[:, p0:p1, :] -= derivative_i[
                    0, kpoint, axis, :, p0:p1, :
                ]
                row_imaginary[:, p0:p1, :] -= derivative_i[
                    1, kpoint, axis, :, p0:p1, :
                ]
                col_real[:, :, p0:p1] -= derivative_j[
                    0, kpoint, axis, :, :, p0:p1
                ]
                col_imaginary[:, :, p0:p1] -= derivative_j[
                    1, kpoint, axis, :, :, p0:p1
                ]
                output[kpoint] += _contract_complex_fourier_block(
                    potential_real[g0:g1],
                    potential_imaginary[g0:g1],
                    row_real + col_real,
                    row_imaginary + col_imaginary,
                )
        return output

    def center_matrix(self, atom_id: int, axis: int) -> np.ndarray:
        """Differentiate the phase of fake nuclear Gaussian A."""

        atom_id, axis = self._indices(atom_id, axis)
        return self.center_matrices(atom_id)[axis]

    def center_matrices(self, atom_id: int) -> np.ndarray:
        '''Return all three center derivatives from one AO-pair FT traversal.'''
        atom_id, _axis = self._indices(atom_id, 0)
        q0, q1 = np.asarray(self.fakenuc.aoslice_by_atom(), dtype=int)[
            atom_id, 2:
        ]
        if q1 - q0 != 1:
            raise NotImplementedError(
                "Gaussian-DF local part1 expects one fake nuclear AO per atom"
            )
        coulomb = _long_range_coulomb(
            self.cell, float(self.builder.omega), self.mesh, self.gv
        )
        derivative_potential = (
            self.charges[atom_id]
            * (-1j * self.gv.T)
            * self.auxiliary_ft[:, q0][None]
            * coulomb[None]
            * self.weights
        )

        packed = np.zeros(
            (3, self.nkpts, self.nao_pair), dtype=np.complex128
        )
        block_size = self._block_size(16 * self.nao_pair * self.nkpts)
        buffer = np.empty((2, self.nkpts, block_size, self.nao_pair))
        for g0, g1 in lib.prange(0, len(self.gv), block_size):
            ao_pair = self._pair_kernel(
                self.gv[g0:g1],
                self.gxyz[g0:g1],
                self.gvbase,
                np.zeros(3),
                out=buffer,
            )
            for kpoint, (pair_real, pair_imaginary) in enumerate(zip(*ao_pair)):
                potential = derivative_potential[:, g0:g1]
                packed[:, kpoint] += lib.einsum(
                    "xg,gp->xp", potential.real, pair_real)
                packed[:, kpoint] += lib.einsum(
                    "xg,gp->xp", potential.imag, pair_imaginary)
                if not is_zero(self.kpts[kpoint]):
                    packed[:, kpoint] += 1j * lib.einsum(
                        "xg,gp->xp", potential.real, pair_imaginary)
                    packed[:, kpoint] -= 1j * lib.einsum(
                        "xg,gp->xp", potential.imag, pair_real)
        return np.asarray(
            [
                [
                    lib.unpack_tril(value.real)
                    if is_zero(kpoint)
                    else lib.unpack_tril(value)
                    for value, kpoint in zip(values, self.kpts)
                ]
                for values in packed
            ]
        )

    def contract_batch(self, densities, atmlst=None) -> np.ndarray:
        '''Contract requested LR derivatives without retaining full H1.'''
        batch = _density_batch(densities, self.nkpts, self.nao)
        if atmlst is None:
            atmlst = range(self.cell.natm)
        atmlst = [int(atom_id) for atom_id in atmlst]
        output = np.zeros((len(batch), len(atmlst), 3), dtype=float)
        for position, atom_id in enumerate(atmlst):
            output[:, position] += self._contract_basis_batch(
                atom_id, batch)
            output[:, position] += lib.einsum(
                'xkij,skji->sx', self.center_matrices(atom_id), batch).real
        return output

    def _contract_basis_batch(self, atom_id, batch):
        '''Contract all three AO-center directions in one FT traversal.'''
        atom_id, _axis = self._indices(atom_id, 0)
        p0, p1 = self._ao_slices[atom_id, 2:]
        block_size = self._block_size(
            96 * self.nao * self.nao * self.nkpts)
        buffer_i = np.empty(
            (2, self.nkpts, 3, block_size, self.nao, self.nao))
        buffer_j = np.empty_like(buffer_i)
        output = np.zeros((len(batch), 3), dtype=float)
        potential_real = self.weighted_potential.real
        potential_imaginary = self.weighted_potential.imag

        for g0, g1 in lib.prange(0, len(self.gv), block_size):
            common = (
                self.gv[g0:g1], self.gxyz[g0:g1], self.gvbase,
                np.zeros(3))
            derivative_i = self._basis_i(*common, out=buffer_i)
            derivative_j = self._basis_j(*common, out=buffer_j)
            for kpoint in range(self.nkpts):
                for axis in range(3):
                    row = _contract_complex_fourier_block(
                        potential_real[g0:g1],
                        potential_imaginary[g0:g1],
                        -derivative_i[0, kpoint, axis, :, p0:p1],
                        -derivative_i[1, kpoint, axis, :, p0:p1],
                    )
                    column = _contract_complex_fourier_block(
                        potential_real[g0:g1],
                        potential_imaginary[g0:g1],
                        -derivative_j[0, kpoint, axis, :, :, p0:p1],
                        -derivative_j[1, kpoint, axis, :, :, p0:p1],
                    )
                    output[:, axis] += lib.einsum(
                        'pj,sjp->s', row,
                        batch[:, kpoint, :, p0:p1]).real
                    output[:, axis] += lib.einsum(
                        'ip,spi->s', column,
                        batch[:, kpoint, p0:p1]).real
        return output

    def _build_weighted_potential(self) -> np.ndarray:
        coulomb = _long_range_coulomb(
            self.cell, float(self.builder.omega), self.mesh, self.gv
        )
        potential = lib.einsum(
            "a,ga,g->g", self.charges, self.auxiliary_ft, coulomb * self.weights
        )
        if self.cell.dimension == 3 or (
            self.cell.dimension == 2
            and self.cell.low_dim_ft_type != "inf_vacuum"
        ):
            exponents = np.hstack(self.fakenuc.bas_exps())
            potential[0] -= self.charges.dot(np.pi / exponents) * self.weights
        return potential

    def _block_size(self, bytes_per_grid: int) -> int:
        '''Choose a reciprocal block from the actual explicit live set.'''
        remaining = max(
            0., float(self.builder.max_memory) - lib.current_memory()[0])
        budget = max(int(bytes_per_grid), int(remaining * .2e6))
        size = max(1, budget // int(bytes_per_grid))
        if size >= 8:
            size = max(8, size // 8 * 8)
        return min(size, len(self.gv), 4096)

    def _zeros(self):
        return np.zeros((self.nkpts, self.nao, self.nao), dtype=np.complex128)

    def _indices(self, atom_id: int, axis: int):
        atom_id = int(atom_id)
        axis = int(axis)
        if not 0 <= atom_id < self.cell.natm:
            raise IndexError(f"atom_id={atom_id} is outside this cell")
        if not 0 <= axis < 3:
            raise IndexError(f"axis={axis} is outside [0, 3)")
        return atom_id, axis


def _long_range_coulomb(cell, omega: float, mesh, gv):
    """Return the exact LR kernel selected by `_RSNucBuilder`."""

    if cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum":
        full = pbc_tools.get_coulG(cell, np.zeros(3), mesh=mesh, Gv=gv)
        with lib.temporary_env(cell, dimension=3):
            short = pbc_tools.get_coulG(
                cell, np.zeros(3), mesh=mesh, Gv=gv, omega=-omega
            )
        return full - short
    return pbc_tools.get_coulG(
        cell, np.zeros(3), mesh=mesh, Gv=gv, omega=omega
    )


def _contract_complex_fourier_block(
    potential_real,
    potential_imaginary,
    derivative_real,
    derivative_imaginary,
):
    """Contract ``V(G)`` with a complex AO-pair derivative block."""

    real = lib.einsum("g,gij->ij", potential_real, derivative_real)
    real += lib.einsum("g,gij->ij", potential_imaginary, derivative_imaginary)
    imaginary = lib.einsum(
        "g,gij->ij", potential_real, derivative_imaginary
    )
    imaginary -= lib.einsum(
        "g,gij->ij", potential_imaginary, derivative_real
    )
    return real + 1j * imaginary


def _density_batch(densities, nkpts, nao):
    '''Normalize one or more density sets to ``(nset,nkpts,nao,nao)``.'''
    batch = np.asarray(densities)
    if batch.ndim == 2:
        batch = batch[None, None]
    elif batch.ndim == 3:
        batch = batch[:, None] if nkpts == 1 else batch[None]
    if batch.shape[1:] != (nkpts, nao, nao):
        raise ValueError(
            'density batch must have shape '
            f'(nset,{nkpts},{nao},{nao}), got {batch.shape}')
    return batch
