"""GDF-consistent one-electron derivatives for the GPU gradient route."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad
from pyscf_pbc_dcdft.df.gdf.local_part2 import LocalPart2Derivative
from pyscf_pbc_dcdft.grad.gdf_one_electron import (
    build_gdf_one_electron_derivatives as build_host_derivatives,
)

from ..df.gdf.local_part1_derivative import NativeGDFLocalPart1Derivative
from .nonlocal_pseudo import contract_nonlocal_pseudo_batch


@dataclass(frozen=True)
class GDFOneElectronDerivatives:
    """Device one-electron matrices and explicit host-generation timing."""

    cell: object
    kpts: np.ndarray
    hcore_by_atom: cp.ndarray | None
    overlap_rows: cp.ndarray
    host_integral_seconds: float
    local_part1: NativeGDFLocalPart1Derivative | None = None
    density_contracted_nonlocal: bool = False
    native_hcore_builder: object | None = None

    def hcore(self, atom_id: int) -> cp.ndarray:
        """Return ``dh_core/dR_A`` with ``(xyz,k,AO,AO)`` ordering."""

        if self.hcore_by_atom is None:
            start = 3 * int(atom_id)
            return self.hcore_batch(start, start + 3).reshape(
                3, len(self.kpts), self.cell.nao_nr(), self.cell.nao_nr()
            )
        return self.hcore_by_atom[int(atom_id)]

    def hcore_batch(self, start: int, stop: int) -> cp.ndarray:
        """Return a bounded Cartesian H1 batch as ``(nset,k,AO,AO)``."""

        component_count = 3 * self.cell.natm
        if not 0 <= int(start) < int(stop) <= component_count:
            raise ValueError(
                f"invalid H1 component range [{start}, {stop}) for "
                f"{component_count} components"
            )
        if self.hcore_by_atom is not None:
            return self.hcore_by_atom.reshape(
                component_count,
                len(self.kpts),
                self.cell.nao_nr(),
                self.cell.nao_nr(),
            )[start:stop]
        if self.native_hcore_builder is None:
            raise RuntimeError("no H1 matrix provider is available")
        return self.native_hcore_builder.build(start, stop)

    def overlap(self, atom_id: int, axis: int) -> cp.ndarray:
        """Return the Hermitian full overlap derivative ``(k,AO,AO)``."""

        p0, p1 = np.asarray(self.cell.aoslice_by_atom(), dtype=int)[
            int(atom_id), 2:
        ]
        output = cp.zeros(
            (len(self.kpts), self.cell.nao_nr(), self.cell.nao_nr()),
            dtype=self.overlap_rows.dtype,
        )
        output[:, p0:p1, :] = self.overlap_rows[
            :, int(axis), p0:p1, :
        ]
        return output + output.transpose(0, 2, 1).conj()

    def contract_local_part1(self, density) -> cp.ndarray:
        """Contract native local part 1, or return zero in reference mode."""

        if self.local_part1 is None:
            return cp.zeros((self.cell.natm, 3), dtype=cp.float64)
        return self.local_part1.contract(density)

    def contract_nonlocal(self, density) -> cp.ndarray:
        """Contract the exact analytic nonlocal GTH derivative with a density.

        Native GDF gradients do not need the full
        ``(atom,xyz,k,AO,AO)`` nonlocal derivative tensor.  GPU4PySCF's
        density-contracted primitive uses exact projector half-integrals at
        Gamma and avoids both the reciprocal-mesh approximation and the large
        atom-by-atom matrix materialization.
        """

        if not self.density_contracted_nonlocal:
            return cp.zeros((self.cell.natm, 3), dtype=cp.float64)
        kpts = None
        if len(self.kpts) != 1 or np.max(np.abs(self.kpts)) > 1.0e-12:
            kpts = self.kpts
        value = vppnl_nuc_grad(self.cell, cp.asarray(density), kpts=kpts)
        return cp.asarray(value, dtype=cp.float64) / len(self.kpts)

    def contract_pseudo_batch(self, densities) -> cp.ndarray:
        """Contract local-part-1 and nonlocal PP for several densities."""

        batch = cp.asarray(densities)
        if batch.ndim == 3:
            batch = batch[None]
        if self.local_part1 is None:
            local = cp.zeros(
                (len(batch), self.cell.natm, 3),
                dtype=cp.float64,
            )
        else:
            local = self.local_part1.contract_batch(batch)
        if self.density_contracted_nonlocal:
            points = None
            if len(self.kpts) != 1 or np.max(np.abs(self.kpts)) > 1.0e-12:
                points = self.kpts
            nonlocal_value = contract_nonlocal_pseudo_batch(
                self.cell,
                batch,
                kpts=points,
            ) / len(self.kpts)
        else:
            nonlocal_value = cp.zeros_like(local)
        return local + nonlocal_value


def build_gdf_one_electron_derivatives(
    cell,
    kpts,
    *,
    native_local_part1=False,
):
    """Build matrix H1 terms and an optional native local-part-1 contractor."""

    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    if native_local_part1:
        return _build_native_derivatives(cell, points)
    start = perf_counter()
    host = build_host_derivatives(cell, points)
    hcore_host = np.stack(
        [host.hcore(atom_id) for atom_id in range(cell.natm)]
    )
    host_seconds = perf_counter() - start
    # Overlap derivative has a native GPU4PySCF integral route and therefore
    # does not cross the host boundary used by the GDF pseudopotential H1.
    overlap_rows = -int1e.int1e_ipovlp(cell, points)
    return GDFOneElectronDerivatives(
        cell=cell,
        kpts=points,
        hcore_by_atom=cp.asarray(hcore_host),
        overlap_rows=overlap_rows,
        host_integral_seconds=host_seconds,
    )


def _build_native_derivatives(cell, points):
    """Build bounded H1 providers for native GPU4PySCF GDF mode."""

    start = perf_counter()
    local_part2 = LocalPart2Derivative(cell, points)
    kinetic_rows = int1e.int1e_ipkin(cell, points)
    hcore_builder = NativeHCoreBatchBuilder(
        cell=cell,
        points=points,
        kinetic_rows=kinetic_rows,
        local_part2=local_part2,
    )
    host_seconds = perf_counter() - start

    overlap_rows = -int1e.int1e_ipovlp(cell, points)
    return GDFOneElectronDerivatives(
        cell=cell,
        kpts=points,
        hcore_by_atom=None,
        overlap_rows=overlap_rows,
        host_integral_seconds=host_seconds,
        local_part1=NativeGDFLocalPart1Derivative(cell, points),
        density_contracted_nonlocal=True,
        native_hcore_builder=hcore_builder,
    )


@dataclass(frozen=True)
class NativeHCoreBatchBuilder:
    """Generate kinetic plus local-part-2 H1 matrices by bounded ranges."""

    cell: object
    points: np.ndarray
    kinetic_rows: cp.ndarray
    local_part2: object

    def build(self, start: int, stop: int) -> cp.ndarray:
        matrices = []
        for component in range(int(start), int(stop)):
            atom_id, axis = divmod(component, 3)
            kinetic = _ao_center_derivative(
                self.cell, self.kinetic_rows, atom_id, axis
            )
            local_part2 = cp.asarray(
                self.local_part2.matrix(atom_id, axis)
            )
            matrices.append(kinetic + local_part2)
        return cp.stack(matrices)


def _full_ao_center_derivatives(cell, points, row_derivatives):
    """Expand bra AO derivatives into Hermitian nuclear derivatives."""

    rows = cp.asarray(row_derivatives)
    output = cp.zeros(
        (
            cell.natm,
            3,
            len(points),
            cell.nao_nr(),
            cell.nao_nr(),
        ),
        dtype=rows.dtype,
    )
    slices = np.asarray(cell.aoslice_by_atom(), dtype=int)
    for atom_id in range(cell.natm):
        p0, p1 = slices[atom_id, 2:]
        for axis in range(3):
            row = rows[:, axis]
            output[atom_id, axis, :, p0:p1, :] -= row[:, p0:p1, :]
            output[atom_id, axis, :, :, p0:p1] -= row[
                :, p0:p1, :
            ].transpose(0, 2, 1).conj()
    return output


def _ao_center_derivative(cell, row_derivatives, atom_id, axis):
    """Expand one compact AO-row derivative into a Hermitian H1 matrix."""

    rows = cp.asarray(row_derivatives)
    p0, p1 = np.asarray(cell.aoslice_by_atom(), dtype=int)[int(atom_id), 2:]
    row = rows[:, int(axis)]
    output = cp.zeros(
        (len(rows), cell.nao_nr(), cell.nao_nr()), dtype=rows.dtype
    )
    output[:, p0:p1, :] -= row[:, p0:p1, :]
    output[:, :, p0:p1] -= row[:, p0:p1, :].transpose(0, 2, 1).conj()
    return output
