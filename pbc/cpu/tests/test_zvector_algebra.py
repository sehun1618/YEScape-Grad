"""Unit and small numerical tests for the shared Z-vector core."""

from __future__ import annotations

import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft
from pyscf_pbc_dcdft.grad.orbitals import (
    build_mo_partitions,
    target_fock_ao_blocks,
    target_fock_vo_blocks,
)
from pyscf_pbc_dcdft.grad.zvector import (
    BlockLayout,
    build_real_linear_matrix,
    complex_to_real,
    real_to_complex,
    solve_z_vector,
)


def _h2_cell():
    cell = gto.Cell()
    cell.atom = "H 0 0 0; H 0 0 0.74"
    cell.a = np.eye(3) * 6.0
    cell.unit = "Angstrom"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf"
    cell.ke_cutoff = 30.0
    cell.verbose = 0
    return cell.build()


class RealLinearAlgebraTest(unittest.TestCase):
    def test_complex_real_round_trip(self) -> None:
        vector = np.array([1.0 + 2.0j, -3.0 + 0.25j])
        self.assertTrue(np.allclose(real_to_complex(complex_to_real(vector)), vector))

    def test_block_layout_round_trip(self) -> None:
        blocks = [np.ones((2, 1)), np.arange(4).reshape(2, 2) * (1.0 + 1.0j)]
        layout = BlockLayout.from_blocks(blocks)
        restored = layout.split(layout.flatten(blocks))
        self.assertTrue(all(np.allclose(a, b) for a, b in zip(blocks, restored)))

    def test_materialized_real_linear_operator(self) -> None:
        template = [np.zeros((1, 1), dtype=complex)]

        def operator(blocks):
            # y = 2 z + z* maps (x, y) -> (3x, y), demonstrating why a
            # complex-linear matrix alone is insufficient.
            z = blocks[0]
            return [2.0 * z + z.conj()]

        matrix, _ = build_real_linear_matrix(operator, template)
        self.assertTrue(np.allclose(matrix, np.diag([3.0, 1.0])))


class PhysicalZVectorTest(unittest.TestCase):
    def _run_and_solve(self, mf) -> None:
        mf.kernel()
        partitions = build_mo_partitions(mf.source_scf)
        rhs = target_fock_vo_blocks(target_fock_ao_blocks(mf), partitions)
        result = solve_z_vector(
            mf.source_scf,
            partitions,
            rhs,
            equation="adjoint",
        )
        self.assertLess(result.residual_max_abs, 1.0e-9)
        self.assertTrue(np.isfinite(result.condition_number))
        self.assertGreater(sum(block.size for block in rhs), 0)

    def test_gamma_h2(self) -> None:
        self._run_and_solve(dft.DCRKS(_h2_cell(), xc="PBE"))

    def test_nongamma_h2(self) -> None:
        cell = _h2_cell()
        self._run_and_solve(
            dft.DCKRKS(cell, kpts=cell.make_kpts([2, 1, 1]), xc="PBE")
        )


if __name__ == "__main__":
    unittest.main()
