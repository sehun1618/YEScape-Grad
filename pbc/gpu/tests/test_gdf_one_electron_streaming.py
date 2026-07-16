"""Fast parity tests for bounded native GDF H1 generation."""

import unittest

import cupy as cp
import numpy as np

from gpu4pyscf_pbc_dcdft.grad.gdf_one_electron import (
    NativeHCoreBatchBuilder,
    _full_ao_center_derivatives,
)


class NativeHCoreBatchTest(unittest.TestCase):
    def test_streamed_batches_match_full_reference_tensor(self):
        rng = np.random.default_rng(31)
        cell = _Cell()
        rows = cp.asarray(
            rng.normal(size=(2, 3, 5, 5))
            + 1j * rng.normal(size=(2, 3, 5, 5))
        )
        local = _LocalPart2(rng, natm=2, nkpts=2, nao=5)
        builder = NativeHCoreBatchBuilder(
            cell=cell,
            points=np.zeros((2, 3)),
            kinetic_rows=rows,
            local_part2=local,
        )
        kinetic = _full_ao_center_derivatives(
            cell, np.zeros((2, 3)), rows
        ).reshape(6, 2, 5, 5)
        expected = kinetic + cp.asarray(local.values).reshape(6, 2, 5, 5)
        actual = cp.concatenate(
            [builder.build(0, 2), builder.build(2, 5), builder.build(5, 6)]
        )
        self.assertLess(
            float(cp.max(cp.abs(expected - actual)).get()), 1.0e-13
        )


class _Cell:
    natm = 2

    def nao_nr(self):
        return 5

    def aoslice_by_atom(self):
        return np.asarray([[0, 0, 0, 2], [0, 0, 2, 5]])


class _LocalPart2:
    def __init__(self, rng, *, natm, nkpts, nao):
        raw = rng.normal(size=(natm, 3, nkpts, nao, nao))
        raw = raw + 1j * rng.normal(size=raw.shape)
        self.values = raw + raw.transpose(0, 1, 2, 4, 3).conj()

    def matrix(self, atom_id, axis):
        return self.values[atom_id, axis]


if __name__ == "__main__":
    unittest.main()
