"""Small numerical gates for the independent fixed-density energy core."""

from __future__ import annotations

import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import df, dft


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


class FixedDensityEnergyTest(unittest.TestCase):
    """Validate energy algebra, not converged production accuracy."""

    def _assert_consistent(self, mf) -> None:
        energy = mf.kernel()
        result = mf.dc_energy_result
        self.assertTrue(np.isfinite(energy))
        self.assertTrue(mf.converged)
        self.assertLess(
            abs(energy - result.components.reconstructed_total),
            1.0e-10,
        )
        self.assertEqual(result.target_xc, "PBE")

    def test_gamma_fftdf(self) -> None:
        mf = dft.DCRKS(_h2_cell(), xc="PBE")
        self._assert_consistent(mf)
        self.assertEqual(mf.dc_energy_result.metadata["df_backend"], "fftdf")

    def test_nongamma_fftdf(self) -> None:
        cell = _h2_cell()
        kpts = cell.make_kpts([2, 1, 1])
        mf = dft.DCKRKS(cell, kpts=kpts, xc="PBE")
        self._assert_consistent(mf)
        self.assertEqual(mf.dc_energy_result.nkpts, 2)

    def test_gamma_gdf(self) -> None:
        cell = _h2_cell()
        mf = dft.DCRKS(cell, xc="PBE")
        mf.with_df = df.GDF(cell)
        self._assert_consistent(mf)
        self.assertEqual(mf.dc_energy_result.metadata["df_backend"], "gdf")


if __name__ == "__main__":
    unittest.main()
