"""Public-driver lifecycle and restricted-scope regression tests."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import df, dft
from pyscf_pbc_dcdft.dft.state import validate_restricted_occupations


def _h2_cell(*, spin=0):
    cell = gto.Cell()
    cell.atom = "H 0 0 0; H 0 0 0.74"
    cell.a = np.eye(3) * 6.0
    cell.unit = "Angstrom"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf"
    cell.ke_cutoff = 30.0
    cell.spin = spin
    cell.verbose = 0
    return cell.build()


class DriverStateTest(unittest.TestCase):
    def test_provider_modifier_invalidates_completed_state(self):
        method = dft.DCRKS(_h2_cell(), xc="PBE")
        method.kernel()
        self.assertTrue(method.converged)

        method.density_fit("gth-szv")
        self.assertFalse(method.converged)
        self.assertIsNone(method.source_scf)
        with self.assertRaisesRegex(RuntimeError, "run the DC-DFT"):
            method.nuc_grad_method()

    def test_direct_provider_replacement_is_detected_by_signature(self):
        cell = _h2_cell()
        method = dft.DCRKS(cell, xc="PBE")
        method.kernel()
        method.with_df = df.GDF(cell, auxbasis="gth-szv")
        with self.assertRaisesRegex(RuntimeError, "inputs changed"):
            method.make_rdm1()

    def test_geometry_change_requires_new_energy_kernel(self):
        cell = _h2_cell()
        method = dft.DCRKS(cell, xc="PBE")
        method.kernel()
        coordinates = cell.atom_coords(unit="Bohr")
        coordinates[1, 2] += 1.0e-3
        cell.set_geom_(coordinates, unit="Bohr")
        with self.assertRaisesRegex(RuntimeError, "inputs changed"):
            method.nuc_grad_method()

    def test_failed_second_kernel_clears_previous_success(self):
        method = dft.DCRKS(_h2_cell(), xc="PBE")
        method.kernel()

        class FailedReference:
            converged = False
            verbose = 0

            def kernel(self, dm0=None):
                return 0.0

        with mock.patch.object(
            method, "_build_reference", return_value=FailedReference()
        ):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                method.kernel()
        self.assertFalse(method.converged)
        self.assertIsNone(method.source_scf)
        self.assertIsNone(method.e_tot)

    def test_custom_fftdf_mesh_mismatch_is_rejected(self):
        method = dft.DCRKS(_h2_cell(), xc="PBE")
        method.fft_density_fit(mesh=[9, 11, 13])
        with self.assertRaisesRegex(ValueError, "with_df.mesh to match"):
            method.kernel()

    def test_nonzero_spin_is_rejected_before_scf(self):
        method = dft.DCRKS(_h2_cell(spin=2), xc="PBE")
        with self.assertRaisesRegex(NotImplementedError, "spin=0"):
            method.kernel()

    def test_fractional_occupations_are_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "0 or 2"):
            validate_restricted_occupations(
                np.asarray([[2.0, 1.0, 1.0, 0.0]]),
                nkpts=1,
                nelectron=4,
            )

    def test_cpu_multigrid_target_is_rejected_explicitly(self):
        method = dft.DCRKS(_h2_cell(), xc="PBE")
        with self.assertRaisesRegex(NotImplementedError, "GPU-only"):
            method.set_target_integrator("multigrid_xc")

    def test_target_integrator_mutation_is_part_of_state_signature(self):
        method = dft.DCRKS(_h2_cell(), xc="PBE")
        method.kernel()
        method.target_integrator = "multigrid_xc"
        with self.assertRaisesRegex(RuntimeError, "inputs changed"):
            method.make_rdm1()


if __name__ == "__main__":
    unittest.main()
