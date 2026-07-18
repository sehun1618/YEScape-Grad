"""Fast ownership tests for reusable target-DFA intermediates."""

from types import SimpleNamespace
import unittest
from unittest import mock

import cupy as cp
import numpy as np

from gpu4pyscf_pbc_dcdft.dft.workspace import (
    TargetEvaluationWorkspace,
    compatible_target_workspace,
)
from gpu4pyscf_pbc_dcdft.grad.target import target_fock_kpts


class TargetWorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.cell = _Cell()
        self.source = SimpleNamespace(
            cell=self.cell,
            mo_coeff=cp.eye(2),
            mo_occ=cp.asarray([2.0, 0.0]),
            mo_energy=cp.asarray([-0.5, 0.3]),
            e_tot=-1.0,
        )
        self.target = SimpleNamespace(xc="PBE")
        self.provider = SimpleNamespace(backend_name="gdf")
        self.hcore = cp.asarray([[1.0, 0.1], [0.1, 2.0]])
        self.veff = cp.asarray([[0.4, 0.2], [0.2, 0.8]])
        self.workspace = TargetEvaluationWorkspace.create(
            source_mf=self.source,
            target_ks=self.target,
            with_df=self.provider,
            xc="PBE",
            kpts=np.zeros((1, 3)),
            density=cp.eye(2),
            hcore=self.hcore,
            veff=self.veff,
        )
        self.method = SimpleNamespace(
            source_scf=self.source,
            target_ks=self.target,
            target_evaluation=self.workspace,
            with_df=self.provider,
            xc="PBE",
            kpts=np.zeros((1, 3)),
            cell=self.cell,
        )

    def test_gradient_reuses_energy_fock_without_numint_call(self):
        context = SimpleNamespace(
            target_evaluation=compatible_target_workspace(self.method),
            target_ks=self.target,
            source_mf=self.source,
            dc_mf=self.method,
            kpts=np.zeros((1, 3)),
        )
        with mock.patch(
            "gpu4pyscf_pbc_dcdft.grad.target.evaluate_target_veff",
            side_effect=AssertionError("target Veff was recomputed"),
        ):
            fock, reused = target_fock_kpts(context, return_reuse=True)
        self.assertTrue(reused)
        self.assertEqual(fock.shape, (1, 2, 2))
        self.assertLess(
            float(cp.max(cp.abs(fock[0] - self.hcore - self.veff)).get()),
            1.0e-14,
        )

    def test_in_place_geometry_change_invalidates_workspace(self):
        self.cell.coordinates[0, 2] += 0.1
        self.assertIsNone(compatible_target_workspace(self.method))

    def test_target_integrator_change_invalidates_workspace(self):
        self.method.target_integrator = "multigrid_xc"
        self.method.target_multigrid_mesh = np.asarray([8, 8, 8])
        self.assertIsNone(compatible_target_workspace(self.method))

    def test_implicit_multigrid_mesh_keeps_workspace_compatible(self):
        workspace = TargetEvaluationWorkspace.create(
            source_mf=self.source,
            target_ks=self.target,
            with_df=self.provider,
            xc="PBE",
            kpts=np.zeros((1, 3)),
            density=cp.eye(2),
            hcore=self.hcore,
            veff=self.veff,
            target_integrator="multigrid_xc",
            target_multigrid_mesh=None,
        )
        self.method.target_integrator = "multigrid_xc"
        self.method.target_multigrid_mesh = None
        self.method.target_evaluation = workspace

        self.assertIs(compatible_target_workspace(self.method), workspace)
        self.assertEqual(workspace.target_multigrid_mesh, tuple(self.cell.mesh))


class _Cell:
    def __init__(self):
        self.natm = 1
        self.coordinates = np.asarray([[0.0, 0.0, 0.0]])
        self.mesh = np.asarray([8, 8, 8])
        self._basis = {"H": "test"}
        self._pseudo = {"H": "test"}
        self.precision = 1.0e-10
        self.omega = 0.0

    def atom_symbol(self, _index):
        return "H"

    def atom_coords(self, unit="Bohr"):
        if unit != "Bohr":
            raise ValueError(unit)
        return self.coordinates

    def lattice_vectors(self):
        return np.eye(3) * 6.0


if __name__ == "__main__":
    unittest.main()
