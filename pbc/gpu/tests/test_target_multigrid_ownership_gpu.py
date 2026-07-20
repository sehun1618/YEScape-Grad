"""Fast ownership contracts for target multigrid energy and gradients."""

from types import SimpleNamespace
import unittest
from unittest import mock

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import tag_array

from gpu4pyscf_pbc_dcdft.dft.target_numint import evaluate_target_veff
from gpu4pyscf_pbc_dcdft.grad.target import (
    release_completed_target_multigrid,
    target_xc_atom_gradient,
)


class _FakeMultiGridNumInt:
    def __init__(self, cell):
        self.mesh = np.asarray(cell.mesh)
        self.calls = []

    def nr_rks(self, _cell, _grids, _xc, density, **kwargs):
        self.calls.append(dict(kwargs))
        value = cp.zeros_like(cp.asarray(density))
        if kwargs["with_j"]:
            value = tag_array(value, ecoul=cp.asarray(3.0))
        return 2.0, cp.asarray(-0.25), value


class _SelectedDF:
    def __init__(self):
        self.calls = 0

    def get_jk(self, density, **_kwargs):
        self.calls += 1
        return cp.ones_like(cp.asarray(density)), None


class _ResettableMultiGridNumInt:
    def __init__(self):
        self.tasks = [object(), object(), object()]
        self.sorted_gaussian_pairs = object()
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        self.tasks = None
        self.sorted_gaussian_pairs = None


class TargetMultigridOwnershipTest(unittest.TestCase):
    def setUp(self):
        self.cell = SimpleNamespace(mesh=np.asarray([7, 9, 11]), natm=2)
        self.target = SimpleNamespace(
            cell=self.cell,
            grids=object(),
            xc="PBE",
            verbose=0,
        )
        self.density = cp.stack([cp.eye(2), cp.eye(2)])
        self.kpts = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])

    @mock.patch(
        "gpu4pyscf_pbc_dcdft.dft.target_numint.multigrid_v2.MultiGridNumInt",
        _FakeMultiGridNumInt,
    )
    def test_m1_keeps_selected_df_target_j(self):
        provider = _SelectedDF()
        result = evaluate_target_veff(
            self.target,
            self.density,
            with_df=provider,
            kpts=self.kpts,
            target_integrator="multigrid_xc",
        )

        self.assertEqual(provider.calls, 1)
        self.assertFalse(result.numerical_integrator.calls[0]["with_j"])
        self.assertEqual(result.metadata["target_j_owner"], "selected_df_backend")

    @mock.patch(
        "gpu4pyscf_pbc_dcdft.dft.target_numint.multigrid_v2.MultiGridNumInt",
        _FakeMultiGridNumInt,
    )
    def test_m2_does_not_call_selected_df_target_j(self):
        provider = _SelectedDF()
        result = evaluate_target_veff(
            self.target,
            self.density,
            with_df=provider,
            kpts=self.kpts,
            target_integrator="multigrid_j_xc",
        )

        self.assertEqual(provider.calls, 0)
        self.assertTrue(result.numerical_integrator.calls[0]["with_j"])
        self.assertEqual(result.metadata["target_j_owner"], "multigrid_v2")

    def test_gradient_with_j_and_nk_contract(self):
        workspace = SimpleNamespace(
            target_integrator="multigrid_j_xc",
            target_numint=object(),
            ownership={},
        )
        context = SimpleNamespace(
            target_evaluation=workspace,
            target_ks=self.target,
            density=self.density,
            kpts=self.kpts,
            nkpts=2,
            dc_mf=SimpleNamespace(verbose=0),
        )
        raw = cp.ones((2, 3)) * 4.0
        with mock.patch(
            "gpu4pyscf_pbc_dcdft.grad.target.multigrid_v2.get_veff_ip1",
            return_value=raw,
        ) as derivative:
            value = target_xc_atom_gradient(context)

        self.assertTrue(derivative.call_args.kwargs["with_j"])
        self.assertFalse(
            derivative.call_args.kwargs[
                "with_pseudo_vloc_orbital_derivative"
            ]
        )
        self.assertLess(float(cp.max(cp.abs(value - 2.0)).get()), 1.0e-14)

    def test_completed_multigrid_workspace_is_released(self):
        numerical_integrator = _ResettableMultiGridNumInt()
        workspace = SimpleNamespace(
            target_integrator="multigrid_xc",
            target_numint=numerical_integrator,
            ownership={},
        )
        context = SimpleNamespace(target_evaluation=workspace)

        metadata = release_completed_target_multigrid(context)

        self.assertEqual(numerical_integrator.reset_calls, 1)
        self.assertIsNone(numerical_integrator.tasks)
        self.assertIsNone(numerical_integrator.sorted_gaussian_pairs)
        self.assertTrue(
            metadata["target_multigrid_cache_released_after_gradient"]
        )
        self.assertEqual(metadata["released_multigrid_task_count"], 3)
        self.assertEqual(workspace.ownership, metadata)


if __name__ == "__main__":
    unittest.main()
