"""Tests for size-aware GPU DF derivative memory planning."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gpu4pyscf")
from dcforge.gpu.df.grad import memory


class FakeDM:
    def __init__(self, left_rank, right_rank, symmetrize):
        self.factor_l = np.empty((1, left_rank))
        self.factor_r = np.empty((1, right_rank))
        self.symmetrize = symmetrize


class FakeInt3c2eOpt:
    def __init__(self, nao, naux, nao_pair, laux=4):
        self.mol = SimpleNamespace(nao=nao)
        self.auxmol = SimpleNamespace(
            nao=naux, uniq_l_ctr=np.asarray([[laux, 1]])
        )
        self.nao_pair = nao_pair

    def pair_and_diag_indices(self, **kwargs):
        return range(self.nao_pair), None


def test_small_system_keeps_four_gib():
    opt = FakeInt3c2eOpt(2100, 4820, 500_000)
    dz = FakeDM(100, 100, symmetrize=1)
    dm = FakeDM(100, 100, symmetrize=0)
    plan = memory.adaptive_derivative_planner_limit(opt, [[dz, dm]])
    assert plan["resolved_limit_bytes"] == (
        memory.DEFAULT_FOCK_DERIV_PLANNER_WORKSPACE_BYTES
    )


def test_large_transition_density_raises_cap():
    opt = FakeInt3c2eOpt(1600, 8512, 1_000_000)
    dz = FakeDM(320, 320, symmetrize=1)
    dm = FakeDM(320, 320, symmetrize=0)
    plan = memory.adaptive_derivative_planner_limit(opt, [[dz, dm]])
    assert plan["factor_ranks"][0]["left"] == 640
    assert plan["factor_ranks"][0]["right"] == 320
    assert plan["resolved_limit_bytes"] == 32 * 1024**3


def test_small_uhf_density_set_keeps_four_gib():
    opt = FakeInt3c2eOpt(815, 1873, 335_000)
    pairs = [
        [FakeDM(79, 79, 1), FakeDM(79, 79, 0)],
        [FakeDM(40, 40, 1), FakeDM(40, 40, 0)],
        [FakeDM(39, 39, 1), FakeDM(39, 39, 0)],
    ]
    plan = memory.adaptive_derivative_planner_limit(opt, pairs)
    assert plan["resolved_limit_bytes"] == (
        memory.DEFAULT_FOCK_DERIV_PLANNER_WORKSPACE_BYTES
    )


def test_explicit_limit_remains_strict():
    assert memory.normalize_fock_deriv_planner_limit("adaptive") == "adaptive"
    assert memory.normalize_fock_deriv_planner_limit(1234) == 1234
    assert memory.normalize_fock_deriv_planner_limit(None) is None


@pytest.mark.parametrize("sum_results", [False, True])
@pytest.mark.parametrize("n_dm", [1, 3])
def test_190_batch_fits_upstream_minimum(sum_results, n_dm):
    opt = FakeInt3c2eOpt(5000, 100, 10_000_000)
    dm = FakeDM(10, 10, 0)
    plan = memory.adaptive_derivative_planner_limit(
        opt, [[dm, dm]] * n_dm, sum_results=sum_results)
    buffers = 2 if sum_results else n_dm
    batch = int((plan["resolved_limit_bytes"] - plan["fixed_intermediate_bytes"])
                * .5 / (buffers * opt.nao_pair * 8))
    assert batch >= plan["minimum_aux_batch_size"]


@pytest.mark.parametrize("pair_form", ["list", "single"])
def test_overcomplete_factors_are_refactored_without_changing_density(monkeypatch, pair_form):
    import cupy as cp
    from gpu4pyscf.lib.cupy_helper import tag_array
    dm = tag_array(cp.eye(3), factor_l=cp.ones((3, 2)),
                   factor_r=cp.ones((3, 2)), symmetrize=1)
    regular = cp.eye(3)
    pairs = [[dm, regular]] if pair_form == "list" else [dm]
    def upstream(opt, received, **kwargs):
        normalized = received[0][0] if pair_form == "list" else received[0]
        assert not hasattr(normalized, "factor_l")
        cp.testing.assert_array_equal(normalized, dm)
        assert normalized.data.ptr == dm.data.ptr
        if pair_form == "list":
            assert received[0][1] is regular
        return 123
    monkeypatch.setattr(memory.df_tdrhf_grad, "_jk_energies_per_atom", upstream)
    grad = SimpleNamespace(fock_deriv_planner_limit_bytes=None)
    assert memory.jk_energies_per_atom_bounded(
        grad, FakeInt3c2eOpt(3, 5, 6), pairs, j_factor=[1],
        k_factor=[1], sum_results=True, verbose=0) == 123
