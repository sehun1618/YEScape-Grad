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
