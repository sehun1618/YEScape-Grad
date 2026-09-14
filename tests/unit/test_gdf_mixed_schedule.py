'''Precision scheduling checks without integral builds or GPU calculations.'''

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dcforge.gpu.pbc.scf import gdf_mixed as mixed
from dcforge.gpu.pbc.energy import RHFDFT
from tests.common import make_water_cell


def controller(monkeypatch, thresholds=None, persistent=True):
    monkeypatch.setattr(mixed, 'GammaFP32JK', lambda *a, **kw: Mock())
    hf = SimpleNamespace(with_df=SimpleNamespace(), verbose=0, stdout=sys.stdout)
    obj = mixed.SourcePrecisionController(
        hf, None, block_size=256, adaptive_norm_fraction=.9,
        thresholds=thresholds, persistent=persistent)
    obj._ao_commutator_norm = lambda env: env['commutator']
    return obj


def cycle(obj, index=0, residual=1.):
    diis = SimpleNamespace(_buffer={'old': 1}, _bookkeep=[1], _head=2)
    obj.callback(dict(cycle=index, e_tot=-1., last_hf_e=-1.,
                      norm_ddm=residual, norm_gorb=residual,
                      commutator=residual, mf_diis=diis))
    return diis


@pytest.mark.parametrize('persistent', [False, True])
def test_twenty_cycles_per_tier_survive_kernel_restart(monkeypatch, persistent):
    obj = controller(monkeypatch, persistent=persistent)
    for i in range(19):
        cycle(obj, i % 7)  # Restart native SCF numbering without losing budget.
    assert obj.tier == 0
    diis = cycle(obj)
    assert obj.tier == 1 and obj.tier_cycles == 0
    assert diis._buffer == {} and diis._bookkeep == []
    assert obj.pending_full_rebuild
    obj.original_get_veff = Mock(return_value='rebuilt')
    assert obj.get_veff(dm='density', dm_last='old', vhf_last='old') == 'rebuilt'
    assert obj.original_get_veff.call_args.args == (None, 'density', None, None, 1)
    for i in range(19):
        cycle(obj)
    assert obj.tier == 1
    cycle(obj)
    assert obj.tier == 2 and obj.current_policy == obj.sequence[-1]
    for i in range(50):
        cycle(obj)
    assert len(obj.transitions) == 2
    assert [x['reason'] for x in obj.transitions] == ['max_cycle', 'max_cycle']
    assert [x['trigger']['tier_cycle'] for x in obj.transitions] == [20, 20]
    assert obj.profile()['thresholds']['fine']['max_cycle'] == 20


def test_stationarity_can_promote_early(monkeypatch):
    obj = controller(monkeypatch)
    cycle(obj, 0, 0.)
    assert obj.tier == 0
    cycle(obj, 1, 0.)
    assert obj.tier == 1
    cycle(obj, 2, 0.)
    assert obj.tier == 1
    cycle(obj, 3, 0.)
    assert obj.tier == 2
    assert all(x['reason'] == 'stationarity' for x in obj.transitions)


def test_custom_limits_disable_and_cleanup(monkeypatch):
    obj = controller(monkeypatch, {'coarse': {'max_cycle': 3},
                                   'fine': {'max_cycle': None}})
    for i in range(3):
        cycle(obj)
    assert obj.tier == 1
    for i in range(50):
        cycle(obj)
    assert obj.tier == 1
    assert obj.force_final(reason='test cleanup')
    assert obj.tier_cycles == 0 and obj.current_policy == obj.sequence[-1]
    assert not obj.force_final(reason='already final')
    # A cap is independent of the pre-existing minimum-cycle gate.
    obj = controller(monkeypatch, {'coarse': {'max_cycle': 1},
                                   'fine': {'max_cycle': 2}})
    cycle(obj)
    cycle(obj)
    assert obj.tier == 1
    cycle(obj)
    assert obj.tier == 2


@pytest.mark.parametrize('value', [0, -1, 1.5, True, '20', float('inf')])
def test_invalid_limit_rejected(monkeypatch, value):
    with pytest.raises(ValueError, match='positive integer or None'):
        controller(monkeypatch, {'coarse': {'max_cycle': value}})


def test_public_options_and_default_isolation():
    mf = RHFDFT(make_water_cell())
    gates = mf._source_precision_thresholds()
    assert gates['coarse']['max_cycle'] == gates['fine']['max_cycle'] == 20
    mf.source_precision_coarse_max_cycle = 12
    mf.source_precision_fine_max_cycle = 25
    gates = mf._source_precision_thresholds()
    assert gates['coarse']['max_cycle'] == 12
    assert gates['fine']['max_cycle'] == 25
    assert 'source_precision_coarse_max_cycle' in mf._keys
    assert 'source_precision_fine_max_cycle' in mf._keys
    normalized = mixed._normalize_thresholds(gates)
    normalized['coarse']['max_cycle'] = 99
    assert mixed.DEFAULT_THRESHOLDS['coarse']['max_cycle'] == 20
    assert gates['coarse']['max_cycle'] == 12
