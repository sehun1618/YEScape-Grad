'''Unit checks for the warning-only physical Z-residual policy.'''

import pytest

from dcforge.tools.zvec import warn_z_residual


class _Log:
    def __init__(self):
        self.messages = []

    def warn(self, message, *args):
        self.messages.append(message % args)


def test_large_physical_z_residual_warns_without_raising():
    log = _Log()
    info = {
        'residual_max_abs': 2e-6,
        'residual_relative': 4e-5,
    }

    warned = warn_z_residual(log, info, 1e-8)

    assert warned is True
    assert info['physical_residual_warning'] is True
    assert info['physical_residual_warn_tol'] == 1e-8
    assert 'gradient is returned unchanged' in log.messages[0]


def test_small_physical_z_residual_is_only_recorded():
    log = _Log()
    info = {
        'residual_max_abs': 2e-10,
        'residual_relative': 4e-9,
    }

    warned = warn_z_residual(log, info, 1e-8)

    assert warned is False
    assert info['physical_residual_warning'] is False
    assert not log.messages


def test_none_disables_only_the_warning():
    log = _Log()
    info = {
        'residual_max_abs': 2e-6,
        'residual_relative': 4e-5,
    }

    warned = warn_z_residual(log, info, None)

    assert warned is False
    assert info['physical_residual_warn_tol'] is None
    assert info['physical_residual_warning'] is False
    assert not log.messages


def test_invalid_physical_z_residual_warning_threshold_is_rejected():
    with pytest.raises(ValueError, match='positive and finite'):
        warn_z_residual(_Log(), {
            'residual_max_abs': 0.,
            'residual_relative': 0.,
        }, 0.)
