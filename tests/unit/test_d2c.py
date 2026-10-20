"""Published D2C dispersion parameter and provider checks."""

import numpy as np
import pytest

from dcforge.cpu.disp.d2c import (
    evaluate_d2c,
    get_d2c_parameters,
    normalize_dispersion,
)
from tests.common import displaced_mol, make_mol


def test_only_explicit_d2c_mode_is_accepted():
    assert normalize_dispersion(None) is None
    assert normalize_dispersion("d2c") == "d2c"
    for value in (False, True, "D2C", "d2c-d4", "off"):
        with pytest.raises(ValueError, match="None or 'd2c'"):
            normalize_dispersion(value)


def test_r2scan_uses_published_d2c_parameters():
    assert get_d2c_parameters("r2SCAN") == {
        "functional": "r2SCAN",
        "s6": 1.0,
        "s8": 0.02734375,
        "s9": 1.0,
        "a1": 0.74707031,
        "a2": 3.34667969,
        "alp": 16.0,
    }


def test_unknown_d2c_functional_is_not_replaced_by_standard_d4():
    with pytest.raises(ValueError, match="no published D2C dispersion parameters"):
        get_d2c_parameters("made-up-xc")


@pytest.mark.fd
def test_d2c_analytic_gradient_matches_energy_finite_difference():
    mol = make_mol("water_dimer_shared_proton.xyz", basis="sto-3g")
    result = evaluate_d2c(mol, "r2SCAN", gradient=True)
    step = 1e-4
    ep = evaluate_d2c(displaced_mol(mol, 2, 2, step), "r2SCAN").energy
    em = evaluate_d2c(displaced_mol(mol, 2, 2, -step), "r2SCAN").energy
    fd = (ep - em) / (2 * step)
    assert np.isfinite(result.energy)
    assert result.gradient.shape == (mol.natm, 3)
    assert abs(result.gradient[2, 2] - fd) < 1e-8
