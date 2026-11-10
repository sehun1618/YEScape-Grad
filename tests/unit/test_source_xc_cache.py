"""Cached density products must never be recomputed inside feature derivatives."""
import numpy as np
import pytest
from dcforge.cpu.source.xc import _features

@pytest.mark.parametrize('xc', ['LDA', 'GGA', 'MGGA'])
@pytest.mark.parametrize('differentiate', [False, True])
def test_cached_product_reuse(xc, differentiate):
    rng = np.random.default_rng(13)
    ao = rng.normal(size=(4, 9, 5))
    dm = rng.normal(size=(5, 5)); dm = dm + dm.T
    deriv = rng.normal(size=ao.shape) if differentiate else None
    expected = _features(ao, dm, deriv, xc)
    # The sentinel cannot participate in matrix multiplication. A recomputation fails.
    actual = _features(ao, object(), deriv, xc, ad=ao @ dm)
    np.testing.assert_array_equal(actual, expected)
