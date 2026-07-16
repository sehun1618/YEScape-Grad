from pyscf_dcdft_grad import RGradients, ROGradients, UGradients, dc_grad


def test_public_imports():
    assert callable(dc_grad)
    assert RGradients is not None
    assert ROGradients is not None
    assert UGradients is not None
