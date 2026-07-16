from gpu4pyscf_dcdft_grad import RDCGradient, UDCGradient, kernel


def test_public_imports():
    assert RDCGradient is not None
    assert UDCGradient is not None
    assert callable(kernel)
