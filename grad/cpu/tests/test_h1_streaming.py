import numpy as np
from pyscf import gto, scf

from pyscf_dcdft_grad.grad.rhf import _iter_h1ao as iter_rhf_h1ao
from pyscf_dcdft_grad.grad.uhf import _iter_h1ao as iter_uhf_h1ao


def test_rhf_df_h1_stream_matches_make_h1():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-12)
    reference = mf.Hessian().make_h1(mf.mo_coeff, mf.mo_occ)
    streamed = list(iter_rhf_h1ao(mf, mf.mo_coeff, mf.mo_occ))

    assert [ia for ia, _ in streamed] == list(range(mol.natm))
    for ia, h1 in streamed:
        np.testing.assert_allclose(h1, reference[ia], atol=1e-12, rtol=0)


def test_uhf_df_h1_stream_matches_make_h1():
    mol = gto.M(
        atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0
    )
    mf = scf.UHF(mol).density_fit().run(conv_tol=1e-12)
    reference_a, reference_b = mf.Hessian().make_h1(
        mf.mo_coeff, mf.mo_occ
    )
    streamed = list(iter_uhf_h1ao(mf, mf.mo_coeff, mf.mo_occ))

    assert [ia for ia, _, _ in streamed] == list(range(mol.natm))
    for ia, h1a, h1b in streamed:
        np.testing.assert_allclose(h1a, reference_a[ia], atol=1e-12, rtol=0)
        np.testing.assert_allclose(h1b, reference_b[ia], atol=1e-12, rtol=0)
