import unittest

import cupy as cp
import numpy as np
from pyscf import gto

from gpu4pyscf.scf import uhf as gpu_uhf
from gpu4pyscf_dcdft_grad.grad.uhf import (
    DCGradientGPU,
    _build_uhf_s1_oo_and_z_se,
)


def _gpu_available():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


@unittest.skipUnless(_gpu_available(), "CUDA device required")
class UHFStreamingTest(unittest.TestCase):
    def test_streamed_spin_blocks_match_full_transform(self):
        rng = np.random.default_rng(19)
        nao, nmo = 8, 7
        coeff_a, _ = np.linalg.qr(rng.normal(size=(nao, nmo)))
        coeff_b, _ = np.linalg.qr(rng.normal(size=(nao, nmo)))
        Ca, Cb = cp.asarray(coeff_a), cp.asarray(coeff_b)
        Coa, Cva = Ca[:, :3], Ca[:, 3:]
        Cob, Cvb = Cb[:, :2], Cb[:, 2:]
        Za = cp.asarray(rng.normal(size=(4, 3)))
        Zb = cp.asarray(rng.normal(size=(5, 2)))
        eoa = cp.asarray([-0.9, -0.6, -0.3])
        eob = cp.asarray([-0.8, -0.4])
        s1_ao = cp.asarray(rng.normal(size=(3, nao, nao)))
        atom_slices = [(0, 3), (3, 8)]

        soo_a, soo_b, z_se = _build_uhf_s1_oo_and_z_se(
            s1_ao, Coa, Cob, Cva, Cvb, Za, Zb, eoa, eob, atom_slices
        )
        ref_a, ref_b, ref_se = [], [], []
        for p0, p1 in atom_slices:
            for ic in range(3):
                block = cp.zeros((nao, nao), dtype=Ca.dtype)
                block[p0:p1] = s1_ao[ic, p0:p1]
                block = block + block.T
                trans_a = Ca.T @ block @ Coa
                trans_b = Cb.T @ block @ Cob
                ref_a.append(trans_a[:3])
                ref_b.append(trans_b[:2])
                ref_se.append(
                    cp.einsum('ai,ai->', Za * eoa[None], trans_a[3:])
                    + cp.einsum('ai,ai->', Zb * eob[None], trans_b[2:])
                )

        np.testing.assert_allclose(cp.asnumpy(soo_a), cp.asnumpy(cp.stack(ref_a)))
        np.testing.assert_allclose(cp.asnumpy(soo_b), cp.asnumpy(cp.stack(ref_b)))
        np.testing.assert_allclose(cp.asnumpy(z_se), cp.asnumpy(cp.stack(ref_se)))


@unittest.skipUnless(_gpu_available(), "CUDA device required")
class OptimizedUHFGradientTest(unittest.TestCase):
    @staticmethod
    def _run_gradient(mol, **options):
        mf = gpu_uhf.UHF(mol).density_fit().run(conv_tol=1e-11)
        driver = DCGradientGPU(
            mf, xc="PBE", atom_grid=(50, 194), grid_response=True, **options
        )
        return driver.kernel(verbose=0), driver

    def test_optimized_path_matches_full_reference(self):
        mol = gto.M(
            atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0
        )
        optimized, driver = self._run_gradient(
            mol, response_backend="mo_direct", share_df=True,
            direct_z_f1=True, df_deriv_memory_gb=1.0,
        )
        reference, _ = self._run_gradient(
            mol, response_backend="ao", share_df=False, direct_z_f1=False
        )

        np.testing.assert_allclose(optimized, reference, atol=2e-10, rtol=0)
        self.assertTrue(driver.optimization_diagnostics["shared_df_identity"])
        self.assertEqual(
            driver.optimization_diagnostics["response_backend"], "mo_direct"
        )
        self.assertTrue(driver.optimization_diagnostics["direct_z_f1"])


if __name__ == "__main__":
    unittest.main()
