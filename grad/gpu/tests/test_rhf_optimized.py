import unittest

import cupy as cp
import numpy as np
from pyscf import gto

from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf_dcdft_grad.grad.rhf import (
    DCGradientGPU,
    _build_s1_oo_and_z_se,
)


def _gpu_available():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


@unittest.skipUnless(_gpu_available(), "CUDA device required")
class S1StreamingTest(unittest.TestCase):
    def test_streamed_blocks_match_full_transform(self):
        rng = np.random.default_rng(7)
        nao, nmo, nocc = 7, 6, 2
        coeff, _ = np.linalg.qr(rng.normal(size=(nao, nmo)))
        coeff = cp.asarray(coeff)
        Co, Cv = coeff[:, :nocc], coeff[:, nocc:]
        Z = cp.asarray(rng.normal(size=(nmo - nocc, nocc)))
        e_occ = cp.asarray([-0.8, -0.4])
        s1_ao = cp.asarray(rng.normal(size=(3, nao, nao)))
        atom_slices = [(0, 3), (3, 7)]

        s1_oo, z_se = _build_s1_oo_and_z_se(
            s1_ao, Co, Cv, Z, e_occ, atom_slices
        )

        full_oo = []
        full_z_se = []
        z_e = Z * e_occ[None, :]
        for p0, p1 in atom_slices:
            for ic in range(3):
                block = cp.zeros((nao, nao), dtype=coeff.dtype)
                block[p0:p1] = s1_ao[ic, p0:p1]
                block = block + block.T
                transformed = coeff.T @ block @ Co
                full_oo.append(transformed[:nocc])
                full_z_se.append(cp.einsum("ai,ai->", z_e, transformed[nocc:]))

        np.testing.assert_allclose(
            cp.asnumpy(s1_oo), cp.asnumpy(cp.stack(full_oo)), atol=1e-12
        )
        np.testing.assert_allclose(
            cp.asnumpy(z_se), cp.asnumpy(cp.stack(full_z_se)), atol=1e-12
        )


@unittest.skipUnless(_gpu_available(), "CUDA device required")
class OptimizedRHFGradientTest(unittest.TestCase):
    @staticmethod
    def _run_gradient(mol, **options):
        mf = gpu_hf.RHF(mol).density_fit().run(conv_tol=1e-11)
        driver = DCGradientGPU(
            mf, xc="PBE", atom_grid=(50, 194), grid_response=True, **options
        )
        return driver.kernel(verbose=0), driver

    def test_optimized_path_matches_ao_dense_reference(self):
        mol = gto.M(
            atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
            basis="def2-svp",
            verbose=0,
        )
        optimized, driver = self._run_gradient(
            mol, response_backend="mo_direct", share_df=True,
            factorized_z_f1=True, df_deriv_memory_gb=1.0,
        )
        reference, _ = self._run_gradient(
            mol, response_backend="ao", share_df=False,
            factorized_z_f1=False,
        )

        np.testing.assert_allclose(optimized, reference, atol=2e-10, rtol=0)
        self.assertTrue(driver.optimization_diagnostics["shared_df_identity"])
        self.assertEqual(
            driver.optimization_diagnostics["response_backend"], "mo_direct"
        )
        self.assertEqual(
            driver.optimization_diagnostics["df_deriv_memory_gb"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
