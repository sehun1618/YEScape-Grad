"""GPU RSDF energy, gradient, conversion, and device-parity gates."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import cupy as cp
import numpy as np
from pyscf.pbc import gto
from pyscf_pbc_dcdft import df as cpu_df
from pyscf_pbc_dcdft import dft as cpu_dft

from gpu4pyscf_pbc_dcdft import df as gpu_df
from gpu4pyscf_pbc_dcdft import dft as gpu_dft
from gpu4pyscf_pbc_dcdft.df.rsdf import (
    component_stream as rsdf_stream_module,
)
from gpu4pyscf_pbc_dcdft.df.rsdf import (
    build_rsdf_contracted_response,
    build_rsdf_response_tensors,
)
from gpu4pyscf_pbc_dcdft.df.gdf.jk_derivative import (
    auxiliary_density_contraction,
    contract_canonical_pair_derivative,
    contract_coulomb_energy_derivative,
    contract_restricted_hf_bilinear_energy_derivative,
    gdf_jk_derivative,
)


def _cell():
    return gto.Cell(
        atom="H 0 0 0; H 0 0 1.4",
        a=np.eye(3) * 6.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-hf",
        ke_cutoff=15.0,
        precision=1.0e-9,
        verbose=0,
    ).build()


def _method(cell, kmesh, *, device):
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    dft_module = gpu_dft if device == "gpu" else cpu_dft
    df_module = gpu_df if device == "gpu" else cpu_df
    if tuple(kmesh) == (1, 1, 1):
        method = dft_module.DCRKS(cell, xc="PBE")
    else:
        method = dft_module.DCKRKS(cell, kpts=kpts, xc="PBE")
    method.with_df = df_module.RSDF(
        cell, kpts=kpts, auxbasis="gth-szv"
    )
    method.conv_tol = 1.0e-11
    method.conv_tol_grad = 1.0e-9
    method.kernel()
    return method


class RSDFGPUParityTest(unittest.TestCase):
    def test_central_pair_preflight_rejects_insufficient_memory(self):
        primitives = SimpleNamespace(
            source=np.zeros((8, 4)),
            metric=np.zeros((4, 4)),
        )
        with self.assertRaisesRegex(
            MemoryError, "central pair factorization"
        ):
            rsdf_stream_module.plan_central_pair_factorization(
                primitives,
                memory_fraction=1.0,
                minimum_reserve_bytes=0,
                free_device_bytes=128,
            )

    def test_contracted_response_matches_full_tensor_reference(self):
        method = _method(_cell(), (1, 1, 1), device="gpu")
        reference = cp.asarray(method.make_rdm1())
        response = 0.037 * reference
        full = build_rsdf_response_tensors(method.with_df)
        reference_derivative = gdf_jk_derivative(
            full.cderi, full.dcderi, reference
        )
        response_derivative = gdf_jk_derivative(
            full.cderi, full.dcderi, response
        )
        target_witness = contract_coulomb_energy_derivative(
            reference, reference_derivative.coulomb
        )
        bilinear_witness = (
            contract_restricted_hf_bilinear_energy_derivative(
                reference,
                response,
                reference_derivative,
                response_derivative,
            )
        )

        contracted = build_rsdf_contracted_response(
            method.with_df, reference, response
        )
        single_component = build_rsdf_contracted_response(
            method.with_df,
            reference,
            response,
            max_component_batch_size=1,
        )
        self.assertLess(
            float(
                cp.max(
                    cp.abs(contracted.target_coulomb - target_witness)
                ).get()
            ),
            1.0e-10,
        )
        self.assertLess(
            float(
                cp.max(
                    cp.abs(
                        contracted.restricted_hf_bilinear
                        - bilinear_witness
                    )
                ).get()
            ),
            1.0e-10,
        )
        self.assertEqual(
            contracted.profile.retained_device_derivative_bytes, 0
        )
        self.assertLess(
            float(
                cp.max(
                    cp.abs(
                        contracted.target_coulomb
                        - single_component.target_coulomb
                    )
                ).get()
            ),
            1.0e-12,
        )
        self.assertLess(
            float(
                cp.max(
                    cp.abs(
                        contracted.restricted_hf_bilinear
                        - single_component.restricted_hf_bilinear
                    )
                ).get()
            ),
            1.0e-12,
        )
        self.assertEqual(
            single_component.profile.initial_component_batch_size, 1
        )

    def test_component_oom_retries_without_double_commit(self):
        source = np.zeros((4, 1))
        metric = np.zeros((4, 1))
        target = cp.zeros((2, 3))
        bilinear = cp.zeros_like(target)
        batch_sizes = []

        def worker(*_args, source_components, **_kwargs):
            batch_sizes.append(len(source_components))
            if len(source_components) == 4:
                raise cp.cuda.memory.OutOfMemoryError(1, 1)
            return cp.ones(len(source_components)), cp.ones(
                len(source_components)
            )

        with mock.patch.object(
            rsdf_stream_module,
            "_component_batch_worker",
            side_effect=worker,
        ):
            execution = rsdf_stream_module.execute_component_batches(
                None,
                pair=(0, 0),
                central=None,
                reference=None,
                response=None,
                auxiliary_reference=None,
                auxiliary_response=None,
                source_components=source,
                metric_components=metric,
                target_coulomb=target,
                bilinear=bilinear,
                initial_batch_size=4,
                explicit_working_bytes=1024,
                additional_budget_bytes=2048,
            )
        self.assertEqual(batch_sizes, [4, 2, 2])
        self.assertEqual(execution["final_batch_size"], 2)
        self.assertEqual(execution["rebatch_count"], 1)
        np.testing.assert_allclose(cp.asnumpy(target).ravel()[:4], 1.0)
        np.testing.assert_allclose(cp.asnumpy(bilinear).ravel()[:4], 1.0)

    def test_canonical_pair_contraction_matches_full_jk_witness(self):
        rng = np.random.default_rng(12)
        nkpts, naux, nao = 2, 3, 4

        def hermitian_stack():
            value = rng.normal(size=(nkpts, nao, nao))
            value = value + 1j * rng.normal(size=value.shape)
            return value + value.conj().transpose(0, 2, 1)

        reference = cp.asarray(hermitian_stack())
        response = cp.asarray(hermitian_stack())
        pairs = ((0, 0), (0, 1), (1, 1))
        cderi = {}
        dcderi = {}
        for pair in pairs:
            value = rng.normal(size=(naux, nao, nao))
            value = value + 1j * rng.normal(size=value.shape)
            derivative = rng.normal(size=(naux, nao, nao))
            derivative = derivative + 1j * rng.normal(
                size=derivative.shape
            )
            cderi[pair] = cp.asarray(value)
            block = cp.zeros((1, 3, naux, nao, nao), dtype=cp.complex128)
            block[0, 0] = cp.asarray(derivative)
            dcderi[pair] = block

        reference_derivative = gdf_jk_derivative(
            cderi, dcderi, reference
        )
        response_derivative = gdf_jk_derivative(
            cderi, dcderi, response
        )
        target_witness = contract_coulomb_energy_derivative(
            reference, reference_derivative.coulomb
        )[0, 0]
        bilinear_witness = (
            contract_restricted_hf_bilinear_energy_derivative(
                reference,
                response,
                reference_derivative,
                response_derivative,
            )[0, 0]
        )

        auxiliary_reference = auxiliary_density_contraction(
            cderi, reference
        )
        auxiliary_response = auxiliary_density_contraction(cderi, response)
        target = cp.asarray(0.0)
        bilinear = cp.asarray(0.0)
        for pair in pairs:
            contracted = contract_canonical_pair_derivative(
                cderi,
                pair,
                dcderi[pair][0, 0],
                reference,
                response,
                auxiliary_reference=auxiliary_reference,
                auxiliary_response=auxiliary_response,
            )
            target += contracted.target_coulomb
            bilinear += contracted.restricted_hf_bilinear
        self.assertLess(float(cp.abs(target - target_witness).get()), 1.0e-10)
        self.assertLess(
            float(cp.abs(bilinear - bilinear_witness).get()), 1.0e-10
        )

    def test_gamma_energy_gradient_parity(self):
        self._assert_parity((1, 1, 1))

    def test_kpoint_energy_gradient_parity(self):
        self._assert_parity((2, 1, 1))

    def test_cpu_to_gpu_and_back_preserves_backend(self):
        cell = _cell()
        kpts = cell.make_kpts((2, 1, 1), wrap_around=True)
        cpu = cpu_dft.DCKRKS(cell, kpts=kpts, xc="PBE")
        cpu.with_df = cpu_df.RSDF(
            cell, kpts=kpts, auxbasis="gth-szv"
        )
        gpu = cpu.to_gpu()
        self.assertEqual(gpu.with_df.backend_name, "rsdf")
        self.assertEqual(gpu.to_cpu().with_df.backend_name, "rsdf")

    def _assert_parity(self, kmesh):
        cell = _cell()
        cpu = _method(cell, kmesh, device="cpu")
        gpu = _method(cell, kmesh, device="gpu")
        cpu_gradient = cpu.nuc_grad_method().kernel()
        gpu_gradient_object = gpu.nuc_grad_method()
        gpu_gradient = gpu_gradient_object.kernel()
        self.assertLess(abs(cpu.e_tot - gpu.e_tot), 1.0e-8)
        self.assertLess(np.max(np.abs(cpu_gradient - gpu_gradient)), 1.0e-7)
        metadata = gpu_gradient_object.result.metadata()
        self.assertEqual(metadata["backend"], "rsdf")
        self.assertFalse(metadata["silent_cpu_fallback"])
        self.assertTrue(metadata["native_one_electron"])
        self.assertLess(metadata["cderi_value_parity_max_abs"], 1.0e-7)
        profile = metadata["df_response_profile"]
        nkpts = metadata["nkpts"]
        self.assertEqual(
            profile["canonical_pair_count"], nkpts * (nkpts + 1) // 2
        )
        self.assertEqual(
            profile["derivative_component_count"],
            profile["canonical_pair_count"] * cell.natm * 3,
        )
        self.assertGreater(profile["host_to_device_bytes"], 0)
        self.assertEqual(
            profile["response_mode"],
            "source_contracted_component_stream",
        )
        self.assertEqual(profile["device_derivative_output_bytes"], 0)
        self.assertEqual(profile["retained_device_derivative_bytes"], 0)
        self.assertGreater(profile["full_derivative_equivalent_bytes"], 0)
        self.assertGreater(
            profile["device_peak_explicit_working_bytes"], 0
        )
        self.assertGreaterEqual(profile["component_batch_size"], 1)
        self.assertGreaterEqual(
            profile["initial_component_batch_size"],
            profile["component_batch_size"],
        )


if __name__ == "__main__":
    unittest.main()
