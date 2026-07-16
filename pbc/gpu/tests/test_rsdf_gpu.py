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
from pyscf_pbc_dcdft.df.gdf.storage import canonical_pairs
from pyscf_pbc_dcdft.df.rsdf.parameters import parameters_from_provider
from pyscf_pbc_dcdft.df.rsdf.metric import rsdf_metric_derivatives
from pyscf_pbc_dcdft.df.rsdf.source import (
    rsdf_source_derivatives,
)

from gpu4pyscf_pbc_dcdft import df as gpu_df
from gpu4pyscf_pbc_dcdft import dft as gpu_dft
from gpu4pyscf_pbc_dcdft.df.rsdf import (
    component_stream as rsdf_stream_module,
)
from gpu4pyscf_pbc_dcdft.df.rsdf import (
    source_reciprocal as source_reciprocal_module,
)
from gpu4pyscf_pbc_dcdft.df.rsdf import (
    build_rsdf_contracted_response,
    build_rsdf_response_tensors,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.source_adjoint import (
    build_pair_source_adjoints,
    contract_source_derivative,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.metric_adjoint import (
    build_pair_metric_adjoints,
    contract_metric_derivative,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.metric_native import (
    build_metric_contraction_context,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.source_native import (
    build_source_optimizer,
    contract_gamma_reciprocal_source,
    contract_gamma_short_range_source,
    contract_gamma_sources,
    contract_kpoint_reciprocal_source,
    contract_kpoint_short_range_source,
    contract_kpoint_sources,
)
from gpu4pyscf_pbc_dcdft.df.gdf.cholesky import CDERIFactorization
from gpu4pyscf_pbc_dcdft.df.rsdf.response import _host_pair_primitives
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
    def test_source_reciprocal_rebatch_is_transactional(self):
        calls = []
        plan = SimpleNamespace(
            initial_block_size=4,
            block_alignment=1,
        )
        ao_derivative = cp.zeros((2, 3))
        auxiliary_partial = cp.zeros((3, 4))

        def worker(start, stop):
            calls.append((start, stop))
            if stop - start == 4:
                raise cp.cuda.memory.OutOfMemoryError(1, 1)
            return source_reciprocal_module._BlockResult(
                cp.ones_like(ao_derivative),
                cp.ones_like(auxiliary_partial),
            )

        execution = source_reciprocal_module._execute_blocks(
            4,
            plan,
            worker,
            ao_derivative,
            auxiliary_partial,
            phase="test_source_reciprocal",
        )
        self.assertEqual(calls, [(0, 4), (0, 2), (2, 4)])
        self.assertEqual(execution.rebatch_count, 1)
        self.assertEqual(execution.final_block_size, 2)
        np.testing.assert_allclose(cp.asnumpy(ao_derivative), 2.0)
        np.testing.assert_allclose(cp.asnumpy(auxiliary_partial), 2.0)

    def test_source_reciprocal_forced_block_one_completes_all_ranges(self):
        calls = []
        plan = SimpleNamespace(
            initial_block_size=1,
            block_alignment=1,
        )
        ao_derivative = cp.zeros((1, 3))
        auxiliary_partial = cp.zeros((3, 1))

        def worker(start, stop):
            calls.append((start, stop))
            return source_reciprocal_module._BlockResult(
                cp.ones_like(ao_derivative),
                cp.ones_like(auxiliary_partial),
            )

        execution = source_reciprocal_module._execute_blocks(
            3,
            plan,
            worker,
            ao_derivative,
            auxiliary_partial,
            phase="test_source_reciprocal",
        )
        self.assertEqual(calls, [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(execution.final_block_size, 1)
        np.testing.assert_allclose(cp.asnumpy(ao_derivative), 3.0)
        np.testing.assert_allclose(cp.asnumpy(auxiliary_partial), 3.0)

    def test_kpoint_source_contractions_match_cpu_primitives(self):
        method = _method(_cell(), (2, 1, 1), device="gpu")
        provider = method.with_df
        witness = provider.cpu_witness
        parameters = parameters_from_provider(witness)
        kpts = np.asarray(provider.kpts)
        reference = cp.asarray(method.make_rdm1())
        response = 0.037 * reference
        target_adjoints = {}
        bilinear_adjoints = {}
        cpu_values = {
            "short_target": np.zeros((provider.cell.natm, 3)),
            "short_bilinear": np.zeros((provider.cell.natm, 3)),
            "reciprocal_target": np.zeros((provider.cell.natm, 3)),
            "reciprocal_bilinear": np.zeros((provider.cell.natm, 3)),
        }

        for pair in canonical_pairs(len(kpts)):
            primitives = _host_pair_primitives(
                witness.cell,
                witness.auxcell,
                kpts,
                pair,
                parameters,
            )
            factorization = CDERIFactorization.build(
                cp.asarray(primitives.source),
                cp.asarray(primitives.metric),
                nao=provider.cell.nao_nr(),
            )
            adjoints = build_pair_source_adjoints(
                provider.central_cderi_blocks,
                pair,
                factorization.factor,
                reference,
                response,
            )
            target_adjoints[pair] = adjoints.target_coulomb
            bilinear_adjoints[pair] = adjoints.restricted_hf_bilinear
            parts = rsdf_source_derivatives(
                witness.cell,
                witness.auxcell,
                parameters,
                kpti=kpts[pair[0]],
                kptj=kpts[pair[1]],
            )
            primitive_groups = (
                ("short", parts.short_range),
                ("reciprocal", parts.long_range + parts.vbar),
            )
            for group_name, primitive in primitive_groups:
                for response_name, source_adjoint in (
                    ("target", adjoints.target_coulomb),
                    ("bilinear", adjoints.restricted_hf_bilinear),
                ):
                    output = cpu_values[f"{group_name}_{response_name}"]
                    for atom_id in range(provider.cell.natm):
                        for axis in range(3):
                            output[atom_id, axis] += float(
                                contract_source_derivative(
                                    source_adjoint,
                                    cp.asarray(
                                        primitive[atom_id, axis]
                                    ),
                                ).get()
                            )

        optimizer = build_source_optimizer(
            provider.cell,
            witness.auxcell,
            parameters,
            bvk_kmesh=parameters.bvk_kmesh,
        )
        gpu_values = {
            "short_target": contract_kpoint_short_range_source(
                optimizer,
                kpts,
                target_adjoints,
            ),
            "short_bilinear": contract_kpoint_short_range_source(
                optimizer,
                kpts,
                bilinear_adjoints,
            ),
            "reciprocal_target": contract_kpoint_reciprocal_source(
                optimizer,
                kpts,
                target_adjoints,
            ),
            "reciprocal_bilinear": contract_kpoint_reciprocal_source(
                optimizer,
                kpts,
                bilinear_adjoints,
            ),
        }
        for name, cpu_value in cpu_values.items():
            difference = cp.asnumpy(gpu_values[name]) - cpu_value
            self.assertLess(float(np.max(np.abs(difference))), 1.0e-9)
        fused = contract_kpoint_sources(
            optimizer,
            kpts,
            (target_adjoints, bilinear_adjoints),
        )
        np.testing.assert_allclose(
            cp.asnumpy(fused[0]),
            cp.asnumpy(
                gpu_values["short_target"]
                + gpu_values["reciprocal_target"]
            ),
            atol=1.0e-9,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            cp.asnumpy(fused[1]),
            cp.asnumpy(
                gpu_values["short_bilinear"]
                + gpu_values["reciprocal_bilinear"]
            ),
            atol=1.0e-9,
            rtol=0.0,
        )

    def test_gamma_short_range_source_contraction_matches_cpu_primitive(self):
        method = _method(_cell(), (1, 1, 1), device="gpu")
        provider = method.with_df
        witness = provider.cpu_witness
        parameters = parameters_from_provider(witness)
        pair = (0, 0)
        primitives = _host_pair_primitives(
            witness.cell,
            witness.auxcell,
            np.asarray(provider.kpts),
            pair,
            parameters,
        )
        factorization = CDERIFactorization.build(
            cp.asarray(primitives.source),
            cp.asarray(primitives.metric),
            nao=provider.cell.nao_nr(),
        )
        reference = cp.asarray(method.make_rdm1())
        response = 0.037 * reference
        adjoints = build_pair_source_adjoints(
            provider.central_cderi_blocks,
            pair,
            factorization.factor,
            reference,
            response,
        )
        cpu_parts = rsdf_source_derivatives(
            witness.cell,
            witness.auxcell,
            parameters,
            kpti=np.asarray(provider.kpts)[0],
            kptj=np.asarray(provider.kpts)[0],
        )
        optimizer = build_source_optimizer(
            provider.cell,
            witness.auxcell,
            parameters,
        )
        primitive_routes = (
            (
                cpu_parts.short_range,
                contract_gamma_short_range_source,
            ),
            (
                cpu_parts.long_range + cpu_parts.vbar,
                contract_gamma_reciprocal_source,
            ),
        )
        for cpu_primitive, gpu_contract in primitive_routes:
            for source_adjoint in (
                adjoints.target_coulomb,
                adjoints.restricted_hf_bilinear,
            ):
                cpu_value = np.empty((provider.cell.natm, 3))
                for atom_id in range(provider.cell.natm):
                    for axis in range(3):
                        cpu_value[atom_id, axis] = float(
                            contract_source_derivative(
                                source_adjoint,
                                cp.asarray(
                                    cpu_primitive[atom_id, axis]
                                ),
                            ).get()
                        )
                gpu_value = cp.asnumpy(
                    gpu_contract(optimizer, source_adjoint)
                )
                self.assertLess(
                    float(np.max(np.abs(gpu_value - cpu_value))),
                    1.0e-9,
                )
        fused = contract_gamma_sources(
            optimizer,
            (
                adjoints.target_coulomb,
                adjoints.restricted_hf_bilinear,
            ),
        )
        for source_id, source_adjoint in enumerate(
            (
                adjoints.target_coulomb,
                adjoints.restricted_hf_bilinear,
            )
        ):
            cpu_total = np.empty((provider.cell.natm, 3))
            primitive = (
                cpu_parts.short_range
                + cpu_parts.long_range
                + cpu_parts.vbar
            )
            for atom_id in range(provider.cell.natm):
                for axis in range(3):
                    cpu_total[atom_id, axis] = float(
                        contract_source_derivative(
                            source_adjoint,
                            cp.asarray(primitive[atom_id, axis]),
                        ).get()
                    )
            np.testing.assert_allclose(
                cp.asnumpy(fused[source_id]),
                cpu_total,
                atol=1.0e-9,
                rtol=0.0,
            )

    def test_source_adjoints_match_complex_cderi_response(self):
        rng = np.random.default_rng(41)
        nkpts, naux, nao = 2, 3, 4

        def complex_random(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        def hermitian_density():
            value = complex_random((nkpts, nao, nao))
            return cp.asarray(value + value.conj().transpose(0, 2, 1))

        reference = hermitian_density()
        response = hermitian_density()
        pairs = ((0, 0), (0, 1), (1, 1))
        central = {
            pair: cp.asarray(complex_random((naux, nao, nao)))
            for pair in pairs
        }
        auxiliary_reference = auxiliary_density_contraction(
            central, reference
        )
        auxiliary_response = auxiliary_density_contraction(
            central, response
        )

        for pair in pairs:
            factor = np.tril(complex_random((naux, naux)))
            factor[np.diag_indices(naux)] += 4.0
            factor = cp.asarray(factor)
            source_derivative = cp.asarray(
                complex_random((nao**2, naux))
            )
            cderi_derivative = cp.linalg.solve(
                factor, source_derivative.T
            ).reshape(naux, nao, nao)
            witness = contract_canonical_pair_derivative(
                central,
                pair,
                cderi_derivative,
                reference,
                response,
                auxiliary_reference=auxiliary_reference,
                auxiliary_response=auxiliary_response,
            )
            adjoints = build_pair_source_adjoints(
                central,
                pair,
                factor,
                reference,
                response,
                auxiliary_reference=auxiliary_reference,
                auxiliary_response=auxiliary_response,
            )
            target = contract_source_derivative(
                adjoints.target_coulomb, source_derivative
            )
            bilinear = contract_source_derivative(
                adjoints.restricted_hf_bilinear, source_derivative
            )
            self.assertLess(
                float(cp.abs(target - witness.target_coulomb).get()),
                1.0e-10,
            )
            self.assertLess(
                float(
                    cp.abs(
                        bilinear - witness.restricted_hf_bilinear
                    ).get()
                ),
                1.0e-10,
            )

    def test_metric_adjoints_match_complex_cderi_response(self):
        rng = np.random.default_rng(43)
        nkpts, naux, nao = 2, 3, 4

        def complex_random(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        def hermitian_density():
            value = complex_random((nkpts, nao, nao))
            return cp.asarray(value + value.conj().transpose(0, 2, 1))

        reference = hermitian_density()
        response = hermitian_density()
        pairs = ((0, 0), (0, 1), (1, 1))
        central = {
            pair: cp.asarray(complex_random((naux, nao, nao)))
            for pair in pairs
        }
        auxiliary_reference = auxiliary_density_contraction(
            central, reference
        )
        auxiliary_response = auxiliary_density_contraction(
            central, response
        )
        for pair in pairs:
            factor = np.tril(complex_random((naux, naux)))
            factor[np.diag_indices(naux)] += 4.0
            factorization = CDERIFactorization(
                factor=cp.asarray(factor),
                value=central[pair],
                nao=nao,
            )
            random_metric = complex_random((naux, naux))
            metric_derivative = cp.asarray(
                random_metric + random_metric.conj().T
            )
            cderi_derivative = factorization.differentiate_metric(
                metric_derivative
            )
            witness = contract_canonical_pair_derivative(
                central,
                pair,
                cderi_derivative,
                reference,
                response,
                auxiliary_reference=auxiliary_reference,
                auxiliary_response=auxiliary_response,
            )
            adjoints = build_pair_metric_adjoints(
                central,
                pair,
                factorization,
                reference,
                response,
                auxiliary_reference=auxiliary_reference,
                auxiliary_response=auxiliary_response,
            )
            target = contract_metric_derivative(
                adjoints.target_coulomb,
                metric_derivative,
            )
            bilinear = contract_metric_derivative(
                adjoints.restricted_hf_bilinear,
                metric_derivative,
            )
            self.assertLess(
                float(cp.abs(target - witness.target_coulomb).get()),
                1.0e-10,
            )
            self.assertLess(
                float(
                    cp.abs(
                        bilinear - witness.restricted_hf_bilinear
                    ).get()
                ),
                1.0e-10,
            )

    def test_native_metric_contraction_matches_cpu_primitives(self):
        rng = np.random.default_rng(47)
        for kmesh in ((1, 1, 1), (2, 1, 1)):
            method = _method(_cell(), kmesh, device="gpu")
            provider = method.with_df
            witness = provider.cpu_witness
            parameters = parameters_from_provider(witness)
            kpts = np.asarray(provider.kpts)
            pair = canonical_pairs(len(kpts))[-1]
            q = kpts[pair[1]] - kpts[pair[0]]
            naux = int(witness.auxcell.nao_nr())
            adjoints = cp.asarray(
                rng.normal(size=(2, naux, naux))
                + 1j * rng.normal(size=(2, naux, naux))
            )
            cpu_derivatives = cp.asarray(
                rsdf_metric_derivatives(
                    witness.cell,
                    witness.auxcell,
                    parameters,
                    q=q,
                ).total
            )
            reference = cp.real(
                cp.einsum(
                    "spq,axpq->sax",
                    adjoints,
                    cpu_derivatives,
                    optimize=True,
                )
            )
            context = build_metric_contraction_context(
                witness.cell,
                witness.auxcell,
                parameters,
                q=q,
            )
            result = context.contract(adjoints)
            np.testing.assert_allclose(
                cp.asnumpy(result.values),
                cp.asnumpy(reference),
                atol=1.0e-9,
                rtol=0.0,
            )
            self.assertEqual(context.ip1.shape, (3, naux, naux))
            self.assertEqual(context.ip2.shape, (3, naux, naux))
            self.assertFalse(hasattr(context, "metric_derivatives"))

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
            method.with_df,
            reference,
            response,
            native_source_derivatives=False,
            native_metric_derivatives=False,
        )
        single_component = build_rsdf_contracted_response(
            method.with_df,
            reference,
            response,
            max_component_batch_size=1,
            native_source_derivatives=False,
            native_metric_derivatives=False,
        )
        native_source = build_rsdf_contracted_response(
            method.with_df,
            reference,
            response,
            native_source_derivatives=True,
            native_metric_derivatives=False,
        )
        native_metric = build_rsdf_contracted_response(
            method.with_df,
            reference,
            response,
            native_source_derivatives=False,
            native_metric_derivatives=True,
        )
        native_both = build_rsdf_contracted_response(
            method.with_df,
            reference,
            response,
            native_source_derivatives=True,
            native_metric_derivatives=True,
        )
        fused = build_rsdf_contracted_response(
            method.with_df,
            reference,
            response,
            native_source_derivatives=True,
            native_metric_derivatives=True,
            observable_mode="fused",
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
        self.assertLess(
            float(
                cp.max(
                    cp.abs(
                        contracted.target_coulomb
                        - native_source.target_coulomb
                    )
                ).get()
            ),
            1.0e-9,
        )
        self.assertLess(
            float(
                cp.max(
                    cp.abs(
                        contracted.restricted_hf_bilinear
                        - native_source.restricted_hf_bilinear
                    )
                ).get()
            ),
            1.0e-9,
        )
        self.assertEqual(
            native_source.profile.host_source_derivative_seconds,
            0.0,
        )
        self.assertLess(
            native_source.profile.host_to_device_bytes,
            contracted.profile.host_to_device_bytes,
        )
        for candidate in (native_metric, native_both):
            np.testing.assert_allclose(
                cp.asnumpy(candidate.target_coulomb),
                cp.asnumpy(contracted.target_coulomb),
                atol=1.0e-9,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                cp.asnumpy(candidate.restricted_hf_bilinear),
                cp.asnumpy(contracted.restricted_hf_bilinear),
                atol=1.0e-9,
                rtol=0.0,
            )
        self.assertEqual(
            native_metric.profile.response_mode,
            "native_metric_contracted_component_stream",
        )
        self.assertEqual(
            native_both.profile.response_mode,
            "native_source_native_metric_contracted_adjoint",
        )
        self.assertIsNone(native_both.profile.component_batch_size)
        self.assertIsNone(native_both.profile.initial_component_batch_size)
        self.assertIsNone(fused.target_coulomb)
        self.assertIsNone(fused.restricted_hf_bilinear)
        np.testing.assert_allclose(
            cp.asnumpy(fused.coupled_target_j_bilinear),
            cp.asnumpy(
                native_both.target_coulomb
                + native_both.restricted_hf_bilinear
            ),
            atol=1.0e-9,
            rtol=0.0,
        )
        self.assertEqual(
            fused.profile.response_mode,
            "native_source_native_metric_fused_adjoint",
        )
        self.assertEqual(fused.profile.metadata()["observable_count"], 1)

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
            "native_source_native_metric_fused_adjoint",
        )
        self.assertEqual(profile["observable_mode"], "fused")
        self.assertEqual(profile["observable_count"], 1)
        self.assertEqual(
            metadata["gradient_partition"],
            "explicit_response_coupled",
        )
        self.assertFalse(metadata["target_coulomb_component_resolved"])
        self.assertEqual(
            profile["host_source_derivative_seconds"],
            0.0,
        )
        self.assertEqual(profile["device_derivative_output_bytes"], 0)
        self.assertEqual(profile["retained_device_derivative_bytes"], 0)
        self.assertGreater(profile["full_derivative_equivalent_bytes"], 0)
        self.assertGreater(
            profile["device_peak_explicit_working_bytes"], 0
        )
        self.assertIsNone(profile["component_batch_size"])
        self.assertIsNone(profile["initial_component_batch_size"])


if __name__ == "__main__":
    unittest.main()
