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
from pyscf_pbc_dcdft.df.gdf.storage import (
    canonical_pairs,
    load_cderi_storage,
)
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
    contracted_response as contracted_response_module,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.central_exact import (
    _compressed_device_cache_plan,
    _source_builder_memory_plan,
)
from gpu4pyscf_pbc_dcdft.df.gdf import gdf as gdf_provider_module
from gpu4pyscf_pbc_dcdft.df.gdf import (
    jk_derivative as gdf_jk_derivative_module,
)
from gpu4pyscf_pbc_dcdft.df.rsdf import rsdf as rsdf_provider_module
from gpu4pyscf_pbc_dcdft.df.rsdf import (
    source_reciprocal as source_reciprocal_module,
)
from gpu4pyscf_pbc_dcdft.df.rsdf import source_adjoint as source_adjoint_module
from gpu4pyscf_pbc_dcdft.df.rsdf import source_native as source_native_module
from gpu4pyscf_pbc_dcdft.df.rsdf import pp_value as pp_value_module
from gpu4pyscf_pbc_dcdft.df.rsdf import (
    build_rsdf_contracted_response,
    build_rsdf_response_tensors,
    build_exact_cderi,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.source_adjoint import (
    build_pair_cderi_adjoints,
    build_pair_fused_cderi_adjoint,
    build_pair_fused_gamma_cderi_adjoint,
    build_pair_source_adjoints,
    contract_source_derivative,
    plan_fused_adjoint_auxiliary_blocks,
    push_cderi_adjoint_through_source,
    push_packed_gamma_adjoint_through_source,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.metric_adjoint import (
    build_pair_metric_adjoints,
    contract_metric_derivative,
    push_cderi_adjoint_through_metric,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.metric_native import (
    build_metric_contraction_context,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.packed_gamma import (
    PackedGammaCDERIAdjoint,
    pack_gamma_cderi_adjoint,
    transform_gamma_sources_to_sorted_packed,
    unpack_lower_triangle,
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
    auxiliary_density_contractions,
    contract_canonical_pair_derivative,
    contract_coulomb_energy_derivative,
    contract_restricted_hf_bilinear_energy_derivative,
    gdf_jk_derivative,
    plan_auxiliary_density_blocks,
)
from gpu4pyscf_pbc_dcdft.grad import gdf as gdf_gradient_module


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


def _water_cell():
    return gto.Cell(
        atom="O 0 0 0; H 0 1.4 0; H 0 -1.4 0",
        a=np.eye(3) * 8.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-hf",
        ke_cutoff=15.0,
        precision=1.0e-9,
        verbose=0,
    ).build()


def _water_dzvp_cell():
    return gto.Cell(
        atom="O 0 0 0; H 0 1.4 0; H 0 -1.4 0",
        a=np.eye(3) * 8.0,
        unit="Bohr",
        basis="DZVP-GTH",
        pseudo="GTH-PBE",
        ke_cutoff=15.0,
        precision=1.0e-8,
        verbose=0,
    ).build()


def _method(cell, kmesh, *, device, central_setup=None):
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    dft_module = gpu_dft if device == "gpu" else cpu_dft
    df_module = gpu_df if device == "gpu" else cpu_df
    if tuple(kmesh) == (1, 1, 1):
        method = dft_module.DCRKS(cell, xc="PBE")
    else:
        method = dft_module.DCKRKS(cell, kpts=kpts, xc="PBE")
    provider_kwargs = {
        "kpts": kpts,
        "auxbasis": "gth-szv",
    }
    if device == "gpu" and central_setup is not None:
        provider_kwargs["central_setup"] = central_setup
    method.with_df = df_module.RSDF(cell, **provider_kwargs)
    method.conv_tol = 1.0e-11
    method.conv_tol_grad = 1.0e-9
    method.kernel()
    return method


class RSDFGPUParityTest(unittest.TestCase):
    def test_gamma_packed_source_sorted_transform_matches_dense(self):
        method = _method(_cell(), (1, 1, 1), device="gpu")
        provider = method.with_df
        witness = provider.cpu_witness
        parameters = parameters_from_provider(witness)
        optimizer = build_source_optimizer(
            provider.cell,
            witness.auxcell,
            parameters,
        )
        rng = np.random.default_rng(29)
        nao = int(provider.cell.nao_nr())
        naux = int(witness.auxcell.nao_nr())
        raw = rng.normal(size=(naux, nao, nao))
        adjoint = cp.asarray(raw + raw.transpose(0, 2, 1))
        source_order = adjoint.reshape(naux, -1).T
        expected = source_native_module._sorted_source(
            optimizer,
            source_order,
        )
        packed = pack_gamma_cderi_adjoint(adjoint)
        transformed = transform_gamma_sources_to_sorted_packed(
            optimizer,
            (packed,),
            auxiliary_block_size=1,
        )
        actual = unpack_lower_triangle(
            transformed.values[0],
            transformed.nao,
        )
        np.testing.assert_allclose(
            cp.asnumpy(actual),
            cp.asnumpy(expected),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        pair_addresses = optimizer.pair_and_diag_indices(
            cart=True,
            original_ao_order=False,
        )[0]
        dense_short = source_native_module.source_short_range.contract_gamma_batch(
            optimizer,
            expected[None],
            pair_addresses,
        )
        packed_short = (
            source_native_module.source_short_range.contract_gamma_packed_batch(
                optimizer,
                transformed,
                pair_addresses,
            )
        )
        dense_reciprocal = (
            source_native_module.source_reciprocal.contract_gamma_batch(
                optimizer,
                expected[None],
            )
        )
        packed_reciprocal = (
            source_native_module.source_reciprocal.contract_gamma_packed_batch(
                optimizer,
                transformed,
            )
        )
        np.testing.assert_allclose(
            cp.asnumpy(packed_short),
            cp.asnumpy(dense_short),
            atol=1.0e-9,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            cp.asnumpy(packed_reciprocal),
            cp.asnumpy(dense_reciprocal),
            atol=1.0e-9,
            rtol=0.0,
        )
        self.assertEqual(
            packed.values.shape[0],
            nao * (nao + 1) // 2,
        )
        self.assertEqual(
            packed.nbytes,
            source_order.nbytes * (nao + 1) // (2 * nao),
        )

    def test_blocked_auxiliary_density_matches_strided_einsum(self):
        rng = np.random.default_rng(30)
        naux, nao = 5, 4
        storage = cp.asarray(
            rng.normal(size=(nao, 1, nao, naux)),
            dtype=cp.float64,
        )
        factor = storage.transpose(1, 3, 0, 2)[0]
        reference = cp.asarray(rng.normal(size=(1, nao, nao)))
        response = cp.asarray(
            rng.normal(size=(1, nao, nao))
            + 1j * rng.normal(size=(1, nao, nao))
        )
        diagnostics = {}
        actual = auxiliary_density_contractions(
            {(0, 0): factor},
            (reference, response),
            max_auxiliary_block=2,
            memory_diagnostics=diagnostics,
        )
        expected = tuple(
            cp.einsum("pij,ji->p", factor, density[0], optimize=True)
            for density in (reference, response)
        )

        for value, witness in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                cp.asnumpy(value),
                cp.asnumpy(witness),
                atol=1.0e-11,
                rtol=1.0e-11,
            )
        self.assertEqual(diagnostics["density_count"], 2)
        self.assertEqual(diagnostics["execution"]["block_count"], 3)

    def test_auxiliary_density_oom_retries_without_partial_commit(self):
        rng = np.random.default_rng(33)
        naux, nao = 4, 3
        factor = cp.asarray(rng.normal(size=(naux, nao, nao)))
        reference = cp.asarray(rng.normal(size=(1, nao, nao)))
        response = cp.asarray(rng.normal(size=(1, nao, nao)))
        expected = auxiliary_density_contractions(
            {(0, 0): factor},
            (reference, response),
            max_auxiliary_block=1,
        )
        original = gdf_jk_derivative_module._plain_matrix_vector
        failed = False

        def fail_first_large_block(matrix, vector):
            nonlocal failed
            if not failed and matrix.shape[0] > 1:
                failed = True
                raise RuntimeError("cudaErrorMemoryAllocation: out of memory")
            return original(matrix, vector)

        diagnostics = {}
        with mock.patch.object(
            gdf_jk_derivative_module,
            "_plain_matrix_vector",
            side_effect=fail_first_large_block,
        ):
            actual = auxiliary_density_contractions(
                {(0, 0): factor},
                (reference, response),
                memory_fraction=1.0,
                minimum_reserve_bytes=0,
                max_auxiliary_block=naux,
                memory_diagnostics=diagnostics,
            )

        for value, witness in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                cp.asnumpy(value),
                cp.asnumpy(witness),
                atol=1.0e-11,
                rtol=1.0e-11,
            )
        execution = diagnostics["execution"]
        self.assertEqual(execution["rebatch_count"], 1)
        self.assertEqual(execution["completed_auxiliary_count"], naux)
        self.assertEqual(execution["final_block_size"], 2)

    def test_water64_dz_auxiliary_density_plan_is_bounded(self):
        plan = plan_auxiliary_density_blocks(
            1472,
            1472,
            np.float64,
            free_device_bytes=int(22 * 1024**3),
        )

        self.assertLessEqual(plan["initial_block_size"], 32)
        self.assertGreaterEqual(plan["reserve_bytes"], 2 * 1024**3)
        self.assertLessEqual(
            plan["initial_block_size"]
            * plan["working_bytes_per_auxiliary"],
            plan["budget_bytes"],
        )

    def test_auxiliary_density_vector_discards_only_roundoff_imaginary(self):
        real = cp.arange(9, dtype=cp.float64).reshape(3, 3)
        roundoff = real.astype(cp.complex128) + 1.0e-13j
        genuine = real.astype(cp.complex128) + 1.0e-6j

        real_vector = gdf_jk_derivative_module._auxiliary_density_vector(
            roundoff
        )
        complex_vector = gdf_jk_derivative_module._auxiliary_density_vector(
            genuine
        )
        self.assertEqual(real_vector.dtype, cp.float64)
        self.assertEqual(complex_vector.dtype, cp.complex128)
        np.testing.assert_allclose(
            cp.asnumpy(real_vector),
            cp.asnumpy(real.T.reshape(-1)),
        )

    def test_fused_gamma_adjoint_matches_resolved_and_stays_real(self):
        rng = np.random.default_rng(31)
        naux, nao = 3, 4
        central = {(0, 0): cp.asarray(rng.normal(size=(naux, nao, nao)))}
        reference = cp.asarray(rng.normal(size=(1, nao, nao)))
        response = cp.asarray(rng.normal(size=(1, nao, nao)))
        auxiliary_reference = cp.asarray(
            rng.normal(size=naux), dtype=cp.complex128
        )
        auxiliary_response = cp.asarray(
            rng.normal(size=naux), dtype=cp.complex128
        )

        target, bilinear = build_pair_cderi_adjoints(
            central,
            (0, 0),
            reference,
            response,
            auxiliary_reference,
            auxiliary_response,
        )
        diagnostics = {}
        fused = build_pair_fused_cderi_adjoint(
            central,
            (0, 0),
            reference,
            response,
            auxiliary_reference,
            auxiliary_response,
            max_auxiliary_block=1,
            memory_diagnostics=diagnostics,
        )

        self.assertEqual(fused.dtype, cp.float64)
        np.testing.assert_allclose(
            cp.asnumpy(fused),
            cp.asnumpy(target + bilinear),
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        self.assertEqual(diagnostics["execution"]["block_count"], naux)
        self.assertEqual(
            diagnostics["execution"]["completed_auxiliary_count"], naux
        )

    def test_fused_gamma_discards_only_roundoff_imaginary_density(self):
        rng = np.random.default_rng(36)
        naux, nao = 3, 4
        central = {(0, 0): cp.asarray(rng.normal(size=(naux, nao, nao)))}
        reference_real = cp.asarray(rng.normal(size=(1, nao, nao)))
        response_real = cp.asarray(rng.normal(size=(1, nao, nao)))
        reference = reference_real.astype(cp.complex128)
        response = response_real.astype(cp.complex128)
        reference += 1.0e-13j
        response -= 2.0e-13j
        auxiliary_reference = auxiliary_density_contraction(
            central, reference_real
        ).astype(cp.complex128)
        auxiliary_response = auxiliary_density_contraction(
            central, response_real
        ).astype(cp.complex128)

        expected = build_pair_fused_cderi_adjoint(
            central,
            (0, 0),
            reference_real,
            response_real,
            auxiliary_reference.real,
            auxiliary_response.real,
            max_auxiliary_block=1,
        )
        actual = build_pair_fused_cderi_adjoint(
            central,
            (0, 0),
            reference,
            response,
            auxiliary_reference,
            auxiliary_response,
            max_auxiliary_block=1,
        )

        self.assertEqual(actual.dtype, cp.float64)
        np.testing.assert_allclose(
            cp.asnumpy(actual),
            cp.asnumpy(expected),
            atol=1.0e-11,
            rtol=1.0e-11,
        )

        complex_response = response_real.astype(cp.complex128)
        complex_response += 1.0e-6j
        complex_auxiliary_response = auxiliary_density_contraction(
            central, complex_response
        )
        retained = build_pair_fused_cderi_adjoint(
            central,
            (0, 0),
            reference,
            complex_response,
            auxiliary_reference,
            complex_auxiliary_response,
            max_auxiliary_block=1,
        )
        self.assertEqual(retained.dtype, cp.complex128)

    def test_fused_gamma_builds_packed_from_origin(self):
        rng = np.random.default_rng(37)
        naux, nao = 4, 5
        central = {(0, 0): cp.asarray(rng.normal(size=(naux, nao, nao)))}
        central[(0, 0)] += central[(0, 0)].transpose(0, 2, 1)
        reference = cp.asarray(rng.normal(size=(1, nao, nao)))
        reference += reference.transpose(0, 2, 1)
        response = cp.asarray(rng.normal(size=(1, nao, nao)))
        response += response.transpose(0, 2, 1)
        auxiliary_reference = auxiliary_density_contraction(
            central, reference
        )
        auxiliary_response = auxiliary_density_contraction(
            central, response
        )
        dense = build_pair_fused_cderi_adjoint(
            central,
            (0, 0),
            reference,
            response,
            auxiliary_reference,
            auxiliary_response,
            max_auxiliary_block=1,
        )
        diagnostics = {}
        packed = build_pair_fused_gamma_cderi_adjoint(
            central,
            (0, 0),
            reference,
            response,
            auxiliary_reference,
            auxiliary_response,
            max_auxiliary_block=1,
            memory_diagnostics=diagnostics,
        )
        expected = pack_gamma_cderi_adjoint(dense)

        self.assertIsInstance(packed, PackedGammaCDERIAdjoint)
        self.assertTrue(packed.values.T.flags.c_contiguous)
        np.testing.assert_allclose(
            cp.asnumpy(packed.values),
            cp.asnumpy(expected.values),
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        storage = diagnostics["storage"]
        self.assertEqual(storage["mode"], "gamma_lower_triangle_from_origin")
        self.assertLess(
            storage["packed_output_bytes"],
            storage["dense_output_equivalent_bytes"],
        )

    def test_prepacked_gamma_source_solve_reuses_storage(self):
        rng = np.random.default_rng(38)
        naux, nao = 4, 3
        factor = np.tril(rng.normal(size=(naux, naux)))
        factor[np.diag_indices(naux)] += 5.0
        dense = cp.asarray(rng.normal(size=(naux, nao, nao)))
        dense += dense.transpose(0, 2, 1)
        expected = push_packed_gamma_adjoint_through_source(
            cp.asarray(factor),
            dense.copy(),
        )
        lower = pack_gamma_cderi_adjoint(dense)
        storage = cp.ascontiguousarray(lower.values.T)
        packed = PackedGammaCDERIAdjoint(values=storage.T, nao=nao)
        pointer = packed.values.data.ptr
        actual = push_packed_gamma_adjoint_through_source(
            cp.asarray(factor),
            packed,
        )

        self.assertEqual(actual.values.data.ptr, pointer)
        np.testing.assert_allclose(
            cp.asnumpy(actual.values),
            cp.asnumpy(expected.values),
            atol=1.0e-11,
            rtol=1.0e-11,
        )

    def test_fused_gamma_packed_oom_retries_without_partial_commit(self):
        rng = np.random.default_rng(39)
        naux, nao = 4, 3
        central = {(0, 0): cp.asarray(rng.normal(size=(naux, nao, nao)))}
        central[(0, 0)] += central[(0, 0)].transpose(0, 2, 1)
        reference = cp.asarray(rng.normal(size=(1, nao, nao)))
        reference += reference.transpose(0, 2, 1)
        response = cp.asarray(rng.normal(size=(1, nao, nao)))
        response += response.transpose(0, 2, 1)
        auxiliary_reference = auxiliary_density_contraction(
            central, reference
        )
        auxiliary_response = auxiliary_density_contraction(
            central, response
        )
        expected = build_pair_fused_gamma_cderi_adjoint(
            central,
            (0, 0),
            reference,
            response,
            auxiliary_reference,
            auxiliary_response,
            max_auxiliary_block=1,
        )
        original = source_adjoint_module._build_fused_adjoint_block
        failed = False

        def fail_first_large_block(*args, start, stop, **kwargs):
            nonlocal failed
            if not failed and stop - start > 1:
                failed = True
                raise RuntimeError("cudaErrorMemoryAllocation: out of memory")
            return original(*args, start=start, stop=stop, **kwargs)

        diagnostics = {}
        with mock.patch.object(
            source_adjoint_module,
            "_build_fused_adjoint_block",
            side_effect=fail_first_large_block,
        ):
            actual = build_pair_fused_gamma_cderi_adjoint(
                central,
                (0, 0),
                reference,
                response,
                auxiliary_reference,
                auxiliary_response,
                memory_fraction=1.0,
                minimum_reserve_bytes=0,
                max_auxiliary_block=naux,
                memory_diagnostics=diagnostics,
            )

        np.testing.assert_allclose(
            cp.asnumpy(actual.values),
            cp.asnumpy(expected.values),
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        execution = diagnostics["execution"]
        self.assertEqual(execution["rebatch_count"], 1)
        self.assertEqual(execution["completed_auxiliary_count"], naux)
        self.assertEqual(execution["final_block_size"], 2)

    def test_fused_adjoint_oom_retries_without_partial_commit(self):
        rng = np.random.default_rng(32)
        naux, nao = 4, 3
        central = {(0, 0): cp.asarray(rng.normal(size=(naux, nao, nao)))}
        reference = cp.asarray(rng.normal(size=(1, nao, nao)))
        response = cp.asarray(rng.normal(size=(1, nao, nao)))
        auxiliary_reference = cp.asarray(rng.normal(size=naux))
        auxiliary_response = cp.asarray(rng.normal(size=naux))
        expected = build_pair_fused_cderi_adjoint(
            central,
            (0, 0),
            reference,
            response,
            auxiliary_reference,
            auxiliary_response,
            max_auxiliary_block=1,
        )
        original = source_adjoint_module._build_fused_adjoint_block
        failed = False

        def fail_first_large_block(*args, start, stop, **kwargs):
            nonlocal failed
            if not failed and stop - start > 1:
                failed = True
                raise RuntimeError("cudaErrorMemoryAllocation: out of memory")
            return original(*args, start=start, stop=stop, **kwargs)

        diagnostics = {}
        with mock.patch.object(
            source_adjoint_module,
            "_build_fused_adjoint_block",
            side_effect=fail_first_large_block,
        ):
            actual = build_pair_fused_cderi_adjoint(
                central,
                (0, 0),
                reference,
                response,
                auxiliary_reference,
                auxiliary_response,
                memory_fraction=1.0,
                minimum_reserve_bytes=0,
                max_auxiliary_block=naux,
                memory_diagnostics=diagnostics,
            )

        np.testing.assert_allclose(
            cp.asnumpy(actual),
            cp.asnumpy(expected),
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        execution = diagnostics["execution"]
        self.assertEqual(execution["rebatch_count"], 1)
        self.assertEqual(execution["completed_auxiliary_count"], naux)
        self.assertEqual(execution["final_block_size"], 2)

    def test_blocked_fused_complex_pair_matches_resolved(self):
        rng = np.random.default_rng(34)
        nkpts, naux, nao = 2, 4, 3

        def complex_random(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        central = {
            (0, 0): cp.asarray(complex_random((naux, nao, nao))),
            (0, 1): cp.asarray(complex_random((naux, nao, nao))),
            (1, 1): cp.asarray(complex_random((naux, nao, nao))),
        }
        reference = cp.asarray(complex_random((nkpts, nao, nao)))
        response = cp.asarray(complex_random((nkpts, nao, nao)))
        auxiliary_reference = cp.asarray(complex_random((naux,)))
        auxiliary_response = cp.asarray(complex_random((naux,)))

        target, bilinear = build_pair_cderi_adjoints(
            central,
            (0, 1),
            reference,
            response,
            auxiliary_reference,
            auxiliary_response,
        )
        fused = build_pair_fused_cderi_adjoint(
            central,
            (0, 1),
            reference,
            response,
            auxiliary_reference,
            auxiliary_response,
            max_auxiliary_block=1,
        )

        np.testing.assert_allclose(
            cp.asnumpy(fused),
            cp.asnumpy(target + bilinear),
            atol=1.0e-11,
            rtol=1.0e-11,
        )

    def test_water64_dz_fused_adjoint_plan_is_bounded(self):
        free = int(21.5 * 1024**3)
        plan = plan_fused_adjoint_auxiliary_blocks(
            1472,
            1472,
            np.float64,
            free_device_bytes=free,
        )

        self.assertLessEqual(plan["initial_block_size"], 32)
        self.assertGreaterEqual(plan["reserve_bytes"], 2 * 1024**3)
        self.assertLessEqual(
            plan["initial_block_size"]
            * plan["working_bytes_per_auxiliary"],
            plan["budget_bytes"],
        )

    def test_inplace_source_reverse_solve_matches_allocating_route(self):
        rng = np.random.default_rng(35)
        for dtype in (np.float64, np.complex128):
            naux, nao = 4, 3
            factor = np.tril(rng.normal(size=(naux, naux)))
            factor[np.diag_indices(naux)] += 5.0
            adjoint = rng.normal(size=(naux, nao, nao))
            if dtype == np.complex128:
                factor = factor + 0.1j * np.tril(
                    rng.normal(size=factor.shape), k=-1
                )
                adjoint = adjoint + 1j * rng.normal(size=adjoint.shape)
            factor = cp.asarray(factor, dtype=dtype)
            adjoint = cp.asarray(adjoint, dtype=dtype)
            expected = push_cderi_adjoint_through_source(
                factor,
                adjoint,
            )
            working = adjoint.copy()
            actual = push_cderi_adjoint_through_source(
                factor,
                working,
                overwrite=True,
            )

            self.assertTrue(cp.shares_memory(actual, working))
            np.testing.assert_allclose(
                cp.asnumpy(actual),
                cp.asnumpy(expected),
                atol=1.0e-11,
                rtol=1.0e-11,
            )

    def test_gamma_source_sorting_releases_cderi_under_pressure(self):
        releases = []

        def release(*, reason):
            releases.append(reason)
            return {
                "released": True,
                "reason": reason,
                "released_cderi_bytes": 4096,
            }

        provider = SimpleNamespace(release_cderi_cache=release)
        optimizer = SimpleNamespace(
            cell=SimpleNamespace(nao=4),
            auxcell=SimpleNamespace(nao=5),
        )
        groups = ({(0, 0): cp.zeros((9, 3), dtype=cp.float64)},)
        with mock.patch.object(
            contracted_response_module.cp.cuda.runtime,
            "memGetInfo",
            return_value=(1, 2),
        ):
            metadata = (
                contracted_response_module._release_gamma_central_if_needed(
                    provider,
                    optimizer,
                    np.zeros((1, 3)),
                    groups,
                    minimum_reserve_bytes=0,
                )
            )

        self.assertEqual(
            releases, ["native_gamma_source_adjoint_complete"]
        )
        self.assertTrue(metadata["released"])

    def test_gamma_source_releases_cderi_even_when_sorting_fits(self):
        releases = []

        def release(*, reason):
            releases.append(reason)
            return {"released": True, "reason": reason}

        provider = SimpleNamespace(release_cderi_cache=release)
        optimizer = SimpleNamespace(
            cell=SimpleNamespace(nao=4),
            auxcell=SimpleNamespace(nao=5),
        )
        groups = ({(0, 0): cp.zeros((9, 3), dtype=cp.float64)},)
        with mock.patch.object(
            contracted_response_module.cp.cuda.runtime,
            "memGetInfo",
            return_value=(10**9, 2 * 10**9),
        ):
            metadata = (
                contracted_response_module._release_gamma_central_if_needed(
                    provider,
                    optimizer,
                    np.zeros((1, 3)),
                    groups,
                    minimum_reserve_bytes=0,
                )
            )

        self.assertTrue(metadata["source_sorting_would_fit_without_release"])
        self.assertEqual(
            releases, ["native_gamma_source_adjoint_complete"]
        )

    def test_single_gamma_source_consumes_owner_without_stack_copy(self):
        class IdentityTransform:
            def __init__(self, count):
                self.cell = SimpleNamespace(nao_nr=lambda: count)

            def apply_C_mat_CT(self, value):
                return value.copy()

            def apply_C_dot(self, value, axis=0):
                self.assert_axis = axis
                return value.copy()

        optimizer = SimpleNamespace(
            cell=IdentityTransform(2),
            auxcell=IdentityTransform(3),
        )
        owner = [cp.arange(12, dtype=cp.float64).reshape(4, 3)]
        with mock.patch.object(
            source_native_module.cp,
            "stack",
            side_effect=AssertionError("single source must not be stacked"),
        ):
            transformed = source_native_module._gamma_source_batch(
                optimizer,
                owner,
                consume=True,
            )

        self.assertEqual(owner, [])
        self.assertEqual(transformed.shape, (1, 2, 2, 3))

    def test_dzvp_consumed_source_transform_matches_reference(self):
        cell = _water_dzvp_cell()
        provider = gpu_df.RSDF(
            cell,
            kpts=np.zeros((1, 3)),
            auxbasis="DZVP-GTH",
        )
        witness = provider._build_parameter_witness(np.zeros((1, 3)))
        parameters = parameters_from_provider(witness)
        optimizer = build_source_optimizer(
            cell,
            witness.auxcell,
            parameters,
            bvk_kmesh=parameters.bvk_kmesh,
        )
        rng = np.random.default_rng(33)
        source = cp.asarray(
            rng.normal(size=(cell.nao**2, witness.auxcell.nao_nr()))
        )
        expected = source_native_module._sorted_gamma_source(
            optimizer,
            source,
        )
        owner = [source.copy()]
        actual = source_native_module._gamma_source_batch(
            optimizer,
            owner,
            consume=True,
        )[0]

        self.assertEqual(owner, [])
        np.testing.assert_allclose(
            cp.asnumpy(actual),
            cp.asnumpy(expected),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_exact_central_source_builder_memory_is_bounded(self):
        reported, plan = _source_builder_memory_plan(
            100 * 1024**3,
            source_mesh=(25, 25, 25),
            naux=384,
        )

        self.assertEqual(reported, 512 * 1024**2)
        self.assertEqual(plan["reported_free_device_bytes"], reported)
        self.assertLess(
            plan["estimated_full_auxiliary_grid_cache_bytes"],
            reported,
        )

    def test_compressed_device_cache_plan_is_bounded_and_gamma_only(self):
        enabled = _compressed_device_cache_plan(
            128 * 1024**2,
            available_bytes=32 * 1024**3,
            policy="auto",
            is_gamma=True,
        )
        oversized = _compressed_device_cache_plan(
            768 * 1024**2,
            available_bytes=32 * 1024**3,
            policy="auto",
            is_gamma=True,
        )
        sampled_k = _compressed_device_cache_plan(
            128 * 1024**2,
            available_bytes=32 * 1024**3,
            policy="auto",
            is_gamma=False,
        )

        self.assertTrue(enabled["enabled"])
        self.assertFalse(oversized["enabled"])
        self.assertEqual(oversized["reason"], "over_auto_budget")
        self.assertFalse(sampled_k["enabled"])
        self.assertEqual(sampled_k["reason"], "sampled_k_not_implemented")

    def test_exact_gpu_source_provider_avoids_canonical_host_cderi(self):
        cell = _cell()
        kpts = cell.make_kpts((1, 1, 1), wrap_around=True)
        provider = gpu_df.RSDF(
            cell,
            kpts=kpts,
            auxbasis="gth-szv",
            central_setup="exact_gpu_source",
        )

        provider.build(j_only=False)

        self.assertIsNone(provider.cpu_witness._cderi)
        self.assertEqual(provider.central_setup, "exact_gpu_source")
        self.assertTrue(provider.uses_native_jk)
        self.assertIsNone(provider._dc_cderi_blocks)
        self.assertIs(
            provider._cderi,
            provider._dc_exact_central.compressed_blocks,
        )
        self.assertIsNotNone(provider._cderi_idx)
        self.assertTrue(
            all(
                not isinstance(value, cp.ndarray)
                for value in provider._cderi.values()
            )
        )
        self.assertFalse(
            provider._dc_central_build_metadata["canonical_materialized"]
        )
        self.assertEqual(
            provider._dc_central_build_metadata["compressed_storage_location"],
            "mapped_host",
        )
        self.assertEqual(
            provider._dc_central_build_metadata["compressed_device_cache"][
                "reason"
            ],
            "policy_host",
        )
        self.assertEqual(
            provider.capability()["central_build_metadata"]["source_owner"],
            "GPU4PySCF CUDA three-center builder",
        )

    def test_exact_gpu_source_dispatches_jk_to_native_compressed_route(self):
        cell = _cell()
        kpts = cell.make_kpts((1, 1, 1), wrap_around=True)
        provider = gpu_df.RSDF(
            cell,
            kpts=kpts,
            auxbasis="gth-szv",
            central_setup="exact_gpu_source",
        )
        provider.build(j_only=False)
        expected = (object(), object())

        with mock.patch.object(
            gdf_provider_module.GPU4PySCFGDF,
            "get_jk",
            return_value=expected,
        ) as native_get_jk:
            actual = provider.get_jk(
                cp.eye(cell.nao_nr()),
                with_j=True,
                with_k=True,
            )

        self.assertIs(actual, expected)
        native_get_jk.assert_called_once()
        self.assertIsNone(provider._dc_cderi_blocks)

    def test_exact_gpu_source_splits_complex_gamma_jk_into_real_calls(self):
        cell = _cell()
        kpts = cell.make_kpts((1, 1, 1), wrap_around=True)
        provider = gpu_df.RSDF(
            cell,
            kpts=kpts,
            auxbasis="gth-szv",
            central_setup="exact_gpu_source",
        )
        provider.build(j_only=False)
        shape = (cell.nao_nr(), cell.nao_nr())
        real_j = cp.full(shape, 1.0)
        real_k = cp.full(shape, 2.0)
        imag_j = cp.full(shape, 3.0)
        imag_k = cp.full(shape, 4.0)

        with mock.patch.object(
            gdf_provider_module.GPU4PySCFGDF,
            "get_jk",
            side_effect=((real_j, real_k), (imag_j, imag_k)),
        ) as native_get_jk:
            actual_j, actual_k = provider.get_jk(
                cp.eye(cell.nao_nr(), dtype=cp.complex128) * (1.0 + 2.0j),
                with_j=True,
                with_k=True,
            )

        self.assertEqual(native_get_jk.call_count, 2)
        np.testing.assert_allclose(
            cp.asnumpy(actual_j),
            cp.asnumpy(real_j + 1j * imag_j),
        )
        np.testing.assert_allclose(
            cp.asnumpy(actual_k),
            cp.asnumpy(real_k + 1j * imag_k),
        )

    def test_source_hf_profile_resolves_jk_and_scf_phases(self):
        method = _method(
            _cell(),
            (1, 1, 1),
            device="gpu",
            central_setup="exact_gpu_source",
        )

        profile = method.metadata()["source_hf_profile"]
        jk_profile = profile["jk"]
        self.assertGreater(jk_profile["logical_call_count"], 0)
        self.assertGreater(jk_profile["device_total_s"], 0.0)
        self.assertIsNotNone(jk_profile["first_call_device_s"])
        self.assertGreaterEqual(jk_profile["repeated_call_device_s"], 0.0)
        self.assertGreater(profile["phases"]["diagonalization"]["call_count"], 0)
        self.assertGreater(profile["phases"]["density_update"]["call_count"], 0)
        self.assertGreaterEqual(profile["unattributed_scf_wall_s"], 0.0)
        self.assertIs(
            method.with_df.capability()["jk_profile"],
            jk_profile,
        )

    def test_exact_gpu_source_canonical_cache_is_lazy_and_rebuildable(self):
        cell = _cell()
        kpts = cell.make_kpts((1, 1, 1), wrap_around=True)
        provider = gpu_df.RSDF(
            cell,
            kpts=kpts,
            auxbasis="gth-szv",
            central_setup="exact_gpu_source",
        )
        provider.build(j_only=False)
        first = cp.asnumpy(provider.central_cderi_blocks[(0, 0)]).copy()
        self.assertTrue(
            provider._dc_central_build_metadata["canonical_materialized"]
        )
        self.assertIsNone(provider._cderi)
        self.assertFalse(
            provider._dc_central_build_metadata["compressed_resident"]
        )
        eviction = provider.release_cderi_cache(reason="unit-test")

        self.assertIsNone(provider._cderi)
        self.assertIsNone(provider._dc_cderi_blocks)
        self.assertFalse(eviction["compressed_jk_factors_retained"])
        self.assertFalse(
            provider._dc_central_build_metadata["canonical_materialized"]
        )
        np.testing.assert_allclose(
            cp.asnumpy(provider.central_cderi_blocks[(0, 0)]),
            first,
            atol=1.0e-11,
            rtol=0.0,
        )

    def test_reference_cderi_release_requires_rebuild_for_next_jk(self):
        cell = _cell()
        kpts = cell.make_kpts((1, 1, 1), wrap_around=True)
        provider = gpu_df.RSDF(
            cell,
            kpts=kpts,
            auxbasis="gth-szv",
            central_setup="reference",
        )
        provider.build(j_only=False)

        provider.release_cderi_cache(reason="unit-test")

        self.assertIsNone(provider._cderi)
        self.assertIsNone(provider._dc_cderi_blocks)

    def test_completed_response_releases_provider_cderi(self):
        release = mock.Mock(
            return_value={"released": True, "released_cderi_bytes": 17}
        )
        provider = SimpleNamespace(release_cderi_cache=release)

        metadata = gdf_gradient_module._release_completed_rsdf_response(
            provider
        )

        release.assert_called_once_with(
            reason="analytic_gradient_response_complete"
        )
        self.assertEqual(
            metadata["provider_cderi_release"]["released_cderi_bytes"],
            17,
        )
        self.assertGreaterEqual(
            metadata["pool_reserved_bytes_before"],
            metadata["pool_reserved_bytes_after"],
        )

    def test_exact_central_gamma_matches_cpu_storage(self):
        cell = _cell()
        kpts = cell.make_kpts((1, 1, 1), wrap_around=True)
        witness = cpu_df.RSDF(cell, kpts=kpts, auxbasis="gth-szv")
        witness.build(j_only=False)
        parameters = parameters_from_provider(witness)
        reference = load_cderi_storage(witness).blocks

        result = build_exact_cderi(
            cell,
            witness.auxcell,
            kpts,
            parameters,
        )

        np.testing.assert_allclose(
            cp.asnumpy(result.blocks[(0, 0)]),
            reference[(0, 0)],
            atol=1.0e-9,
            rtol=0.0,
        )
        self.assertIsNone(result.negative_blocks)
        self.assertEqual(result.metadata["pair_count"], 1)
        self.assertEqual(
            result.metadata["short_range_metric_owner"],
            "PySCF screened two-center primitive",
        )

    def test_exact_central_kpoint_matches_cpu_storage(self):
        cell = _cell()
        kpts = cell.make_kpts((2, 1, 1), wrap_around=True)
        witness = cpu_df.RSDF(cell, kpts=kpts, auxbasis="gth-szv")
        witness.build(j_only=False)
        parameters = parameters_from_provider(witness)
        reference = load_cderi_storage(witness).blocks

        result = build_exact_cderi(
            cell,
            witness.auxcell,
            kpts,
            parameters,
        )

        for pair in canonical_pairs(len(kpts)):
            np.testing.assert_allclose(
                cp.asnumpy(result.blocks[pair]),
                reference[pair],
                atol=1.0e-9,
                rtol=0.0,
            )
        self.assertEqual(result.metadata["pair_count"], 3)

    def test_exact_central_handles_heterogeneous_auxiliary_basis(self):
        cell = _water_cell()
        kpts = cell.make_kpts((1, 1, 1), wrap_around=True)
        witness = cpu_df.RSDF(cell, kpts=kpts, auxbasis="gth-szv")
        witness.build(j_only=False)
        parameters = parameters_from_provider(witness)
        reference = load_cderi_storage(witness).blocks[(0, 0)]

        result = build_exact_cderi(
            cell,
            witness.auxcell,
            kpts,
            parameters,
        )

        np.testing.assert_allclose(
            cp.asnumpy(result.blocks[(0, 0)]),
            reference,
            atol=1.0e-9,
            rtol=0.0,
        )

    def test_pseudopotential_grid_plan_caps_large_device_block(self):
        with mock.patch.object(
            pp_value_module,
            "get_avail_mem",
            return_value=100 * 1024**3,
        ):
            plan = pp_value_module._plan_grid_blocks(
                100_000,
                nao_pairs=73_920,
            )

        self.assertLessEqual(plan.initial_block_size, 4096)
        self.assertLessEqual(
            plan.initial_block_size * plan.bytes_per_grid,
            plan.budget_bytes,
        )

    def test_reference_pseudopotential_value_is_cached_and_reset(self):
        method = _method(_cell(), (1, 1, 1), device="gpu")
        provider = method.with_df
        provider._clear_pp_cache()
        with mock.patch.object(
            rsdf_provider_module,
            "build_pp_value",
            wraps=rsdf_provider_module.build_pp_value,
        ) as get_pp:
            first = provider.get_pp()
            reference = first.copy()
            first += 1.0
            second = provider.get_pp()

        np.testing.assert_allclose(
            cp.asnumpy(second),
            cp.asnumpy(reference),
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(get_pp.call_count, 1)
        self.assertEqual(provider._dc_pp_cache_hits, 1)
        self.assertEqual(provider._dc_pp_cache_misses, 1)
        provider.reset()
        self.assertEqual(provider._dc_pp_cache, {})
        self.assertEqual(provider._dc_pp_cache_hits, 0)
        self.assertEqual(provider._dc_pp_cache_misses, 0)

    def test_gpu_pseudopotential_value_matches_cpu_witness(self):
        for kmesh in ((1, 1, 1), (2, 1, 1)):
            method = _method(_cell(), kmesh, device="gpu")
            provider = method.with_df
            kpts = np.asarray(provider.kpts)
            cpu_value = provider.cpu_witness.get_pp(kpts)
            provider._clear_pp_cache()
            gpu_value = provider.get_pp(kpts)

            np.testing.assert_allclose(
                cp.asnumpy(gpu_value),
                np.asarray(cpu_value),
                atol=1.0e-9,
                rtol=0.0,
            )

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

    def test_source_reciprocal_plan_caps_large_device_block(self):
        with mock.patch.object(
            source_reciprocal_module,
            "get_avail_mem",
            return_value=100 * 1024**3,
        ):
            plan = source_reciprocal_module._memory_plan(
                15_625,
                nao=384,
                naux=384,
            )

        self.assertLessEqual(plan.initial_block_size, 512)
        self.assertEqual(plan.memory_fraction, 0.15)

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

    def test_metric_adjoint_avoids_full_mixed_dtype_promotion(self):
        rng = np.random.default_rng(44)
        naux, nao = 4, 3
        factor = np.tril(rng.normal(size=(naux, naux)))
        factor[np.diag_indices(naux)] += 5.0
        # Match the auxiliary-fast strided Gamma layout returned by the
        # GPU4PySCF CDERI unpacker after AO padding is removed.
        storage = cp.asarray(
            rng.normal(size=(nao, 1, nao, naux)),
            dtype=cp.float64,
        )
        value = storage.transpose(1, 3, 0, 2)[0]
        adjoint = cp.asarray(
            rng.normal(size=value.shape)
            + 1j * rng.normal(size=value.shape),
            dtype=cp.complex128,
        )
        factorization = CDERIFactorization(
            factor=cp.asarray(factor),
            value=value,
            nao=nao,
        )

        actual = push_cderi_adjoint_through_metric(
            factorization,
            adjoint,
        )
        promoted = CDERIFactorization(
            factor=cp.asarray(factor, dtype=cp.complex128),
            value=value.astype(cp.complex128),
            nao=nao,
        )
        expected = push_cderi_adjoint_through_metric(
            promoted,
            adjoint,
        )

        np.testing.assert_allclose(
            cp.asnumpy(actual),
            cp.asnumpy(expected),
            atol=1.0e-11,
            rtol=1.0e-11,
        )

        mixed_precision = CDERIFactorization(
            factor=cp.asarray(factor),
            value=value.astype(cp.float32),
            nao=nao,
        )
        mixed_actual = push_cderi_adjoint_through_metric(
            mixed_precision,
            adjoint,
        )
        mixed_promoted = CDERIFactorization(
            factor=cp.asarray(factor, dtype=cp.complex128),
            value=value.astype(cp.complex128),
            nao=nao,
        )
        mixed_expected = push_cderi_adjoint_through_metric(
            mixed_promoted,
            adjoint,
        )
        np.testing.assert_allclose(
            cp.asnumpy(mixed_actual),
            cp.asnumpy(mixed_expected),
            atol=1.0e-6,
            rtol=1.0e-6,
        )

    def test_packed_gamma_metric_adjoint_matches_dense(self):
        rng = np.random.default_rng(45)
        naux, nao = 5, 4
        factor = np.tril(rng.normal(size=(naux, naux)))
        factor[np.diag_indices(naux)] += 6.0
        value = cp.asarray(rng.normal(size=(naux, nao, nao)))
        value += value.transpose(0, 2, 1)
        dense = cp.asarray(rng.normal(size=(naux, nao, nao)))
        dense += dense.transpose(0, 2, 1)
        factorization = CDERIFactorization(
            factor=cp.asarray(factor),
            value=value,
            nao=nao,
        )
        lower = pack_gamma_cderi_adjoint(dense)
        packed = PackedGammaCDERIAdjoint(
            values=cp.ascontiguousarray(lower.values.T).T,
            nao=nao,
        )

        actual = push_cderi_adjoint_through_metric(
            factorization,
            packed,
        )
        expected = push_cderi_adjoint_through_metric(
            factorization,
            dense,
        )
        np.testing.assert_allclose(
            cp.asnumpy(actual),
            cp.asnumpy(expected),
            atol=1.0e-10,
            rtol=1.0e-10,
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
        production = build_rsdf_contracted_response(
            method.with_df,
            reference,
            response,
            native_source_derivatives=True,
            native_metric_derivatives=True,
            observable_mode="fused",
            validate_central_source=False,
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
        np.testing.assert_allclose(
            cp.asnumpy(production.coupled_target_j_bilinear),
            cp.asnumpy(fused.coupled_target_j_bilinear),
            atol=1.0e-10,
            rtol=0.0,
        )
        self.assertIsNone(production.value_parity_max_abs)
        self.assertEqual(production.profile.host_source_value_seconds, 0.0)
        self.assertFalse(production.profile.central_source_validation)
        fused_memory = production.profile.metadata()[
            "device_fused_adjoint_memory"
        ]
        self.assertEqual(len(fused_memory["pairs"]), 1)
        self.assertEqual(fused_memory["pairs"][0]["pair"], [0, 0])
        self.assertEqual(
            fused_memory["pairs"][0]["execution"][
                "completed_auxiliary_count"
            ],
            int(method.with_df.auxcell.nao_nr()),
        )
        self.assertEqual(
            fused_memory["pairs"][0]["storage"]["mode"],
            "gamma_lower_triangle_from_origin",
        )
        self.assertTrue(
            fused_memory["pairs"][0]["packed_gamma"][
                "built_packed_from_origin"
            ]
        )
        self.assertFalse(
            fused_memory["pairs"][0]["packed_gamma_metric"][
                "dense_cderi_adjoint_reconstructed"
            ]
        )
        self.assertNotIn(
            "RSDF three-center source value",
            production.profile.metadata()["host_owned_operations"],
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

    def test_exact_gpu_source_gamma_energy_gradient_parity(self):
        self._assert_parity(
            (1, 1, 1),
            central_setup="exact_gpu_source",
        )

    def test_exact_gpu_source_kpoint_energy_gradient_parity(self):
        self._assert_parity(
            (2, 1, 1),
            central_setup="exact_gpu_source",
        )

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

    def _assert_parity(self, kmesh, *, central_setup=None):
        cell = _cell()
        cpu = _method(cell, kmesh, device="cpu")
        gpu = _method(
            cell,
            kmesh,
            device="gpu",
            central_setup=central_setup,
        )
        cpu_gradient = cpu.nuc_grad_method().kernel()
        gpu_gradient_object = gpu.nuc_grad_method()
        self.assertEqual(gpu_gradient_object.rsdf_derivative_omega_scale, 1.0)
        gpu_gradient = gpu_gradient_object.kernel()
        self.assertLess(abs(cpu.e_tot - gpu.e_tot), 1.0e-8)
        self.assertLess(np.max(np.abs(cpu_gradient - gpu_gradient)), 1.0e-7)
        metadata = gpu_gradient_object.result.metadata()
        capability = gpu.with_df.capability()
        if central_setup is not None:
            self.assertEqual(capability["central_setup"], central_setup)
            self.assertEqual(
                metadata["rsdf_central_setup"],
                central_setup,
            )
        self.assertEqual(metadata["backend"], "rsdf")
        self.assertFalse(metadata["silent_cpu_fallback"])
        self.assertTrue(metadata["native_one_electron"])
        self.assertIsNone(metadata["cderi_value_parity_max_abs"])
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
        self.assertFalse(profile["central_source_validation"])
        self.assertEqual(profile["host_source_value_seconds"], 0.0)
        self.assertIsNotNone(
            profile["device_native_source_reciprocal_memory"]
        )
        self.assertEqual(
            metadata["gradient_partition"],
            "explicit_response_coupled",
        )
        self.assertFalse(metadata["target_coulomb_component_resolved"])
        self.assertEqual(
            capability["response_mode"],
            "native_source_native_metric_fused_adjoint",
        )
        self.assertFalse(capability["silent_cpu_fallback"])
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
