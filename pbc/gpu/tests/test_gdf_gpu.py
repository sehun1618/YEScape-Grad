"""Public energy and analytic-gradient gates for the GPU GDF route.

Native mode builds periodic factors with GPU4PySCF and owns source HF, target
PBE, response, local-part-1 pseudopotential response, and final contractions
on the GPU for Gamma and sampled k-points without a CPU DC-DFT fallback.
"""

from __future__ import annotations

import unittest
import warnings
from unittest.mock import patch

import cupy as cp
import numpy as np
from pyscf.pbc import gto
from pyscf_pbc_dcdft import df as cpu_df
from pyscf_pbc_dcdft import dft as cpu_dft
from pyscf_pbc_dcdft.df.gdf.cholesky import differentiate_cderi
from pyscf_pbc_dcdft.df.gdf.jk_derivative import (
    gdf_jk_derivative as cpu_gdf_jk_derivative,
)
from pyscf_pbc_dcdft.df.gdf.local_part1 import LocalPart1Derivative

from gpu4pyscf_pbc_dcdft import df as gpu_df
from gpu4pyscf_pbc_dcdft import dft as gpu_dft
from gpu4pyscf_pbc_dcdft.df.gdf.build_signature import (
    make_gdf_build_signature,
)
from gpu4pyscf_pbc_dcdft.df.gdf.cholesky import CDERIFactorization
from gpu4pyscf_pbc_dcdft.df.gdf.contracted_derivative import (
    _bounded_sr_derivative_batches,
    _build_gamma_derivative_optimizer,
    coulomb_energy_derivative,
    gpu_gdf_grad_krhf,
    gpu_gdf_grad_rhf,
    restricted_hf_bilinear_derivative,
    restricted_hf_bilinear_derivative_one_pass,
    restricted_hf_bilinear_derivative_one_pass_gamma,
    target_j_plus_restricted_hf_bilinear_derivative_one_pass,
)
from gpu4pyscf.gto import mole as gpu_mole
from gpu4pyscf.pbc.df.int3c2e import POOL_SIZE
from gpu4pyscf_pbc_dcdft.df.gdf.jk_derivative import (
    contract_restricted_hf_bilinear_energy_derivative,
    contract_restricted_hf_energy_derivative,
    ewald_exchange_derivative,
    gdf_jk_derivative as gpu_gdf_jk_derivative,
)
from gpu4pyscf_pbc_dcdft.df.gdf.local_part1_derivative import (
    NativeGDFLocalPart1Derivative,
    _validate_short_range_batches,
)
from gpu4pyscf_pbc_dcdft.df.gdf.reciprocal_memory import (
    plan_gamma_reciprocal_blocks,
)
from gpu4pyscf_pbc_dcdft.df.gdf.signed_gamma_native import (
    _gamma_coulomb_vector,
)
from gpu4pyscf_pbc_dcdft.df.gdf.signed_density import (
    build_gamma_signed_density_source,
    build_gamma_factorization_adjoint,
    contract_gamma_signed_density_source,
    contract_kpoint_bilinear_cderi_derivative,
    ewald_restricted_hf_bilinear_derivative,
)


def _gpu_available() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def _cell():
    return gto.Cell(
        atom="H 0 0 0; H 0 0 1.4",
        a=np.eye(3) * 6.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-hf",
        ke_cutoff=15.0,
        verbose=0,
    ).build()


def _water_cell():
    return gto.Cell(
        atom="O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11",
        a=np.eye(3) * 8.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-hf",
        ke_cutoff=15.0,
        verbose=0,
    ).build()


def _water_dzvp_cell():
    return gto.Cell(
        atom="O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11",
        a=np.eye(3) * 8.0,
        unit="Bohr",
        basis="DZVP-GTH",
        pseudo="GTH-PBE",
        ke_cutoff=15.0,
        precision=1.0e-8,
        verbose=0,
    ).build()


def _methods(kmesh, *, setup_mode="native"):
    cell = _cell()
    if kmesh == (1, 1, 1):
        kpts = np.zeros((1, 3))
        cpu = cpu_dft.DCRKS(cell, xc="PBE")
        gpu = gpu_dft.DCRKS(cell, xc="PBE")
    else:
        kpts = cell.make_kpts(kmesh)
        cpu = cpu_dft.DCKRKS(cell, kpts=kpts, xc="PBE")
        gpu = gpu_dft.DCKRKS(cell, kpts=kpts, xc="PBE")
    cpu.with_df = cpu_df.GDF(cell, kpts=kpts, auxbasis="gth-szv")
    gpu.with_df = gpu_df.GDF(
        cell,
        kpts=kpts,
        auxbasis="gth-szv",
        setup_mode=setup_mode,
    )
    cpu.conv_tol = gpu.conv_tol = 1.0e-11
    cpu.conv_tol_grad = gpu.conv_tol_grad = 1.0e-9
    return cpu, gpu


@unittest.skipUnless(_gpu_available(), "CUDA device is required")
class GDFGPUEnergyTest(unittest.TestCase):
    """Check native GPU GDF source and target energies against CPU GDF."""

    def assert_energy_parity(self, kmesh):
        cpu, gpu = _methods(kmesh)
        cpu_energy = cpu.kernel()
        gpu_energy = gpu.kernel()
        self.assertLess(abs(cpu_energy - gpu_energy), 1.0e-8)
        self.assertEqual(gpu.with_df.capability()["backend"], "gdf")
        self.assertEqual(gpu.with_df.capability()["device"], "gpu")
        self.assertTrue(gpu.with_df.capability()["energy"])
        self.assertTrue(gpu.with_df.capability()["analytic_gradient"])
        self.assertEqual(gpu.with_df.capability()["gdf_setup_mode"], "native")
        self.assertTrue(gpu.with_df.uses_native_integrals)
        self.assertIsNone(gpu.with_df._cpu_witness)
        self.assertIsInstance(gpu.source_scf.with_df, gpu_df.GDF)
        self.assertIsInstance(gpu.target_ks.with_df, gpu_df.GDF)

    def test_gamma_h2_energy(self):
        self.assert_energy_parity((1, 1, 1))

    def test_kpoint_h2_energy(self):
        self.assert_energy_parity((2, 1, 1))

    def test_reference_integral_setup_remains_available(self):
        cell = _cell()
        provider = gpu_df.GDF(
            cell,
            kpts=np.zeros((1, 3)),
            auxbasis="gth-szv",
            setup_mode="reference",
        )
        provider.build()
        self.assertFalse(provider.uses_native_integrals)
        self.assertIsNotNone(provider.cpu_witness)
        self.assertGreater(len(provider.central_cderi_blocks), 0)

    def test_native_get_j_and_k_e1_contract(self):
        _cpu, gpu = _methods((1, 1, 1))
        gpu.kernel()
        density = gpu.source_scf.make_rdm1()
        coulomb = gpu.with_df.get_j_e1(density)
        exchange = gpu.with_df.get_k_e1(density, exxdiv=gpu.exxdiv)
        restricted_hf = gpu.with_df.get_hf_e1(
            density,
            exxdiv=gpu.exxdiv,
        )
        self.assertEqual(coulomb.shape, (gpu.cell.natm, 3))
        self.assertEqual(exchange.shape, (gpu.cell.natm, 3))
        self.assertLess(
            float(cp.max(cp.abs(restricted_hf - (coulomb - exchange)))),
            1.0e-9,
        )

    def test_native_cderi_cache_rebuilds_after_in_place_geometry_change(self):
        cell = _cell()
        provider = gpu_df.GDF(
            cell,
            kpts=np.zeros((1, 3)),
            auxbasis="gth-szv",
            setup_mode="native",
        )
        provider.build()
        original_signature = provider._dc_build_signature
        original_cderi = provider._cderi

        atoms = [
            (cell.atom_symbol(index), coordinate.copy())
            for index, coordinate in enumerate(cell.atom_coords(unit="Bohr"))
        ]
        atoms[1][1][2] += 0.05
        cell.set_geom_(atoms, unit="Bohr")
        self.assertNotEqual(
            original_signature,
            make_gdf_build_signature(provider),
        )

        provider._ensure_built()
        rebuilt_cderi = provider._cderi
        self.assertIsNot(rebuilt_cderi, original_cderi)
        self.assertEqual(
            provider._dc_build_signature,
            make_gdf_build_signature(provider),
        )

        provider._ensure_built()
        self.assertIs(provider._cderi, rebuilt_cderi)


@unittest.skipUnless(_gpu_available(), "CUDA device is required")
class GDFGPUAlgebraTest(unittest.TestCase):
    """Check device metric and J/K response algebra against CPU equations."""

    def test_gamma_coulomb_weights_slice_the_reciprocal_axis(self):
        full_weights = cp.arange(15, dtype=cp.float64)[None, :]
        weights = _gamma_coulomb_vector(full_weights, ngrids=15)

        self.assertEqual(weights.shape, (15,))
        block = cp.ones((7, 8)) * weights[:8]
        self.assertEqual(block.shape, (7, 8))

    def test_gamma_coulomb_weights_reject_multiple_kpoint_rows(self):
        with self.assertRaisesRegex(ValueError, "one Coulomb-weight k-point"):
            _gamma_coulomb_vector(cp.ones((2, 15)), ngrids=15)

    def test_native_local_part1_batches_reject_pool_overflow(self):
        _validate_short_range_batches(
            np.asarray([0, 273, 300]),
            np.asarray([0, 60]),
            bvk_ncells=1,
        )
        with self.assertRaisesRegex(RuntimeError, "POOL_SIZE"):
            _validate_short_range_batches(
                np.asarray([0, 300]),
                np.asarray([0, 60]),
                bvk_ncells=1,
            )

    def test_native_derivative_batches_are_bounded_and_restored(self):
        provider = type("Provider", (), {})()
        provider.auxcell = type("AuxCell", (), {"nbas": 60})()
        original_aggregate = gpu_mole.SortedCell.aggregate_shl_pairs
        original_rhf_splitter = gpu_gdf_grad_rhf._split_l_ctr_pattern
        original_krhf_splitter = gpu_gdf_grad_krhf._split_l_ctr_pattern
        offsets = np.asarray([0, 60])
        patterns = np.asarray([[0, 1]])

        with _bounded_sr_derivative_batches(provider, kpts=None):
            self.assertIsNot(
                gpu_mole.SortedCell.aggregate_shl_pairs,
                original_aggregate,
            )
            bounded_offsets, _ = gpu_gdf_grad_rhf._split_l_ctr_pattern(
                offsets,
                patterns,
                batch_size=120,
            )
            self.assertLessEqual(int(np.max(np.diff(bounded_offsets))), 60)
            self.assertIs(
                gpu_gdf_grad_rhf._split_l_ctr_pattern,
                gpu_gdf_grad_krhf._split_l_ctr_pattern,
            )

        self.assertIs(
            gpu_mole.SortedCell.aggregate_shl_pairs,
            original_aggregate,
        )
        self.assertIs(
            gpu_gdf_grad_rhf._split_l_ctr_pattern,
            original_rhf_splitter,
        )
        self.assertIs(
            gpu_gdf_grad_krhf._split_l_ctr_pattern,
            original_krhf_splitter,
        )

    def test_native_gamma_signed_one_pass_matches_two_pass(self):
        _cpu, gpu = _methods((1, 1, 1))
        gpu.kernel()
        reference = cp.asarray(gpu.source_scf.make_rdm1())
        rng = np.random.default_rng(107)
        raw = cp.asarray(rng.normal(size=reference.shape))
        response = (raw + raw.T) * 0.05

        two_pass = restricted_hf_bilinear_derivative(
            gpu.with_df,
            reference,
            response,
            exxdiv=gpu.exxdiv,
        )
        one_pass = restricted_hf_bilinear_derivative_one_pass_gamma(
            gpu.with_df,
            reference,
            response,
            exxdiv=gpu.exxdiv,
        )
        self.assertLess(
            float(cp.max(cp.abs(one_pass - two_pass))),
            1.0e-9,
        )

    def test_native_gamma_signed_one_pass_h2o_matrix_parity(self):
        cell = _water_cell()
        provider = gpu_df.GDF(
            cell,
            kpts=np.zeros((1, 3)),
            auxbasis="gth-szv",
            setup_mode="native",
        ).build()
        rng = np.random.default_rng(109)
        raw_reference = cp.asarray(rng.normal(size=(cell.nao, cell.nao)))
        raw_response = cp.asarray(rng.normal(size=(cell.nao, cell.nao)))
        reference = cp.eye(cell.nao) + 0.02 * (
            raw_reference + raw_reference.T
        )
        response = 0.02 * (raw_response + raw_response.T)

        two_pass = restricted_hf_bilinear_derivative(
            provider,
            reference,
            response,
            exxdiv="ewald",
        )
        one_pass = restricted_hf_bilinear_derivative_one_pass_gamma(
            provider,
            reference,
            response,
            exxdiv="ewald",
        )
        self.assertLess(
            float(cp.max(cp.abs(one_pass - two_pass))),
            1.0e-9,
        )

    def test_native_gamma_dzvp_sorted_cartesian_ao_matrix_parity(self):
        cell = _water_dzvp_cell()
        provider = gpu_df.GDF(
            cell,
            kpts=np.zeros((1, 3)),
            auxbasis="DZVP-GTH",
            setup_mode="native",
        ).build()
        optimizer = _build_gamma_derivative_optimizer(provider)
        transformed = optimizer.cell.apply_C_dot(
            cp.eye(cell.nao), axis=0
        )
        self.assertNotEqual(cell.nao, transformed.shape[0])
        rng = np.random.default_rng(110)
        raw_reference = cp.asarray(rng.normal(size=(cell.nao, cell.nao)))
        raw_response = cp.asarray(rng.normal(size=(cell.nao, cell.nao)))
        reference = cp.eye(cell.nao) + 0.02 * (
            raw_reference + raw_reference.T
        )
        response = 0.02 * (raw_response + raw_response.T)

        two_pass = restricted_hf_bilinear_derivative(
            provider,
            reference,
            response,
            exxdiv="ewald",
        )
        one_pass = restricted_hf_bilinear_derivative_one_pass_gamma(
            provider,
            reference,
            response,
            exxdiv="ewald",
        )

        self.assertLess(
            float(cp.max(cp.abs(one_pass - two_pass))),
            1.0e-8,
        )

    def test_native_gamma_forced_reciprocal_multiblock_parity(self):
        _cpu, gpu = _methods((1, 1, 1))
        gpu.kernel()
        reference = cp.asarray(gpu.source_scf.make_rdm1())
        rng = np.random.default_rng(111)
        raw = cp.asarray(rng.normal(size=reference.shape))
        response = 0.02 * (raw + raw.T)
        expected = restricted_hf_bilinear_derivative_one_pass_gamma(
            gpu.with_df,
            reference,
            response,
            exxdiv=gpu.exxdiv,
        )

        def force_multiple_blocks(*args, **kwargs):
            ngrids = int(args[0])
            return plan_gamma_reciprocal_blocks(
                *args,
                **kwargs,
                block_alignment=1,
                max_block_size=max(1, ngrids // 2),
            )

        diagnostics = {}
        with patch(
            "gpu4pyscf_pbc_dcdft.df.gdf.signed_gamma_native."
            "plan_gamma_reciprocal_blocks",
            side_effect=force_multiple_blocks,
        ):
            actual = restricted_hf_bilinear_derivative_one_pass_gamma(
                gpu.with_df,
                reference,
                response,
                exxdiv=gpu.exxdiv,
                memory_diagnostics=diagnostics,
            )

        self.assertLess(float(cp.max(cp.abs(actual - expected))), 1.0e-9)
        self.assertLess(
            diagnostics["plan"]["initial_block_size"],
            diagnostics["plan"]["ngrids"],
        )
        self.assertEqual(diagnostics["derivative_omega_scale"], 1.5)
        self.assertGreater(diagnostics["short_range_omega"], 0.0)
        for phase in ("zero_order_execution", "derivative_execution"):
            self.assertEqual(
                diagnostics[phase]["completed_count"],
                diagnostics["plan"]["ngrids"],
            )
            self.assertGreater(diagnostics[phase]["block_count"], 1)
        short_range = diagnostics["short_range_device_timing"]
        self.assertLessEqual(
            short_range["maximum_tasks_per_batch"],
            POOL_SIZE,
        )

    def test_native_kpoint_signed_one_pass_matches_two_pass(self):
        _cpu, gpu = _methods((2, 1, 1))
        gpu.kernel()
        reference = cp.asarray(gpu.source_scf.make_rdm1())
        rng = np.random.default_rng(113)
        raw = cp.asarray(
            rng.normal(size=reference.shape)
            + 1j * rng.normal(size=reference.shape)
        )
        response = 0.02 * (raw + raw.swapaxes(-1, -2).conj())

        two_pass = restricted_hf_bilinear_derivative(
            gpu.with_df,
            reference,
            response,
            exxdiv=gpu.exxdiv,
        )
        one_pass = restricted_hf_bilinear_derivative_one_pass(
            gpu.with_df,
            reference,
            response,
            exxdiv=gpu.exxdiv,
        )
        self.assertLess(
            float(cp.max(cp.abs(one_pass - two_pass))),
            1.0e-9,
        )

    def test_native_kpoint_signed_one_pass_h2o_matrix_parity(self):
        cell = _water_cell()
        kpts = cell.make_kpts((2, 1, 1))
        provider = gpu_df.GDF(
            cell,
            kpts=kpts,
            auxbasis="gth-szv",
            setup_mode="native",
        ).build()
        rng = np.random.default_rng(127)
        raw_reference = cp.asarray(
            rng.normal(size=(len(kpts), cell.nao, cell.nao))
            + 1j * rng.normal(size=(len(kpts), cell.nao, cell.nao))
        )
        raw_response = cp.asarray(
            rng.normal(size=(len(kpts), cell.nao, cell.nao))
            + 1j * rng.normal(size=(len(kpts), cell.nao, cell.nao))
        )
        reference = cp.eye(cell.nao)[None] + 0.02 * (
            raw_reference + raw_reference.swapaxes(-1, -2).conj()
        )
        response = 0.02 * (
            raw_response + raw_response.swapaxes(-1, -2).conj()
        )

        two_pass = restricted_hf_bilinear_derivative(
            provider,
            reference,
            response,
            exxdiv="ewald",
        )
        one_pass = restricted_hf_bilinear_derivative_one_pass(
            provider,
            reference,
            response,
            exxdiv="ewald",
        )
        self.assertLess(
            float(cp.max(cp.abs(one_pass - two_pass))),
            1.0e-9,
        )

    def test_native_kpoint_dzvp_sorted_cartesian_ao_matrix_parity(self):
        cell = _water_dzvp_cell()
        kpts = cell.make_kpts((2, 1, 1))
        provider = gpu_df.GDF(
            cell,
            kpts=kpts,
            auxbasis="DZVP-GTH",
            setup_mode="native",
        ).build()
        rng = np.random.default_rng(129)
        shape = (len(kpts), cell.nao, cell.nao)
        raw_reference = cp.asarray(
            rng.normal(size=shape) + 1j * rng.normal(size=shape)
        )
        raw_response = cp.asarray(
            rng.normal(size=shape) + 1j * rng.normal(size=shape)
        )
        reference = cp.eye(cell.nao)[None] + 0.02 * (
            raw_reference + raw_reference.swapaxes(-1, -2).conj()
        )
        response = 0.02 * (
            raw_response + raw_response.swapaxes(-1, -2).conj()
        )

        two_pass = restricted_hf_bilinear_derivative(
            provider,
            reference,
            response,
            exxdiv="ewald",
        )
        one_pass = restricted_hf_bilinear_derivative_one_pass(
            provider,
            reference,
            response,
            exxdiv="ewald",
        )

        self.assertLess(
            float(cp.max(cp.abs(one_pass - two_pass))),
            1.0e-8,
        )

    def test_native_gamma_target_j_and_bilinear_fused_parity(self):
        _cpu, gpu = _methods((1, 1, 1))
        gpu.kernel()
        reference = cp.asarray(gpu.source_scf.make_rdm1())
        rng = np.random.default_rng(131)
        raw = cp.asarray(rng.normal(size=reference.shape))
        response = 0.02 * (raw + raw.T)
        separate = coulomb_energy_derivative(gpu.with_df, reference)
        separate += restricted_hf_bilinear_derivative_one_pass(
            gpu.with_df, reference, response, exxdiv=gpu.exxdiv
        )
        fused = target_j_plus_restricted_hf_bilinear_derivative_one_pass(
            gpu.with_df, reference, response, exxdiv=gpu.exxdiv
        )
        self.assertLess(float(cp.max(cp.abs(fused - separate))), 1.0e-9)

    def test_native_kpoint_target_j_and_bilinear_fused_parity(self):
        _cpu, gpu = _methods((2, 1, 1))
        gpu.kernel()
        reference = cp.asarray(gpu.source_scf.make_rdm1())
        rng = np.random.default_rng(137)
        raw = cp.asarray(
            rng.normal(size=reference.shape)
            + 1j * rng.normal(size=reference.shape)
        )
        response = 0.02 * (raw + raw.swapaxes(-1, -2).conj())
        separate = coulomb_energy_derivative(gpu.with_df, reference)
        separate += restricted_hf_bilinear_derivative_one_pass(
            gpu.with_df, reference, response, exxdiv=gpu.exxdiv
        )
        fused = target_j_plus_restricted_hf_bilinear_derivative_one_pass(
            gpu.with_df, reference, response, exxdiv=gpu.exxdiv
        )
        self.assertLess(float(cp.max(cp.abs(fused - separate))), 1.0e-9)

    def test_cholesky_response_matches_cpu(self):
        rng = np.random.default_rng(18)
        nao = 2
        naux = 4
        raw = rng.normal(size=(naux, naux)) + 1j * rng.normal(
            size=(naux, naux)
        )
        metric = raw @ raw.conj().T + np.eye(naux)
        source = rng.normal(size=(nao**2, naux)) + 1j * rng.normal(
            size=(nao**2, naux)
        )
        d_source = rng.normal(size=source.shape) + 1j * rng.normal(
            size=source.shape
        )
        d_raw = rng.normal(size=(naux, naux)) + 1j * rng.normal(
            size=(naux, naux)
        )
        d_metric = d_raw + d_raw.conj().T
        cpu = differentiate_cderi(
            source, metric, d_source, d_metric, nao
        )
        factorization = CDERIFactorization.build(
            source, metric, nao=nao
        )
        gpu = factorization.differentiate(d_source, d_metric)
        self.assertLess(
            float(np.max(np.abs(cpu.value - cp.asnumpy(gpu.value)))),
            1.0e-10,
        )
        self.assertLess(
            float(np.max(np.abs(cpu.total - cp.asnumpy(gpu.total)))),
            1.0e-10,
        )

    def test_real_factor_preserves_complex_derivative_response(self):
        rng = np.random.default_rng(181)
        nao = 2
        naux = 4
        raw = rng.normal(size=(naux, naux))
        metric = raw @ raw.T + np.eye(naux)
        source = rng.normal(size=(nao**2, naux))
        d_source = rng.normal(size=source.shape) + 1j * rng.normal(
            size=source.shape
        )
        d_raw = rng.normal(size=(naux, naux)) + 1j * rng.normal(
            size=(naux, naux)
        )
        d_metric = d_raw + d_raw.conj().T
        cpu = differentiate_cderi(
            source, metric, d_source, d_metric, nao
        )
        factorization = CDERIFactorization.build(
            source, metric, nao=nao
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.exceptions.ComplexWarning)
            gpu = factorization.differentiate(d_source, d_metric)
        gpu_total = cp.asnumpy(gpu.total)
        self.assertTrue(np.iscomplexobj(gpu_total))
        self.assertGreater(float(np.max(np.abs(gpu_total.imag))), 1.0e-3)
        self.assertLess(
            float(np.max(np.abs(cpu.total - gpu_total))),
            1.0e-10,
        )

    def test_jk_response_matches_cpu(self):
        rng = np.random.default_rng(31)
        nkpts = 2
        nao = 2
        naux = 3
        natm = 2
        pairs = ((0, 0), (1, 0), (1, 1))
        cderi = {
            pair: rng.normal(size=(naux, nao, nao))
            + 1j * rng.normal(size=(naux, nao, nao))
            for pair in pairs
        }
        dcderi = {
            pair: rng.normal(size=(natm, 3, naux, nao, nao))
            + 1j * rng.normal(size=(natm, 3, naux, nao, nao))
            for pair in pairs
        }
        density = []
        for _ in range(nkpts):
            raw = rng.normal(size=(nao, nao)) + 1j * rng.normal(
                size=(nao, nao)
            )
            density.append(raw + raw.conj().T)
        density = np.asarray(density)
        cpu = cpu_gdf_jk_derivative(cderi, dcderi, density)
        gpu = gpu_gdf_jk_derivative(
            {key: cp.asarray(value) for key, value in cderi.items()},
            {key: cp.asarray(value) for key, value in dcderi.items()},
            cp.asarray(density),
        )
        self.assertLess(
            float(
                np.max(
                    np.abs(cpu.coulomb - cp.asnumpy(gpu.coulomb))
                )
            ),
            1.0e-10,
        )
        self.assertLess(
            float(
                np.max(
                    np.abs(cpu.exchange - cp.asnumpy(gpu.exchange))
                )
            ),
            1.0e-10,
        )

    def test_full_matrix_bilinear_reference_matches_polarization(self):
        rng = np.random.default_rng(73)
        for nkpts in (1, 2):
            nao = 3
            naux = 4
            natm = 2
            pairs = tuple(
                (ki, kj)
                for ki in range(nkpts)
                for kj in range(ki + 1)
            )
            cderi = {
                pair: cp.asarray(
                    rng.normal(size=(naux, nao, nao))
                    + 1j * rng.normal(size=(naux, nao, nao))
                )
                for pair in pairs
            }
            dcderi = {
                pair: cp.asarray(
                    rng.normal(size=(natm, 3, naux, nao, nao))
                    + 1j
                    * rng.normal(size=(natm, 3, naux, nao, nao))
                )
                for pair in pairs
            }

            def hermitian_density():
                raw = rng.normal(size=(nkpts, nao, nao))
                raw = raw + 1j * rng.normal(size=raw.shape)
                return cp.asarray(raw + raw.conj().transpose(0, 2, 1))

            reference = hermitian_density()
            response = hermitian_density()
            derivative_at_reference = gpu_gdf_jk_derivative(
                cderi, dcderi, reference
            )
            derivative_at_response = gpu_gdf_jk_derivative(
                cderi, dcderi, response
            )
            direct = contract_restricted_hf_bilinear_energy_derivative(
                reference,
                response,
                derivative_at_reference,
                derivative_at_response,
            )

            plus_density = reference + response
            minus_density = reference - response
            plus = contract_restricted_hf_energy_derivative(
                plus_density,
                gpu_gdf_jk_derivative(cderi, dcderi, plus_density),
            )
            minus = contract_restricted_hf_energy_derivative(
                minus_density,
                gpu_gdf_jk_derivative(cderi, dcderi, minus_density),
            )
            self.assertLess(
                float(cp.max(cp.abs(direct - (plus - minus)))),
                1.0e-9,
            )

            contracted_source = contract_kpoint_bilinear_cderi_derivative(
                cderi,
                dcderi,
                reference,
                response,
            )
            self.assertLess(
                float(cp.max(cp.abs(contracted_source - direct))),
                1.0e-9,
            )

    def test_gamma_signed_density_source_matches_full_matrix_reference(self):
        rng = np.random.default_rng(91)
        naux = 5
        nao = 4
        natm = 3

        raw_factors = rng.normal(size=(naux, nao, nao))
        factors = raw_factors + raw_factors.transpose(0, 2, 1)
        raw_derivative = rng.normal(size=(natm, 3, naux, nao, nao))
        derivative = raw_derivative + raw_derivative.transpose(0, 1, 2, 4, 3)

        def symmetric_density():
            raw = rng.normal(size=(nao, nao))
            return raw + raw.T

        reference = symmetric_density()
        response = symmetric_density()
        cderi = {(0, 0): cp.asarray(factors)}
        dcderi = {(0, 0): cp.asarray(derivative)}
        derivative_at_reference = gpu_gdf_jk_derivative(
            cderi,
            dcderi,
            cp.asarray(reference)[None],
        )
        derivative_at_response = gpu_gdf_jk_derivative(
            cderi,
            dcderi,
            cp.asarray(response)[None],
        )
        full_matrix = contract_restricted_hf_bilinear_energy_derivative(
            cp.asarray(reference)[None],
            cp.asarray(response)[None],
            derivative_at_reference,
            derivative_at_response,
        )

        source = build_gamma_signed_density_source(
            cp.asarray(factors),
            cp.asarray(reference),
            cp.asarray(response),
        )
        contracted = contract_gamma_signed_density_source(
            source,
            cp.asarray(derivative),
        )
        self.assertEqual(source.restricted_hf.shape, (naux, nao, nao))
        self.assertLess(
            float(cp.max(cp.abs(contracted - full_matrix))),
            1.0e-9,
        )

    def test_ewald_signed_density_term_matches_polarization(self):
        rng = np.random.default_rng(97)
        cell = _cell()
        for kmesh in ((1, 1, 1), (2, 1, 1)):
            kpts = (
                np.zeros((1, 3))
                if kmesh == (1, 1, 1)
                else cell.make_kpts(kmesh)
            )
            nkpts = len(kpts)
            nao = cell.nao_nr()

            def hermitian_density():
                raw = rng.normal(size=(nkpts, nao, nao))
                raw = raw + 1j * rng.normal(size=raw.shape)
                value = raw + raw.conj().transpose(0, 2, 1)
                if nkpts == 1:
                    value = value.real
                return cp.asarray(value)

            reference = hermitian_density()
            response = hermitian_density()
            raw_overlap = rng.normal(
                size=(cell.natm, nkpts, 3, nao, nao)
            )
            raw_overlap = raw_overlap + 1j * rng.normal(
                size=raw_overlap.shape
            )
            overlap_derivative = cp.asarray(
                raw_overlap + raw_overlap.conj().transpose(0, 1, 2, 4, 3)
            )

            def quadratic(density):
                exchange = ewald_exchange_derivative(
                    cell,
                    kpts,
                    density,
                    overlap_derivative,
                )
                return (
                    -0.25
                    * cp.einsum(
                        "akxij,kji->ax",
                        exchange,
                        density,
                        optimize=True,
                    ).real
                    / nkpts
                )

            polarization = quadratic(reference + response) - quadratic(
                reference - response
            )
            direct = ewald_restricted_hf_bilinear_derivative(
                cell,
                kpts,
                reference,
                response,
                overlap_derivative,
            )
            self.assertLess(
                float(cp.max(cp.abs(direct - polarization))),
                1.0e-9,
            )

    def test_gamma_factorization_adjoint_matches_forward_response(self):
        rng = np.random.default_rng(101)
        naux = 5
        nao = 3

        raw_metric = rng.normal(size=(naux, naux))
        metric = raw_metric @ raw_metric.T + np.eye(naux)
        factor = np.linalg.cholesky(metric)
        raw_cderi = rng.normal(size=(naux, nao, nao))
        cderi = raw_cderi + raw_cderi.transpose(0, 2, 1)
        source = (factor @ cderi.reshape(naux, -1)).T
        factorization = CDERIFactorization.build(
            cp.asarray(source),
            cp.asarray(metric),
            nao=nao,
        )

        def symmetric_density():
            raw = rng.normal(size=(nao, nao))
            return cp.asarray(raw + raw.T)

        signed_source = build_gamma_signed_density_source(
            factorization.value,
            symmetric_density(),
            symmetric_density(),
        )
        source_derivative = cp.asarray(
            rng.normal(size=(nao * nao, naux))
        )
        raw_metric_derivative = rng.normal(size=(naux, naux))
        metric_derivative = cp.asarray(
            raw_metric_derivative + raw_metric_derivative.T
        )
        forward = factorization.differentiate(
            source_derivative,
            metric_derivative,
        )
        source_forward = cp.einsum(
            "pij,pij->",
            signed_source.restricted_hf,
            forward.source,
            optimize=True,
        ).real
        metric_forward = cp.einsum(
            "pij,pij->",
            signed_source.restricted_hf,
            forward.metric,
            optimize=True,
        ).real

        adjoint = build_gamma_factorization_adjoint(
            factorization,
            signed_source,
        )
        source_reverse = cp.einsum(
            "ip,ip->", adjoint.source, source_derivative
        ).real
        metric_reverse = cp.einsum(
            "pq,pq->", adjoint.metric, metric_derivative
        ).real
        self.assertLess(float(cp.abs(source_forward - source_reverse)), 1.0e-9)
        self.assertLess(float(cp.abs(metric_forward - metric_reverse)), 1.0e-9)
        self.assertLess(
            float(
                cp.abs(
                    forward.total.ravel().dot(
                        signed_source.restricted_hf.ravel()
                    )
                    - adjoint.contract(
                        source_derivative,
                        metric_derivative,
                    )
                )
            ),
            1.0e-9,
        )

    def test_native_local_part1_contraction_matches_full_h1(self):
        cell = _cell()
        for kmesh in ((1, 1, 1), (2, 1, 1)):
            kpts = (
                np.zeros((1, 3))
                if kmesh == (1, 1, 1)
                else cell.make_kpts(kmesh)
            )
            density = np.asarray(
                cell.pbc_intor("int1e_ovlp", kpts=kpts)
            )
            full = LocalPart1Derivative(cell, kpts)
            reference = np.zeros((cell.natm, 3))
            for atom_id in range(cell.natm):
                for axis in range(3):
                    reference[atom_id, axis] = np.einsum(
                        "kij,kji->",
                        full.matrix(atom_id, axis),
                        density,
                    ).real / len(kpts)
            contracted = NativeGDFLocalPart1Derivative(
                cell,
                kpts,
            ).contract(density)
            self.assertLess(
                float(np.max(np.abs(cp.asnumpy(contracted) - reference))),
                1.0e-8,
            )


@unittest.skipUnless(_gpu_available(), "CUDA device is required")
class GDFGPUGradientTest(unittest.TestCase):
    """Check the public analytic route against the completed CPU package."""

    def assert_gradient_parity(self, kmesh):
        cpu, gpu = _methods(kmesh)
        cpu.kernel()
        gpu.kernel()
        cpu_gradient = cpu.nuc_grad_method().kernel()
        gpu_gradient_object = gpu.nuc_grad_method()
        gpu_gradient = gpu_gradient_object.kernel()
        self.assertLess(
            float(np.max(np.abs(cpu_gradient - gpu_gradient))),
            1.0e-7,
        )
        self.assertLess(
            gpu_gradient_object.result.z_vector.residual_max_abs,
            1.0e-9,
        )
        self.assertFalse(
            gpu_gradient_object.result.metadata()["silent_cpu_fallback"]
        )
        self.assertEqual(
            gpu_gradient_object.result.metadata()["gdf_setup_mode"],
            "native",
        )
        self.assertGreater(
            gpu_gradient_object.result.z_nuclear_response_batch_size,
            1,
        )
        metadata = gpu_gradient_object.result.metadata()
        self.assertEqual(metadata["gdf_bilinear_mode"], "one_pass")
        if kmesh == (1, 1, 1):
            reciprocal = metadata["gdf_reciprocal_memory"]
            self.assertEqual(reciprocal["route"], "gamma")
            self.assertEqual(
                reciprocal["zero_order_execution"]["completed_count"],
                reciprocal["plan"]["ngrids"],
            )
            self.assertEqual(
                reciprocal["derivative_execution"]["completed_count"],
                reciprocal["plan"]["ngrids"],
            )
        else:
            self.assertIsNone(metadata["gdf_reciprocal_memory"])
        self.assertEqual(
            metadata["z_nuclear_response_batch_plan"]["component_count"],
            3 * gpu.cell.natm,
        )
        self.assertEqual(
            metadata["z_nuclear_response_batch_execution"][
                "completed_count"
            ],
            3 * gpu.cell.natm,
        )

    def test_gamma_h2_gradient_parity(self):
        self.assert_gradient_parity((1, 1, 1))

    def test_gamma_one_pass_full_gradient_parity(self):
        _cpu, gpu = _methods((1, 1, 1))
        gpu.kernel()

        two_pass = gpu.nuc_grad_method()
        two_pass.gdf_bilinear_mode = "two_pass"
        two_pass_gradient = two_pass.kernel()
        one_pass = gpu.nuc_grad_method()
        one_pass.gdf_bilinear_mode = "one_pass"
        one_pass_gradient = one_pass.kernel()

        self.assertLess(
            float(np.max(np.abs(one_pass_gradient - two_pass_gradient))),
            1.0e-9,
        )
        self.assertEqual(
            one_pass.result.metadata()["gdf_bilinear_mode"],
            "one_pass",
        )

    def test_kpoint_one_pass_full_gradient_parity(self):
        _cpu, gpu = _methods((2, 1, 1))
        gpu.kernel()

        two_pass = gpu.nuc_grad_method()
        two_pass.gdf_bilinear_mode = "two_pass"
        two_pass_gradient = two_pass.kernel()
        one_pass = gpu.nuc_grad_method()
        one_pass.gdf_bilinear_mode = "one_pass"
        one_pass_gradient = one_pass.kernel()

        self.assertLess(
            float(np.max(np.abs(one_pass_gradient - two_pass_gradient))),
            1.0e-9,
        )
        self.assertEqual(
            one_pass.result.metadata()["gdf_bilinear_mode"],
            "one_pass",
        )

    def assert_fused_full_gradient_parity(self, kmesh):
        _cpu, gpu = _methods(kmesh)
        gpu.kernel()
        resolved = gpu.nuc_grad_method()
        resolved.gdf_bilinear_mode = "one_pass"
        resolved_gradient = resolved.kernel()

        fused = gpu.nuc_grad_method()
        fused.gdf_bilinear_mode = "one_pass_fused"
        fused_gradient = fused.kernel()
        self.assertLess(
            float(np.max(np.abs(fused_gradient - resolved_gradient))),
            1.0e-9,
        )
        metadata = fused.result.metadata()
        self.assertEqual(metadata["gdf_bilinear_mode"], "one_pass_fused")
        self.assertEqual(
            metadata["gradient_partition"],
            "explicit_response_coupled",
        )
        self.assertFalse(metadata["target_coulomb_component_resolved"])
        self.assertTrue(metadata["target_fock_workspace_reused"])
        self.assertIsNone(fused.result.target_coulomb)
        self.assertIsNotNone(fused.result.fused_target_j_z_bilinear)

    def test_gamma_fused_full_gradient_parity(self):
        self.assert_fused_full_gradient_parity((1, 1, 1))

    def test_kpoint_fused_full_gradient_parity(self):
        self.assert_fused_full_gradient_parity((2, 1, 1))

    def test_one_pass_gamma_rejects_sampled_kpoints(self):
        _cpu, gpu = _methods((2, 1, 1))
        gpu.kernel()
        gradient = gpu.nuc_grad_method()
        gradient.gdf_bilinear_mode = "one_pass_gamma"
        with self.assertRaisesRegex(NotImplementedError, "sampled k-points"):
            gradient.kernel()

    def test_kpoint_h2_gradient_parity(self):
        self.assert_gradient_parity((2, 1, 1))

    def assert_forced_small_batch_parity(self, kmesh):
        _cpu, gpu = _methods(kmesh)
        gpu.kernel()

        automatic = gpu.nuc_grad_method()
        automatic_gradient = automatic.kernel()

        forced = gpu.nuc_grad_method()
        forced.z_nuclear_batch_max_components = 1
        forced_gradient = forced.kernel()

        self.assertLess(
            float(np.max(np.abs(automatic_gradient - forced_gradient))),
            1.0e-9,
        )
        metadata = forced.result.metadata()
        self.assertEqual(forced.result.z_nuclear_response_batch_size, 1)
        self.assertEqual(
            metadata["z_nuclear_response_batch_plan"]["max_batch_size"],
            1,
        )
        execution = metadata["z_nuclear_response_batch_execution"]
        self.assertEqual(execution["completed_count"], 3 * gpu.cell.natm)
        self.assertEqual(execution["rebatch_count"], 0)
        self.assertIn("device_memory", execution)

    def test_gamma_forced_small_batch_parity(self):
        self.assert_forced_small_batch_parity((1, 1, 1))

    def test_kpoint_forced_small_batch_parity(self):
        self.assert_forced_small_batch_parity((2, 1, 1))

    def test_reference_gamma_gradient_parity(self):
        cpu, gpu = _methods((1, 1, 1), setup_mode="reference")
        cpu.kernel()
        gpu.kernel()
        gpu_gradient_object = gpu.nuc_grad_method()
        self.assertEqual(gpu_gradient_object.gdf_bilinear_mode, "two_pass")
        difference = (
            cpu.nuc_grad_method().kernel()
            - gpu_gradient_object.kernel()
        )
        self.assertLess(float(np.max(np.abs(difference))), 1.0e-7)
        self.assertIsNone(
            gpu_gradient_object.result.z_nuclear_response_batch_size
        )
        self.assertIsNone(
            gpu_gradient_object.result.z_nuclear_response_batch_plan
        )
        self.assertIsNone(
            gpu_gradient_object.result.z_nuclear_response_batch_execution
        )

    def assert_z_solver_parity(self, kmesh):
        _cpu, gpu = _methods(kmesh)
        gpu.kernel()

        dense_object = gpu.nuc_grad_method()
        dense_object.z_solver = "dense"
        dense_gradient = dense_object.kernel()

        matrix_free_object = gpu.nuc_grad_method()
        matrix_free_object.z_solver = "matrix_free"
        matrix_free_gradient = matrix_free_object.kernel()

        self.assertLess(
            float(np.max(np.abs(dense_gradient - matrix_free_gradient))),
            1.0e-9,
        )
        self.assertEqual(matrix_free_object.result.z_vector.solver, "matrix_free")
        self.assertLess(
            matrix_free_object.result.z_vector.residual_max_abs,
            matrix_free_object.z_residual_tolerance,
        )

    def test_gamma_dense_matrix_free_z_solver_parity(self):
        self.assert_z_solver_parity((1, 1, 1))

    def test_kpoint_dense_matrix_free_z_solver_parity(self):
        self.assert_z_solver_parity((2, 1, 1))


if __name__ == "__main__":
    unittest.main()
