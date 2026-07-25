import unittest
from types import MappingProxyType, SimpleNamespace

import cupy as cp
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.df import incore

from gpu4pyscf_pbc_dcdft.df.gdf.cholesky import CDERIFactorization
from gpu4pyscf_pbc_dcdft.df.rsdf.metric_adjoint import (
    push_cderi_adjoint_through_metric,
)
from gpu4pyscf_pbc_dcdft.df.rsdf.source_adjoint import (
    build_pair_cderi_adjoints,
    push_cderi_adjoint_through_source,
)
from gpu4pyscf_pbc_dcdft_stress.df_response import (
    DeviceDFStrainResponse,
    PairStrainPrimitive,
    device_host_factorized_projection,
    device_host_factorized_projection_bounded,
    _resolve_gpu_ft_dispatch,
)
from gpu4pyscf_pbc_dcdft_stress.ft_source import (
    _AUTO_AO_PAIR_WORK_BUDGET_BYTES,
    _automatic_block_size,
    gamma_general_ao_pair_strain,
    gamma_long_range_source_adjoint_strain,
    gamma_separable_ao_pair_strain,
    kpoint_general_ao_pair_strain,
    kpoint_long_range_source_adjoint_strain as
    kpoint_long_range_source_adjoint_strain_gpu,
)
from pyscf_pbc_dcdft_stress import (
    kpoint_gdf_jk_strain,
    kpoint_long_range_source_adjoint_strain,
)
from pyscf_pbc_dcdft_stress.gdf_source_strain import (
    _build_ao_pair_source_context,
    _contract_ao_pair_source_adjoint_block,
    auxiliary_fourier_strain,
    weighted_coulomb_strain,
)
from pyscf_pbc_dcdft_stress.gdf_metric_strain import matrix_tensor_to_voigt


class FakeFactorization:
    def __init__(self, value):
        self.value = cp.asarray(value)

    def differentiate(self, source, metric):
        del metric
        return SimpleNamespace(total=cp.asarray(source))


class DeviceDFResponseTest(unittest.TestCase):
    def test_automatic_ft_block_bounds_large_ao_pair_workspace(self):
        block = _automatic_block_size(
            sorted_nao=120,
            ngrid=13_651_919,
            available_bytes=100 << 30,
        )
        self.assertEqual(block, 4656)
        work_bytes = 2*120*120*np.dtype(np.complex128).itemsize*block
        self.assertLessEqual(work_bytes, _AUTO_AO_PAIR_WORK_BUDGET_BYTES)
        self.assertLessEqual(
            _AUTO_AO_PAIR_WORK_BUDGET_BYTES-work_bytes,
            2*120*120*np.dtype(np.complex128).itemsize*16,
        )

    def test_automatic_ft_block_keeps_small_system_throughput_cap(self):
        self.assertEqual(
            _automatic_block_size(
                sorted_nao=48,
                ngrid=1_000_000,
                available_bytes=100 << 30,
            ),
            16384,
        )
        self.assertEqual(
            _automatic_block_size(
                sorted_nao=48,
                ngrid=97,
                available_bytes=100 << 30,
            ),
            97,
        )

    def test_general_gpu_ft_dispatches_gamma_without_bvk(self):
        gamma = np.zeros((1, 3))
        sampled = np.asarray(((0.0, 0.0, 0.0), (0.2, 0.0, 0.0)))
        self.assertEqual(
            _resolve_gpu_ft_dispatch(
                gamma, gamma_gpu_ft=False, gpu_ft=True
            ),
            "gamma",
        )
        self.assertEqual(
            _resolve_gpu_ft_dispatch(
                sampled, gamma_gpu_ft=False, gpu_ft=True
            ),
            "sampled_k",
        )
        self.assertEqual(
            _resolve_gpu_ft_dispatch(
                gamma, gamma_gpu_ft=False, gpu_ft=False
            ),
            "host",
        )
        with self.assertRaises(NotImplementedError):
            _resolve_gpu_ft_dispatch(
                sampled, gamma_gpu_ft=True, gpu_ft=False
            )

    def test_sampled_k_gpu_long_range_source_adjoint_matches_host(self):
        cell = gto.Cell(
            atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
            a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3),
                          (0.1, 0.0, 6.0))),
            unit="Bohr", basis="gth-szv", pseudo="gth-hf",
            mesh=np.asarray((7, 9, 11)), precision=1e-10, verbose=0,
        ).build()
        auxcell = incore.make_auxcell(cell, "gth-szv")
        kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
        kpti, kptj = kpts[[0, 1]]
        rng = np.random.default_rng(43)
        adjoints = (
            rng.normal(
                size=(2, cell.nao_nr()**2, auxcell.nao_nr())
            )
            + 1j*rng.normal(
                size=(2, cell.nao_nr()**2, auxcell.nao_nr())
            )
        )
        expected = kpoint_long_range_source_adjoint_strain(
            cell, auxcell, 0.9, cell.mesh, kpti, kptj, adjoints,
            reciprocal_block_size=97,
        ).values
        observed = kpoint_long_range_source_adjoint_strain_gpu(
            cell, auxcell, 0.9, cell.mesh, kpts, kpti, kptj, adjoints,
            block_size=97,
        ).values
        self.assertLess(
            float(np.max(np.abs(expected-observed))), 5e-10
        )

    def test_sampled_k_general_gpu_ft_source_matches_host_pair(self):
        cell = gto.Cell(
            atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
            a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3),
                          (0.1, 0.0, 6.0))),
            unit="Bohr", basis="gth-szv", pseudo="gth-hf",
            mesh=np.asarray((7, 9, 11)), precision=1e-10, verbose=0,
        ).build()
        auxcell = incore.make_auxcell(cell, "gth-szv")
        kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
        kpti, kptj = kpts[[0, 1]]
        vectors = cell.get_Gv(cell.mesh)
        momentum = vectors+kptj-kpti
        auxiliary = auxiliary_fourier_strain(auxcell, momentum)
        kernel = weighted_coulomb_strain(cell, momentum, omega=0.9)
        weighted_auxiliary = (auxiliary.value*kernel.value).T
        rng = np.random.default_rng(41)
        adjoint = (
            rng.normal(size=(cell.nao_nr()**2, auxcell.nao_nr()))
            + 1j*rng.normal(
                size=(cell.nao_nr()**2, auxcell.nao_nr())
            )
        )
        expected = np.zeros((1, 6), dtype=np.complex128)
        context = _build_ao_pair_source_context(cell, kpti, kptj)
        _contract_ao_pair_source_adjoint_block(
            context, vectors, weighted_auxiliary, adjoint[None], expected
        )
        response_weight = np.einsum(
            "gp,qp->gq", weighted_auxiliary.conj(), adjoint, optimize=True
        )
        ao_grid_weight = response_weight.reshape(
            len(vectors), cell.nao_nr(), cell.nao_nr()
        ).transpose(1, 2, 0)
        observed = kpoint_general_ao_pair_strain(
            cell, ao_grid_weight, vectors, kpts, kpti, kptj,
            block_size=83,
        )
        self.assertLess(
            float(np.max(np.abs(np.real(expected[0])-observed))), 5e-10
        )

    def test_gamma_gpu_long_range_source_adjoint_matches_host(self):
        cell = gto.Cell(
            atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
            a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3),
                          (0.1, 0.0, 6.0))),
            unit="Bohr", basis="gth-szv", pseudo="gth-hf",
            mesh=np.asarray((7, 9, 11)), verbose=0,
        ).build()
        auxcell = incore.make_auxcell(cell, "gth-szv")
        rng = np.random.default_rng(37)
        adjoints = rng.normal(
            size=(2, cell.nao_nr(), cell.nao_nr(), auxcell.nao_nr())
        )
        adjoints = 0.5*(adjoints+adjoints.transpose(0, 2, 1, 3))
        adjoints = adjoints.reshape(
            2, cell.nao_nr()**2, auxcell.nao_nr()
        )
        expected = kpoint_long_range_source_adjoint_strain(
            cell, auxcell, 0.9, cell.mesh, np.zeros(3), np.zeros(3),
            adjoints, reciprocal_block_size=97,
        ).values
        observed = gamma_long_range_source_adjoint_strain(
            cell, auxcell, 0.9, cell.mesh, adjoints, block_size=97
        ).values
        self.assertLess(
            float(np.max(np.abs(expected-observed))), 5e-10
        )

    def test_gamma_general_gpu_ft_source_matches_host_analytic_pair(self):
        cell = gto.Cell(
            atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
            a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3),
                          (0.1, 0.0, 6.0))),
            unit="Bohr", basis="gth-szv", pseudo="gth-hf",
            mesh=np.asarray((7, 9, 11)), verbose=0,
        ).build()
        auxcell = incore.make_auxcell(cell, "gth-szv")
        vectors = cell.get_Gv(cell.mesh)
        auxiliary = auxiliary_fourier_strain(auxcell, vectors)
        kernel = weighted_coulomb_strain(cell, vectors, omega=0.9)
        weighted_auxiliary = (auxiliary.value*kernel.value).T
        rng = np.random.default_rng(31)
        adjoint = rng.normal(
            size=(cell.nao_nr(), cell.nao_nr(), auxcell.nao_nr())
        )
        adjoint = 0.5*(adjoint+adjoint.transpose(1, 0, 2))
        adjoint = adjoint.reshape(cell.nao_nr()**2, auxcell.nao_nr())
        expected = np.zeros((1, 6), dtype=np.complex128)
        context = _build_ao_pair_source_context(
            cell, np.zeros(3), np.zeros(3)
        )
        _contract_ao_pair_source_adjoint_block(
            context, vectors, weighted_auxiliary, adjoint[None], expected
        )
        response_weight = np.einsum(
            "gp,qp->gq", weighted_auxiliary.conj(), adjoint, optimize=True
        )
        ao_grid_weight = response_weight.reshape(
            len(vectors), cell.nao_nr(), cell.nao_nr()
        ).transpose(1, 2, 0)
        observed = gamma_general_ao_pair_strain(
            cell, ao_grid_weight, vectors, block_size=83
        )
        self.assertLess(
            float(np.max(np.abs(np.real(expected[0])-observed))), 3e-10
        )

    def test_gamma_separable_gpu_ft_source_matches_host_analytic_pair(self):
        cell = gto.Cell(
            atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
            a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3),
                          (0.1, 0.0, 6.0))),
            unit="Bohr", basis="gth-szv", pseudo="gth-hf",
            mesh=np.asarray((7, 9, 11)), verbose=0,
        ).build()
        auxcell = incore.make_auxcell(cell, "gth-szv")
        vectors = cell.get_Gv(cell.mesh)
        auxiliary = auxiliary_fourier_strain(auxcell, vectors)
        kernel = weighted_coulomb_strain(cell, vectors, omega=0.9)
        weighted_auxiliary = (auxiliary.value*kernel.value).T
        rng = np.random.default_rng(29)
        density = rng.normal(size=(cell.nao_nr(), cell.nao_nr()))
        density = 0.5*(density+density.T)
        auxiliary_weight = rng.normal(size=auxcell.nao_nr())
        adjoint = (
            density.reshape(-1, 1)*auxiliary_weight.reshape(1, -1)
        )
        expected = np.zeros((1, 6), dtype=np.complex128)
        context = _build_ao_pair_source_context(
            cell, np.zeros(3), np.zeros(3)
        )
        _contract_ao_pair_source_adjoint_block(
            context, vectors, weighted_auxiliary, adjoint[None], expected
        )
        reciprocal_weight = weighted_auxiliary.conj()@auxiliary_weight
        observed = gamma_separable_ao_pair_strain(
            cell, density, reciprocal_weight, vectors, block_size=97
        )
        self.assertLess(
            float(np.max(np.abs(np.real(expected[0])-observed))), 2e-10
        )

    def test_bounded_factorized_projection_matches_full_device_projection(self):
        rng = np.random.default_rng(23)
        ngrid, nao, naux = 11, 3, 4
        shape = (3, ngrid, nao, nao)
        center_i = rng.normal(size=shape)+1j*rng.normal(size=shape)
        center_j = rng.normal(size=shape)+1j*rng.normal(size=shape)
        moment = rng.normal(size=shape)+1j*rng.normal(size=shape)
        momentum = rng.normal(size=(ngrid, 3))
        weights = (
            rng.normal(size=(ngrid, naux))
            + 1j*rng.normal(size=(ngrid, naux))
        )
        expected = device_host_factorized_projection(
            center_i, center_j, moment, momentum, weights
        )
        observed = device_host_factorized_projection_bounded(
            center_i, center_j, moment, momentum, weights,
            transfer_block_bytes=1200,
        )
        for expected_part, observed_part in zip(expected, observed):
            self.assertLess(
                float(np.max(np.abs(expected_part-observed_part))), 2e-12
            )
        with self.assertRaisesRegex(MemoryError, "current free VRAM"):
            device_host_factorized_projection_bounded(
                center_i, center_j, moment, momentum, weights,
                transfer_block_bytes=1200, device_memory_fraction=0.0,
            )

    def test_reverse_cholesky_source_metric_contraction(self):
        rng = np.random.default_rng(19)
        naux, nao = 4, 3
        metric_seed = (
            rng.normal(size=(naux, naux))
            + 1j*rng.normal(size=(naux, naux))
        )
        metric = metric_seed@metric_seed.conj().T+2.0*np.eye(naux)
        source = (
            rng.normal(size=(nao*nao, naux))
            + 1j*rng.normal(size=(nao*nao, naux))
        )
        factorization = CDERIFactorization.build(
            cp.asarray(source), cp.asarray(metric), nao=nao
        )
        central = {(0, 0): factorization.value}
        reference = (
            rng.normal(size=(1, nao, nao))
            + 1j*rng.normal(size=(1, nao, nao))
        )
        response = (
            rng.normal(size=(1, nao, nao))
            + 1j*rng.normal(size=(1, nao, nao))
        )
        reference = 0.5*(
            reference+reference.conj().transpose(0, 2, 1)
        )
        response = 0.5*(
            response+response.conj().transpose(0, 2, 1)
        )
        reference_gpu = cp.asarray(reference)
        response_gpu = cp.asarray(response)
        reference_aux = cp.einsum(
            "pij,ji->p", factorization.value, reference_gpu[0]
        )
        response_aux = cp.einsum(
            "pij,ji->p", factorization.value, response_gpu[0]
        )
        cderi_adjoints = build_pair_cderi_adjoints(
            central, (0, 0), reference_gpu, response_gpu,
            reference_aux, response_aux,
        )
        source_seed = (
            rng.normal(size=source.shape)
            + 1j*rng.normal(size=source.shape)
        )
        metric_derivative_seed = (
            rng.normal(size=metric.shape)
            + 1j*rng.normal(size=metric.shape)
        )
        metric_derivative = (
            metric_derivative_seed+metric_derivative_seed.conj().T
        )
        cderi_derivative = factorization.differentiate(
            cp.asarray(source_seed), cp.asarray(metric_derivative)
        ).total

        for cderi_adjoint in cderi_adjoints:
            with self.subTest(observable=id(cderi_adjoint)):
                forward = cp.real(cp.einsum(
                    "pij,pij->", cderi_adjoint, cderi_derivative
                ))
                source_adjoint = push_cderi_adjoint_through_source(
                    factorization.factor, cderi_adjoint
                )
                metric_adjoint = push_cderi_adjoint_through_metric(
                    factorization, cderi_adjoint
                )
                reverse = cp.real(cp.einsum(
                    "qp,qp->", source_adjoint, cp.asarray(source_seed)
                ))
                reverse += cp.real(cp.einsum(
                    "pq,pq->", metric_adjoint,
                    cp.asarray(metric_derivative),
                ))
                self.assertLess(float(cp.abs(forward-reverse).get()), 2e-12)

    def test_single_pass_voigt_matches_cpu_tensor_algebra(self):
        rng = np.random.default_rng(11)
        nkpts, naux, nao = 2, 3, 2
        cpu_responses = {}
        gpu_primitives = {}
        for first in range(nkpts):
            for second in range(first+1):
                value = rng.normal(size=(naux, nao, nao))
                value = value+1j*rng.normal(size=value.shape)
                derivative = rng.normal(size=(3, 3, naux, nao, nao))
                derivative = derivative+1j*rng.normal(size=derivative.shape)
                if first == second:
                    value = 0.5*(value+value.conj().transpose(0, 2, 1))
                    derivative = 0.5*(
                        derivative+derivative.conj().transpose(0, 1, 2, 4, 3)
                    )
                pair = (first, second)
                cpu_responses[pair] = SimpleNamespace(
                    value=value, raw_derivative=derivative
                )
                gpu_primitives[pair] = PairStrainPrimitive(
                    source_derivative=matrix_tensor_to_voigt(derivative),
                    metric_derivative=np.zeros(
                        (6, naux, naux), dtype=np.complex128
                    ),
                    factorization=FakeFactorization(value),
                )
        density = rng.normal(size=(nkpts, nao, nao))
        density = density+1j*rng.normal(size=density.shape)
        density = 0.5*(density+density.conj().transpose(0, 2, 1))
        expected = kpoint_gdf_jk_strain(cpu_responses, density)
        for cache_enabled in (True, False):
            with self.subTest(cache_enabled=cache_enabled):
                response = DeviceDFStrainResponse(
                    backend="gdf",
                    primitives=MappingProxyType(gpu_primitives),
                    nkpts=nkpts,
                    nao=nao,
                    parameters={},
                    memory_estimate={},
                    diagonal_response_bytes=0,
                    diagonal_cache_enabled=cache_enabled,
                    preflight_free_device_bytes=0,
                    device_memory_fraction=0.85,
                    host_primitive_seconds=0.0,
                    host_worker_count=1,
                    device_factorization_seconds=0.0,
                    central_parity_max_abs=None,
                )
                observed = response.contract_jk_voigt(cp.asarray(density))
                self.assertLess(
                    float(cp.max(cp.abs(
                        observed.coulomb
                        - cp.asarray(matrix_tensor_to_voigt(expected.coulomb))
                    )).get()),
                    2e-11,
                )
                self.assertLess(
                    float(cp.max(cp.abs(
                        observed.exchange
                        - cp.asarray(matrix_tensor_to_voigt(expected.exchange))
                    )).get()),
                    2e-11,
                )
                self.assertEqual(observed.derivative_evaluations, 18)

                transition = rng.normal(size=(nkpts, nao, nao))
                transition = transition+1j*rng.normal(size=transition.shape)
                transition = 0.5*(
                    transition+transition.conj().transpose(0, 2, 1)
                )
                direct = response.contract_dc_energy_voigt(
                    cp.asarray(density), cp.asarray(transition)
                )
                target = 0.5*cp.einsum(
                    "xkij,kji->x", observed.coulomb,
                    cp.asarray(density), optimize=True,
                ).real/nkpts
                adjoint = 2.0*cp.einsum(
                    "xkij,kji->x", observed.restricted_hf,
                    cp.asarray(transition), optimize=True,
                ).real/nkpts
                self.assertLess(
                    float(cp.max(cp.abs(
                        direct.target_coulomb-target
                    )).get()),
                    2e-11,
                )
                self.assertLess(
                    float(cp.max(cp.abs(
                        direct.adjoint_restricted_hf-adjoint
                    )).get()),
                    2e-11,
                )
                self.assertEqual(direct.derivative_evaluations, 18)
                self.assertFalse(direct.metadata["materialized_jk_response"])


if __name__ == "__main__":
    unittest.main()
