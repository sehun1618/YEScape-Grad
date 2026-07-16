"""Fast policy tests for batched GPU nuclear CPHF sources."""

import unittest
from types import SimpleNamespace

import cupy as cp
import numpy as np

from gpu4pyscf_pbc_dcdft.grad.nuclear_response import (
    AdaptiveBatchScheduler,
    build_adjoint_nuclear_response_state,
    contract_adjoint_z_nuclear_sources,
    contract_z_vector_sets,
    execute_adaptive_batches,
    nuclear_response_source_vo_sets,
    nuclear_response_batch_size,
    plan_nuclear_response_batches,
)
from gpu4pyscf_pbc_dcdft.grad.orbitals import MOPartition


class NuclearResponseBatchPolicyTest(unittest.TestCase):
    def test_adjoint_contraction_matches_explicit_metric_responses(self):
        rng = np.random.default_rng(18)
        nao, nocc, nvir, nsets = 7, 3, 4, 5
        q, _ = np.linalg.qr(
            rng.normal(size=(nao, nao))
            + 1j * rng.normal(size=(nao, nao))
        )
        occupied = cp.asarray(q[:, :nocc])
        virtual = cp.asarray(q[:, nocc:])
        energies = cp.asarray(np.linspace(-1.2, 0.8, nao))
        partition = MOPartition(
            occupied_coeff=occupied,
            virtual_coeff=virtual,
            occupied_mask=cp.asarray([True] * nocc + [False] * nvir),
            virtual_mask=cp.asarray([False] * nocc + [True] * nvir),
            energy_gap=(
                energies[nocc:, None] - energies[:nocc][None, :]
            ),
        )
        z = cp.asarray(
            rng.normal(size=(nvir, nocc))
            + 1j * rng.normal(size=(nvir, nocc))
        )
        h1 = _random_hermitian_sets(rng, nsets, nao)[:, None]
        overlap = _random_hermitian_sets(rng, nsets, nao)[:, None]
        source_mf = _SelfAdjointResponse(nao, q)
        context = SimpleNamespace(
            source_mf=source_mf,
            mo_energy=energies[None],
        )

        explicit_sources = nuclear_response_source_vo_sets(
            context, [partition], h1, overlap
        )
        explicit = contract_z_vector_sets([z], explicit_sources, nkpts=1)
        state = build_adjoint_nuclear_response_state(
            context, [partition], [z]
        )
        adjoint = contract_adjoint_z_nuclear_sources(
            state, [partition], h1, overlap, nkpts=1
        )
        self.assertLess(
            float(cp.max(cp.abs(explicit - adjoint)).get()), 1.0e-10
        )

    def test_batch_size_is_bounded_by_tensor_memory(self):
        matrix_bytes = 2 * 20 * 20 * 16
        working_bytes = 12 * matrix_bytes
        batch = nuclear_response_batch_size(
            100,
            2,
            20,
            free_device_bytes=10 * working_bytes,
            memory_fraction=1.0,
        )
        self.assertEqual(batch, 10)

    def test_batch_size_stays_between_one_and_component_count(self):
        self.assertEqual(
            nuclear_response_batch_size(
                7,
                1,
                100,
                free_device_bytes=1,
                memory_fraction=1.0,
            ),
            1,
        )
        self.assertEqual(
            nuclear_response_batch_size(
                7,
                1,
                2,
                free_device_bytes=1024**3,
                memory_fraction=1.0,
            ),
            7,
        )

    def test_batch_plan_records_memory_preflight_and_explicit_cap(self):
        plan = plan_nuclear_response_batches(
            18,
            2,
            20,
            free_device_bytes=1024**3,
            total_device_bytes=2 * 1024**3,
            memory_fraction=0.25,
            max_batch_size=3,
        )
        self.assertEqual(plan.batch_size, 3)
        self.assertEqual(plan.batch_count, 6)
        self.assertEqual(plan.max_batch_size, 3)
        self.assertEqual(plan.total_device_bytes, 2 * 1024**3)
        self.assertEqual(
            plan.estimated_batch_working_bytes,
            3 * plan.estimated_bytes_per_component,
        )

    def test_adaptive_scheduler_retries_without_dropping_components(self):
        scheduler = AdaptiveBatchScheduler(component_count=7, batch_size=4)
        self.assertEqual(scheduler.next_range(), (0, 4))
        self.assertEqual(scheduler.shrink_after_oom(), 2)

        accepted = []
        while not scheduler.done:
            start, stop = scheduler.next_range()
            accepted.extend(range(start, stop))
            scheduler.accept(stop)

        self.assertEqual(accepted, list(range(7)))
        self.assertEqual(scheduler.completed_count, 7)
        self.assertEqual(scheduler.rebatch_count, 1)
        self.assertEqual(scheduler.minimum_batch_size, 2)

    def test_adaptive_scheduler_reports_minimum_batch_oom(self):
        scheduler = AdaptiveBatchScheduler(component_count=2, batch_size=1)
        with self.assertRaisesRegex(RuntimeError, "minimum batch size 1"):
            scheduler.shrink_after_oom()

    def test_adaptive_executor_retries_failed_range_without_truncation(self):
        output = [None] * 7
        oom_ranges = []

        def worker(start, stop):
            if stop - start > 2:
                raise cp.cuda.memory.OutOfMemoryError(1, 0, 0)
            return list(range(start, stop))

        def accept_result(start, stop, values):
            output[start:stop] = values

        scheduler = execute_adaptive_batches(
            7,
            4,
            worker,
            accept_result,
            on_oom=lambda start, stop, _error: oom_ranges.append(
                (start, stop)
            ),
        )

        self.assertEqual(oom_ranges, [(0, 4)])
        self.assertEqual(output, list(range(7)))
        self.assertEqual(scheduler.rebatch_count, 1)
        self.assertEqual(scheduler.batch_size, 2)


def _random_hermitian_sets(rng, nsets, nao):
    raw = rng.normal(size=(nsets, nao, nao))
    raw = raw + 1j * rng.normal(size=raw.shape)
    return cp.asarray(raw + raw.transpose(0, 2, 1).conj())


class _SelfAdjointResponse:
    def __init__(self, nao, mo_coeff):
        self.mo_coeff = cp.asarray(mo_coeff)
        self.with_df = SimpleNamespace(backend_name="test")
        self._identity = cp.eye(nao)

    def get_jk(self, dm, **_kwargs):
        density = cp.asarray(dm)
        trace = cp.trace(density, axis1=-2, axis2=-1)
        coulomb = 0.7 * density + 0.2 * trace[..., None, None] * self._identity
        exchange = 0.4 * density
        return coulomb, exchange


if __name__ == "__main__":
    unittest.main()
