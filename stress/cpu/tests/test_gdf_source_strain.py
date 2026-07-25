import unittest

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.df import ft_ao, incore

from pyscf_pbc_dcdft.df.gdf.source_long_range import long_range_source
from pyscf_pbc_dcdft.df.gdf.source_short_range import short_range_source
from pyscf_pbc_dcdft.df.gdf.source import three_center_source
from pyscf_pbc_dcdft.df.gdf.vbar import vbar_source
from pyscf_pbc_dcdft_stress import (
    gamma_ao_pair_fourier_strain,
    gamma_long_range_source_strain,
    gamma_short_range_source_strain,
    gamma_three_center_source_strain,
    gamma_vbar_source_strain,
    kpoint_ao_pair_fourier_strain,
    kpoint_long_range_source_adjoint_strain,
    kpoint_long_range_source_strain,
    kpoint_short_range_source_strain,
    kpoint_three_center_source_adjoint_strain,
    kpoint_three_center_source_strain,
    kpoint_vbar_source_strain,
)
from pyscf_pbc_dcdft_stress.strain import deform_cell, deform_kpoints


def make_cell():
    return gto.Cell(
        atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
        a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3), (0.1, 0.0, 6.0))),
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((7, 9, 11)), verbose=0,
    ).build()


class GDFSourceStrainTest(unittest.TestCase):
    def test_complete_source_adjoint_contraction_matches_full_gamma(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        full = gamma_three_center_source_strain(
            cell, auxcell, omega, mesh, reciprocal_block_size=17,
            retain_components=False, compact_voigt=True,
        )
        rng = np.random.default_rng(83)
        adjoints = (
            rng.normal(size=(2, cell.nao_nr()**2, auxcell.nao_nr()))
            + 1j*rng.normal(
                size=(2, cell.nao_nr()**2, auxcell.nao_nr())
            )
        )
        expected = np.real(np.einsum(
            "sqp,xqp->sx", adjoints, full.raw_derivative, optimize=True
        ))
        observed = kpoint_three_center_source_adjoint_strain(
            cell, auxcell, omega, mesh, np.zeros(3), np.zeros(3), adjoints,
            reciprocal_block_size=17,
        )
        self.assertLess(
            float(np.max(np.abs(observed.values-expected))), 5e-10
        )
        self.assertEqual(
            observed.memory_estimate["retained_source_derivative_bytes"], 0
        )

    def test_complete_source_adjoint_contraction_matches_complex_kpair(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        scaled = np.asarray(((0.17, 0.08, 0.03), (-0.11, 0.06, 0.02)))
        kpti, kptj = cell.get_abs_kpts(scaled)
        full = kpoint_three_center_source_strain(
            cell, auxcell, omega, mesh, kpti, kptj,
            reciprocal_block_size=17, retain_components=False,
            compact_voigt=True,
        )
        rng = np.random.default_rng(84)
        adjoints = (
            rng.normal(size=(2, cell.nao_nr()**2, auxcell.nao_nr()))
            + 1j*rng.normal(
                size=(2, cell.nao_nr()**2, auxcell.nao_nr())
            )
        )
        expected = np.real(np.einsum(
            "sqp,xqp->sx", adjoints, full.raw_derivative, optimize=True
        ))
        observed = kpoint_three_center_source_adjoint_strain(
            cell, auxcell, omega, mesh, kpti, kptj, adjoints,
            reciprocal_block_size=17,
        )
        self.assertLess(
            float(np.max(np.abs(observed.values-expected))), 7e-10
        )

    def test_long_range_adjoint_contraction_matches_full_gamma_response(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        full = gamma_long_range_source_strain(
            cell, auxcell, omega, mesh, reciprocal_block_size=17,
            retain_components=False, compact_voigt=True,
        )
        rng = np.random.default_rng(81)
        adjoints = (
            rng.normal(size=(2, cell.nao_nr()**2, auxcell.nao_nr()))
            + 1j*rng.normal(
                size=(2, cell.nao_nr()**2, auxcell.nao_nr())
            )
        )
        expected = np.real(np.einsum(
            "sqp,xqp->sx", adjoints, full.raw_derivative, optimize=True
        ))
        observed = kpoint_long_range_source_adjoint_strain(
            cell, auxcell, omega, mesh, np.zeros(3), np.zeros(3), adjoints,
            reciprocal_block_size=17,
        )
        self.assertEqual(observed.values.shape, (2, 6))
        self.assertLess(
            float(np.max(np.abs(observed.values-expected))), 3e-10
        )
        self.assertGreater(observed.full_response_bytes_avoided, 0)

    def test_long_range_adjoint_contraction_matches_complex_kpair(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        scaled = np.asarray(((0.17, 0.08, 0.03), (-0.11, 0.06, 0.02)))
        kpti, kptj = cell.get_abs_kpts(scaled)
        full = kpoint_long_range_source_strain(
            cell, auxcell, omega, mesh, kpti, kptj,
            reciprocal_block_size=17, retain_components=False,
            compact_voigt=True,
        )
        rng = np.random.default_rng(82)
        adjoints = (
            rng.normal(size=(2, cell.nao_nr()**2, auxcell.nao_nr()))
            + 1j*rng.normal(
                size=(2, cell.nao_nr()**2, auxcell.nao_nr())
            )
        )
        expected = np.real(np.einsum(
            "sqp,xqp->sx", adjoints, full.raw_derivative, optimize=True
        ))
        observed = kpoint_long_range_source_adjoint_strain(
            cell, auxcell, omega, mesh, kpti, kptj, adjoints,
            reciprocal_block_size=17,
        )
        self.assertLess(
            float(np.max(np.abs(observed.values-expected))), 3e-10
        )

    def test_long_range_source_blocking_and_compact_mode_preserve_result(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        grid_points = int(np.prod(mesh))
        full = gamma_long_range_source_strain(
            cell, auxcell, omega, mesh,
            reciprocal_block_size=grid_points,
        )
        blocked = gamma_long_range_source_strain(
            cell, auxcell, omega, mesh, reciprocal_block_size=17,
        )
        compact = gamma_long_range_source_strain(
            cell, auxcell, omega, mesh, reciprocal_block_size=17,
            retain_components=False,
        )
        compact_voigt = gamma_long_range_source_strain(
            cell, auxcell, omega, mesh, reciprocal_block_size=17,
            retain_components=False, compact_voigt=True,
        )
        self.assertEqual(blocked.reciprocal_block_size, 17)
        self.assertEqual(blocked.grid_points, grid_points)
        self.assertIsNone(compact.ao_pair_response)
        self.assertIsNone(compact.auxiliary_response)
        self.assertIsNone(compact.kernel_response)
        for candidate in (blocked, compact):
            self.assertLess(
                float(np.max(np.abs(candidate.value-full.value))), 2e-11
            )
            self.assertLess(
                float(np.max(np.abs(
                    candidate.raw_derivative-full.raw_derivative
                ))),
                2e-10,
            )
        self.assertEqual(compact_voigt.raw_derivative.shape[0], 6)
        self.assertLess(
            float(np.max(np.abs(
                compact_voigt.raw_derivative-full.voigt_derivative
            ))),
            2e-10,
        )

    def test_complete_compact_voigt_source_matches_cartesian_projection(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        full = gamma_three_center_source_strain(
            cell, auxcell, 0.9, mesh, reciprocal_block_size=17,
            retain_components=False,
        )
        compact = gamma_three_center_source_strain(
            cell, auxcell, 0.9, mesh, reciprocal_block_size=17,
            retain_components=False, compact_voigt=True,
        )
        self.assertEqual(compact.raw_derivative.shape[0], 6)
        self.assertLess(float(np.max(np.abs(compact.value-full.value))), 2e-11)
        self.assertLess(
            float(np.max(np.abs(
                compact.raw_derivative-full.voigt_derivative
            ))),
            2e-10,
        )

    def test_complete_gamma_source_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        analytic = gamma_three_center_source_strain(
            cell, auxcell, omega, mesh
        )
        central = three_center_source(
            cell, auxcell, omega, mesh=mesh, kpti=np.zeros(3),
            kptj=np.zeros(3),
        )
        self.assertLess(
            float(np.max(np.abs(analytic.value - central))), 3e-9
        )

        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus = three_center_source(
                plus_cell, plus_aux, omega, mesh=mesh, kpti=np.zeros(3),
                kptj=np.zeros(3),
            )
            minus = three_center_source(
                minus_cell, minus_aux, omega, mesh=mesh, kpti=np.zeros(3),
                kptj=np.zeros(3),
            )
            reference = (plus - minus) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 8e-9, msg=index)

    def test_complete_complex_kpair_source_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        scaled = np.asarray(((0.17, 0.08, 0.03), (-0.11, 0.06, 0.02)))
        kpti, kptj = cell.get_abs_kpts(scaled)
        analytic = kpoint_three_center_source_strain(
            cell, auxcell, omega, mesh, kpti, kptj
        )
        central = three_center_source(
            cell, auxcell, omega, mesh=mesh, kpti=kpti, kptj=kptj
        )
        self.assertLess(
            float(np.max(np.abs(analytic.value-central))), 5e-9
        )
        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus_kpts = deform_kpoints(cell, plus_cell, (kpti, kptj))
            minus_kpts = deform_kpoints(cell, minus_cell, (kpti, kptj))
            plus = three_center_source(
                plus_cell, plus_aux, omega, mesh=mesh,
                kpti=plus_kpts[0], kptj=plus_kpts[1],
            )
            minus = three_center_source(
                minus_cell, minus_aux, omega, mesh=mesh,
                kpti=minus_kpts[0], kptj=minus_kpts[1],
            )
            reference = (plus-minus) / (2.0*step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index]-reference)
            )
            self.assertLess(float(error), 5e-8, msg=index)

    def test_diagonal_kpoint_vbar_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        kpt = cell.get_abs_kpts(np.asarray(((0.19, 0.07, 0.03),)))[0]
        analytic = kpoint_vbar_source_strain(
            cell, auxcell, omega, kpt, kpt
        )
        central = vbar_source(cell, auxcell, omega, kpt=kpt)
        self.assertLess(
            float(np.max(np.abs(analytic.value-central))), 2e-9
        )
        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus_kpt = deform_kpoints(cell, plus_cell, (kpt,))[0]
            minus_kpt = deform_kpoints(cell, minus_cell, (kpt,))[0]
            plus = vbar_source(plus_cell, plus_aux, omega, kpt=plus_kpt)
            minus = vbar_source(minus_cell, minus_aux, omega, kpt=minus_kpt)
            reference = (plus-minus) / (2.0*step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index]-reference)
            )
            self.assertLess(float(error), 2e-8, msg=index)

    def test_short_range_source_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        analytic = gamma_short_range_source_strain(
            cell, auxcell, omega
        )
        central = short_range_source(
            cell, auxcell, omega, kpti=np.zeros(3), kptj=np.zeros(3)
        )
        self.assertLess(
            float(np.max(np.abs(analytic.value - central))), 1e-12
        )

        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus = short_range_source(
                plus_cell, plus_aux, omega, kpti=np.zeros(3),
                kptj=np.zeros(3),
            )
            minus = short_range_source(
                minus_cell, minus_aux, omega, kpti=np.zeros(3),
                kptj=np.zeros(3),
            )
            reference = (plus - minus) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 2e-9, msg=index)

    def test_complex_kpair_short_range_source_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        scaled = np.asarray(((0.17, 0.08, 0.03), (-0.11, 0.06, 0.02)))
        kpti, kptj = cell.get_abs_kpts(scaled)
        analytic = kpoint_short_range_source_strain(
            cell, auxcell, omega, kpti, kptj
        )
        central = short_range_source(
            cell, auxcell, omega, kpti=kpti, kptj=kptj
        )
        self.assertLess(
            float(np.max(np.abs(analytic.value - central))), 3e-10
        )

        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus_kpts = deform_kpoints(cell, plus_cell, (kpti, kptj))
            minus_kpts = deform_kpoints(cell, minus_cell, (kpti, kptj))
            plus = short_range_source(
                plus_cell, plus_aux, omega,
                kpti=plus_kpts[0], kptj=plus_kpts[1],
            )
            minus = short_range_source(
                minus_cell, minus_aux, omega,
                kpti=minus_kpts[0], kptj=minus_kpts[1],
            )
            reference = (plus - minus) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 4e-9, msg=index)

    def test_periodic_ao_pair_fourier_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        grid_vectors = cell.get_Gv(mesh)
        analytic = gamma_ao_pair_fourier_strain(cell, grid_vectors)
        central = ft_ao.ft_aopair(
            cell, grid_vectors, aosym="s1", kpti_kptj=np.zeros((2, 3)),
            intor="GTO_ft_ovlp",
        )
        self.assertLess(float(np.max(np.abs(analytic.value - central))), 1e-9)

        step = 1e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus = ft_ao.ft_aopair(
                plus_cell, plus_cell.get_Gv(mesh), aosym="s1",
                kpti_kptj=np.zeros((2, 3)), intor="GTO_ft_ovlp",
            )
            minus = ft_ao.ft_aopair(
                minus_cell, minus_cell.get_Gv(mesh), aosym="s1",
                kpti_kptj=np.zeros((2, 3)), intor="GTO_ft_ovlp",
            )
            reference = (plus - minus) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 1e-8, msg=index)

    def test_complex_kpair_ao_pair_fourier_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        grid_vectors = cell.get_Gv(mesh)
        scaled = np.asarray(((0.17, 0.08, 0.03), (-0.11, 0.06, 0.02)))
        kpti, kptj = cell.get_abs_kpts(scaled)
        analytic = kpoint_ao_pair_fourier_strain(
            cell, grid_vectors, kpti, kptj
        )
        central = ft_ao.ft_aopair(
            cell, grid_vectors, aosym="s1", kpti_kptj=(kpti, kptj),
            q=kptj-kpti, intor="GTO_ft_ovlp",
        )
        self.assertLess(float(np.max(np.abs(analytic.value-central))), 2e-9)

        step = 1e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_kpts = deform_kpoints(cell, plus_cell, (kpti, kptj))
            minus_kpts = deform_kpoints(cell, minus_cell, (kpti, kptj))
            plus = ft_ao.ft_aopair(
                plus_cell, plus_cell.get_Gv(mesh), aosym="s1",
                kpti_kptj=plus_kpts, q=plus_kpts[1]-plus_kpts[0],
                intor="GTO_ft_ovlp",
            )
            minus = ft_ao.ft_aopair(
                minus_cell, minus_cell.get_Gv(mesh), aosym="s1",
                kpti_kptj=minus_kpts, q=minus_kpts[1]-minus_kpts[0],
                intor="GTO_ft_ovlp",
            )
            reference = (plus-minus) / (2.0*step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index]-reference)
            )
            self.assertLess(float(error), 2e-8, msg=index)

    def test_long_range_source_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        analytic = gamma_long_range_source_strain(
            cell, auxcell, omega, mesh
        )
        central = long_range_source(
            cell, auxcell, omega, mesh=mesh, kpti=np.zeros(3),
            kptj=np.zeros(3),
        )
        self.assertLess(
            float(np.max(np.abs(analytic.value - central))), 2e-9
        )

        step = 1e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus = long_range_source(
                plus_cell, plus_aux, omega, mesh=mesh, kpti=np.zeros(3),
                kptj=np.zeros(3),
            )
            minus = long_range_source(
                minus_cell, minus_aux, omega, mesh=mesh, kpti=np.zeros(3),
                kptj=np.zeros(3),
            )
            reference = (plus - minus) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 5e-9, msg=index)

    def test_complex_kpair_long_range_source_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        scaled = np.asarray(((0.17, 0.08, 0.03), (-0.11, 0.06, 0.02)))
        kpti, kptj = cell.get_abs_kpts(scaled)
        analytic = kpoint_long_range_source_strain(
            cell, auxcell, omega, mesh, kpti, kptj
        )
        central = long_range_source(
            cell, auxcell, omega, mesh=mesh, kpti=kpti, kptj=kptj
        )
        self.assertLess(
            float(np.max(np.abs(analytic.value-central))), 4e-9
        )

        step = 1e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus_kpts = deform_kpoints(cell, plus_cell, (kpti, kptj))
            minus_kpts = deform_kpoints(cell, minus_cell, (kpti, kptj))
            plus = long_range_source(
                plus_cell, plus_aux, omega, mesh=mesh,
                kpti=plus_kpts[0], kptj=plus_kpts[1],
            )
            minus = long_range_source(
                minus_cell, minus_aux, omega, mesh=mesh,
                kpti=minus_kpts[0], kptj=minus_kpts[1],
            )
            reference = (plus-minus) / (2.0*step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index]-reference)
            )
            self.assertLess(float(error), 4e-8, msg=index)

    def test_vbar_source_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        analytic = gamma_vbar_source_strain(cell, auxcell, omega)
        central = vbar_source(
            cell, auxcell, omega, kpt=np.zeros(3)
        )
        self.assertLess(
            float(np.max(np.abs(analytic.value - central))), 1e-9
        )

        step = 1e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus = vbar_source(
                plus_cell, plus_aux, omega, kpt=np.zeros(3)
            )
            minus = vbar_source(
                minus_cell, minus_aux, omega, kpt=np.zeros(3)
            )
            reference = (plus - minus) / (2.0 * step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index] - reference)
            )
            self.assertLess(float(error), 2e-9, msg=index)


if __name__ == "__main__":
    unittest.main()
