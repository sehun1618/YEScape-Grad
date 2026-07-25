import unittest
from types import SimpleNamespace

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.df import incore

from pyscf_pbc_dcdft.df.gdf.cholesky import solve_cderi
from pyscf_pbc_dcdft.df.gdf.metric import auxiliary_metric
from pyscf_pbc_dcdft.df.gdf.source import three_center_source
from pyscf_pbc_dcdft_stress import (
    gamma_cderi_strain,
    gamma_ewald_exchange_strain,
    gamma_gdf_jk_strain,
    gamma_madelung_strain,
    kpoint_cderi_strain,
    kpoint_gdf_jk_strain,
    kpoint_ewald_exchange_strain,
    kpoint_madelung_strain,
)
from pyscf_pbc_dcdft_stress.strain import deform_cell, deform_kpoints


def make_cell():
    return gto.Cell(
        atom="H 0.2 0.3 0.4; H 1.2 0.8 1.6",
        a=np.asarray(((5.0, 0.2, 0.1), (0.0, 5.5, 0.3), (0.1, 0.0, 6.0))),
        unit="Bohr", basis="gth-szv", pseudo="gth-hf",
        mesh=np.asarray((7, 9, 11)), verbose=0,
    ).build()


def cderi_value(cell, auxcell, omega, source_mesh, metric_mesh):
    source = three_center_source(
        cell, auxcell, omega, mesh=source_mesh, kpti=np.zeros(3),
        kptj=np.zeros(3),
    )
    metric = auxiliary_metric(
        cell, auxcell, omega, mesh=metric_mesh, q=np.zeros(3)
    )
    return solve_cderi(source, metric, cell.nao_nr())


def kpoint_cderi_value(
    cell, auxcell, omega, source_mesh, metric_mesh, kpti, kptj
):
    source = three_center_source(
        cell, auxcell, omega, mesh=source_mesh, kpti=kpti, kptj=kptj
    )
    metric = auxiliary_metric(
        cell, auxcell, omega, mesh=metric_mesh, q=kptj-kpti
    )
    return solve_cderi(source, metric, cell.nao_nr())


def jk_value(cderi, density):
    auxiliary_density = np.einsum("pij,ji->p", cderi, density)
    coulomb = np.einsum("p,pij->ij", auxiliary_density, cderi)
    exchange = np.einsum("pik,kl,plj->ij", cderi, density, cderi)
    return coulomb, exchange


def sampled_k_jk_value(blocks, density):
    nkpts = len(density)

    def pair(ki, kj):
        if (ki, kj) in blocks:
            return blocks[(ki, kj)]
        return blocks[(kj, ki)].conj().transpose(0, 2, 1)

    jaux = sum(
        np.einsum("pij,ji->p", pair(k, k), density[k], optimize=True)
        for k in range(nkpts)
    )
    coulomb = np.asarray([
        np.einsum("p,pij->ij", jaux, pair(k, k), optimize=True)/nkpts
        for k in range(nkpts)
    ])
    exchange = np.zeros_like(coulomb, dtype=np.complex128)
    for ki in range(nkpts):
        for kj in range(nkpts):
            exchange[ki] += np.einsum(
                "pij,jk,pkl->il", pair(ki, kj), density[kj],
                pair(kj, ki), optimize=True,
            )/nkpts
    return coulomb, exchange


def matrix_tensor_to_voigt_for_test(value):
    tensor = np.asarray(value)
    return np.asarray((
        tensor[0, 0], tensor[1, 1], tensor[2, 2],
        0.5*(tensor[1, 2]+tensor[2, 1]),
        0.5*(tensor[0, 2]+tensor[2, 0]),
        0.5*(tensor[0, 1]+tensor[1, 0]),
    ))


class GDFCDERIStrainTest(unittest.TestCase):
    def test_sampled_k_madelung_and_exchange_match_strain_fd(self):
        from pyscf.pbc.tools import pbc as pbc_tools

        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        kpts = cell.make_kpts((2, 1, 1))
        density = np.asarray([
            ((1.2, 0.15), (0.15, 0.8)),
            ((1.1, 0.08j), (-0.08j, 0.9)),
        ], dtype=np.complex128)
        madelung = kpoint_madelung_strain(cell, kpts)
        exchange = kpoint_ewald_exchange_strain(cell, kpts, density)
        step = 1e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_kpts = deform_kpoints(cell, plus_cell, kpts)
            minus_kpts = deform_kpoints(cell, minus_cell, kpts)
            plus_m = pbc_tools.madelung(plus_cell, plus_kpts)
            minus_m = pbc_tools.madelung(minus_cell, minus_kpts)
            scalar_reference = (plus_m-minus_m)/(2.0*step)
            self.assertLess(
                abs(madelung.voigt_derivative[index]-scalar_reference),
                8e-9,
            )
            plus_s = plus_cell.pbc_intor(
                "int1e_ovlp", hermi=1, kpts=plus_kpts
            )
            minus_s = minus_cell.pbc_intor(
                "int1e_ovlp", hermi=1, kpts=minus_kpts
            )
            plus_value = np.asarray([
                plus_m*s@dm@s for s, dm in zip(plus_s, density)
            ])
            minus_value = np.asarray([
                minus_m*s@dm@s for s, dm in zip(minus_s, density)
            ])
            reference = (plus_value-minus_value)/(2.0*step)
            error = np.max(
                np.abs(matrix_tensor_to_voigt_for_test(exchange)[index]-reference)
            )
            self.assertLess(float(error), 3e-8, msg=index)

    def test_sampled_k_jk_strain_tensor_algebra(self):
        rng = np.random.default_rng(8)
        nkpts, naux, nao = 2, 3, 2
        responses = {}
        for ki in range(nkpts):
            for kj in range(ki+1):
                value = rng.normal(size=(naux, nao, nao))
                value = value+1j*rng.normal(size=value.shape)
                derivative = rng.normal(size=(3, 3, naux, nao, nao))
                derivative = derivative+1j*rng.normal(size=derivative.shape)
                if ki == kj:
                    value = 0.5*(value+value.conj().transpose(0, 2, 1))
                    derivative = 0.5*(
                        derivative+derivative.conj().transpose(0, 1, 2, 4, 3)
                    )
                responses[(ki, kj)] = SimpleNamespace(
                    value=value, raw_derivative=derivative
                )
        density = rng.normal(size=(nkpts, nao, nao))
        density = density+1j*rng.normal(size=density.shape)
        density = 0.5*(density+density.conj().transpose(0, 2, 1))
        analytic = kpoint_gdf_jk_strain(responses, density)
        step = 1e-6
        for x, y in ((0, 0), (0, 1), (2, 1)):
            plus = {
                pair: response.value+step*response.raw_derivative[x, y]
                for pair, response in responses.items()
            }
            minus = {
                pair: response.value-step*response.raw_derivative[x, y]
                for pair, response in responses.items()
            }
            plus_j, plus_k = sampled_k_jk_value(plus, density)
            minus_j, minus_k = sampled_k_jk_value(minus, density)
            self.assertLess(float(np.max(np.abs(
                analytic.coulomb[x, y]-(plus_j-minus_j)/(2*step)
            ))), 2e-8)
            self.assertLess(float(np.max(np.abs(
                analytic.exchange[x, y]-(plus_k-minus_k)/(2*step)
            ))), 2e-8)

    def test_complex_kpair_cderi_matches_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        scaled = np.asarray(((0.17, 0.08, 0.03), (-0.11, 0.06, 0.02)))
        kpti, kptj = cell.get_abs_kpts(scaled)
        analytic = kpoint_cderi_strain(
            cell, auxcell, omega, mesh, mesh, kpti, kptj
        )
        central = kpoint_cderi_value(
            cell, auxcell, omega, mesh, mesh, kpti, kptj
        )
        self.assertLess(
            float(np.max(np.abs(analytic.value-central))), 2e-9
        )
        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus_kpts = deform_kpoints(cell, plus_cell, (kpti, kptj))
            minus_kpts = deform_kpoints(cell, minus_cell, (kpti, kptj))
            plus = kpoint_cderi_value(
                plus_cell, plus_aux, omega, mesh, mesh,
                plus_kpts[0], plus_kpts[1],
            )
            minus = kpoint_cderi_value(
                minus_cell, minus_aux, omega, mesh, mesh,
                minus_kpts[0], minus_kpts[1],
            )
            reference = (plus-minus) / (2.0*step)
            error = np.max(
                np.abs(analytic.voigt_derivative[index]-reference)
            )
            self.assertLess(float(error), 8e-8, msg=index)

    def test_madelung_and_ewald_exchange_match_strain_fd(self):
        from pyscf.pbc.tools import pbc as pbc_tools

        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        density = np.asarray(((1.2, 0.15), (0.15, 0.8)))
        madelung = gamma_madelung_strain(cell)
        exchange_derivative = gamma_ewald_exchange_strain(cell, density)
        step = 1e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_madelung = pbc_tools.madelung(
                plus_cell, np.zeros((1, 3))
            )
            minus_madelung = pbc_tools.madelung(
                minus_cell, np.zeros((1, 3))
            )
            scalar_reference = (
                plus_madelung - minus_madelung
            ) / (2.0 * step)
            self.assertLess(
                abs(madelung.voigt_derivative[index] - scalar_reference),
                5e-9,
                msg=("madelung", index),
            )
            plus_overlap = plus_cell.pbc_intor("int1e_ovlp", hermi=1)
            minus_overlap = minus_cell.pbc_intor("int1e_ovlp", hermi=1)
            plus = plus_madelung * plus_overlap @ density @ plus_overlap
            minus = minus_madelung * minus_overlap @ density @ minus_overlap
            matrix_reference = (plus - minus) / (2.0 * step)
            voigt = np.asarray(
                [
                    exchange_derivative[0, 0],
                    exchange_derivative[1, 1],
                    exchange_derivative[2, 2],
                    0.5 * (exchange_derivative[1, 2] + exchange_derivative[2, 1]),
                    0.5 * (exchange_derivative[0, 2] + exchange_derivative[2, 0]),
                    0.5 * (exchange_derivative[0, 1] + exchange_derivative[1, 0]),
                ]
            )
            error = np.max(np.abs(voigt[index] - matrix_reference))
            self.assertLess(float(error), 2e-8, msg=("exchange", index))

    def test_cderi_and_jk_match_strain_fd(self):
        cell = make_cell()
        mesh = np.asarray(cell.mesh, dtype=int)
        auxcell = incore.make_auxcell(cell, "gth-szv")
        omega = 0.9
        analytic = gamma_cderi_strain(
            cell, auxcell, omega, mesh, mesh
        )
        central = cderi_value(cell, auxcell, omega, mesh, mesh)
        self.assertLess(
            float(np.max(np.abs(analytic.value - central))), 1e-9
        )
        density = np.asarray(((1.2, 0.15), (0.15, 0.8)))
        analytic_jk = gamma_gdf_jk_strain(analytic, density)

        step = 2e-5
        for index in range(6):
            plus_cell = deform_cell(cell, index, step, mesh=mesh)
            minus_cell = deform_cell(cell, index, -step, mesh=mesh)
            plus_aux = incore.make_auxcell(plus_cell, "gth-szv")
            minus_aux = incore.make_auxcell(minus_cell, "gth-szv")
            plus = cderi_value(plus_cell, plus_aux, omega, mesh, mesh)
            minus = cderi_value(minus_cell, minus_aux, omega, mesh, mesh)
            cderi_reference = (plus - minus) / (2.0 * step)
            cderi_error = np.max(
                np.abs(analytic.voigt_derivative[index] - cderi_reference)
            )
            self.assertLess(float(cderi_error), 2e-8, msg=("cderi", index))

            plus_j, plus_k = jk_value(plus, density)
            minus_j, minus_k = jk_value(minus, density)
            j_reference = (plus_j - minus_j) / (2.0 * step)
            k_reference = (plus_k - minus_k) / (2.0 * step)
            j_error = np.max(
                np.abs(analytic_jk.coulomb_voigt[index] - j_reference)
            )
            k_error = np.max(
                np.abs(analytic_jk.exchange_voigt[index] - k_reference)
            )
            self.assertLess(float(j_error), 3e-8, msg=("j", index))
            self.assertLess(float(k_error), 3e-8, msg=("k", index))


if __name__ == "__main__":
    unittest.main()
