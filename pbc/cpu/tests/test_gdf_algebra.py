"""Unit tests for GDF storage, metric response, and J/K contractions."""

from __future__ import annotations

import unittest

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.gto.pseudo import pp_int

from pyscf_pbc_dcdft import df
from pyscf_pbc_dcdft.df.gdf.cholesky import differentiate_cderi, solve_cderi
from pyscf_pbc_dcdft.df.gdf.jk_derivative import (
    ewald_exchange_derivative,
    gdf_jk_derivative,
)
from pyscf_pbc_dcdft.df.gdf.pair_response import (
    rsgdf_parameters,
)
from pyscf_pbc_dcdft.df.gdf.response import build_gdf_response_tensors
from pyscf_pbc_dcdft.df.gdf.storage import canonical_pairs, load_cderi_storage
from pyscf_pbc_dcdft.grad.one_electron import build_one_electron_derivatives
from pyscf_pbc_dcdft.pseudo import nonlocal_gth_derivative


def _hermitian_positive_definite(rng, size):
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return raw @ raw.conj().T + np.eye(size) * 2.0


def _hermitian(rng, size):
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return raw + raw.conj().T


def _jk_value(cderi, density):
    nkpts, nao, _ = density.shape
    jaux = sum(
        np.einsum("pij,ji->p", _pair(cderi, k, k), density[k])
        for k in range(nkpts)
    )
    coulomb = np.empty((nkpts, nao, nao), dtype=np.complex128)
    exchange = np.empty_like(coulomb)
    for ki in range(nkpts):
        c_ii = _pair(cderi, ki, ki)
        coulomb[ki] = np.einsum("p,pij->ij", jaux, c_ii) / nkpts
        exchange[ki] = sum(
            np.einsum(
                "pij,jk,pkl->il",
                _pair(cderi, ki, kj),
                density[kj],
                _pair(cderi, kj, ki),
            )
            for kj in range(nkpts)
        ) / nkpts
    return coulomb, exchange


def _pair(blocks, ki, kj):
    if (ki, kj) in blocks:
        return blocks[(ki, kj)]
    return blocks[(kj, ki)].conj().transpose(0, 2, 1)


class GDFCholeskyResponseTest(unittest.TestCase):
    def test_source_and_metric_derivative_matches_central_difference(self):
        rng = np.random.default_rng(9)
        nao, naux = 2, 4
        source = rng.normal(size=(nao * nao, naux)) + 1j * rng.normal(
            size=(nao * nao, naux)
        )
        dsource = rng.normal(size=source.shape) + 1j * rng.normal(
            size=source.shape
        )
        metric = _hermitian_positive_definite(rng, naux)
        dmetric = _hermitian(rng, naux)
        parts = differentiate_cderi(source, metric, dsource, dmetric, nao)
        step = 1.0e-5
        plus = solve_cderi(
            source + step * dsource, metric + step * dmetric, nao
        )
        minus = solve_cderi(
            source - step * dsource, metric - step * dmetric, nao
        )
        finite_difference = (plus - minus) / (2.0 * step)
        self.assertLess(np.max(np.abs(parts.total - finite_difference)), 2.0e-8)


class GDFJKDerivativeTest(unittest.TestCase):
    def test_full_jk_derivative_matches_factor_finite_difference(self):
        rng = np.random.default_rng(11)
        nkpts, naux, nao = 2, 3, 2
        cderi = {}
        dcderi = {}
        for pair in canonical_pairs(nkpts):
            value = rng.normal(size=(naux, nao, nao)) + 1j * rng.normal(
                size=(naux, nao, nao)
            )
            if pair[0] == pair[1]:
                value = value + value.conj().transpose(0, 2, 1)
            cderi[pair] = value
            derivative = rng.normal(size=(1, 3, naux, nao, nao)) + 1j * rng.normal(
                size=(1, 3, naux, nao, nao)
            )
            if pair[0] == pair[1]:
                derivative = derivative + derivative.conj().transpose(0, 1, 2, 4, 3)
            dcderi[pair] = derivative
        density = np.asarray([_hermitian(rng, nao) for _ in range(nkpts)])
        analytic = gdf_jk_derivative(cderi, dcderi, density)
        step = 1.0e-6
        for axis in range(3):
            plus = {
                pair: block + step * dcderi[pair][0, axis]
                for pair, block in cderi.items()
            }
            minus = {
                pair: block - step * dcderi[pair][0, axis]
                for pair, block in cderi.items()
            }
            plus_j, plus_k = _jk_value(plus, density)
            minus_j, minus_k = _jk_value(minus, density)
            fd_j = (plus_j - minus_j) / (2.0 * step)
            fd_k = (plus_k - minus_k) / (2.0 * step)
            self.assertLess(
                np.max(np.abs(analytic.coulomb[0, :, axis] - fd_j)),
                2.0e-8,
            )
            self.assertLess(
                np.max(np.abs(analytic.exchange[0, :, axis] - fd_k)),
                2.0e-8,
            )


class GDFStorageTest(unittest.TestCase):
    def test_gamma_and_reverse_kpoint_dense_blocks(self):
        cell = gto.Cell(
            atom="H 0 0 0; H 0 0 1.4",
            a=np.eye(3) * 6.0,
            unit="Bohr",
            basis="gth-szv",
            pseudo="gth-hf",
            ke_cutoff=15.0,
            verbose=0,
        ).build()
        kpts = cell.make_kpts((2, 1, 1))
        provider = df.GDF(cell, kpts=kpts, auxbasis="gth-szv")
        provider.build(j_only=False)
        storage = load_cderi_storage(provider)
        self.assertEqual(tuple(storage.blocks), canonical_pairs(2))
        self.assertEqual(storage.block(1, 0).shape, (storage.naux, 2, 2))
        np.testing.assert_allclose(
            storage.block(0, 1),
            storage.block(1, 0).conj().transpose(0, 2, 1),
        )


class GDFPhysicalCDERIResponseTest(unittest.TestCase):
    """Real-integral gates for Gamma and non-Gamma auxiliary response."""

    @classmethod
    def setUpClass(cls):
        cls.cell = gto.Cell(
            atom="H 0 0 0; H 0 0 1.4",
            a=np.eye(3) * 6.0,
            unit="Bohr",
            basis="gth-szv",
            pseudo="gth-hf",
            ke_cutoff=15.0,
            verbose=0,
        ).build()
        cls.kpts = cls.cell.make_kpts((2, 1, 1))
        cls.provider = cls._build_provider(cls.cell)
        cls.parameters = rsgdf_parameters(
            cls.cell, cls.provider.auxcell, cls.kpts
        )
        cls.response = build_gdf_response_tensors(cls.provider)
        cls.atom_id = 1
        cls.axis = 2
        cls.step = 1.0e-4
        cls.plus_provider = cls._displaced_provider(+cls.step)
        cls.minus_provider = cls._displaced_provider(-cls.step)

    @classmethod
    def _build_provider(cls, cell):
        provider = df.GDF(cell, kpts=cls.kpts, auxbasis="gth-szv")
        provider.build(j_only=False)
        return provider

    @classmethod
    def _displaced_provider(cls, displacement):
        cell = cls.cell.copy()
        coordinates = cell.atom_coords().copy()
        coordinates[cls.atom_id, cls.axis] += displacement
        cell.set_geom_(list(zip(cell.elements, coordinates)), unit="Bohr")
        return cls._build_provider(cell)

    def test_pair_values_match_pyscf_storage(self):
        self.assertLess(self.response.value_parity_max_abs, 3.0e-9)

    def test_gamma_offdiagonal_and_same_k_derivatives_match_storage_fd(self):
        plus = load_cderi_storage(self.plus_provider)
        minus = load_cderi_storage(self.minus_provider)
        for pair in canonical_pairs(len(self.kpts)):
            finite_difference = (
                plus.block(*pair) - minus.block(*pair)
            ) / (2.0 * self.step)
            analytic = self.response.dcderi[pair][self.atom_id, self.axis]
            self.assertLess(
                np.max(np.abs(analytic - finite_difference)),
                3.0e-8,
            )

    def test_physical_jk_derivatives_match_provider_fd(self):
        density = np.asarray(
            [
                [[1.1, 0.2], [0.2, 0.9]],
                [[0.8, 0.1j], [-0.1j, 1.0]],
            ],
            dtype=np.complex128,
        )
        analytic = gdf_jk_derivative(
            self.response.cderi,
            self.response.dcderi,
            density,
        )
        plus_j, plus_k = self.plus_provider.get_jk(
            density, kpts=self.kpts, exxdiv=None
        )
        minus_j, minus_k = self.minus_provider.get_jk(
            density, kpts=self.kpts, exxdiv=None
        )
        for matrix, plus, minus in (
            (analytic.coulomb, plus_j, minus_j),
            (analytic.exchange, plus_k, minus_k),
        ):
            finite_difference = (
                np.asarray(plus) - np.asarray(minus)
            ) / (2.0 * self.step)
            self.assertLess(
                np.max(
                    np.abs(
                        matrix[self.atom_id, :, self.axis]
                        - finite_difference
                    )
                ),
                3.0e-8,
            )

        one_electron = build_one_electron_derivatives(self.cell, self.kpts)
        overlap_derivative = np.empty(
            (
                self.cell.natm,
                len(self.kpts),
                3,
                self.cell.nao_nr(),
                self.cell.nao_nr(),
            ),
            dtype=np.complex128,
        )
        for atom_id in range(self.cell.natm):
            for axis in range(3):
                overlap_derivative[atom_id, :, axis] = one_electron.overlap(
                    atom_id, axis
                )
        exchange = analytic.exchange + ewald_exchange_derivative(
            self.cell,
            self.kpts,
            density,
            overlap_derivative,
        )
        plus_exchange = self.plus_provider.get_jk(
            density, kpts=self.kpts, exxdiv="ewald"
        )[1]
        minus_exchange = self.minus_provider.get_jk(
            density, kpts=self.kpts, exxdiv="ewald"
        )[1]
        finite_difference = (
            np.asarray(plus_exchange) - np.asarray(minus_exchange)
        ) / (2.0 * self.step)
        self.assertLess(
            np.max(
                np.abs(
                    exchange[self.atom_id, :, self.axis]
                    - finite_difference
                )
            ),
            3.0e-8,
        )


class GDFWaterGammaResponseTest(unittest.TestCase):
    """Guard the unpacked Gamma AO-pair and auxiliary-center response."""

    atom_id = 1
    axis = 2
    step = 1.0e-4

    @classmethod
    def setUpClass(cls):
        cls.cell = _water_cell()
        cls.kpts = np.zeros((1, 3))
        cls.provider = cls._provider(cls.cell)
        cls.response = build_gdf_response_tensors(cls.provider)
        cls.plus_provider = cls._displaced_provider(+cls.step)
        cls.minus_provider = cls._displaced_provider(-cls.step)

    @classmethod
    def _provider(cls, cell):
        provider = df.GDF(cell, kpts=cls.kpts, auxbasis="gth-szv")
        provider.build(j_only=False)
        return provider

    @classmethod
    def _displaced_provider(cls, displacement):
        cell = _displaced_cell(
            cls.cell,
            cls.atom_id,
            cls.axis,
            displacement,
        )
        return cls._provider(cell)

    def test_gamma_cderi_derivative_matches_storage_fd(self):
        plus = load_cderi_storage(self.plus_provider).block(0, 0)
        minus = load_cderi_storage(self.minus_provider).block(0, 0)
        finite_difference = (plus - minus) / (2.0 * self.step)
        analytic = self.response.dcderi[(0, 0)][self.atom_id, self.axis]
        self.assertLess(np.max(np.abs(analytic - finite_difference)), 3.0e-8)

    def test_gamma_jk_derivative_matches_provider_fd(self):
        nao = self.cell.nao_nr()
        density = np.eye(nao, dtype=np.complex128)[None]
        density[0, 0, 1] = density[0, 1, 0] = 0.13
        analytic = gdf_jk_derivative(
            self.response.cderi,
            self.response.dcderi,
            density,
        )
        plus_j, plus_k = self.plus_provider.get_jk(
            density, kpts=self.kpts, exxdiv=None
        )
        minus_j, minus_k = self.minus_provider.get_jk(
            density, kpts=self.kpts, exxdiv=None
        )
        for derivative, plus, minus in (
            (analytic.coulomb, plus_j, minus_j),
            (analytic.exchange, plus_k, minus_k),
        ):
            finite_difference = (
                np.asarray(plus) - np.asarray(minus)
            ) / (2.0 * self.step)
            self.assertLess(
                np.max(
                    np.abs(
                        derivative[self.atom_id, :, self.axis]
                        - finite_difference
                    )
                ),
                3.0e-8,
            )


class GTHNonlocalDerivativeTest(unittest.TestCase):
    """Check O/H projector values and derivatives beyond the H-only case."""

    def test_water_gamma_and_kpoint_derivative_matches_fd(self):
        cell = _water_cell()
        atom_id, axis, step = 1, 2, 1.0e-4
        for kpts in (np.zeros((1, 3)), cell.make_kpts((2, 1, 1))):
            result = nonlocal_gth_derivative(cell, kpts)
            reference = np.asarray(pp_int.get_pp_nl(cell, kpts))
            self.assertLess(np.max(np.abs(result.matrix - reference)), 1.0e-12)

            plus = _displaced_cell(cell, atom_id, axis, +step)
            minus = _displaced_cell(cell, atom_id, axis, -step)
            finite_difference = (
                np.asarray(pp_int.get_pp_nl(plus, kpts))
                - np.asarray(pp_int.get_pp_nl(minus, kpts))
            ) / (2.0 * step)
            analytic = result.derivative[atom_id, :, axis]
            self.assertLess(
                np.max(np.abs(analytic - finite_difference)),
                3.0e-8,
            )


def _water_cell():
    return gto.Cell(
        atom=(
            "O 0.1 -0.2 0.3; "
            "H 1.55 0.15 0.05; "
            "H -0.35 1.25 -0.25"
        ),
        a=np.eye(3) * 7.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-hf",
        ke_cutoff=15.0,
        verbose=0,
    ).build()


def _displaced_cell(cell, atom_id, axis, displacement):
    displaced = cell.copy()
    coordinates = displaced.atom_coords().copy()
    coordinates[int(atom_id), int(axis)] += float(displacement)
    displaced.set_geom_(
        list(zip(displaced.elements, coordinates, strict=True)),
        unit="Bohr",
    )
    return displaced


if __name__ == "__main__":
    unittest.main()
