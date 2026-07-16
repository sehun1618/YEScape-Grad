"""End-to-end CPU RSDF analytic-response completion gates."""

from __future__ import annotations

import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import df, dft
from pyscf_pbc_dcdft.df.gdf.storage import (
    canonical_pairs,
    load_cderi_storage,
)
from pyscf_pbc_dcdft.df.rsdf import build_rsdf_response_tensors
from pyscf_pbc_dcdft.df.rsdf.parameters import parameters_from_provider
from pyscf_pbc_dcdft.df.rsdf.metric import (
    rsdf_metric_derivative,
    rsdf_metric_derivatives,
)
from pyscf_pbc_dcdft.df.rsdf.source import (
    rsdf_source_derivative,
    rsdf_source_derivatives,
)


def _cell(distance=1.4):
    return gto.Cell(
        atom=f"H 0 0 0; H 0 0 {float(distance)}",
        a=np.eye(3) * 6.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-hf",
        precision=1.0e-9,
        mesh=[17, 17, 17],
        verbose=0,
    ).build()


def _provider(cell, kpts):
    provider = df.RSDF(cell, kpts=kpts, auxbasis="gth-szv")
    provider.build(j_only=False)
    return provider


def _method(distance, kmesh):
    cell = _cell(distance)
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    if tuple(kmesh) == (1, 1, 1):
        method = dft.DCRKS(cell, xc="PBE")
    else:
        method = dft.DCKRKS(cell, kpts=kpts, xc="PBE")
    method.with_df = df.RSDF(cell, kpts=kpts, auxbasis="gth-szv")
    method.conv_tol = 1.0e-11
    method.kernel()
    return method


class RSDFCDERIResponseTest(unittest.TestCase):
    """Guard public-RSDF source, metric, Cholesky, and k-pair response."""

    def test_kpoint_value_and_derivative_match_storage_fd(self):
        cell = _cell()
        kpts = cell.make_kpts((2, 1, 1), wrap_around=True)
        provider = _provider(cell, kpts)
        response = build_rsdf_response_tensors(provider)
        self.assertLess(response.value_parity_max_abs, 2.0e-10)

        atom_id, axis, step = 1, 2, 1.0e-4
        displaced = []
        for sign in (+1.0, -1.0):
            moved = cell.copy()
            coordinates = moved.atom_coords().copy()
            coordinates[atom_id, axis] += sign * step
            moved.set_geom_(
                list(zip(moved.elements, coordinates, strict=True)),
                unit="Bohr",
            )
            displaced.append(load_cderi_storage(_provider(moved, kpts)))
        for pair in canonical_pairs(len(kpts)):
            finite_difference = (
                displaced[0].blocks[pair] - displaced[1].blocks[pair]
            ) / (2.0 * step)
            analytic = response.dcderi[pair][atom_id, axis]
            self.assertLess(
                np.max(np.abs(analytic - finite_difference)), 2.0e-8
            )

    def test_gamma_bulk_source_derivative_matches_atomwise_route(self):
        cell = _cell()
        kpts = cell.make_kpts((1, 1, 1), wrap_around=True)
        provider = _provider(cell, kpts)
        parameters = parameters_from_provider(provider)
        bulk = rsdf_source_derivatives(
            cell,
            provider.auxcell,
            parameters,
            kpti=kpts[0],
            kptj=kpts[0],
        ).total
        atomwise = np.asarray(
            [
                rsdf_source_derivative(
                    cell,
                    provider.auxcell,
                    atom_id,
                    parameters,
                    kpti=kpts[0],
                    kptj=kpts[0],
                ).total
                for atom_id in range(cell.natm)
            ]
        )
        self.assertLess(np.max(np.abs(bulk - atomwise)), 1.0e-11)

    def test_gamma_bulk_metric_derivative_matches_atomwise_route(self):
        cell = _cell()
        kpts = cell.make_kpts((1, 1, 1), wrap_around=True)
        provider = _provider(cell, kpts)
        parameters = parameters_from_provider(provider)
        bulk = rsdf_metric_derivatives(
            cell,
            provider.auxcell,
            parameters,
            q=np.zeros(3),
        ).total
        atomwise = np.asarray(
            [
                rsdf_metric_derivative(
                    cell,
                    provider.auxcell,
                    atom_id,
                    parameters,
                    q=np.zeros(3),
                ).total
                for atom_id in range(cell.natm)
            ]
        )
        self.assertLess(np.max(np.abs(bulk - atomwise)), 1.0e-11)


class RSDFFullGradientTest(unittest.TestCase):
    """Guard the full nonvariational DC-PBE gradient at Gamma and k-points."""

    def test_gamma_h2_z_matches_energy_fd(self):
        self._assert_selected_fd((1, 1, 1))

    def test_kpoint_h2_z_matches_energy_fd(self):
        self._assert_selected_fd((2, 1, 1))

    def _assert_selected_fd(self, kmesh):
        central = _method(1.4, kmesh)
        analytic = central.nuc_grad_method().kernel()[1, 2]
        step = 1.0e-4
        plus = _method(1.4 + step, kmesh).e_tot
        minus = _method(1.4 - step, kmesh).e_tot
        finite_difference = (plus - minus) / (2.0 * step)
        self.assertLess(abs(analytic - finite_difference), 1.0e-7)


if __name__ == "__main__":
    unittest.main()
