"""End-to-end CPU FFTDF analytic-gradient completion gates."""

from __future__ import annotations

import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft
from pyscf_pbc_dcdft.pseudo import fft_nonlocal_gth_derivative


def _cell(coords, *, cutoff_ha: float, basis: str = "gth-szv"):
    cell = gto.Cell()
    cell.atom = [("H", tuple(coord)) for coord in np.asarray(coords)]
    cell.a = np.eye(3) * 6.0
    cell.unit = "Bohr"
    cell.basis = basis
    cell.pseudo = "gth-hf"
    cell.ke_cutoff = cutoff_ha
    cell.verbose = 0
    return cell.build()


def _energy(coords, *, kmesh, cutoff_ha, basis="gth-szv"):
    cell = _cell(coords, cutoff_ha=cutoff_ha, basis=basis)
    if kmesh == (1, 1, 1):
        method = dft.DCRKS(cell, xc="PBE")
    else:
        method = dft.DCKRKS(
            cell,
            kpts=cell.make_kpts(kmesh),
            xc="PBE",
        )
    return method.kernel()


def _analytic(coords, *, kmesh, cutoff_ha, basis="gth-szv"):
    cell = _cell(coords, cutoff_ha=cutoff_ha, basis=basis)
    if kmesh == (1, 1, 1):
        method = dft.DCRKS(cell, xc="PBE")
    else:
        method = dft.DCKRKS(
            cell,
            kpts=cell.make_kpts(kmesh),
            xc="PBE",
        )
    method.kernel()
    gradient_object = method.nuc_grad_method()
    return gradient_object.kernel(), gradient_object


def _central_fd(
    coords,
    atom_id,
    axis,
    *,
    step,
    kmesh,
    cutoff_ha,
    basis="gth-szv",
):
    plus = np.array(coords, copy=True)
    minus = np.array(coords, copy=True)
    plus[atom_id, axis] += step
    minus[atom_id, axis] -= step
    return (
        _energy(plus, kmesh=kmesh, cutoff_ha=cutoff_ha, basis=basis)
        - _energy(minus, kmesh=kmesh, cutoff_ha=cutoff_ha, basis=basis)
    ) / (2.0 * step)


class FFTDFGradientGateTest(unittest.TestCase):
    """Compare the analytic route with full DC-PBE energy finite differences."""

    coordinates = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])

    def test_gamma_h2_selected_component(self):
        gradient, gradient_object = _analytic(
            self.coordinates,
            kmesh=(1, 1, 1),
            cutoff_ha=20.0,
        )
        finite_difference = _central_fd(
            self.coordinates,
            1,
            2,
            step=1.0e-3,
            kmesh=(1, 1, 1),
            cutoff_ha=20.0,
        )
        self.assertLess(abs(gradient[1, 2] - finite_difference), 2.0e-6)
        self.assertLess(
            gradient_object.result.z_vector.residual_max_abs,
            1.0e-10,
        )

    def test_sampled_kpoint_h2_selected_component(self):
        gradient, gradient_object = _analytic(
            self.coordinates,
            kmesh=(2, 1, 1),
            cutoff_ha=15.0,
        )
        finite_difference = _central_fd(
            self.coordinates,
            1,
            2,
            step=1.0e-3,
            kmesh=(2, 1, 1),
            cutoff_ha=15.0,
        )
        self.assertLess(abs(gradient[1, 2] - finite_difference), 2.0e-6)
        self.assertEqual(gradient_object.result.z_vector.dimension, 4)

    def test_gamma_h2_dzvp_basis_transfer(self):
        gradient, _ = _analytic(
            self.coordinates,
            kmesh=(1, 1, 1),
            cutoff_ha=20.0,
            basis="gth-dzvp",
        )
        finite_difference = _central_fd(
            self.coordinates,
            1,
            2,
            step=5.0e-4,
            kmesh=(1, 1, 1),
            cutoff_ha=20.0,
            basis="gth-dzvp",
        )
        self.assertLess(abs(gradient[1, 2] - finite_difference), 2.0e-6)


class FFTDFNonlocalGTHDerivativeTest(unittest.TestCase):
    """Guard the finite-mesh projector representation used by FFTDF energy."""

    def test_water_gamma_and_kpoint_matrix_derivative(self):
        cell = gto.Cell(
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
        atom_id, axis, step = 1, 2, 1.0e-4
        for kpts in (np.zeros((1, 3)), cell.make_kpts((2, 1, 1))):
            result = fft_nonlocal_gth_derivative(cell, kpts)
            plus = _displaced_cell(cell, atom_id, axis, +step)
            minus = _displaced_cell(cell, atom_id, axis, -step)
            finite_difference = (
                fft_nonlocal_gth_derivative(plus, kpts).matrix
                - fft_nonlocal_gth_derivative(minus, kpts).matrix
            ) / (2.0 * step)
            analytic = result.derivative[atom_id, :, axis]
            self.assertLess(
                np.max(np.abs(analytic - finite_difference)),
                3.0e-8,
            )


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
