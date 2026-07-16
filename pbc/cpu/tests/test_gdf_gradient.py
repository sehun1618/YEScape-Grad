"""End-to-end CPU GDF analytic-gradient completion gates.

The tests differentiate the complete ``E_PBE[n_HF]`` energy, including a new
HF calculation at each displaced geometry.  They therefore guard the GDF
metric/source response, GTH pseudopotential derivative, target PBE derivative,
and nonvariational Z-vector contraction as one public workflow.
"""

from __future__ import annotations

import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


H2_COORDINATES = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.4)))
H2O_COORDINATES = np.asarray(
    ((0.1, -0.2, 0.3), (1.55, 0.15, 0.05), (-0.35, 1.25, -0.25))
)


def _method(atoms, coordinates, *, basis, cutoff_ha, kmesh):
    cell = gto.Cell(
        atom=list(zip(atoms, coordinates, strict=True)),
        a=np.eye(3) * (6.0 if atoms == ("H", "H") else 7.0),
        unit="Bohr",
        basis=basis,
        pseudo="gth-hf",
        ke_cutoff=cutoff_ha,
        verbose=0,
    ).build()
    if kmesh == (1, 1, 1):
        method = dft.DCRKS(cell, xc="PBE")
    else:
        method = dft.DCKRKS(
            cell,
            kpts=cell.make_kpts(kmesh),
            xc="PBE",
        )
    method.density_fit(auxbasis="gth-szv")
    method.conv_tol = 1.0e-11
    method.max_cycle = 100
    return method


def _analytic(atoms, coordinates, **settings):
    method = _method(atoms, coordinates, **settings)
    method.kernel()
    gradient_object = method.nuc_grad_method()
    return gradient_object.kernel(), gradient_object


def _energy(atoms, coordinates, **settings):
    return _method(atoms, coordinates, **settings).kernel()


def _central_fd(atoms, coordinates, atom_id, axis, *, step, **settings):
    plus = np.array(coordinates, copy=True)
    minus = np.array(coordinates, copy=True)
    plus[atom_id, axis] += step
    minus[atom_id, axis] -= step
    return (
        _energy(atoms, plus, **settings)
        - _energy(atoms, minus, **settings)
    ) / (2.0 * step)


class GDFGradientGateTest(unittest.TestCase):
    """Exercise Gamma/non-Gamma and primary-basis transfer routes."""

    def _assert_component(
        self,
        atoms,
        coordinates,
        *,
        atom_id,
        axis,
        step,
        **settings,
    ):
        gradient, gradient_object = _analytic(
            atoms, coordinates, **settings
        )
        finite_difference = _central_fd(
            atoms,
            coordinates,
            atom_id,
            axis,
            step=step,
            **settings,
        )
        self.assertLess(
            abs(gradient[atom_id, axis] - finite_difference),
            2.0e-6,
        )
        self.assertLess(
            gradient_object.result.z_vector.residual_max_abs,
            1.0e-10,
        )
        self.assertLess(
            gradient_object.result.cderi_value_parity_max_abs,
            1.0e-7,
        )

    def test_gamma_h2_szv(self):
        self._assert_component(
            ("H", "H"),
            H2_COORDINATES,
            atom_id=1,
            axis=2,
            step=5.0e-4,
            basis="gth-szv",
            cutoff_ha=20.0,
            kmesh=(1, 1, 1),
        )

    def test_kpoint_h2_dzvp(self):
        self._assert_component(
            ("H", "H"),
            H2_COORDINATES,
            atom_id=1,
            axis=2,
            step=5.0e-4,
            basis="gth-dzvp",
            cutoff_ha=15.0,
            kmesh=(2, 1, 1),
        )

    def test_gamma_distorted_h2o(self):
        self._assert_component(
            ("O", "H", "H"),
            H2O_COORDINATES,
            atom_id=1,
            axis=2,
            step=2.0e-4,
            basis="gth-szv",
            cutoff_ha=25.0,
            kmesh=(1, 1, 1),
        )

    def test_kpoint_distorted_h2o(self):
        self._assert_component(
            ("O", "H", "H"),
            H2O_COORDINATES,
            atom_id=1,
            axis=2,
            step=2.0e-4,
            basis="gth-szv",
            cutoff_ha=15.0,
            kmesh=(2, 1, 1),
        )


if __name__ == "__main__":
    unittest.main()
