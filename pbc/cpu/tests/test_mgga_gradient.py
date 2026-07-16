"""Focused CPU MGGA target-XC gradient regressions.

These tests isolate the kinetic-energy-density response added for SCAN and
r2SCAN.  Full Stage-6 validation covers all Cartesian components separately.
"""

from __future__ import annotations

import unittest

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


def _cell(coordinates, *, basis="gth-szv", cutoff=20.0):
    return gto.Cell(
        atom=list(zip(("H", "H"), coordinates, strict=True)),
        a=np.eye(3) * 6.0,
        unit="Bohr",
        basis=basis,
        pseudo="gth-hf",
        ke_cutoff=cutoff,
        verbose=0,
    ).build()


def _method(coordinates, *, xc, kmesh, backend):
    cell = _cell(coordinates)
    if kmesh == (1, 1, 1):
        mf = dft.DCRKS(cell, xc=xc)
    else:
        mf = dft.DCKRKS(cell, kpts=cell.make_kpts(kmesh), xc=xc)
    if backend == "gdf":
        mf.density_fit("gth-szv")
    elif backend == "rsdf":
        mf.rs_density_fit("gth-szv")
    mf.conv_tol = 1.0e-11
    mf.conv_tol_grad = 1.0e-9
    return mf


def _energy(coordinates, *, xc, kmesh, backend):
    return _method(
        coordinates, xc=xc, kmesh=kmesh, backend=backend
    ).kernel()


class MGGACPUGradientTest(unittest.TestCase):
    def assert_z_fd(self, *, xc, kmesh, backend):
        coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
        mf = _method(coordinates, xc=xc, kmesh=kmesh, backend=backend)
        mf.kernel()
        analytic = mf.nuc_grad_method().kernel()[0, 2]
        step = 5.0e-4
        plus = coordinates.copy()
        minus = coordinates.copy()
        plus[0, 2] += step
        minus[0, 2] -= step
        finite_difference = (
            _energy(plus, xc=xc, kmesh=kmesh, backend=backend)
            - _energy(minus, xc=xc, kmesh=kmesh, backend=backend)
        ) / (2.0 * step)
        self.assertLess(abs(analytic - finite_difference), 1.0e-5)

    def test_scan_gamma_fftdf_z_fd(self):
        self.assert_z_fd(
            xc="SCAN", kmesh=(1, 1, 1), backend="fftdf"
        )

    def test_r2scan_kpoint_gdf_z_fd(self):
        self.assert_z_fd(
            xc="R2SCAN", kmesh=(2, 1, 1), backend="gdf"
        )

    def test_scan_gamma_rsdf_z_fd(self):
        self.assert_z_fd(
            xc="SCAN", kmesh=(1, 1, 1), backend="rsdf"
        )

    def test_r2scan_kpoint_rsdf_z_fd(self):
        self.assert_z_fd(
            xc="R2SCAN", kmesh=(2, 1, 1), backend="rsdf"
        )


if __name__ == "__main__":
    unittest.main()
