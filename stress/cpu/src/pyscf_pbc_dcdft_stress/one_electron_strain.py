"""Analytic Gamma periodic overlap and kinetic homogeneous strain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gdf_metric_strain import matrix_tensor_to_voigt


@dataclass(frozen=True)
class GammaOneElectronStrain:
    """One periodic two-center matrix and its Cartesian strain derivative."""

    integral: str
    value: np.ndarray
    raw_derivative: np.ndarray
    center_response: np.ndarray
    image_response: np.ndarray

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


def gamma_overlap_strain(cell) -> GammaOneElectronStrain:
    """Return analytic Gamma overlap matrix strain response."""

    return gamma_one_electron_strain(cell, "int1e_ovlp")


def gamma_kinetic_strain(cell) -> GammaOneElectronStrain:
    """Return analytic Gamma kinetic matrix strain response."""

    return gamma_one_electron_strain(cell, "int1e_kin")


def gamma_one_electron_strain(cell, integral) -> GammaOneElectronStrain:
    r"""Differentiate one translationally invariant Gamma two-center matrix.

    For an image-resolved integral :math:`I_{ijL}`, translation invariance
    gives

    .. math::

       I^{(1)}_{xy}=\sum_L (\nabla_i I_{ijL})_x
                         (R_j-R_i+L)_y.

    PySCF's ``kderiv=1`` periodic integral is
    :math:`i\sum_L L_y I_{ijL}`.  Applying it to ``int1e_ip*`` therefore
    recovers the image moment without storing every lattice-image matrix.
    """

    names = {
        "int1e_ovlp": "int1e_ipovlp",
        "int1e_kin": "int1e_ipkin",
    }
    integral = str(integral)
    if integral not in names:
        raise ValueError(f"unsupported one-electron strain integral {integral!r}")
    if int(getattr(cell, "dimension", 3)) != 3:
        raise NotImplementedError("Gamma one-electron strain currently supports 3D cells")

    nao = int(cell.nao_nr())
    value = np.asarray(cell.pbc_intor(integral, hermi=0))
    center_gradient = np.asarray(
        cell.pbc_intor(names[integral], comp=3, hermi=0)
    )
    image_moment = np.asarray(
        cell.pbc_intor(
            names[integral], comp=3, hermi=0, kderiv=1
        )
    ).reshape(3, 3, nao, nao) / 1j

    centers = _ao_centers(cell)
    separations = centers[None, :, :] - centers[:, None, :]
    center_response = np.einsum(
        "xij,ijy->xyij", center_gradient, separations, optimize=True
    )
    image_response = image_moment.transpose(1, 0, 2, 3)
    derivative = center_response + image_response
    return GammaOneElectronStrain(
        integral=integral,
        value=_real_if_close(value),
        raw_derivative=_real_if_close(derivative),
        center_response=_real_if_close(center_response),
        image_response=_real_if_close(image_response),
    )


def kpoint_one_electron_strain(cell, kpts, integral) -> GammaOneElectronStrain:
    """Return sampled-k two-center strain at fixed fractional k-points."""

    names = {
        "int1e_ovlp": "int1e_ipovlp",
        "int1e_kin": "int1e_ipkin",
    }
    integral = str(integral)
    if integral not in names:
        raise ValueError(f"unsupported one-electron strain integral {integral!r}")
    points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    nao = int(cell.nao_nr())
    value = np.asarray(cell.pbc_intor(integral, hermi=0, kpts=points))
    center_gradient = np.asarray(
        cell.pbc_intor(names[integral], comp=3, hermi=0, kpts=points)
    )
    image_moment = np.asarray(
        cell.pbc_intor(
            names[integral], comp=3, hermi=0, kpts=points, kderiv=1
        )
    ).reshape(len(points), 3, 3, nao, nao) / 1j
    centers = _ao_centers(cell)
    separations = centers[None, :, :] - centers[:, None, :]
    center_response = np.einsum(
        "kxij,ijy->xykij", center_gradient, separations, optimize=True
    )
    image_response = image_moment.transpose(2, 1, 0, 3, 4)
    derivative = center_response + image_response
    return GammaOneElectronStrain(
        integral=integral,
        value=_real_if_close(value),
        raw_derivative=_real_if_close(derivative),
        center_response=_real_if_close(center_response),
        image_response=_real_if_close(image_response),
    )


def _ao_centers(cell):
    centers = np.empty((cell.nao_nr(), 3), dtype=float)
    for atom_id, (_, _, start, stop) in enumerate(cell.aoslice_by_atom()):
        centers[start:stop] = cell.atom_coord(atom_id)
    return centers


def _real_if_close(value, tolerance=1e-10):
    array = np.asarray(value)
    if array.size and np.max(np.abs(array.imag)) <= tolerance:
        return array.real
    return array
