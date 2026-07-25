"""Analytic Gamma GDF GTH pseudopotential strain components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import gto, lib
from pyscf.pbc.df import aft, rsdf_builder
from pyscf.pbc.tools import pbc as pbc_tools

from .gdf_metric_strain import auxiliary_fourier_strain, matrix_tensor_to_voigt
from .gdf_source_strain import (
    GammaThreeCenterSourceStrain,
    gamma_ao_pair_fourier_strain,
    gamma_three_center_source_strain,
    kpoint_ao_pair_fourier_strain,
    kpoint_three_center_source_strain,
)


@dataclass(frozen=True)
class GammaGDFLocalPart1Strain:
    """GDF local-part1 value and resolved homogeneous-strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    three_center_response: np.ndarray
    pseudo_g0_response: np.ndarray
    source: GammaThreeCenterSourceStrain
    omega: float
    mesh: tuple[int, int, int]

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class GammaGDFLocalPart2Strain:
    """Converged reciprocal GTH local-part2 matrix and strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    mesh: tuple[int, int, int]
    grid_points: int
    block_size: int

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class GammaGDFNonlocalStrain:
    """Converged reciprocal GTH nonlocal-projector matrix strain."""

    value: np.ndarray
    raw_derivative: np.ndarray
    mesh: tuple[int, int, int]
    grid_points: int
    block_size: int

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class GammaGDFPseudopotentialStrain:
    """Complete analytic Gamma GDF GTH pseudopotential matrix strain."""

    value: np.ndarray
    raw_derivative: np.ndarray
    local_part1: GammaGDFLocalPart1Strain
    local_part2: GammaGDFLocalPart2Strain
    nonlocal_part: GammaGDFNonlocalStrain

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class KPointGDFLocalPart1Strain:
    """Sampled-k local-part1 matrices and analytic strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    three_center_response: np.ndarray
    pseudo_g0_response: np.ndarray
    sources: tuple
    omega: float
    mesh: tuple[int, int, int]

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class KPointGDFLocalPart2Strain:
    """Sampled-k local-part2 matrices and analytic strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    mesh: tuple[int, int, int]
    grid_points: int
    block_size: int

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class KPointGDFNonlocalStrain:
    """Sampled-k nonlocal projector matrices and strain response."""

    value: np.ndarray
    raw_derivative: np.ndarray
    mesh: tuple[int, int, int]
    grid_points: int
    block_size: int

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


@dataclass(frozen=True)
class KPointGDFPseudopotentialStrain:
    """Complete sampled-k GTH pseudopotential matrix strain."""

    value: np.ndarray
    raw_derivative: np.ndarray
    local_part1: KPointGDFLocalPart1Strain
    local_part2: KPointGDFLocalPart2Strain
    nonlocal_part: KPointGDFNonlocalStrain

    @property
    def voigt_derivative(self):
        return matrix_tensor_to_voigt(self.raw_derivative)


def gamma_gdf_local_part1_strain(cell, *, reciprocal_block_size=None):
    """Differentiate `_RSNucBuilder.get_pp_loc_part1()` at Gamma.

    The range-separated local potential is the complete three-center source
    with a fake nuclear Gaussian auxiliary basis, contracted by nuclear
    charges.  GTH adds one representation-specific G=0 Gaussian term; its
    overlap and volume response are retained separately below.
    """

    builder = rsdf_builder._RSNucBuilder(
        cell, np.zeros((1, 3))
    ).build()
    fake_nuclear = aft._fake_nuc(cell, with_pseudo=True)
    charges = -np.asarray(cell.atom_charges(), dtype=float)
    mesh = tuple(int(value) for value in builder.mesh)
    overlap = kpoint_ao_pair_fourier_strain(
        cell, np.zeros((1, 3)), np.zeros(3), np.zeros(3)
    )
    source = gamma_three_center_source_strain(
        cell, fake_nuclear, float(builder.omega), mesh,
        reciprocal_block_size=reciprocal_block_size,
        retain_components=False,
        zero_transfer_overlap=overlap,
    )
    nao = int(cell.nao_nr())
    nfake = int(fake_nuclear.nao_nr())
    source_value = np.einsum(
        "ijz,z->ij", source.value.reshape(nao, nao, nfake), charges,
        optimize=True,
    )
    source_response = np.einsum(
        "xyijz,z->xyij",
        source.raw_derivative.reshape(3, 3, nao, nao, nfake),
        charges,
        optimize=True,
    )

    exponents = np.hstack(fake_nuclear.bas_exps())
    coefficient = float(charges.dot(np.pi / exponents) / cell.vol)
    pseudo_g0_value = -coefficient * overlap.value[0]
    pseudo_g0_response = -coefficient * overlap.raw_derivative[:, :, 0]
    for axis in range(3):
        pseudo_g0_response[axis, axis] -= pseudo_g0_value
    return GammaGDFLocalPart1Strain(
        value=_real_if_close(source_value + pseudo_g0_value),
        raw_derivative=_real_if_close(
            source_response + pseudo_g0_response
        ),
        three_center_response=_real_if_close(source_response),
        pseudo_g0_response=_real_if_close(pseudo_g0_response),
        source=source,
        omega=float(builder.omega),
        mesh=mesh,
    )


def gamma_gdf_local_part2_strain(cell, *, mesh=None, block_size=8192):
    """Differentiate GTH local-part2 on a converged reciprocal mesh.

    `_IntPPBuilder` evaluates the same Gaussian-polynomial potential with
    three-center integrals.  Its reciprocal representation converges
    exponentially.  The default mesh follows PySCF's AO cutoff estimator and
    is validated against the integral builder before this result is accepted.
    AO values and their analytic strain response are evaluated in bounded
    real-space grid blocks.
    """

    if mesh is None:
        cutoff = aft.estimate_ke_cutoff(cell, cell.precision)
        mesh_array = cell.symmetrize_mesh(cell.cutoff_to_mesh(cutoff))
    else:
        mesh_array = np.asarray(mesh, dtype=int)
    if mesh_array.shape != (3,) or np.any(mesh_array <= 0):
        raise ValueError(f"mesh must contain three positive integers: {mesh_array}")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    grid_vectors = cell.get_Gv(mesh_array)
    potential_g, potential_derivative_g = (
        _local_part2_potential_fourier_strain(
            cell, grid_vectors, block_size=block_size
        )
    )
    potential_r = pbc_tools.ifft(potential_g, mesh_array).real
    potential_derivative_r = np.empty(
        (3, 3, len(grid_vectors)), dtype=np.float64
    )
    for x in range(3):
        for y in range(3):
            potential_derivative_r[x, y] = pbc_tools.ifft(
                potential_derivative_g[x, y], mesh_array
            ).real
    del potential_g, potential_derivative_g

    coordinates = cell.get_uniform_grids(mesh_array)
    nao = int(cell.nao_nr())
    value = np.zeros((nao, nao), dtype=np.complex128)
    derivative = np.zeros((3, 3, nao, nao), dtype=np.complex128)
    evaluator = "GTOval_cart_deriv1" if cell.cart else "GTOval_sph_deriv1"
    from .upstream import PySCFStressPrimitives

    primitives = PySCFStressPrimitives.load()
    for p0, p1 in lib.prange(0, len(coordinates), block_size):
        block_coordinates = coordinates[p0:p1]
        ao_all = np.asarray(
            cell.pbc_eval_gto(evaluator, block_coordinates)
        )
        ao_value = ao_all[0]
        ao_gradient = ao_all[1:4]
        ao_response = np.asarray(
            primitives.ao_strain_derivatives(
                cell, block_coordinates, deriv=0
            )[0]
        )[:, :, 0]
        ao_response += np.einsum(
            "xgi,gy->xygi", ao_gradient, block_coordinates, optimize=True
        )
        local_potential = potential_r[p0:p1]
        value += np.einsum(
            "gi,g,gj->ij", ao_value.conj(), local_potential, ao_value,
            optimize=True,
        )
        derivative += np.einsum(
            "xygi,g,gj->xyij", ao_response.conj(), local_potential,
            ao_value, optimize=True,
        )
        derivative += np.einsum(
            "gi,g,xygj->xyij", ao_value.conj(), local_potential,
            ao_response, optimize=True,
        )
        derivative += np.einsum(
            "gi,xyg,gj->xyij", ao_value.conj(),
            potential_derivative_r[:, :, p0:p1], ao_value, optimize=True,
        )
    return GammaGDFLocalPart2Strain(
        value=_real_if_close(value),
        raw_derivative=_real_if_close(derivative),
        mesh=tuple(int(component) for component in mesh_array),
        grid_points=int(np.prod(mesh_array)),
        block_size=block_size,
    )


def gamma_gdf_nonlocal_strain(cell, *, mesh=None, block_size=8192):
    """Differentiate the GTH nonlocal projector matrix at Gamma.

    The reciprocal projector representation converges exponentially to
    `pp_int.get_pp_nl`.  Projector gradients, AO Fourier strain, the
    ``1/sqrt(volume)`` AO normalization, and the final ``1/volume`` response
    are all included analytically.
    """

    if mesh is None:
        cutoff = aft.estimate_ke_cutoff(cell, cell.precision)
        mesh_array = cell.symmetrize_mesh(cell.cutoff_to_mesh(cutoff))
    else:
        mesh_array = np.asarray(mesh, dtype=int)
    if mesh_array.shape != (3,) or np.any(mesh_array <= 0):
        raise ValueError(f"mesh must contain three positive integers: {mesh_array}")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    nao = int(cell.nao_nr())
    if not _has_nonlocal_projectors(cell):
        zeros = np.zeros((nao, nao))
        return GammaGDFNonlocalStrain(
            value=zeros,
            raw_derivative=np.zeros((3, 3, nao, nao)),
            mesh=tuple(int(component) for component in mesh_array),
            grid_points=0,
            block_size=block_size,
        )

    grid_vectors = cell.get_Gv(mesh_array)
    fake = _projector_angular_molecule()
    channels = []
    for atom_id in range(cell.natm):
        symbol = cell.atom_symbol(atom_id)
        if symbol not in cell._pseudo:
            continue
        for angular_momentum, projector_data in enumerate(
            cell._pseudo[symbol][5:]
        ):
            radius, count, coupling = projector_data
            if count > 0:
                channels.append(
                    {
                        "atom_id": atom_id,
                        "angular_momentum": angular_momentum,
                        "radius": float(radius),
                        "count": int(count),
                        "coupling": np.asarray(coupling, dtype=float),
                        "overlap": np.zeros(
                            (count, 2 * angular_momentum + 1, nao),
                            dtype=np.complex128,
                        ),
                        "response": np.zeros(
                            (3, 3, count, 2 * angular_momentum + 1, nao),
                            dtype=np.complex128,
                        ),
                    }
                )
    inverse_sqrt_volume = float(cell.vol) ** -0.5

    for p0, p1 in lib.prange(0, len(grid_vectors), block_size):
        vectors = grid_vectors[p0:p1]
        structure_factors = cell.get_SI(Gv=vectors)
        ao_fourier = auxiliary_fourier_strain(cell, vectors)
        ao_value = ao_fourier.value.T * inverse_sqrt_volume
        ao_response = (
            ao_fourier.raw_derivative.transpose(0, 1, 3, 2)
            * inverse_sqrt_volume
        )
        for axis in range(3):
            ao_response[axis, axis] -= 0.5 * ao_value

        for channel in channels:
            phase = structure_factors[channel["atom_id"]].conj()
            projector, projector_gradient = _projector_value_gradient(
                fake, vectors, channel["angular_momentum"],
                channel["count"], channel["radius"]
            )
            projector_response = -np.einsum(
                "gx,ynmg->xynmg", vectors, projector_gradient,
                optimize=True,
            )
            phased_projector = projector * phase[None, None, :]
            phased_response = (
                projector_response * phase[None, None, None, None, :]
            )
            channel["overlap"] += np.einsum(
                "nmg,gp->nmp", phased_projector, ao_value,
                optimize=True,
            )
            channel["response"] += np.einsum(
                "xynmg,gp->xynmp", phased_response, ao_value,
                optimize=True,
            )
            channel["response"] += np.einsum(
                "nmg,xygp->xynmp", phased_projector, ao_response,
                optimize=True,
            )

    value_accumulator = np.zeros((nao, nao), dtype=np.complex128)
    derivative_accumulator = np.zeros(
        (3, 3, nao, nao), dtype=np.complex128
    )
    for channel in channels:
        overlap = channel["overlap"]
        response = channel["response"]
        coupling = channel["coupling"]
        value_accumulator += np.einsum(
            "amp,ab,bmq->pq", overlap.conj(), coupling, overlap,
            optimize=True,
        )
        derivative_accumulator += np.einsum(
            "xyamp,ab,bmq->xypq", response.conj(), coupling, overlap,
            optimize=True,
        )
        derivative_accumulator += np.einsum(
            "amp,ab,xybmq->xypq", overlap.conj(), coupling, response,
            optimize=True,
        )

    value = value_accumulator / float(cell.vol)
    derivative = derivative_accumulator / float(cell.vol)
    for axis in range(3):
        derivative[axis, axis] -= value
    return GammaGDFNonlocalStrain(
        value=_real_if_close(value),
        raw_derivative=_real_if_close(derivative),
        mesh=tuple(int(component) for component in mesh_array),
        grid_points=int(np.prod(mesh_array)),
        block_size=block_size,
    )


def gamma_gdf_pseudopotential_strain(
    cell, *, reciprocal_mesh=None, block_size=8192,
    source_block_size=None,
):
    """Assemble complete analytic GDF GTH pseudopotential strain matrices."""

    local_part1 = gamma_gdf_local_part1_strain(
        cell, reciprocal_block_size=source_block_size
    )
    local_part2 = gamma_gdf_local_part2_strain(
        cell, mesh=reciprocal_mesh, block_size=block_size
    )
    nonlocal_part = gamma_gdf_nonlocal_strain(
        cell, mesh=reciprocal_mesh, block_size=block_size
    )
    return GammaGDFPseudopotentialStrain(
        value=_real_if_close(
            local_part1.value + local_part2.value + nonlocal_part.value
        ),
        raw_derivative=_real_if_close(
            local_part1.raw_derivative
            + local_part2.raw_derivative
            + nonlocal_part.raw_derivative
        ),
        local_part1=local_part1,
        local_part2=local_part2,
        nonlocal_part=nonlocal_part,
    )


def kpoint_gdf_local_part1_strain(
    cell, kpts, *, reciprocal_block_size=None,
):
    """Differentiate range-separated local-part1 at sampled k points."""

    points = _kpoints(kpts)
    builder = rsdf_builder._RSNucBuilder(cell, points).build()
    fake_nuclear = aft._fake_nuc(cell, with_pseudo=True)
    charges = -np.asarray(cell.atom_charges(), dtype=float)
    mesh = tuple(int(value) for value in builder.mesh)
    nao = int(cell.nao_nr())
    nfake = int(fake_nuclear.nao_nr())
    values = np.empty((len(points), nao, nao), dtype=np.complex128)
    derivatives = np.empty(
        (3, 3, len(points), nao, nao), dtype=np.complex128
    )
    source_responses = np.empty_like(derivatives)
    g0_responses = np.empty_like(derivatives)
    sources = []
    exponents = np.hstack(fake_nuclear.bas_exps())
    coefficient = float(charges.dot(np.pi/exponents)/cell.vol)
    for kpoint_id, kpt in enumerate(points):
        overlap = kpoint_ao_pair_fourier_strain(
            cell, np.zeros((1, 3)), kpt, kpt
        )
        source = kpoint_three_center_source_strain(
            cell, fake_nuclear, float(builder.omega), mesh, kpt, kpt,
            reciprocal_block_size=reciprocal_block_size,
            retain_components=False,
            zero_transfer_overlap=overlap,
        )
        sources.append(source)
        source_value = np.einsum(
            "ijz,z->ij", source.value.reshape(nao, nao, nfake), charges,
            optimize=True,
        )
        source_response = np.einsum(
            "xyijz,z->xyij",
            source.raw_derivative.reshape(3, 3, nao, nao, nfake),
            charges, optimize=True,
        )
        pseudo_g0_value = -coefficient*overlap.value[0]
        pseudo_g0_response = -coefficient*overlap.raw_derivative[:, :, 0]
        for axis in range(3):
            pseudo_g0_response[axis, axis] -= pseudo_g0_value
        values[kpoint_id] = source_value+pseudo_g0_value
        source_responses[:, :, kpoint_id] = source_response
        g0_responses[:, :, kpoint_id] = pseudo_g0_response
        derivatives[:, :, kpoint_id] = source_response+pseudo_g0_response
    return KPointGDFLocalPart1Strain(
        value=_real_if_close(values),
        raw_derivative=_real_if_close(derivatives),
        three_center_response=_real_if_close(source_responses),
        pseudo_g0_response=_real_if_close(g0_responses),
        sources=tuple(sources),
        omega=float(builder.omega),
        mesh=mesh,
    )


def kpoint_gdf_local_part2_strain(
    cell, kpts, *, mesh=None, block_size=8192
):
    """Differentiate reciprocal local-part2 with sampled Bloch AOs."""

    points = _kpoints(kpts)
    mesh_array = _reciprocal_mesh(cell, mesh)
    block_size = _block_size(block_size)
    grid_vectors = cell.get_Gv(mesh_array)
    potential_g, potential_derivative_g = (
        _local_part2_potential_fourier_strain(
            cell, grid_vectors, block_size=block_size
        )
    )
    potential_r = pbc_tools.ifft(potential_g, mesh_array).real
    potential_derivative_r = np.empty(
        (3, 3, len(grid_vectors)), dtype=np.float64
    )
    for x in range(3):
        for y in range(3):
            potential_derivative_r[x, y] = pbc_tools.ifft(
                potential_derivative_g[x, y], mesh_array
            ).real
    del potential_g, potential_derivative_g

    coordinates = cell.get_uniform_grids(mesh_array)
    nao = int(cell.nao_nr())
    values = np.zeros((len(points), nao, nao), dtype=np.complex128)
    derivatives = np.zeros(
        (3, 3, len(points), nao, nao), dtype=np.complex128
    )
    evaluator = "GTOval_cart_deriv1" if cell.cart else "GTOval_sph_deriv1"
    from .upstream import PySCFStressPrimitives

    primitives = PySCFStressPrimitives.load()
    for p0, p1 in lib.prange(0, len(coordinates), block_size):
        block_coordinates = coordinates[p0:p1]
        ao_sets = _ao_kpoint_sets(
            cell.pbc_eval_gto(evaluator, block_coordinates, kpts=points),
            len(points),
        )
        strain_sets = primitives.module._eval_ao_strain_derivatives(
            cell, block_coordinates, kpts=points, deriv=0
        )
        for kpoint_id, ao_all in enumerate(ao_sets):
            ao_value = ao_all[0]
            ao_gradient = ao_all[1:4]
            ao_response = np.asarray(strain_sets[kpoint_id])[:, :, 0].copy()
            ao_response += np.einsum(
                "xgi,gy->xygi", ao_gradient, block_coordinates,
                optimize=True,
            )
            local_potential = potential_r[p0:p1]
            values[kpoint_id] += np.einsum(
                "gi,g,gj->ij", ao_value.conj(), local_potential, ao_value,
                optimize=True,
            )
            derivatives[:, :, kpoint_id] += np.einsum(
                "xygi,g,gj->xyij", ao_response.conj(), local_potential,
                ao_value, optimize=True,
            )
            derivatives[:, :, kpoint_id] += np.einsum(
                "gi,g,xygj->xyij", ao_value.conj(), local_potential,
                ao_response, optimize=True,
            )
            derivatives[:, :, kpoint_id] += np.einsum(
                "gi,xyg,gj->xyij", ao_value.conj(),
                potential_derivative_r[:, :, p0:p1], ao_value, optimize=True,
            )
    return KPointGDFLocalPart2Strain(
        value=_real_if_close(values),
        raw_derivative=_real_if_close(derivatives),
        mesh=tuple(int(component) for component in mesh_array),
        grid_points=int(np.prod(mesh_array)),
        block_size=block_size,
    )


def kpoint_gdf_nonlocal_strain(
    cell, kpts, *, mesh=None, block_size=8192
):
    """Differentiate nonlocal GTH projectors for sampled k points."""

    points = _kpoints(kpts)
    mesh_array = _reciprocal_mesh(cell, mesh)
    block_size = _block_size(block_size)
    nao = int(cell.nao_nr())
    if not _has_nonlocal_projectors(cell):
        return KPointGDFNonlocalStrain(
            value=np.zeros((len(points), nao, nao)),
            raw_derivative=np.zeros((3, 3, len(points), nao, nao)),
            mesh=tuple(int(component) for component in mesh_array),
            grid_points=0,
            block_size=block_size,
        )

    grid_vectors = cell.get_Gv(mesh_array)
    fake = _projector_angular_molecule()
    channels = []
    for atom_id in range(cell.natm):
        symbol = cell.atom_symbol(atom_id)
        if symbol not in cell._pseudo:
            continue
        for angular_momentum, projector_data in enumerate(
            cell._pseudo[symbol][5:]
        ):
            radius, count, coupling = projector_data
            if count > 0:
                channels.append({
                    "atom_id": atom_id,
                    "angular_momentum": angular_momentum,
                    "radius": float(radius),
                    "count": int(count),
                    "coupling": np.asarray(coupling, dtype=float),
                    "overlap": np.zeros(
                        (len(points), count, 2*angular_momentum+1, nao),
                        dtype=np.complex128,
                    ),
                    "response": np.zeros(
                        (3, 3, len(points), count,
                         2*angular_momentum+1, nao),
                        dtype=np.complex128,
                    ),
                })
    inverse_sqrt_volume = float(cell.vol)**-0.5
    for p0, p1 in lib.prange(0, len(grid_vectors), block_size):
        base_vectors = grid_vectors[p0:p1]
        structure_factors = cell.get_SI(Gv=base_vectors)
        for kpoint_id, kpt in enumerate(points):
            momentum = base_vectors+kpt
            ao_fourier = auxiliary_fourier_strain(cell, momentum)
            ao_value = ao_fourier.value.T*inverse_sqrt_volume
            ao_response = (
                ao_fourier.raw_derivative.transpose(0, 1, 3, 2)
                * inverse_sqrt_volume
            )
            for axis in range(3):
                ao_response[axis, axis] -= 0.5*ao_value
            for channel in channels:
                phase = structure_factors[channel["atom_id"]].conj()
                projector, projector_gradient = _projector_value_gradient(
                    fake, momentum, channel["angular_momentum"],
                    channel["count"], channel["radius"],
                )
                projector_response = -np.einsum(
                    "gx,ynmg->xynmg", momentum, projector_gradient,
                    optimize=True,
                )
                phased_projector = projector*phase[None, None, :]
                phased_response = (
                    projector_response*phase[None, None, None, None, :]
                )
                channel["overlap"][kpoint_id] += np.einsum(
                    "nmg,gp->nmp", phased_projector, ao_value,
                    optimize=True,
                )
                channel["response"][:, :, kpoint_id] += np.einsum(
                    "xynmg,gp->xynmp", phased_response, ao_value,
                    optimize=True,
                )
                channel["response"][:, :, kpoint_id] += np.einsum(
                    "nmg,xygp->xynmp", phased_projector, ao_response,
                    optimize=True,
                )

    values = np.zeros((len(points), nao, nao), dtype=np.complex128)
    derivatives = np.zeros(
        (3, 3, len(points), nao, nao), dtype=np.complex128
    )
    for channel in channels:
        overlap = channel["overlap"]
        response = channel["response"]
        coupling = channel["coupling"]
        values += np.einsum(
            "kamp,ab,kbmq->kpq", overlap.conj(), coupling, overlap,
            optimize=True,
        )
        derivatives += np.einsum(
            "xykamp,ab,kbmq->xykpq", response.conj(), coupling, overlap,
            optimize=True,
        )
        derivatives += np.einsum(
            "kamp,ab,xykbmq->xykpq", overlap.conj(), coupling, response,
            optimize=True,
        )
    values /= float(cell.vol)
    derivatives /= float(cell.vol)
    for axis in range(3):
        derivatives[axis, axis] -= values
    return KPointGDFNonlocalStrain(
        value=_real_if_close(values),
        raw_derivative=_real_if_close(derivatives),
        mesh=tuple(int(component) for component in mesh_array),
        grid_points=int(np.prod(mesh_array)),
        block_size=block_size,
    )


def kpoint_gdf_pseudopotential_strain(
    cell, kpts, *, reciprocal_mesh=None, block_size=8192,
    source_block_size=None,
):
    """Assemble the complete sampled-k analytic GTH PP strain matrices."""

    local_part1 = kpoint_gdf_local_part1_strain(
        cell, kpts, reciprocal_block_size=source_block_size
    )
    local_part2 = kpoint_gdf_local_part2_strain(
        cell, kpts, mesh=reciprocal_mesh, block_size=block_size
    )
    nonlocal_part = kpoint_gdf_nonlocal_strain(
        cell, kpts, mesh=reciprocal_mesh, block_size=block_size
    )
    return KPointGDFPseudopotentialStrain(
        value=_real_if_close(
            local_part1.value+local_part2.value+nonlocal_part.value
        ),
        raw_derivative=_real_if_close(
            local_part1.raw_derivative+local_part2.raw_derivative
            + nonlocal_part.raw_derivative
        ),
        local_part1=local_part1,
        local_part2=local_part2,
        nonlocal_part=nonlocal_part,
    )


def _local_part2_atomic_fourier_strain(cell, grid_vectors):
    norm2 = np.einsum("gx,gx->g", grid_vectors, grid_vectors)
    dyadic = np.einsum("gx,gy->xyg", grid_vectors, grid_vectors)
    values = np.zeros((cell.natm, len(grid_vectors)))
    derivatives = np.zeros((cell.natm, 3, 3, len(grid_vectors)))
    for atom_id in range(cell.natm):
        symbol = cell.atom_symbol(atom_id)
        if symbol not in cell._pseudo:
            continue
        pseudo = cell._pseudo[symbol]
        radius, count, coefficients = pseudo[1:4]
        reduced = norm2 * radius**2
        polynomial = np.zeros_like(reduced)
        polynomial_derivative = np.zeros_like(reduced)
        if count >= 1:
            polynomial += coefficients[0]
        if count >= 2:
            polynomial += coefficients[1] * (3.0 - reduced)
            polynomial_derivative -= coefficients[1]
        if count >= 3:
            polynomial += coefficients[2] * (
                15.0 - 10.0 * reduced + reduced**2
            )
            polynomial_derivative += coefficients[2] * (
                -10.0 + 2.0 * reduced
            )
        if count >= 4:
            polynomial += coefficients[3] * (
                105.0 - 105.0 * reduced + 21.0 * reduced**2
                - reduced**3
            )
            polynomial_derivative += coefficients[3] * (
                -105.0 + 42.0 * reduced - 3.0 * reduced**2
            )
        amplitude = (2.0 * np.pi) ** 1.5 * radius**3
        damping = np.exp(-0.5 * reduced)
        value = -amplitude * damping * polynomial
        derivative_norm2 = -amplitude * radius**2 * damping * (
            polynomial_derivative - 0.5 * polynomial
        )
        values[atom_id] = value
        derivatives[atom_id] = (
            -2.0 * dyadic * derivative_norm2[None, None, :]
        )
    return values, derivatives


def _local_part2_potential_fourier_strain(
    cell, grid_vectors, *, block_size,
):
    """Contract atom-resolved local-part2 factors without full atom/G storage."""

    vectors = np.asarray(grid_vectors, dtype=float)
    potential = np.empty(len(vectors), dtype=np.complex128)
    derivative = np.empty((3, 3, len(vectors)), dtype=np.complex128)
    for p0, p1 in lib.prange(0, len(vectors), int(block_size)):
        block = vectors[p0:p1]
        structure_factors = cell.get_SI(Gv=block)
        atomic_value, atomic_derivative = _local_part2_atomic_fourier_strain(
            cell, block
        )
        potential[p0:p1] = -np.einsum(
            "ag,ag->g", structure_factors, atomic_value, optimize=True
        )
        derivative[:, :, p0:p1] = -np.einsum(
            "ag,axyg->xyg", structure_factors, atomic_derivative,
            optimize=True,
        )
    return potential, derivative


def _projector_angular_molecule():
    fake = gto.Mole()
    fake._atm = np.zeros((1, gto.ATM_SLOTS), dtype=np.int32)
    fake._bas = np.zeros((1, gto.BAS_SLOTS), dtype=np.int32)
    pointer = gto.PTR_ENV_START
    fake._env = np.zeros(pointer + 10)
    fake._bas[0, gto.NPRIM_OF] = 1
    fake._bas[0, gto.NCTR_OF] = 1
    fake._bas[0, gto.PTR_EXP] = pointer + 3
    fake._bas[0, gto.PTR_COEFF] = pointer + 4
    return fake


def _has_nonlocal_projectors(cell):
    for atom_id in range(cell.natm):
        symbol = cell.atom_symbol(atom_id)
        if symbol not in cell._pseudo:
            continue
        if any(int(projector[1]) > 0 for projector in cell._pseudo[symbol][5:]):
            return True
    return False


def _projector_value_gradient(fake, vectors, angular_momentum, count, radius):
    pointer = gto.PTR_ENV_START
    fake._bas[0, gto.ANG_OF] = int(angular_momentum)
    fake._env[pointer + 3] = 0.5 * radius**2
    fake._env[pointer + 4] = radius ** (angular_momentum + 1.5) * np.pi**1.25
    base = np.asarray(fake.eval_gto("GTOval", vectors)).T
    base_gradient = np.asarray(fake.eval_gto("GTOval_ip", vectors)).transpose(
        0, 2, 1
    )
    norm2_reduced = np.einsum("gx,gx->g", vectors, vectors) * radius**2
    projectors = []
    gradients = []
    for radial_index in range(int(count)):
        polynomial, derivative_t = _qli_polynomial_and_derivative(
            norm2_reduced, angular_momentum, radial_index
        )
        polynomial_gradient = (
            2.0 * radius**2
            * vectors.T[:, None, :]
            * derivative_t[None, None, :]
        )
        projectors.append(base * polynomial[None, :])
        gradients.append(
            base_gradient * polynomial[None, None, :]
            + base[None, :, :] * polynomial_gradient
        )
    return np.asarray(projectors), np.asarray(gradients).transpose(1, 0, 2, 3)


def _qli_polynomial_and_derivative(reduced_norm2, angular_momentum, index):
    sqrt = np.sqrt
    constants = {
        (0, 0): (4 * sqrt(2.0), 1.0, 0.0, 0.0),
        (0, 1): (8 * sqrt(2 / 15), 3.0, 1.0, 0.0),
        (0, 2): (16 / 3 * sqrt(2 / 105), 15.0, 10.0, 1.0),
        (1, 0): (8 * sqrt(1 / 3), 1.0, 0.0, 0.0),
        (1, 1): (16 * sqrt(1 / 105), 5.0, 1.0, 0.0),
        (1, 2): (32 / 3 * sqrt(1 / 1155), 35.0, 14.0, 1.0),
        (2, 0): (8 * sqrt(2 / 15), 1.0, 0.0, 0.0),
        (2, 1): (16 / 3 * sqrt(2 / 105), 7.0, 1.0, 0.0),
        (2, 2): (32 / 3 * sqrt(2 / 15015), 63.0, 18.0, 1.0),
        (3, 0): (16 * sqrt(1 / 105), 1.0, 0.0, 0.0),
        (3, 1): (32 / 3 * sqrt(1 / 1155), 9.0, 1.0, 0.0),
        (3, 2): (64 / 45 * sqrt(1 / 1001), 99.0, 22.0, 1.0),
        (4, 0): (16 / 3 * sqrt(2 / 105), 1.0, 0.0, 0.0),
        (4, 1): (32 / 3 * sqrt(2 / 15015), 11.0, 1.0, 0.0),
        (4, 2): (64 / 45 * sqrt(2 / 17017), 143.0, 26.0, 1.0),
    }
    try:
        scale, constant, linear, quadratic = constants[
            (int(angular_momentum), int(index))
        ]
    except KeyError as exc:
        raise NotImplementedError(
            f"unsupported GTH projector l={angular_momentum}, i={index}"
        ) from exc
    value = scale * (
        constant - linear * reduced_norm2 + quadratic * reduced_norm2**2
    )
    derivative = scale * (
        -linear + 2.0 * quadratic * reduced_norm2
    )
    return value, derivative


def _kpoints(value):
    points = np.asarray(value, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"kpts must have shape (nkpts,3), got {points.shape}")
    return points


def _reciprocal_mesh(cell, mesh):
    if mesh is None:
        cutoff = aft.estimate_ke_cutoff(cell, cell.precision)
        mesh_array = cell.symmetrize_mesh(cell.cutoff_to_mesh(cutoff))
    else:
        mesh_array = np.asarray(mesh, dtype=int)
    if mesh_array.shape != (3,) or np.any(mesh_array <= 0):
        raise ValueError(f"mesh must contain three positive integers: {mesh_array}")
    return np.asarray(mesh_array, dtype=int)


def _block_size(value):
    size = int(value)
    if size <= 0:
        raise ValueError("block_size must be positive")
    return size


def _ao_kpoint_sets(value, nkpts):
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if int(nkpts) == 1 and array.ndim == 3:
            return (array,)
        if array.ndim == 4 and array.shape[0] == int(nkpts):
            return tuple(array)
    values = tuple(np.asarray(item) for item in value)
    if len(values) != int(nkpts):
        raise ValueError(
            f"AO evaluator returned {len(values)} k-point blocks, expected {nkpts}"
        )
    return values


def _real_if_close(value, tolerance=1e-10):
    array = np.asarray(value)
    if array.size and np.max(np.abs(array.imag)) <= tolerance:
        return array.real
    return array
