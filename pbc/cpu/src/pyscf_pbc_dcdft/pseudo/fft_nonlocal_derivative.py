"""Finite-mesh FFTDF derivative of the nonlocal GTH potential.

PySCF :class:`pyscf.pbc.df.fft.FFTDF` evaluates the nonlocal GTH projector on
the provider's reciprocal mesh.  Its finite-mesh value is not identical to the
analytic two-center-integral representation used by GDF.  An FFTDF gradient
must therefore differentiate this reciprocal expression rather than attach a
continuum-limit projector derivative to a finite-grid energy.

For one angular channel the matrix is

``V_nl(k) = vol^-1 * B(k)^H h B(k)``,

where ``B`` contracts the atom structure factor, radial/angular GTH projector,
and Fourier AO.  Nuclear translation changes the projector structure factor
and the AO Fourier columns.  This module applies the product rule to both
channels and returns the uncontracted AO matrices needed by the DC Z-vector
source.  No displaced geometry is used.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto as molecular_gto
from pyscf.pbc.df import ft_ao
from pyscf.pbc.gto.pseudo import pp as gth_pp

from .nonlocal_derivative import NonlocalDerivativeResult


def fft_nonlocal_gth_derivative(cell, kpts) -> NonlocalDerivativeResult:
    """Return the FFTDF nonlocal value and nuclear derivative matrices.

    Parameters
    ----------
    cell
        Built periodic PySCF cell carrying a GTH pseudopotential and the mesh
        used by the associated FFTDF provider.
    kpts
        Explicit k-points with shape ``(nkpts,3)`` in reciprocal Bohr.

    Returns
    -------
    NonlocalDerivativeResult
        ``matrix`` has shape ``(nkpts,nao,nao)``.  The two derivative channels
        have shape ``(natm,nkpts,3,nao,nao)`` and units Hartree/Bohr.

    Notes
    -----
    ``projector_center`` differentiates ``SI_A(G)^*`` and therefore contains
    ``+i G_x``.  ``ao_center`` differentiates atom-owned AO Fourier columns
    and contains ``-i (G+k)_x``.  Keeping ``G`` and ``G+k`` distinct is
    essential at non-Gamma k-points.
    """

    kpoint_array = np.asarray(kpts, dtype=float).reshape(-1, 3)
    grid_vectors = np.asarray(cell.get_Gv(cell.mesh))
    structure_factors = np.asarray(cell.get_SI(mesh=cell.mesh))
    helper, pointer = _projector_molecule()
    nao = int(cell.nao_nr())
    matrix = np.zeros(
        (len(kpoint_array), nao, nao), dtype=np.complex128
    )
    projector_center = np.zeros(
        (cell.natm, len(kpoint_array), 3, nao, nao),
        dtype=np.complex128,
    )
    ao_center = np.zeros_like(projector_center)

    for kpoint_id, kpoint in enumerate(kpoint_array):
        result = _one_kpoint_derivative(
            cell,
            kpoint,
            grid_vectors,
            structure_factors,
            helper,
            pointer,
        )
        matrix[kpoint_id] = result.matrix
        projector_center[:, kpoint_id] = result.projector_center
        ao_center[:, kpoint_id] = result.ao_center

    # This final factor mirrors FFTDF.get_pp after its Fourier AO has already
    # received the sqrt(1/vol) normalization.
    scale = 1.0 / float(cell.vol)
    return NonlocalDerivativeResult(
        matrix=matrix * scale,
        projector_center=projector_center * scale,
        ao_center=ao_center * scale,
    )


def _one_kpoint_derivative(
    cell,
    kpoint,
    grid_vectors,
    structure_factors,
    helper,
    pointer,
) -> NonlocalDerivativeResult:
    """Assemble one k-point before the final ``1/vol`` scaling."""

    shifted_vectors = grid_vectors + np.asarray(kpoint)
    radial_grid = np.linalg.norm(shifted_vectors, axis=1)
    ao_fourier = ft_ao.ft_ao(
        cell,
        grid_vectors,
        kpt=kpoint,
    ) * (1.0 / float(cell.vol)) ** 0.5
    nao = int(cell.nao_nr())
    matrix = np.zeros((nao, nao), dtype=np.complex128)
    projector_center = np.zeros(
        (cell.natm, 3, nao, nao), dtype=np.complex128
    )
    ao_center = np.zeros_like(projector_center)

    for pseudo_atom in range(cell.natm):
        for projectors, coupling in _projector_blocks(
            cell,
            pseudo_atom,
            shifted_vectors,
            radial_grid,
            helper,
            pointer,
        ):
            phased_projectors = (
                structure_factors[pseudo_atom].conj() * projectors
            )
            half_integral = np.einsum(
                "nmg,gp->nmp",
                phased_projectors,
                ao_fourier,
                optimize=True,
            )
            coupled_half = np.einsum(
                "ij,jmp->imp", coupling, half_integral, optimize=True
            )
            matrix += np.einsum(
                "imp,imq->pq",
                half_integral.conj(),
                coupled_half,
                optimize=True,
            )

            for atom_id in range(cell.natm):
                for axis in range(3):
                    if atom_id == pseudo_atom:
                        projector_derivative = np.einsum(
                            "nmg,gp->nmp",
                            1j
                            * grid_vectors[:, axis]
                            * phased_projectors,
                            ao_fourier,
                            optimize=True,
                        )
                        projector_center[atom_id, axis] += _product_rule(
                            half_integral,
                            coupled_half,
                            coupling,
                            projector_derivative,
                        )
                    ao_derivative = _ao_fourier_derivative(
                        cell,
                        ao_fourier,
                        shifted_vectors,
                        atom_id,
                        axis,
                    )
                    derivative_half = np.einsum(
                        "nmg,gp->nmp",
                        phased_projectors,
                        ao_derivative,
                        optimize=True,
                    )
                    ao_center[atom_id, axis] += _product_rule(
                        half_integral,
                        coupled_half,
                        coupling,
                        derivative_half,
                    )
    return NonlocalDerivativeResult(matrix, projector_center, ao_center)


def _projector_blocks(
    cell,
    atom_id,
    shifted_vectors,
    radial_grid,
    helper,
    pointer,
):
    """Yield reciprocal projector arrays and radial couplings for one atom."""

    symbol = cell.atom_symbol(atom_id)
    if symbol not in cell._pseudo:
        return
    for angular_momentum, channel in enumerate(cell._pseudo[symbol][5:]):
        radius, radial_count, coupling = channel
        if radial_count <= 0:
            continue
        helper._bas[0, molecular_gto.ANG_OF] = angular_momentum
        helper._env[pointer + 3] = 0.5 * radius**2
        helper._env[pointer + 4] = (
            radius ** (angular_momentum + 1.5) * np.pi**1.25
        )
        solid_harmonic = helper.eval_gto("GTOval", shifted_vectors)
        projectors = np.empty(
            (
                radial_count,
                2 * angular_momentum + 1,
                len(shifted_vectors),
            ),
            dtype=np.complex128,
        )
        for radial_id in range(radial_count):
            radial = gth_pp._qli(
                radial_grid * radius,
                angular_momentum,
                radial_id,
            )
            projectors[radial_id] = solid_harmonic.T * radial
        yield projectors, np.asarray(coupling)


def _ao_fourier_derivative(
    cell,
    ao_fourier,
    shifted_vectors,
    atom_id,
    axis,
):
    """Differentiate only AO columns centered on ``atom_id``."""

    p0, p1 = np.asarray(cell.aoslice_by_atom(), dtype=int)[atom_id, 2:]
    derivative = np.zeros_like(ao_fourier)
    derivative[:, p0:p1] = (
        -1j
        * shifted_vectors[:, axis, None]
        * ao_fourier[:, p0:p1]
    )
    return derivative


def _product_rule(half, coupled_half, coupling, derivative_half):
    coupled_derivative = np.einsum(
        "ij,jmp->imp", coupling, derivative_half, optimize=True
    )
    return np.einsum(
        "imp,imq->pq",
        derivative_half.conj(),
        coupled_half,
        optimize=True,
    ) + np.einsum(
        "imp,imq->pq",
        half.conj(),
        coupled_derivative,
        optimize=True,
    )


def _projector_molecule():
    """Return PySCF's one-shell helper for solid-harmonic projectors."""

    molecule = molecular_gto.Mole()
    molecule._atm = np.zeros(
        (1, molecular_gto.ATM_SLOTS), dtype=np.int32
    )
    molecule._bas = np.zeros(
        (1, molecular_gto.BAS_SLOTS), dtype=np.int32
    )
    pointer = molecular_gto.PTR_ENV_START
    molecule._env = np.zeros(pointer + 10)
    molecule._bas[0, molecular_gto.NPRIM_OF] = 1
    molecule._bas[0, molecular_gto.NCTR_OF] = 1
    molecule._bas[0, molecular_gto.PTR_EXP] = pointer + 3
    molecule._bas[0, molecular_gto.PTR_COEFF] = pointer + 4
    return molecule, pointer
