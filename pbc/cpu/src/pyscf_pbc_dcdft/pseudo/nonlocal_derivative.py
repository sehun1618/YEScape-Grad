"""Analytic matrix derivative of the periodic nonlocal GTH potential.

PySCF represents every nonlocal GTH channel by projector half-integrals
``I[l,i,m,mu](k)`` and a small radial coupling matrix ``h[l]``:

``V_nl(k) = sum_l I_l(k)^H h_l I_l(k)``.

Ordinary variational gradients contract its derivative with a density matrix
inside :func:`pyscf.pbc.gto.pseudo.pp_int.vppnl_nuc_grad`.  DC-DFT additionally
needs the uncontracted AO matrix for the Z-vector nuclear source.  This module
applies the same product rule before density contraction and returns separate
projector-center and AO-center channels.

The implementation deliberately uses PySCF's analytic two-center integral
projectors rather than reconstructing them on a reciprocal mesh.  This keeps
all radial normalizations and s/p/d/f angular channels identical to
``pp_int.get_pp_nl``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf.pbc.gto.pseudo import pp_int


@dataclass(frozen=True)
class NonlocalDerivativeResult:
    """Central nonlocal matrices and two center-resolved derivative channels.

    Attributes
    ----------
    matrix
        Shape ``(nkpts,nao,nao)`` central nonlocal GTH potential.
    projector_center
        Shape ``(natm,nkpts,3,nao,nao)`` derivative from translating the
        pseudopotential projector center.
    ao_center
        Same shape, containing AO-basis translations.
    """

    matrix: np.ndarray
    projector_center: np.ndarray
    ao_center: np.ndarray

    @property
    def derivative(self) -> np.ndarray:
        """Return the complete analytic nuclear derivative matrix."""

        return self.projector_center + self.ao_center


def nonlocal_gth_derivative(cell, kpts) -> NonlocalDerivativeResult:
    """Return ``V_nl`` and ``dV_nl/dR`` in PySCF's exact integral convention.

    No displaced geometry is built.  ``int1e_*_ip2`` projector half-integrals
    provide the analytic derivative with respect to the GTH projector center;
    translational invariance gives the opposite derivative on AO centers.
    """

    kpoint_array = np.asarray(kpts, dtype=float).reshape(-1, 3)
    fakecell, coupling_blocks = pp_int.fake_cell_vnl(cell)
    projector_values = pp_int._int_vnl(
        cell, fakecell, coupling_blocks, kpoint_array
    )
    derivative_intors = (
        "int1e_ipovlp",
        "int1e_r2_origi_ip2",
        "int1e_r4_origi_ip2",
    )
    projector_derivatives = pp_int._int_vnl(
        cell,
        fakecell,
        coupling_blocks,
        kpoint_array,
        derivative_intors,
        comp=3,
    )

    # The first integral name is historically an ip1 integral.  PySCF applies
    # this sign conversion in vppnl_nuc_grad before treating it as ip2.
    if len(projector_derivatives[0]) > 0:
        for kpoint in range(len(kpoint_array)):
            projector_derivatives[0][kpoint] *= -1

    nkpts = len(kpoint_array)
    nao = int(cell.nao_nr())
    matrix = np.zeros((nkpts, nao, nao), dtype=np.complex128)
    projector_center = np.zeros(
        (cell.natm, nkpts, 3, nao, nao), dtype=np.complex128
    )
    ao_center = np.zeros_like(projector_center)
    ao_slices = np.asarray(cell.aoslice_by_atom(), dtype=int)

    for kpoint in range(nkpts):
        radial_offsets = [0, 0, 0]
        for shell, coupling in enumerate(coupling_blocks):
            angular_momentum = int(fakecell.bas_angular(shell))
            magnetic_dimension = 2 * angular_momentum + 1
            radial_dimension = int(np.asarray(coupling).shape[0])
            values, derivatives = _projector_shell_blocks(
                projector_values,
                projector_derivatives,
                kpoint,
                radial_offsets,
                radial_dimension,
                magnetic_dimension,
                nao,
            )
            coupling = np.asarray(coupling)
            matrix[kpoint] += np.einsum(
                "imp,ij,jmq->pq", values.conj(), coupling, values
            )

            # ``one_sided`` differentiates the bra projector half.  Adding its
            # Hermitian conjugate is the second product-rule term.
            one_sided = np.einsum(
                "idmp,ij,jmq->dpq",
                derivatives.conj(),
                coupling,
                values,
            )
            parent_atom = int(fakecell._bas[shell, 0])
            projector_center[parent_atom, kpoint] += _hermitian_product_rule(
                one_sided
            )

            # Translating an AO and translating the projector center in the
            # same direction have opposite half-integral derivatives.  Only
            # AO rows owned by the selected atom receive the bra term.
            for atom_id, (_s0, _s1, p0, p1) in enumerate(ao_slices):
                ao_center[atom_id, kpoint, :, p0:p1, :] -= one_sided[
                    :, p0:p1, :
                ]
                ao_center[atom_id, kpoint, :, :, p0:p1] -= one_sided[
                    :, p0:p1, :
                ].transpose(0, 2, 1).conj()

    return NonlocalDerivativeResult(
        matrix=matrix,
        projector_center=projector_center,
        ao_center=ao_center,
    )


def _projector_shell_blocks(
    values_by_radial,
    derivatives_by_radial,
    kpoint: int,
    offsets: list[int],
    radial_dimension: int,
    magnetic_dimension: int,
    nao: int,
):
    """Gather one angular shell from PySCF's radial-major storage."""

    values = np.empty(
        (radial_dimension, magnetic_dimension, nao), dtype=np.complex128
    )
    derivatives = np.empty(
        (radial_dimension, 3, magnetic_dimension, nao), dtype=np.complex128
    )
    for radial in range(radial_dimension):
        p0 = offsets[radial]
        p1 = p0 + magnetic_dimension
        values[radial] = np.asarray(values_by_radial[radial][kpoint])[p0:p1]
        derivatives[radial] = np.asarray(
            derivatives_by_radial[radial][kpoint]
        )[:, p0:p1]
        offsets[radial] = p1
    return values, derivatives


def _hermitian_product_rule(one_sided: np.ndarray) -> np.ndarray:
    return one_sided + one_sided.transpose(0, 2, 1).conj()
