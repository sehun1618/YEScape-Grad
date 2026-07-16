"""Direct RSDF metric-derivative contractions on the GPU.

The production PySCF RSDF short-range metric uses a screened driver whose
numerical convention is not reproduced by GPU4PySCF's current public
``int2c2e_ip1`` helper.  We therefore generate only the exact host ``ip1`` and
``ip2`` coordinate arrays and transfer them.  The expensive reciprocal metric
response is evaluated and contracted entirely on the device.  Neither route
materializes the all-atom ``dM`` tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.df import ft_ao
from pyscf.pbc.df import rsdf
from pyscf_pbc_dcdft.df.rsdf.metric import (
    short_range_metric_coordinate_derivatives,
)


@dataclass(frozen=True)
class MetricContractionResult:
    """Two observable gradients and exclusive device timing."""

    values: cp.ndarray
    device_seconds: float
    peak_explicit_bytes: int


@dataclass(frozen=True)
class MetricContractionContext:
    """Reusable exact-SR/GPU-LR data for one momentum-transfer block."""

    auxcell: object
    ip1: cp.ndarray
    ip2: cp.ndarray
    fourier: cp.ndarray
    coulomb: cp.ndarray
    momentum: cp.ndarray
    atom_ids: cp.ndarray
    host_seconds: float
    host_transfer_bytes: int
    device_setup_seconds: float

    @property
    def device_resident_bytes(self) -> int:
        """Explicit bytes retained by this momentum-transfer context."""

        arrays = (
            self.ip1,
            self.ip2,
            self.fourier,
            self.coulomb,
            self.momentum,
            self.atom_ids,
        )
        return sum(int(array.nbytes) for array in arrays)

    def contract(self, metric_adjoints) -> MetricContractionResult:
        """Contract one or more metric adjoints without forming ``dM``."""

        adjoints = cp.asarray(metric_adjoints)
        naux = int(self.fourier.shape[0])
        if adjoints.ndim != 3 or adjoints.shape[1:] != (naux, naux):
            raise ValueError(
                "metric adjoints must have shape (nsource,naux,naux), "
                f"got {adjoints.shape}"
            )
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        values = _contract_short_range(
            adjoints,
            self.ip1,
            self.ip2,
            self.atom_ids,
            int(self.auxcell.natm),
        )
        values += _contract_reciprocal(
            adjoints,
            self.fourier,
            self.coulomb,
            self.momentum,
            self.atom_ids,
            int(self.auxcell.natm),
        )
        end.record()
        end.synchronize()
        working_bytes = self.device_resident_bytes + (
            int(adjoints.nbytes)
            + 4 * int(self.fourier.nbytes) * len(adjoints)
            + int(values.nbytes)
        )
        return MetricContractionResult(
            values=values,
            device_seconds=(
                cp.cuda.get_elapsed_time(start, end) / 1000.0
            ),
            peak_explicit_bytes=working_bytes,
        )


def build_metric_contraction_context(
    cell,
    auxcell,
    parameters,
    *,
    q,
) -> MetricContractionContext:
    """Build exact short-range and device reciprocal data for one ``q``."""

    momentum_transfer = _qpoint(q)
    host_start = perf_counter()
    coordinate = short_range_metric_coordinate_derivatives(
        auxcell,
        parameters.metric_omega,
        q=momentum_transfer,
    )
    grid_vectors, coulomb = _reciprocal_grid(
        cell,
        auxcell,
        parameters.metric_omega,
        parameters.metric_mesh,
        momentum_transfer,
    )
    atom_ids = _auxiliary_atom_ids(auxcell)
    host_seconds = perf_counter() - host_start
    host_arrays = (
        coordinate.ip1,
        coordinate.ip2,
        grid_vectors,
        coulomb,
        atom_ids,
    )

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    device_momentum = cp.asarray(grid_vectors + momentum_transfer)
    fourier = ft_ao.ft_ao(auxcell, device_momentum).T
    ip1 = cp.asarray(coordinate.ip1)
    ip2 = cp.asarray(coordinate.ip2)
    device_coulomb = cp.asarray(coulomb)
    device_atom_ids = cp.asarray(atom_ids, dtype=cp.int32)
    end.record()
    end.synchronize()
    return MetricContractionContext(
        auxcell=auxcell,
        ip1=ip1,
        ip2=ip2,
        fourier=fourier,
        coulomb=device_coulomb,
        momentum=device_momentum,
        atom_ids=device_atom_ids,
        host_seconds=host_seconds,
        host_transfer_bytes=sum(int(array.nbytes) for array in host_arrays),
        device_setup_seconds=(
            cp.cuda.get_elapsed_time(start, end) / 1000.0
        ),
    )


def _contract_short_range(adjoints, ip1, ip2, atom_ids, natm):
    row = -cp.real(
        cp.einsum("spq,xpq->sxp", adjoints, ip1, optimize=True)
    )
    column = -cp.real(
        cp.einsum("spq,xpq->sxq", adjoints, ip2, optimize=True)
    )
    return _group_auxiliary(row + column, atom_ids, natm)


def _contract_reciprocal(
    adjoints,
    fourier,
    coulomb,
    momentum,
    atom_ids,
    natm,
):
    left = cp.einsum(
        "spq,qG->spG",
        adjoints,
        fourier,
        optimize=True,
    )
    right = cp.einsum(
        "sqp,qG->spG",
        adjoints,
        fourier.conj(),
        optimize=True,
    )
    basis = cp.empty((len(adjoints), 3, len(atom_ids)))
    for axis in range(3):
        phase = (
            1j
            * momentum[:, axis][None, None]
            * coulomb[None, None]
        )
        derivative = phase * (
            fourier.conj()[None] * left
            - fourier[None] * right
        )
        basis[:, axis] = cp.real(cp.sum(derivative, axis=2))
    return _group_auxiliary(basis, atom_ids, natm)


def _group_auxiliary(basis_values, atom_ids, natm):
    output = cp.zeros((len(basis_values), int(natm), 3))
    cp.add.at(
        output,
        (slice(None), atom_ids, slice(None)),
        basis_values.transpose(0, 2, 1),
    )
    return output


def _reciprocal_grid(cell, auxcell, omega, mesh, q):
    mesh_array = np.asarray(mesh, dtype=int)
    grid_vectors, _grid_bases, _weights = auxcell.get_Gv_weights(
        mesh_array
    )
    provider = rsdf.RSGDF(cell, kpts=q.reshape(1, 3))
    provider.auxcell = auxcell
    coulomb = provider.weighted_coulG(
        q,
        mesh=mesh_array,
        omega=float(omega),
    )
    return np.asarray(grid_vectors), np.asarray(coulomb)


def _auxiliary_atom_ids(auxcell):
    slices = np.asarray(auxcell.aoslice_nr_by_atom(), dtype=int)
    return np.concatenate(
        [
            np.full(int(p1 - p0), atom_id, dtype=np.int32)
            for atom_id, (_s0, _s1, p0, p1) in enumerate(slices)
        ]
    )


def _qpoint(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"q must have shape (3,), got {point.shape}")
    return point
