'''Direct Gamma RSDF metric-response contractions on the GPU.

The public PySCF RSDF metric uses a screened short-range driver.  GPU4PySCF
does not expose a representation-identical derivative of that driver.  This
module therefore receives only its exact ``ip1``/``ip2`` primitive arrays
from the CPU-side PySCF integral implementation.  The observable contraction,
atom reduction, and the long-range reciprocal derivative remain on the GPU.
No all-atom metric derivative ``dM`` is assembled or transferred.
'''

from time import perf_counter

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.df import ft_ao
from pyscf.pbc.df import rsdf

from dcforge.cpu.pbc.df.rsdf.metric import (
    short_range_metric_coordinate_derivatives,
)


class MetricContractionContext:
    r'''Reusable exact-SR/device-LR metric data at Gamma.

    ``ip1`` and ``ip2`` are the small screened host primitives.  ``fourier``
    and ``coulomb`` define the device-resident reciprocal contribution.  A
    context is independent of the particular observable, so target-J and
    source-Fock metric adjoints share all of its setup work.
    '''

    def __init__(self, auxcell, ip1, ip2, fourier, coulomb, vectors,
                 atom_ids, host_seconds, host_transfer_bytes,
                 device_setup_seconds):
        r'''Initialize the reusable metric-response intermediates.

        Args:
            auxcell : pyscf.pbc.gto.Cell
                Auxiliary cell defining basis-to-atom ownership.
            ip1, ip2 : cupy.ndarray
                Screened short-range metric coordinate derivatives.
            fourier : cupy.ndarray
                Auxiliary Fourier values in the resolved metric mesh.
            coulomb : cupy.ndarray
                Weighted reciprocal Coulomb kernel.
            vectors : cupy.ndarray
                Reciprocal vectors corresponding to ``fourier``.
            atom_ids : cupy.ndarray
                Auxiliary-AO to physical-atom mapping.
            host_seconds : float
                Host primitive and reciprocal-grid setup time.
            host_transfer_bytes : int
                Explicit host arrays copied to the device.
            device_setup_seconds : float
                Device context construction time.
        '''
        self.auxcell = auxcell
        self.ip1 = ip1
        self.ip2 = ip2
        self.fourier = fourier
        self.coulomb = coulomb
        self.vectors = vectors
        self.atom_ids = atom_ids
        self.host_seconds = float(host_seconds)
        self.host_transfer_bytes = int(host_transfer_bytes)
        self.device_setup_seconds = float(device_setup_seconds)

    @property
    def device_resident_bytes(self):
        '''Return explicit context bytes retained on the device.'''
        return sum(
            int(array.nbytes) for array in (
                self.ip1, self.ip2, self.fourier, self.coulomb,
                self.vectors, self.atom_ids))

    def contract(self, metric_adjoints):
        r'''Contract one or more ``dE/dM`` tensors into ``dE/dR``.

        Args:
            metric_adjoints : cupy.ndarray
                Real Gamma metric adjoints with shape
                ``(nobservable,naux,naux)``.

        Returns:
            cupy.ndarray
                Gradient contributions with shape
                ``(nobservable,natm,3)`` in Ha/Bohr.
        '''
        adjoints = cp.asarray(metric_adjoints)
        naux = int(self.fourier.shape[0])
        if adjoints.ndim != 3 or adjoints.shape[1:] != (naux, naux):
            raise ValueError(
                'metric adjoints must have shape '
                f'(nobservable,{naux},{naux}), got {adjoints.shape}')
        if adjoints.dtype.kind == 'c':
            imaginary = float(cp.max(cp.abs(adjoints.imag)).get())
            if imaginary > 1.0e-10:
                raise NotImplementedError(
                    'Gamma RSDF metric adjoint has an unsupported imaginary '
                    f'component {imaginary:.3e}')
            adjoints = adjoints.real
        return (_contract_short_range(
            adjoints, self.ip1, self.ip2, self.atom_ids, self.auxcell.natm)
            + _contract_reciprocal(
                adjoints, self.fourier, self.coulomb, self.vectors,
                self.atom_ids, self.auxcell.natm))


def build_metric_contraction_context(with_df):
    r'''Build exact Gamma RSDF metric primitive data.

    Args:
        with_df : dcforge.gpu.pbc.df.rsdf.RSDF
            Built GPU RSDF provider with independently resolved
            ``omega_j2c`` and ``mesh_j2c``.

    Returns:
        MetricContractionContext
            Exact short-range host primitive plus GPU reciprocal context.
    '''
    if getattr(with_df, 'backend_name', None) != 'rsdf':
        raise TypeError('metric response requires a dcforge GPU RSDF provider')
    if with_df.auxcell is None:
        with_df.build(j_only=False)
    settings = with_df.response_settings()
    metric_omega = settings['metric_omega']
    metric_mesh = np.asarray(settings['metric_mesh'], dtype=int)
    auxcell = with_df.auxcell
    start = perf_counter()
    ip1, ip2 = short_range_metric_coordinate_derivatives(auxcell, metric_omega)
    vectors, coulomb = _reciprocal_grid(
        with_df.cell, auxcell, metric_omega, metric_mesh)
    atom_ids = _auxiliary_atom_ids(auxcell)
    host_seconds = perf_counter() - start

    start_event = cp.cuda.Event()
    stop_event = cp.cuda.Event()
    start_event.record()
    device_vectors = cp.asarray(vectors)
    context = MetricContractionContext(
        auxcell=auxcell,
        ip1=cp.asarray(ip1),
        ip2=cp.asarray(ip2),
        fourier=ft_ao.ft_ao(auxcell, device_vectors).T,
        coulomb=cp.asarray(coulomb),
        vectors=device_vectors,
        atom_ids=cp.asarray(atom_ids, dtype=cp.int32),
        host_seconds=host_seconds,
        host_transfer_bytes=sum(
            int(array.nbytes) for array in (ip1, ip2, vectors, coulomb, atom_ids)),
        device_setup_seconds=0.,
    )
    stop_event.record()
    stop_event.synchronize()
    context.device_setup_seconds = (
        cp.cuda.get_elapsed_time(start_event, stop_event) / 1000.)
    return context


def contract_metric_derivative(with_df, metric_adjoints):
    r'''Convenience wrapper for a complete direct metric contraction.'''
    return build_metric_contraction_context(with_df).contract(metric_adjoints)


def _contract_short_range(adjoints, ip1, ip2, atom_ids, natm):
    row = -cp.real(cp.einsum('spq,xpq->sxp', adjoints, ip1, optimize=True))
    column = -cp.real(cp.einsum('spq,xpq->sxq', adjoints, ip2, optimize=True))
    return _group_auxiliary(row + column, atom_ids, natm)


def _contract_reciprocal(adjoints, fourier, coulomb, vectors, atom_ids, natm):
    left = cp.einsum('spq,qG->spG', adjoints, fourier, optimize=True)
    right = cp.einsum('sqp,qG->spG', adjoints, fourier.conj(), optimize=True)
    basis = cp.empty((len(adjoints), 3, len(atom_ids)))
    for axis in range(3):
        phase = 1j * vectors[:, axis][None, None] * coulomb[None, None]
        derivative = phase * (
            fourier.conj()[None] * left - fourier[None] * right)
        basis[:, axis] = cp.real(cp.sum(derivative, axis=2))
    return _group_auxiliary(basis, atom_ids, natm)


def _group_auxiliary(basis_values, atom_ids, natm):
    output = cp.zeros((len(basis_values), int(natm), 3))
    cp.add.at(
        output,
        (slice(None), atom_ids, slice(None)),
        basis_values.transpose(0, 2, 1))
    return output


def _reciprocal_grid(cell, auxcell, omega, mesh):
    vectors, _bases, _weights = auxcell.get_Gv_weights(mesh)
    provider = rsdf.RSGDF(cell, kpts=np.zeros((1, 3)))
    provider.auxcell = auxcell
    coulomb = provider.weighted_coulG(np.zeros(3), mesh=mesh, omega=omega)
    return np.asarray(vectors), np.asarray(coulomb)


def _auxiliary_atom_ids(auxcell):
    slices = np.asarray(auxcell.aoslice_nr_by_atom(), dtype=int)
    return np.concatenate([
        np.full(int(p1 - p0), atom_id, dtype=np.int32)
        for atom_id, (_s0, _s1, p0, p1) in enumerate(slices)])
