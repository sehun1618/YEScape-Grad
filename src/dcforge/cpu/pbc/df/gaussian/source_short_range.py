r'''Short-range analytic-integral part of a Gaussian-DF three-center source.'''

import numpy as np
from pyscf import lib
from pyscf.pbc.df import rsdf_helper

# PySCF's private RSDF helper selects its exact unpacked ``s1`` driver only
# when the k-point list is not all Gamma.  This nonzero, diagonal k-point pair
# is appended as a storage-mode sentinel; its integral block is discarded.
# Any generic nonzero fraction gives the same retained Gamma result.
_S1_DRIVER_SENTINEL = .173


def short_range_source(
        cell, auxcell, omega, precision=None, max_memory=None):
    r'''Return the Gamma-point source :math:`B^{\mathrm{SR}}`.

    PySCF stores an all-Gamma symmetric AO pair in packed ``s2`` form.  The
    Cholesky response algebra uses the explicit ``(mu,nu)`` pair, so the result
    is unpacked here to ``(nao*nao,naux)``.

    Kwargs:
        precision : float or None
            Real-space three-center screening threshold.
        max_memory : float or None
            PySCF memory budget in MB passed to the short-range integral
            traversal.  ``None`` uses ``cell.max_memory``.
    '''
    tensor = _helper_tensor(
        cell, auxcell, omega, 'int3c2e', 's2', precision,
        max_memory=max_memory)
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f'unexpected packed source shape {tensor.shape}')
    tensor = lib.unpack_tril(tensor[0].T).transpose(1, 2, 0)
    return np.asarray(tensor).reshape(cell.nao_nr() ** 2, auxcell.nao_nr())


def short_range_source_kpts(
        cell, auxcell, omega, kpti, kptj, precision=None,
        max_memory=None):
    r'''Return sampled-k :math:`B^{SR}(k_i,k_j)`.

    Absolute Gamma retains the existing packed-value path.  Every other pair
    uses PySCF's unpacked complex ``s1`` driver so the primary AO orientation
    remains explicit for pair reversal and nuclear response.
    '''
    kpti = _kpoint(kpti)
    kptj = _kpoint(kptj)
    if _absolute_gamma_pair(kpti, kptj):
        return short_range_source(
            cell, auxcell, omega, precision, max_memory)
    tensor = _helper_tensor(
        cell, auxcell, omega, 'int3c2e', 's1', precision,
        pairs=np.asarray(((kpti, kptj),)), max_memory=max_memory)
    if tensor.shape[0] == 1:
        tensor = tensor[0]
    return np.asarray(tensor).reshape(
        cell.nao_nr() ** 2, auxcell.nao_nr())


def grad_short_range_source(cell, auxcell, omega, precision=None,
                            atmlst=None, max_memory=None):
    r'''Return the full fixed-cell derivative :math:`(B^{SR})^\lambda`.

    ``int3c2e_ip1`` differentiates the first primary AO and
    ``int3c2e_ip2`` the auxiliary AO with respect to the electron coordinate.
    The second-primary-center contribution follows from Gamma-point Hermitian
    AO-pair symmetry.  Nuclear-center derivatives carry the opposite sign.

    Returns:
        ndarray of shape (len(atmlst), 3, nao*nao, naux)
            Primary-i, primary-j, and auxiliary-center responses accumulated
            for the selected atoms from one pair of derivative integral
            traversals.

    Notes:
        PySCF forces packed ``s2`` storage if every requested pair is Gamma.
        Packed ``ip1`` loses which center was differentiated for shell pairs
        spanning two atoms.  ``_gamma_s1_tensor`` appends one unused diagonal
        non-Gamma pair, selecting PySCF's exact unpacked ``s1`` driver.  This
        changes storage only; it is not a small-k numerical approximation.

        ``max_memory`` is forwarded to PySCF's short-range integral writer;
        it controls its shell blocking and temporary HDF5 buffering rather
        than changing the mathematical source.
    '''
    source1 = gen_grad_short_range_source(
        cell, auxcell, omega, precision, max_memory)
    return source1(atmlst)


def gen_grad_short_range_source(
        cell, auxcell, omega, precision=None, max_memory=None):
    r'''Generate atom-block assembly of :math:`(B^{SR})^\lambda`.

    The periodic ``ip1`` and ``ip2`` integral traversals are independent of
    which atom owns the resulting AO slices.  They are evaluated once here;
    the returned function only allocates and fills the requested atom block.

    Kwargs:
        precision : float or None
            Real-space three-center screening threshold.
        max_memory : float or None
            PySCF memory budget in MB for both derivative traversals.
    '''
    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    ip1 = _gamma_s1_tensor(
        cell, auxcell, omega, 'int3c2e_ip1', precision, max_memory)
    ip2 = _gamma_s1_tensor(
        cell, auxcell, omega, 'int3c2e_ip2', precision, max_memory)
    ip1 = ip1.reshape(3, nao, nao, naux)
    ip2 = ip2.reshape(3, nao, nao, naux)

    ao_slices = cell.aoslice_nr_by_atom()
    aux_slices = auxcell.aoslice_nr_by_atom()

    def source1(atmlst=None):
        if atmlst is None:
            atmlst = range(cell.natm)
        atmlst = [int(ia) for ia in atmlst]
        value = np.zeros(
            (len(atmlst), 3, nao, nao, naux),
            dtype=np.result_type(ip1, ip2))
        for k, ia in enumerate(atmlst):
            p0, p1 = ao_slices[ia, 2:]
            q0, q1 = aux_slices[ia, 2:]
            value[k, :, p0:p1] -= ip1[:, p0:p1]
            value[k, :, :, p0:p1] -= ip1[:, p0:p1].transpose(
                0, 2, 1, 3).conj()
            value[k, :, :, :, q0:q1] -= ip2[:, :, :, q0:q1]
        return _real_if_close(
            value.reshape(len(atmlst), 3, nao * nao, naux))

    return source1


def grad_short_range_source_kpts(
        cell, auxcell, omega, kpti, kptj, precision=None,
        atmlst=None, max_memory=None):
    r'''Return sampled-k :math:`(B^{SR}(k_i,k_j))^\lambda`.

    The first-primary derivative comes from ``int3c2e_ip1(k_i,k_j)``.
    The second-primary derivative is the conjugate AO transpose of
    ``int3c2e_ip1(k_j,k_i)``.  ``int3c2e_ip2`` owns the auxiliary-center
    response.  Electron-coordinate derivatives acquire a minus sign when
    converted to nuclear-center derivatives.
    '''
    kpti = _kpoint(kpti)
    kptj = _kpoint(kptj)
    if _absolute_gamma_pair(kpti, kptj):
        return grad_short_range_source(
            cell, auxcell, omega, precision, atmlst, max_memory)
    if atmlst is None:
        atmlst = range(cell.natm)
    atmlst = [int(ia) for ia in atmlst]
    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    pairs = np.asarray(((kpti, kptj),))
    swapped = np.asarray(((kptj, kpti),))
    ip1 = _helper_tensor(
        cell, auxcell, omega, 'int3c2e_ip1', 's1', precision,
        pairs=pairs, max_memory=max_memory).reshape(3, nao, nao, naux)
    ip2 = _helper_tensor(
        cell, auxcell, omega, 'int3c2e_ip2', 's1', precision,
        pairs=pairs, max_memory=max_memory).reshape(3, nao, nao, naux)
    ip1_swapped = _helper_tensor(
        cell, auxcell, omega, 'int3c2e_ip1', 's1', precision,
        pairs=swapped, max_memory=max_memory).reshape(
            3, nao, nao, naux)
    value = np.zeros(
        (len(atmlst), 3, nao, nao, naux),
        dtype=np.result_type(ip1, ip2, ip1_swapped))
    ao_slices = cell.aoslice_nr_by_atom()
    aux_slices = auxcell.aoslice_nr_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = ao_slices[ia, 2:]
        q0, q1 = aux_slices[ia, 2:]
        value[k, :, p0:p1] -= ip1[:, p0:p1]
        value[k, :, :, p0:p1] -= ip1_swapped[:, p0:p1].transpose(
            0, 2, 1, 3).conj()
        value[k, :, :, :, q0:q1] -= ip2[:, :, :, q0:q1]
    return _real_if_close(
        value.reshape(len(atmlst), 3, nao * nao, naux))


def _gamma_s1_tensor(
        cell, auxcell, omega, intor, precision, max_memory=None):
    gamma = np.zeros(3)
    dummy = _S1_DRIVER_SENTINEL * cell.reciprocal_vectors()[0]
    pairs = np.asarray(((gamma, gamma), (dummy, dummy)))
    return _helper_tensor(
        cell, auxcell, omega, intor, 's1', precision,
        pairs=pairs, max_memory=max_memory)


def _helper_tensor(
        cell, auxcell, omega, intor, aosym, precision, pairs=None,
        max_memory=None):
    if pairs is None:
        pairs = np.zeros((1, 2, 3))
    temporary = lib.H5TmpFile()
    try:
        if max_memory is None:
            max_memory = getattr(cell, 'max_memory', 2000)
        max_memory = float(max_memory)
        if max_memory <= 0:
            raise ValueError('short-range source max_memory must be positive')
        rsdf_helper._aux_e2_nospltbas(
            cell, auxcell, abs(float(omega)), temporary,
            intor=intor, aosym=aosym, dataname='j3c',
            kptij_lst=np.asarray(pairs), max_memory=max_memory,
            precision=cell.precision if precision is None else precision,
        )
        group = temporary['j3c/0']
        keys = sorted(group.keys(), key=int)
        return np.concatenate(
            [np.asarray(group[key]) for key in keys], axis=-1)
    finally:
        temporary.close()


def _kpoint(value):
    '''Return one Cartesian reciprocal vector.'''
    value = np.asarray(value, dtype=float)
    if value.shape != (3,):
        raise ValueError(f'k-point must have shape (3,), got {value.shape}')
    return value


def _absolute_gamma_pair(kpti, kptj):
    '''Return whether both absolute k-points are Gamma.'''
    return np.linalg.norm(kpti) < 1e-10 and np.linalg.norm(kptj) < 1e-10


def _real_if_close(value, tol=1e-10):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tol:
        return value.real
    return value
