'''Density-batched real-Gamma nonlocal GTH contractions for Gaussian DF.'''

import warnings

import cupy as cp
import numpy as np
from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vnl


def contract_nonlocal_pseudo_batch(cell, densities):
    r'''Contract one projector-derivative build with several AO densities.

    GPU4PySCF's public :func:`vppnl_nuc_grad` accepts one density at a time.
    A DC-DFT gradient needs the same nonlocal GTH derivative contracted with
    the source density ``P`` and the Hermitian Z transition density ``D_Z``.
    This real-Gamma helper builds the shared projector half-integrals once
    and retains a leading density-set axis only in the final contraction.

    Args:
        cell : pyscf.pbc.gto.Cell
            Three-dimensional periodic cell with a GTH pseudopotential.
        densities : cupy.ndarray
            One real-Gamma AO density ``(nao,nao)`` or a density batch
            ``(nset,nao,nao)``.

    Returns:
        cupy.ndarray
            Nonlocal pseudopotential gradients ``(nset,natm,3)`` in
            Ha/Bohr.  No source-Fock coefficient or force sign is applied.

    Notes:
        The projector assembly follows ``gpu4pyscf.pbc.grad.pp``.  Only the
        density contraction is batched; no pseudopotential or finite-grid
        representation is changed.
    '''
    from gpu4pyscf.pbc.gto.pseudo.pp_int import _int_vnl_gpu

    dms = cp.asarray(densities)
    if dms.ndim == 2:
        dms = dms[None]
    nao = int(cell.nao_nr())
    if dms.ndim != 3 or tuple(dms.shape[1:]) != (nao, nao):
        raise ValueError(
            'nonlocal pseudo density batch must have shape '
            f'(nset,{nao},{nao}), got {dms.shape}')
    imaginary = float(cp.max(cp.abs(dms.imag)).get())
    if imaginary > 1e-10:
        raise NotImplementedError(
            'batched nonlocal pseudopotential contraction requires '
            f'real-Gamma densities; maximum imaginary part is {imaginary:.3e}')

    kpts = np.zeros((1, 3))
    fakecell, hl_blocks = fake_cell_vnl(cell)
    integral_names = (
        'int1e_ipovlp',
        'int1e_r2_origi_ip2',
        'int1e_r4_origi_ip2',
    )
    half = _int_vnl_gpu(cell, fakecell, hl_blocks, kpts)
    half_ip2 = _int_vnl_gpu(
        cell, fakecell, hl_blocks, kpts, integral_names, comp=3)
    if len(half_ip2[0]) > 0:
        half_ip2[0][0] *= -1
    half = _to_device_integrals(half)
    half_ip2 = _to_device_integrals(half_ip2)

    # The public kernel uses D + D^T because the projector derivative is
    # supplied as one differentiated half.  Keep the same convention for
    # every density in the batch.
    dms = dms.real
    dms = dms + dms.transpose(0, 2, 1)
    de = cp.zeros((len(dms), cell.natm, 3), dtype=cp.complex128)
    h1ao = cp.zeros((3, nao, nao), dtype=cp.complex128)
    offsets = [0] * 3

    for ib, coefficients in enumerate(hl_blocks):
        l = fakecell.bas_angular(ib)
        nlm = 2 * l + 1
        nr = coefficients.shape[0]
        value = cp.zeros((nr, nlm, nao), dtype=cp.complex128)
        deriv = cp.zeros((nr, 3, nlm, nao), dtype=cp.complex128)
        for ir in range(nr):
            p0 = offsets[ir]
            p1 = p0 + nlm
            if len(half[ir]) > 0:
                value[ir] = half[ir][0, p0:p1]
            if len(half_ip2[ir]) > 0:
                deriv[ir] = half_ip2[ir][0, :, p0:p1]
            offsets[ir] = p1

        h1 = cp.einsum(
            'idlp,ij,jlq->dpq', deriv.conj(),
            cp.asarray(coefficients), value)
        h1ao += h1
        ia = int(fakecell._bas[ib, 0])
        de[:, ia] += cp.einsum('dpq,sqp->sd', h1, dms)

    for ia, aoslice in enumerate(cell.aoslice_by_atom()):
        p0, p1 = aoslice[2:]
        de[:, ia] -= cp.einsum(
            'dpq,sqp->sd', h1ao[:, p0:p1], dms[:, :, p0:p1])

    imaginary = float(cp.max(cp.abs(de.imag)).get())
    if imaginary >= 1e-8:
        warnings.warn(
            'large imaginary nonlocal pseudopotential gradient component: '
            f'{imaginary:.3e}', RuntimeWarning, stacklevel=2)
    return de.real


def _to_device_integrals(values):
    '''Transfer nonempty projector blocks to the active GPU.'''
    return [cp.asarray(value) if len(value) > 0 else value for value in values]
