r'''Gaussian-DF long-range metric and zero-transfer correction.

GDF and RSDF use different short-range metric representations, but share the
reciprocal-space long-range term and the fixed-cell :math:`G=0` correction.
Provider modules add their own short-range value and derivative to these
common terms.
'''

import numpy as np
from pyscf import lib
from pyscf.pbc.df import ft_ao, rsdf
from pyscf.pbc.lib.kpts_helper import is_zero


def long_range_metric(cell, auxcell, omega, mesh, q=None):
    r'''Return the common reciprocal-space metric :math:`M^{\mathrm{LR}}`.'''
    auxG, coulG, q, _Gv = _long_range_context(
        cell, auxcell, omega, mesh, q)
    metric = lib.einsum('PG,G,QG->PQ', auxG.conj(), coulG, auxG)
    return _real_at_gamma(metric, q)


def grad_long_range_metric(cell, auxcell, omega, mesh, q=None, atmlst=None):
    r'''Return all auxiliary-center derivatives of :math:`M^{\mathrm{LR}}`.

    The Fourier transform of an auxiliary AO centered on atom A obeys

    .. math::

        \partial_{R_{A,x}} P(\mathbf G+\mathbf q)
        = -i(G_x+q_x)P(\mathbf G+\mathbf q).

    Returns:
        ndarray of shape (len(atmlst), 3, naux, naux)
            Fixed-cell long-range metric response.
    '''
    metric1 = gen_grad_long_range_metric(
        cell, auxcell, omega, mesh, q)
    return metric1(atmlst)


def gen_grad_long_range_metric(cell, auxcell, omega, mesh, q=None):
    r'''Generate atom-block assembly of :math:`(M^{LR})^\lambda`.'''
    auxG, coulG, q, Gv = _long_range_context(
        cell, auxcell, omega, mesh, q)
    momentum = Gv + q
    naux = auxcell.nao_nr()
    aoslices = auxcell.aoslice_nr_by_atom()

    def metric1(atmlst=None):
        if atmlst is None:
            atmlst = range(auxcell.natm)
        atmlst = [int(ia) for ia in atmlst]
        value = np.zeros(
            (len(atmlst), 3, naux, naux), dtype=np.complex128)
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            for xyz in range(3):
                auxG1 = np.zeros_like(auxG)
                auxG1[p0:p1] = (
                    -1j * momentum[:, xyz][None] * auxG[p0:p1])
                value[k, xyz] = lib.einsum(
                    'PG,G,QG->PQ', auxG1.conj(), coulG, auxG)
                value[k, xyz] += lib.einsum(
                    'PG,G,QG->PQ', auxG.conj(), coulG, auxG1)
        return _real_at_gamma(value, q)

    return metric1


def g0_metric(cell, auxcell, omega, q=None):
    r'''Return the common :math:`G=0` auxiliary-charge correction.'''
    q = _qpoint(q)
    naux = auxcell.nao_nr()
    if not is_zero(q):
        return np.zeros((naux, naux), dtype=np.complex128)
    charge = np.asarray(rsdf.get_aux_chg(auxcell), dtype=float)
    coefficient = np.pi / (float(omega) ** 2 * float(cell.vol))
    return -coefficient * np.outer(charge, charge)


def _long_range_context(cell, auxcell, omega, mesh, q):
    q = _qpoint(q)
    mesh = _mesh(mesh)
    Gv, Gvbase, _weights = auxcell.get_Gv_weights(mesh)
    reciprocal = auxcell.reciprocal_vectors()
    gxyz = lib.cartesian_prod(
        [np.arange(len(values)) for values in Gvbase])
    with_df = rsdf.RSGDF(cell, kpts=q.reshape(1, 3))
    with_df.auxcell = auxcell
    coulG = with_df.weighted_coulG(q, mesh=mesh, omega=float(omega))
    auxG = ft_ao.ft_ao(
        auxcell, Gv, None, reciprocal, gxyz, Gvbase, q).T
    return auxG, coulG, q, Gv


def _real_at_gamma(value, q):
    value = np.asarray(value)
    return value.real if is_zero(q) else value


def _qpoint(q):
    q = np.zeros(3) if q is None else np.asarray(q, dtype=float)
    if q.shape != (3,):
        raise ValueError(f'q must have shape (3,), got {q.shape}')
    return q


def _mesh(mesh):
    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError(
            f'mesh must contain three positive integers, got {mesh}')
    return mesh
