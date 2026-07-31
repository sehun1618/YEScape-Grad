r'''Zero-transfer ``vbar`` correction to a Gamma-point Gaussian-DF source.'''

import numpy as np
from pyscf.pbc.df import rsdf


def vbar_source(cell, auxcell, omega):
    r'''Return :math:`B^{vbar}_{\mu\nu,P}=-S_{\mu\nu}\bar v_P`.

    Here :math:`\bar v_P=\pi q_P/(\omega^2\Omega)`, ``q_P`` is the
    integrated auxiliary Gaussian charge, and the lattice volume is fixed.
    '''
    overlap = np.asarray(cell.pbc_intor(
        'int1e_ovlp', hermi=1, kpts=np.zeros((1, 3))))[0]
    return -overlap.reshape(cell.nao_nr() ** 2, 1) * _vbar(
        cell, auxcell, omega)[None]


def grad_vbar_source(cell, auxcell, omega, atmlst=None):
    r'''Return the fixed-cell derivative :math:`(B^{vbar})^\lambda`.

    Auxiliary charges, ``omega``, and the cell volume do not change under a
    nuclear-center displacement.  Only the primary AO overlap response enters.

    Returns:
        ndarray of shape (len(atmlst), 3, nao*nao, naux)
            Overlap-center response multiplied by ``-vbar``.
    '''
    source1 = gen_grad_vbar_source(cell, auxcell, omega)
    return source1(atmlst)


def gen_grad_vbar_source(cell, auxcell, omega):
    r'''Generate atom-block assembly of :math:`(B^{vbar})^\lambda`.'''
    # PySCF's int1e_ipovlp is an electron-coordinate derivative.  Negating it
    # once here produces the nuclear derivative on the first AO center.
    s1 = -np.asarray(cell.pbc_intor(
        'int1e_ipovlp', kpts=np.zeros((1, 3))))[0]
    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    coefficient = _vbar(cell, auxcell, omega)
    aoslices = cell.aoslice_nr_by_atom()

    def source1(atmlst=None):
        if atmlst is None:
            atmlst = range(cell.natm)
        atmlst = [int(ia) for ia in atmlst]
        value = np.zeros(
            (len(atmlst), 3, nao, nao, naux), dtype=s1.dtype)
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            overlap1 = np.zeros_like(s1)
            overlap1[:, p0:p1] += s1[:, p0:p1]
            overlap1[:, :, p0:p1] += s1[:, p0:p1].transpose(
                0, 2, 1).conj()
            value[k] = -overlap1[:, :, :, None] * coefficient
        return value.reshape(len(atmlst), 3, nao * nao, naux)

    return source1


def _vbar(cell, auxcell, omega):
    charge = np.asarray(rsdf.get_aux_chg(auxcell), dtype=float)
    return np.pi * charge / (float(omega) ** 2 * float(cell.vol))
