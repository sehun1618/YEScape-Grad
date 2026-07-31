'''XC-only multigrid derivative adapters for periodic DC-DFT gradients.'''

import numpy as np
from pyscf.pbc import tools
from pyscf.pbc.dft.multigrid import multigrid_pair


def get_vxc_deriv(mf_grad, dm0):
    '''Dispatch the target-XC derivative to the selected MG algorithm.'''
    method = mf_grad.multigrid_xc_deriv_method
    if method == 'single_pass':
        value = get_vxc_deriv_single_pass(mf_grad, dm0)
    elif method == 'double_pass':
        value = get_vxc_deriv_double_pass(mf_grad, dm0)
    else:
        raise ValueError(
            'multigrid_xc_deriv_method must be single_pass or double_pass')
    mf_grad.multigrid_xc_deriv_method_used = method
    return value


def get_vxc_deriv_single_pass(mf_grad, dm0):
    r'''Return the XC AO-row derivative with one MG density/task pass.

    PySCF ``multigrid_pair.get_veff_ip1`` first builds the reciprocal XC
    potential :math:`w_{\rm xc}(G)`, adds the Hartree and cached local-PP
    potential, and finally transforms the combined potential with
    ``_get_j_pass2_ip1``.  DC-Forge assigns Hartree and the complete GTH
    one-electron Hamiltonian to GDF/RSDF, so only the first quantity belongs
    in the target-XC derivative.

    This implementation reuses PySCF's density and pass-2 kernels but sends
    only :math:`w_{\rm xc}(G)` to pass 2.  It is algebraically identical to

    .. math::

        (J + V_{\rm xc})^{I\alpha}_{\rm MG}
        - J^{I\alpha}_{\rm MG},

    without evaluating the density, Hartree potential and AO derivative a
    second time.  ``get_vxc_deriv_double_pass`` remains the numerical oracle.

    Args:
        mf_grad : dcforge.cpu.pbc.grad.rhf_dft.Gradients
            Active Gamma-point restricted PBC gradient object.
        dm0 : ndarray of shape (nao, nao)
            Source RHF density including occupation factor two.

    Returns:
        ndarray of shape (3, 1, nao, nao)
            Nuclear-center AO-row derivative of target XC.

    Notes:
        ``_eval_rhoG``, ``_get_j_pass2_ip1`` and the non-FFT GGA pass are
        private PySCF 2.14.0 kernels.  Their use is intentionally localized
        here and guarded by shape/backend regression tests.
    '''
    mf = mf_grad.base
    ni = mf._numint
    cell = mf.cell
    mesh = np.asarray(ni.mesh, dtype=int)
    ngrids = int(np.prod(mesh))
    nao = cell.nao_nr()
    dm0 = np.asarray(dm0, order='C')
    if dm0.shape != (nao, nao):
        raise ValueError(
            f'multigrid RHF density shape {dm0.shape}; expected {(nao, nao)}')

    xctype = ni._xc_type(mf.xc)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    else:
        raise NotImplementedError(
            f'single-pass multigrid XC derivative does not support {xctype}')

    kpts = np.zeros((1, 3))
    rhoG = multigrid_pair._eval_rhoG(
        ni, dm0, hermi=1, kpts=kpts, deriv=deriv)
    expected_rhodim = 1 if xctype == 'LDA' else 4
    expected = (1, expected_rhodim, ngrids)
    if rhoG.shape != expected:
        raise ValueError(
            f'unexpected multigrid density shape {rhoG.shape}; '
            f'expected {expected}')
    ni.rhoG = rhoG

    weight = cell.vol / ngrids
    rhoR = tools.ifft(rhoG.reshape(-1, ngrids), mesh).real / weight
    rhoR = rhoR.reshape(1, expected_rhodim, ngrids)
    vxc = ni.eval_xc_eff(
        mf.xc, rhoR[0], deriv=1, xctype=xctype, spin=0)[1]
    wv = tools.fft((weight * vxc).reshape(-1, ngrids), mesh)
    wv = wv.reshape(expected_rhodim, ngrids)

    if xctype == 'GGA' and multigrid_pair.GGA_METHOD.upper() == 'FFT':
        Gv = cell.get_Gv(mesh)
        wv[0] -= np.einsum('xg,gx->g', wv[1:4], 1j * Gv)
        wv = wv[:1]
        vxc_ip1 = multigrid_pair._get_j_pass2_ip1(
            ni, wv[None], hermi=0, kpts=kpts, deriv=1)
    elif xctype == 'GGA':
        vxc_ip1 = multigrid_pair._get_gga_pass2_ip1(
            ni, wv[None], hermi=0, kpts=kpts, deriv=1)
    else:
        vxc_ip1 = multigrid_pair._get_j_pass2_ip1(
            ni, wv[None], hermi=0, kpts=kpts, deriv=1)

    vxc_ip1 = np.asarray(vxc_ip1)
    expected = (1, 3, nao, nao)
    if vxc_ip1.shape != expected:
        raise ValueError(
            f'unexpected multigrid XC ip1 shape {vxc_ip1.shape}; '
            f'expected {expected}')
    return -vxc_ip1[0, :, None]


def get_vxc_deriv_double_pass(mf_grad, dm0):
    r'''Return the AO-row derivative of multigrid :math:`V_{\rm xc}`.

    PySCF ``MultiGridNumInt2.get_veff_ip1`` returns the AO-row derivative of
    the combined multigrid potential

    .. math::

        (J + V_{\rm xc})^{I\alpha}_{\mu\nu}.

    DC-Forge assigns Hartree to the active GDF/RSDF provider.  The multigrid
    XC derivative is therefore isolated with the exact same multigrid density
    and mesh by

    .. math::

        V_{\rm xc}^{I\alpha}
        = (J + V_{\rm xc})^{I\alpha}_{\rm MG}
          - J^{I\alpha}_{\rm MG}.

    The second call uses ``xc_code='HF'``; it contains no semilocal XC
    potential and thus supplies the multigrid J derivative to subtract.  Any
    cached long-range local-pseudopotential contribution is present in both
    calls and cancels as well.  The leading minus sign converts PySCF's
    electron-coordinate ``ip1`` convention to the AO-row nuclear derivative
    convention returned by ``pyscf.pbc.grad.krks.get_vxc``.

    Args:
        mf_grad : dcforge.cpu.pbc.grad.rhf_dft.Gradients
            Active PBC gradient object.  Its base must own a Gamma-point
            ``MultiGridNumInt2`` configured with ``xc_with_j=False``.
        dm0 : ndarray of shape (nao, nao)
            Source RHF density matrix, including the restricted occupation
            factor two.

    Returns:
        ndarray of shape (3, 1, nao, nao)
            Cartesian, density-set, AO-row, AO-column target-XC derivative.
    '''
    mf = mf_grad.base
    ni = mf._numint
    kpts = np.zeros((1, 3))
    vj_xc_ip1 = np.asarray(ni.get_veff_ip1(
        dm0, xc_code=mf.xc, kpts=kpts))
    vj_ip1 = np.asarray(ni.get_veff_ip1(
        dm0, xc_code='HF', kpts=kpts))
    vxc = -(vj_xc_ip1 - vj_ip1)
    if vxc.shape == (3, mf.cell.nao_nr(), mf.cell.nao_nr()):
        vxc = vxc[:, None]
    expected = (3, 1, mf.cell.nao_nr(), mf.cell.nao_nr())
    if vxc.shape != expected:
        raise ValueError(
            f'unexpected multigrid XC derivative shape {vxc.shape}; '
            f'expected {expected}')
    return vxc
