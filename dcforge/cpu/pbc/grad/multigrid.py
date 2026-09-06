'''Target XC or fused J+XC multigrid derivatives for periodic gradients.'''

import numpy as np
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.dft.multigrid import multigrid_pair


def get_veff_deriv(mf_grad, dm0):
    '''Dispatch the selected target MG derivative algorithm.

    ``mf_grad.base.multigrid_with_j`` selects either the XC-only derivative
    or the fused target J+XC derivative.  Source-HF response remains outside
    this adapter and continues to use the active DF provider.
    '''
    method = mf_grad.multigrid_xc_deriv_method
    if method == 'single_pass':
        value = get_veff_deriv_single_pass(mf_grad, dm0)
    elif method == 'double_pass':
        value = get_veff_deriv_double_pass(mf_grad, dm0)
    else:
        raise ValueError(
            'multigrid_xc_deriv_method must be single_pass or double_pass')
    mf_grad.multigrid_xc_deriv_method_used = method
    return value


def get_vxc_deriv(mf_grad, dm0):
    '''Backward-compatible alias for :func:`get_veff_deriv`.'''
    return get_veff_deriv(mf_grad, dm0)


def get_veff_deriv_single_pass(mf_grad, dm0):
    r'''Return target XC or J+XC AO-row derivative in one MG pass.

    PySCF ``multigrid_pair.get_veff_ip1`` first builds the reciprocal XC
    potential :math:`w_{\rm xc}(G)`, adds the Hartree and cached local-PP
    potential, and finally transforms the combined potential with
    ``_get_j_pass2_ip1``.  DC-Forge assigns Hartree and the complete GTH
    one-electron Hamiltonian to the active FFTDF/GDF provider, so only
    the first quantity belongs in the target-XC derivative.

    This implementation reuses PySCF's density and pass-2 kernels but sends
    only :math:`w_{\rm xc}(G)` to pass 2 in XC-only mode.  With
    ``multigrid_with_j=True``, it adds :math:`\rho(G)v_C(G)` before that same
    pass, reproducing the fused target J+XC potential.  The local GTH
    pseudopotential is excluded in both modes because complete Hcore is a
    separate skeleton term.

    The XC-only route is algebraically identical to

    .. math::

        (J + V_{\rm xc})^{I\alpha}_{\rm MG}
        - J^{I\alpha}_{\rm MG},

    without evaluating the density, Hartree potential and AO derivative a
    second time.  ``get_veff_deriv_double_pass`` remains the numerical reference.

    Args:
        mf_grad : dcforge.cpu.pbc.grad.rhf_dft.Gradients
            Active Gamma-point restricted PBC gradient object.
        dm0 : ndarray of shape (nao,nao) or (nkpts,nao,nao)
            Source RHF/KRHF density including occupation factor two.

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
    kpts = np.asarray(getattr(mf, 'kpts', np.zeros((1, 3))))
    kpts = kpts.reshape(-1, 3)
    dm0 = np.asarray(dm0, order='C')
    if dm0.ndim == 2:
        dm0 = dm0[None]
    if dm0.shape != (len(kpts), nao, nao):
        raise ValueError(
            'multigrid restricted density shape '
            f'{dm0.shape}; expected {(len(kpts), nao, nao)}')

    xctype = ni._xc_type(mf.xc)
    if xctype == 'LDA':
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    else:
        raise NotImplementedError(
            f'single-pass multigrid XC derivative does not support {xctype}')

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

    if mf.multigrid_with_j:
        coulG = tools.get_coulG(cell, mesh=mesh)
        wv[0] += rhoG[0, 0] * coulG

    if xctype == 'GGA' and multigrid_pair.GGA_METHOD.upper() == 'FFT':
        Gv = cell.get_Gv(mesh)
        wv[0] -= lib.einsum('xg,gx->g', wv[1:4], 1j * Gv)
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
    expected = (len(kpts), 3, nao, nao)
    if vxc_ip1.shape != expected:
        raise ValueError(
            f'unexpected multigrid XC ip1 shape {vxc_ip1.shape}; '
            f'expected {expected}')
    return -vxc_ip1.transpose(1, 0, 2, 3)


def get_veff_deriv_double_pass(mf_grad, dm0):
    r'''Return a validation-reference MG XC or J+XC AO-row derivative.

    PySCF ``MultiGridNumInt2.get_veff_ip1`` returns the AO-row derivative of
    the combined multigrid potential

    .. math::

        (J + V_{\rm xc})^{I\alpha}_{\mu\nu}.

    DC-Forge assigns Hartree to the active FFTDF/GDF provider.  The multigrid
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
        dm0 : ndarray of shape (nao,nao) or (nkpts,nao,nao)
            Source RHF/KRHF density including occupation factor two.

    Returns:
        ndarray of shape (3, 1, nao, nao)
            Cartesian, density-set, AO-row, AO-column target-XC derivative.
    '''
    mf = mf_grad.base
    ni = mf._numint
    kpts = np.asarray(getattr(mf, 'kpts', np.zeros((1, 3)))).reshape(-1, 3)
    vj_xc_ip1 = np.asarray(ni.get_veff_ip1(
        dm0, xc_code=mf.xc, kpts=kpts))
    if mf.multigrid_with_j:
        # The zero-density HF call contains only the cached local-PP part.
        # Subtracting it leaves J+XC while preserving the independent PySCF
        # double-pass implementation as a validation reference.
        pp_ip1 = np.asarray(ni.get_veff_ip1(
            np.zeros_like(dm0), xc_code='HF', kpts=kpts))
        vxc = -(vj_xc_ip1 - pp_ip1)
    else:
        vj_ip1 = np.asarray(ni.get_veff_ip1(
            dm0, xc_code='HF', kpts=kpts))
        vxc = -(vj_xc_ip1 - vj_ip1)
    nao = mf.cell.nao_nr()
    if vxc.shape == (3, nao, nao):
        vxc = vxc[None]
    expected = (len(kpts), 3, nao, nao)
    if vxc.shape != expected:
        raise ValueError(
            f'unexpected multigrid XC derivative shape {vxc.shape}; '
            f'expected {expected}')
    return vxc.transpose(1, 0, 2, 3)


# PySCF-style compatibility names retained for downstream validation code.
get_vxc_deriv_single_pass = get_veff_deriv_single_pass
get_vxc_deriv_double_pass = get_veff_deriv_double_pass
