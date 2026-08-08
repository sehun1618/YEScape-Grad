'''GPU target XC or fused J+XC multigrid derivative adapter.'''

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.dft import multigrid_v2

from dcforge.gpu.pbc import multigrid as dc_multigrid


def get_veff_deriv(mf_grad, dm0, kpts=None):
    r'''Return the selected density-contracted target MG derivative.

    GPU4PySCF ``multigrid_v2.get_veff_ip1`` can combine XC, Hartree and the
    local GTH pseudopotential derivative.  DC-Forge always excludes the local
    pseudopotential because complete Hcore is a separate skeleton term.
    ``multigrid_with_j=False`` leaves only

    .. math::

       \left. \frac{\partial E_{\rm xc}[P_{\rm HF}]}{
       {\partial R_{A\alpha}} \right|_{P_{\rm HF}}

    on the multigrid mesh.  ``True`` returns the fused explicit target J+XC
    derivative instead.  Unlike the uniform-grid ``krks.get_vxc`` API,
    this GPU4PySCF primitive is already contracted with the supplied density
    and returns one scalar per atom and Cartesian component.

    Args:
        mf_grad : Gradients
            Active Gamma-point or sampled-k gradient object whose numerical
            integrator is ``multigrid_v2.MultiGridNumInt``.
        dm0 : cupy.ndarray
            Closed-shell source density with shape ``(nao,nao)`` at Gamma
            or ``(nkpts,nao,nao)`` for explicit equal-weight sampling.

    Kwargs:
        kpts : array_like or None
            Explicit sampled k-points.  ``None`` selects one Gamma point.

    Returns:
        cupy.ndarray of shape (natm, 3)
            Explicit fixed-density target-XC derivative in Ha/Bohr.

    Raises:
        TypeError
            If the target integrator is not the supported GPU multigrid-v2
            object.
        ValueError
            If the upstream derivative has an unexpected atom/Cartesian
            layout.
    '''
    mf = mf_grad.base
    ni = mf._numint
    if not isinstance(ni, multigrid_v2.MultiGridNumInt):
        raise TypeError('GPU multigrid derivative requires MultiGridNumInt')
    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    de = dc_multigrid.get_veff_ip1(
        ni, mf.xc, cp.asarray(dm0), hermi=1, kpts=kpts,
        with_j=bool(mf.multigrid_with_j),
        with_pseudo_vloc_orbital_derivative=False,
        verbose=mf_grad.verbose) / len(kpts)
    de = cp.asarray(de)
    expected = (mf.cell.natm, 3)
    if de.shape != expected:
        raise ValueError(
            f'unexpected GPU multigrid derivative shape {de.shape}; '
            f'expected {expected}')
    mf_grad.xc_deriv_backend = (
        'multigrid_v2_j_xc' if mf.multigrid_with_j
        else 'multigrid_v2_xc_only')
    return de


def get_vxc_deriv(mf_grad, dm0, kpts=None):
    '''Backward-compatible alias for :func:`get_veff_deriv`.'''
    return get_veff_deriv(mf_grad, dm0, kpts=kpts)
