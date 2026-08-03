'''XC-only GPU multigrid derivative adapter for periodic DFA@RHF.'''

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.dft import multigrid_v2


def get_vxc_deriv(mf_grad, dm0):
    r'''Return the density-contracted multigrid target-XC derivative.

    GPU4PySCF ``multigrid_v2.get_veff_ip1`` can combine XC, Hartree and the
    local GTH pseudopotential derivative.  DC-Forge intentionally assigns
    the latter two quantities to the active GDF source/target provider.
    Requesting both switches below therefore leaves only

    .. math::

       \left. \frac{\partial E_{\rm xc}[P_{\rm HF}]}{
       {\partial R_{A\alpha}} \right|_{P_{\rm HF}}

    on the multigrid mesh.  Unlike the uniform-grid ``krks.get_vxc`` API,
    this GPU4PySCF primitive is already contracted with the supplied density
    and returns one scalar per atom and Cartesian component.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.rhf_dft.Gradients
            Active Gamma-point gradient object whose target numerical
            integrator is ``multigrid_v2.MultiGridNumInt``.
        dm0 : cupy.ndarray of shape (nao, nao)
            Closed-shell source RHF density including its occupation factor.

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
        raise TypeError('GPU XC-only multigrid derivative requires MultiGridNumInt')
    kpts = np.zeros((1, 3))
    de = multigrid_v2.get_veff_ip1(
        ni, mf.xc, cp.asarray(dm0), hermi=1, kpts=kpts,
        with_j=False, with_pseudo_vloc_orbital_derivative=False,
        verbose=mf_grad.verbose)
    de = cp.asarray(de)
    expected = (mf.cell.natm, 3)
    if de.shape != expected:
        raise ValueError(
            f'unexpected GPU multigrid XC derivative shape {de.shape}; '
            f'expected {expected}')
    mf_grad.xc_deriv_backend = 'multigrid_v2_xc_only'
    return de
