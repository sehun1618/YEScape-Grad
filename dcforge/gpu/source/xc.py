# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''GPU analytic mixed density/nuclear derivative of molecular semilocal XC.

For fixed AO matrices P and Q, contract d/dR (V_xc[P]:Q). At each grid point,
let e be the XC energy per volume, v=de/d(rho features), and fxc=dv/d(rho
features), as returned by eval_xc_eff (not derivatives of energy per particle).
The moving-grid chain rule is w' v.rho_Q + w [(fxc rho_Q).rho_P' + v.rho_Q'].
Features are linear in P/Q; tau=1/2 sum_mu,nu P_mu,nu grad(phi_mu).grad(phi_nu).
Primes include AO-center and grid-point motion, but no orbital relaxation.
There is no finite difference or numerical density perturbation.
PySCF supplies AO derivatives, XC derivatives and Becke partition derivatives.
'''

import cupy as cp
from gpu4pyscf.lib.cutensor import contract
from gpu4pyscf.hessian.rks import get_dweight_dA
from gpu4pyscf.dft.numint import _GDFTOpt


def _features(ao, dm, deriv=None, xctype='GGA'):
    '''Density, spatial gradient and tau; optionally their nuclear derivative.

    ao and deriv have shape (4, ngrid, nao); dm is real symmetric and may be
    indefinite (the Z transition density). No positive-density assumption is
    made here. LibXC itself is evaluated only at the physical source density.
    '''
    ad = ao @ dm
    dot = lambda a, b: contract('gi,gi->g', a, b)
    if deriv is None:
        values = [dot(ao[0], ad[0])]
        if xctype != 'LDA':
            values.extend(2 * dot(ao[k], ad[0]) for k in range(1, 4))
        if xctype == 'MGGA':
            values.append(sum(dot(ao[k], ad[k]) for k in range(1, 4)) * .5)
    else:
        values = [2 * dot(deriv[0], ad[0])]
        if xctype != 'LDA':
            values.extend(2 * (dot(deriv[k], ad[0]) + dot(deriv[0], ad[k]))
                          for k in range(1, 4))
        if xctype == 'MGGA':
            values.append(sum(dot(deriv[k], ad[k]) for k in range(1, 4)))
    return cp.stack(values)


def grad_xc_fock(source, dm, transition_dm, atmlst, block_size=2048,
                 grid_response=True):
    '''Return sum_σ Q_σ:F_xc,σ^[R], including source grid response.

    P and Q are AO matrices with matching restricted/unrestricted layout.
    Supports LDA, GGA, tau-MGGA, including the semilocal part of hybrids/RSH.
    The finite quadrature must use PySCF atom-centered grids without density
    pruning (source.small_rho_cutoff=0). Orbital response is NOT included here.

    Args:
        source : native RKS or UKS
            Stationary source with its own built grids, XC and NumInt.
        dm, transition_dm : cupy.ndarray
            Real symmetric AO matrices P and Q, (nao,nao) for restricted
            or (2,nao,nao) for unrestricted. Q need not be positive.
        atmlst : sequence of int
            Nuclear centers to differentiate, in output order.

    Kwargs:
        block_size : int
            Maximum grid points in an AO/XC contraction block.
        grid_response : bool
            Include moving points and Becke-weight derivatives. False keeps
            only AO-center motion and is not the full energy derivative.

    Returns:
        array of shape (len(atmlst),3)
            Q:F_xc^[R] on the input array backend, before the adjoint minus
            sign. rho features are (rho), (rho,grad rho), or (rho,grad rho,tau).
            Spin and feature axes are flattened together only for the fxc
            contraction, retaining all alpha/beta cross terms.
    '''
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    ni, mol, grids = source._numint, source.mol, source.grids
    opt = _GDFTOpt.from_mol(mol)

    def grid_iterator(grids):
        # Use the current weight derivative primitive, not the deprecated
        # GPU grids_response_cc wrapper. Preserve the actual source points
        # and quadrature weights; only reorder them by their owning atom.
        for atom in range(mol.natm):
            idx = cp.flatnonzero(grids.atm_idx == atom)
            block = grids.copy()
            for name in ('coords', 'weights', 'quadrature_weights', 'atm_idx'):
                setattr(block, name, getattr(grids, name)[idx])
            yield block.coords, block.weights, get_dweight_dA(mol, block)

    def eval_ao(coords):
        return ni.eval_ao(mol, coords, deriv=2, gdftopt=opt)
    xctype = ni._xc_type(source.xc)
    if xctype == 'HF':
        return cp.zeros((len(atmlst), 3))
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        raise NotImplementedError(f'Unsupported source XC type {xctype}')
    spin = int(dm.ndim == 3)
    p = cp.asarray(dm).reshape(-1, mol.nao, mol.nao)
    q = cp.asarray(transition_dm).reshape(p.shape)
    slices = mol.aoslice_by_atom()[:, 2:]
    # AO derivatives: x, y, z; xx, xy, xz; xy, yy, yz; xz, yz, zz.
    second = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
    out = cp.zeros((len(atmlst), 3))
    for owner, (coords, weights, weight1) in enumerate(grid_iterator(grids)):
        for lo in range(0, len(weights), block_size):
            hi = min(lo + block_size, len(weights))
            ao = eval_ao(coords[lo:hi])
            a = ao[:4]
            rho = cp.stack([_features(a, d, xctype=xctype) for d in p])
            rq = cp.stack([_features(a, d, xctype=xctype) for d in q])
            rho_xc = rho if spin else rho[0]
            if xctype == 'LDA':
                rho_xc = rho_xc[:, 0] if spin else rho_xc[0]
            _, vxc, fxc, _ = ni.eval_xc_eff(source.xc, rho_xc, deriv=2,
                                           xctype=xctype, spin=spin)
            v = cp.asarray(vxc).reshape(-1, hi-lo)
            f = cp.asarray(fxc).reshape(v.shape[0], v.shape[0], hi-lo)
            rq = rq.reshape(v.shape)
            fq = contract('abg,bg->ag', f, rq)
            if grid_response:
                out += contract('nxg,g->nx', weight1[atmlst, :, lo:hi],
                                contract('ag,ag->g', v, rq))
            for n, atom in enumerate(atmlst):
                # ponytail: explicit atom/axis loops repeat AO-density products;
                # this small-system reference favors auditability over speed.
                # Cache AO@P/Q and batch center contractions before scaling up.
                shift = cp.full(mol.nao, float(grid_response and atom == owner))
                p0, p1 = slices[atom]
                shift[p0:p1] -= 1
                for axis in range(3):
                    da = ao[[axis+1, *second[axis]]] * shift
                    rp1 = cp.stack([_features(a, d, da, xctype) for d in p])
                    rq1 = cp.stack([_features(a, d, da, xctype) for d in q])
                    out[n, axis] += cp.sum(weights[lo:hi] * (
                        cp.sum(fq * rp1.reshape(v.shape), axis=0)
                        + cp.sum(v * rq1.reshape(v.shape), axis=0)))
    return out
