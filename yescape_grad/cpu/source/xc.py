# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''CPU analytic mixed density/nuclear derivative of molecular semilocal XC.

For fixed AO matrices P and Q, contract d/dR (V_xc[P]:Q). At each grid point,
let e be the XC energy per volume, v=de/d(rho features), and fxc=dv/d(rho
features), as returned by eval_xc_eff (not derivatives of energy per particle).
The moving-grid chain rule is w' v.rho_Q + w [(fxc rho_Q).rho_P' + v.rho_Q'].
Features are linear in P/Q; tau=1/2 sum_mu,nu P_mu,nu grad(phi_mu).grad(phi_nu).
Primes include AO-center and grid-point motion, but no orbital relaxation.
There is no finite difference or numerical density perturbation.
PySCF supplies AO derivatives, XC derivatives and Becke partition derivatives.
'''

from time import perf_counter, process_time
import numpy as np
from pyscf import lib
from pyscf.grad.rks import grids_response_cc


def _features(ao, dm, deriv=None, xctype='GGA', ad=None):
    '''Density, spatial gradient and tau; optionally their nuclear derivative.

    ao and deriv have shape (4, ngrid, nao); dm is real symmetric and may be
    indefinite (the Z transition density). No positive-density assumption is
    made here. LibXC itself is evaluated only at the physical source density.
    ad optionally supplies the block-local ao @ dm product for reuse.
    '''
    if ad is None:
        ad = ao @ dm
    dot = lambda a, b: lib.einsum('gi,gi->g', a, b)
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
    return np.stack(values)


def _ao_nuclear_terms(ao, ap, aq, fq, v, weights, xctype):
    """Contract once per AO; atom dependence is only its center and grid owner.

    For features (rho, grad rho, tau), combine their derivatives into
    sum_k dAO_k * Y_k. P/Q and spin contributions share the same AO derivatives.
    The tau derivative has no factor 2 because tau itself contains 1/2.
    """
    ncomp = 1 if xctype == 'LDA' else 4
    y = np.zeros_like(ap[0][:ncomp])
    for products, coefficients in ((ap, fq), (aq, v)):
        cs = coefficients.reshape(len(products), -1, len(weights))
        for ad, c in zip(products, cs):
            w = c * weights
            y[0] += 2 * w[0, :, None] * ad[0]
            if xctype != 'LDA':
                for k in range(1, 4):
                    y[0] += 2 * w[k, :, None] * ad[k]
                    y[k] += 2 * w[k, :, None] * ad[0]
                    if xctype == 'MGGA':
                        y[k] += w[4, :, None] * ad[k]
    second = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
    return np.stack([sum(np.einsum('gi,gi->i', ao[j], y[k])
                         for k, j in enumerate((axis+1, *second[axis])[:ncomp]))
                     for axis in range(3)])


def grad_xc_fock(source, dm, transition_dm, atmlst, block_size=2048,
                 grid_response=True, grid_response_backend='pyscf_default', timings=None):
    '''Return sum_σ Q_σ:F_xc,σ^[R], including source grid response.

    P and Q are AO matrices with matching restricted/unrestricted layout.
    Supports LDA, GGA, tau-MGGA, including the semilocal part of hybrids/RSH.
    The finite quadrature must use PySCF atom-centered grids without density
    pruning (source.small_rho_cutoff=0). Orbital response is NOT included here.

    Args:
        source : native RKS or UKS
            Stationary source with its own built grids, XC and NumInt.
        dm, transition_dm : numpy.ndarray
            Real symmetric AO matrices P and Q, (nao,nao) for restricted
            or (2,nao,nao) for unrestricted. Q need not be positive.
        atmlst : sequence of int
            Nuclear centers to differentiate, in output order.

    Kwargs:
        block_size : int
            Maximum grid points in an AO/XC contraction block.
        timings : dict or None
            Optional output of non-overlapping wall times and process CPU times
            (cpu_ prefix), in seconds. CPU/wall is average active CPU equivalents.
        grid_response_backend : str
            CPU partition derivative provider: pyscf_default or becke_omp.
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
    started = perf_counter()
    cpu_started = process_time()
    parts = dict(grid_partition=0., ao_evaluation=0., ao_density_products=0.,
                 density_and_xc=0., nuclear_contractions=0.)
    cpu_parts = dict.fromkeys(parts, 0.)
    if timings is not None:
        timings.clear()
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    ni, mol, grids = source._numint, source.mol, source.grids
    if grid_response_backend not in ('pyscf_default', 'becke_omp'):
        raise ValueError('Unknown source grid_response_backend')
    grid_iterator = grids_response_cc
    if grid_response and grid_response_backend == 'becke_omp':
        from yescape_grad.cpu.grad.grid_response import grids_response_becke
        grid_iterator = grids_response_becke

    def eval_ao(coords):
        return ni.eval_ao(mol, coords, deriv=2, cutoff=grids.cutoff)
    xctype = ni._xc_type(source.xc)
    if xctype == 'HF':
        return np.zeros((len(atmlst), 3))
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        raise NotImplementedError(f'Unsupported source XC type {xctype}')
    spin = int(dm.ndim == 3)
    p = np.asarray(dm).reshape(-1, mol.nao, mol.nao)
    q = np.asarray(transition_dm).reshape(p.shape)
    slices = mol.aoslice_by_atom()[:, 2:]
    out = np.zeros((len(atmlst), 3))
    def timed_grids():
        iterator = iter(grid_iterator(grids))
        while True:
            t = perf_counter()
            ct = process_time()
            try:
                block = next(iterator)
            except StopIteration:
                parts['grid_partition'] += perf_counter()-t
                cpu_parts['grid_partition'] += process_time()-ct
                return
            parts['grid_partition'] += perf_counter()-t
            cpu_parts['grid_partition'] += process_time()-ct
            yield block

    for owner, (coords, weights, weight1) in enumerate(timed_grids()):
        for lo in range(0, len(weights), block_size):
            hi = min(lo + block_size, len(weights))
            t = perf_counter()
            ct = process_time()
            ao = eval_ao(coords[lo:hi])
            a = ao[:4]
            parts['ao_evaluation'] += perf_counter()-t
            cpu_parts['ao_evaluation'] += process_time()-ct
            t = perf_counter()
            ct = process_time()
            # P and Q are fixed throughout this block's nuclear-coordinate loop.
            ap = [a @ d for d in p]
            aq = [a @ d for d in q]
            parts['ao_density_products'] += perf_counter()-t
            cpu_parts['ao_density_products'] += process_time()-ct
            t = perf_counter()
            ct = process_time()
            rho = np.stack([_features(a, d, xctype=xctype, ad=ad) for d, ad in zip(p, ap)])
            rq = np.stack([_features(a, d, xctype=xctype, ad=ad) for d, ad in zip(q, aq)])
            rho_xc = rho if spin else rho[0]
            if xctype == 'LDA':
                rho_xc = rho_xc[:, 0] if spin else rho_xc[0]
            _, vxc, fxc, _ = ni.eval_xc_eff(source.xc, rho_xc, deriv=2,
                                           xctype=xctype, spin=spin)
            v = np.asarray(vxc).reshape(-1, hi-lo)
            f = np.asarray(fxc).reshape(v.shape[0], v.shape[0], hi-lo)
            rq = rq.reshape(v.shape)
            fq = lib.einsum('abg,bg->ag', f, rq)
            parts['density_and_xc'] += perf_counter()-t
            cpu_parts['density_and_xc'] += process_time()-ct
            t = perf_counter()
            ct = process_time()
            if grid_response:
                out += lib.einsum('nxg,g->nx', weight1[atmlst, :, lo:hi],
                                lib.einsum('ag,ag->g', v, rq))
            d = _ao_nuclear_terms(ao, ap, aq, fq, v, weights[lo:hi], xctype)
            moving_grid = d.sum(axis=1) if grid_response else np.zeros(3)
            for n, atom in enumerate(atmlst):
                p0, p1 = slices[atom]
                out[n] -= d[:, p0:p1].sum(axis=1)
                if atom == owner:
                    out[n] += moving_grid
            parts['nuclear_contractions'] += perf_counter()-t
            cpu_parts['nuclear_contractions'] += process_time()-ct
    if timings is not None:
        timings.update(parts)
        timings['total'] = perf_counter()-started
        timings.update({'cpu_'+k: value for k, value in cpu_parts.items()})
        timings['cpu_total'] = process_time()-cpu_started
    return out
