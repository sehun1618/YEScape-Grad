'''Real DF-HF MO-block response, shared by restricted and unrestricted CPUs.

Metric-whitened three-center integrals are transformed once per gradient call.
UHF shares the Coulomb response across spins and keeps exchange spin-local.
No AO response potential is formed by this provider; the solver still checks
its final physical residual independently with the ordinary AO provider.
'''

from time import perf_counter
import numpy as np
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger


class MOResponse:
    '''Cache OO/VO/VV DF factors for one geometry and supplied orbital set.'''

    def __init__(self, dfobj, Co, Cv, unrestricted=False):
        self.unrestricted = unrestricted
        self.Co = tuple(Co) if unrestricted else (Co,)
        self.Cv = tuple(Cv) if unrestricted else (Cv,)
        if any(np.iscomplexobj(c) for c in (*self.Co, *self.Cv)):
            raise ValueError('CPU MO-direct response requires real orbitals')
        self.shapes = [(v.shape[1], o.shape[1]) for o, v in zip(self.Co, self.Cv)]
        coeffs = [np.asarray(np.column_stack((o, v)), order='F')
                  for o, v in zip(self.Co, self.Cv)]
        self.blocks = []
        # ponytail: O(naux*nmo^2) in-memory cache; budget gate below selects AO
        # for larger systems. Disk-streamed MO factors would lift this ceiling.
        for packed in dfobj.loop(blksize=64):
            spin_blocks = []
            for c, (nv, no) in zip(coeffs, self.shapes):
                nm = no + nv
                b = _ao2mo.nr_e2(packed, c, (0, nm, 0, nm),
                                aosym='s2', mosym='s1').reshape(-1, nm, nm)
                spin_blocks.append((b[:, :no, :no].copy(),
                                    b[:, no:, :no].copy(), b[:, no:, no:].copy()))
                del b
            self.blocks.append(spin_blocks)
        self.cache_bytes = sum(a.nbytes for block in self.blocks
                               for spin in block for a in spin)

    def potential(self, xs, block):
        '''Return G[D]_vo or G[D]_oo, D_sigma=Cv X_sigma Co.T + transpose.'''
        out = [np.zeros(shape if block == 'vo' else (shape[1], shape[1]))
               for shape in self.shapes]
        for spins in self.blocks:
            rho = sum((lib.einsum('Lai,ai->L', vo, x)
                       if x.size else np.zeros(len(vo)))
                      for (_, vo, _), x in zip(spins, xs))
            for s, ((oo, vo, vv), x) in enumerate(zip(spins, xs)):
                if not out[s].size:
                    continue
                b = vo if block == 'vo' else oo
                out[s] += 2 * lib.einsum('Lpi,L->pi', b, rho)
                if not x.size:
                    continue
                if block == 'vo':
                    k = lib.einsum('Laj,Lji->ai',
                                   lib.einsum('Lab,bj->Laj', vv, x), oo)
                    k += lib.einsum('Laj,Lji->ai', vo,
                                    lib.einsum('bj,Lbi->Lji', x, vo))
                else:
                    q = lib.einsum('Lai,aj->Lij', vo, x)
                    k1 = lib.einsum('Lij,Ljk->ik', q, oo)
                    k = k1 + k1.T
                out[s] -= k * (1 if self.unrestricted else .5)
        return out

    def vind(self, x):
        '''Apply the two-electron VO Hessian in the upstream solver layout.'''
        x = np.asarray(x)
        if not x.size:
            return np.zeros_like(x)
        sizes = [nv * no for nv, no in self.shapes]
        out = []
        for row in x.reshape(-1, sum(sizes)):
            xs = [part.reshape(shape) for part, shape in
                  zip(np.split(row, np.cumsum(sizes)[:-1]), self.shapes)]
            v = self.potential(xs, 'vo')
            out.append(np.concatenate([a.ravel() for a in v]))
        return np.asarray(out).reshape(x.shape) * (1 if self.unrestricted else 2)

    def oo(self, z):
        blocks = self.potential(z if self.unrestricted else (z,), 'oo')
        return tuple(blocks) if self.unrestricted else blocks[0]


def make_response(grad, Co, Cv, unrestricted=False):
    '''Select DF response, with an explicit logged AO fallback on memory budget.'''
    grad.response_backend_used = None
    grad.response_mo_cache_bytes = 0
    grad.response_mo_setup_s = 0.
    if grad.response_backend == 'ao':
        grad.response_backend_used = 'ao'
        return None
    if grad.response_backend != 'mo_direct':
        raise ValueError("DF CPU response_backend must be 'ao' or 'mo_direct'")
    if hasattr(grad.base.mf_source, 'xc'):
        raise ValueError("DFA source requires response_backend='ao' to include fxc")
    occupied = Co if unrestricted else (Co,)
    virtual = Cv if unrestricted else (Cv,)
    if any(np.iscomplexobj(c) for c in (*occupied, *virtual)):
        raise ValueError('CPU MO-direct response requires real orbitals')
    dfobj = grad.base.mf_source.with_df
    sizes = [(o.shape[1], v.shape[1]) for o, v in zip(occupied, virtual)]
    cache = 8 * dfobj.get_naoaux() * sum(no*no + no*nv + nv*nv for no, nv in sizes)
    # Reserve transformation/contraction workspace as well as the retained factors.
    workspace = 8 * 64 * max((no+nv)**2 for no, nv in sizes) * 3
    available = max(0., grad.max_memory - lib.current_memory()[0]) * 1e6
    if cache + workspace > .8 * available:
        logger.warn(grad, 'MO-direct cache/workspace exceeds max_memory; using AO response')
        grad.response_backend_used = 'ao'
        return None
    t = perf_counter()
    response = MOResponse(dfobj, Co, Cv, unrestricted)
    grad.response_mo_setup_s = perf_counter() - t
    grad.response_mo_cache_bytes = response.cache_bytes
    grad.response_backend_used = 'mo_direct'
    logger.info(grad, 'MO-direct response cache %.3f MB; setup %.3f s',
                response.cache_bytes / 1e6, grad.response_mo_setup_s)
    return response
