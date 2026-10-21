'''Check the documented overlap-adjoint signs and spin factors independently.'''

import numpy as np
from dcforge.cpu.grad import r_nv, u_nv


def test_overlap_adjoint_identity():
    rng = np.random.default_rng(23)
    coeff = [np.linalg.qr(rng.normal(size=(5, 5)))[0] for _ in range(2)]
    co = [coeff[0][:, :2], coeff[1][:, :1]]
    cv = [coeff[0][:, 2:], coeff[1][:, 1:]]
    z = [rng.normal(size=(v.shape[1], o.shape[1])) for o, v in zip(co, cv)]
    s = [rng.normal(size=(2, 3, o.shape[1], o.shape[1])) for o in co]
    s = [a + a.swapaxes(-1, -2) for a in s]
    # Self-adjoint AO response with nonzero inter-spin coupling.
    coupling = np.array([[1.3, .4], [.4, .8]])
    response_u = lambda dm: np.einsum('st,t...ij->s...ij', coupling, dm)
    p_oo = np.asarray([-np.einsum('pi,kxij,qj->kxpq', o, a, o)
                       for o, a in zip(co, s)])
    v = response_u(p_oo)
    direct = sum(-2*np.einsum('ai,pa,kxpq,qi->kx', zz, vv, pot, oo)
                 for zz, vv, pot, oo in zip(z, cv, v, co))
    np.testing.assert_allclose(u_nv.grad_ovlp_response(co, cv, z, s, response_u),
                               direct, atol=1e-12)
    response_r = lambda dm: 1.3*dm + .4*np.swapaxes(dm, -1, -2)
    v = response_r(2*p_oo[0])
    direct = -4*np.einsum('ai,pa,kxpq,qi->kx', z[0], cv[0], v, co[0])
    np.testing.assert_allclose(r_nv.grad_ovlp_response(co[0], cv[0], z[0], s[0], response_r),
                               direct, atol=1e-12)
    np.testing.assert_allclose(r_nv.grad_ovlp_response_full(co[0], cv[0], z[0], s[0], response_r),
                               direct, atol=1e-12)
