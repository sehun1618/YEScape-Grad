# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''RKS source Fock derivative at fixed AO density.

Q=2(Cv Z Co.T+h.c.), so -Tr[Q F^lambda]=-4 Z:F_vo^lambda.
Full-range exchange uses hyb and long-range exchange alpha-hyb.
Source XC includes AO-center/grid motion but not orbital relaxation.
'''

import numpy as np
from pyscf import lib
from yescape_grad.tools.logger import gradient_stage, source_xc_timing
from yescape_grad.cpu.source import rhf
from yescape_grad.cpu.source.xc import grad_xc_fock


def grad_fock(mf_grad, Co, Cv, zvec, dm0, atmlst):
    '''Contract the explicit RKS source Fock derivative with the Z density.

    Only a fresh gradient instance is adapted. Both DF polarization orders
    and auxiliary derivative tags are preserved; SCF methods are unchanged.
    '''
    if dm0.ndim != 2:
        raise ValueError('RKS source density has the wrong spin layout')
    source = mf_grad.base.mf_source
    if mf_grad.fock_deriv_method != 'direct':
        raise NotImplementedError('DFA source requires fock_deriv_method=direct; '
                                  'native make_h1 omits full moving-grid response')
    grad = source.nuc_grad_method()
    grad.auxbasis_response = getattr(mf_grad, 'auxbasis_response', True)
    omega, alpha, hyb = source._numint.rsh_and_hybrid_coeff(source.xc, source.mol.spin)
    get_jk, get_j = grad.get_jk, grad.get_j
    lr_coeff = alpha-hyb if omega != 0 else 0.
    with_exchange = hyb != 0 or lr_coeff != 0

    def weighted_jk(mol, dm, hermi=1):
        if hyb != 0:
            vj, raw = get_jk(mol, dm, hermi=hermi)
        else:
            vj, raw = get_j(mol, dm, hermi=hermi), None
        vk = np.asarray(raw)*hyb if raw is not None else None
        aux = np.asarray(raw.aux)*hyb if hasattr(raw, 'aux') else None
        if lr_coeff != 0:
            # Retain the native combined J/K path for long-range exchange.
            _, lr = get_jk(mol, dm, hermi=hermi, omega=omega)
            vk = lr_coeff*np.asarray(lr) if vk is None else vk+lr_coeff*np.asarray(lr)
            if hasattr(lr, 'aux'):
                aux = lr_coeff*np.asarray(lr.aux) if aux is None else aux+lr_coeff*np.asarray(lr.aux)
        return vj, lib.tag_array(vk, aux=aux) if aux is not None else vk

    grad.get_jk = weighted_jk
    with gradient_stage(mf_grad, 'Source one-electron and J/K derivative'):
        hjk = rhf.grad_fock_direct(grad, Co, Cv, zvec, dm0, atmlst,
                                 with_k=with_exchange)
    q = Cv @ zvec @ Co.T
    q = 2*(q+q.T)
    mf_grad.source_xc_stage_times = {}
    with gradient_stage(mf_grad, 'Source XC derivative (AO, kernel and moving grid)'):
        xc = -grad_xc_fock(source, dm0, q, atmlst,
                           grid_response=mf_grad.source_grid_response,
                           grid_response_backend=mf_grad.grid_response_backend,
                           timings=mf_grad.source_xc_stage_times)
    source_xc_timing(mf_grad)
    mf_grad.source_grid_response_backend_used = mf_grad.grid_response_backend if mf_grad.source_grid_response else 'inactive'
    mf_grad.de_source_hjk, mf_grad.de_source_xc = hjk, xc
    mf_grad.fock_deriv_method_used = 'direct_source_ks'
    mf_grad.fock_deriv_intermediate_bytes = grad.fock_deriv_intermediate_bytes
    return hjk + xc
