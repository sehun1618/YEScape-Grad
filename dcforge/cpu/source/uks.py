# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''UKS source Fock derivative at fixed AO density.

For σ in {α, β}, Q_σ=Cv_σ Z_σ Co_σ.T+h.c., so -sum_σ Tr[Q_σ F_σ^lambda]
=-2 sum_σ Z_σ:F_σ,vo^lambda. XC includes inter-spin kernel coupling.
Full-range exchange uses hyb and long-range exchange alpha-hyb.
Source XC includes AO-center/grid motion but not orbital relaxation.
'''

import numpy as np
from pyscf import lib
from dcforge.cpu.source import uhf
from dcforge.cpu.source.xc import grad_xc_fock


def grad_fock(mf_grad, Co, Cv, zvec, dm0, atmlst):
    '''Contract the explicit UKS source Fock derivative with the Z density.

    Only a fresh gradient instance is adapted. Both DF polarization orders
    and auxiliary derivative tags are preserved; SCF methods are unchanged.
    '''
    if dm0.ndim != 3 or dm0.shape[0] != 2:
        raise ValueError('UKS source density has the wrong spin layout')
    source = mf_grad.base.mf_source
    if mf_grad.fock_deriv_method != 'direct':
        raise NotImplementedError('DFA source requires fock_deriv_method=direct; '
                                  'native make_h1 omits full moving-grid response')
    grad = source.nuc_grad_method()
    grad.auxbasis_response = getattr(mf_grad, 'auxbasis_response', True)
    omega, alpha, hyb = source._numint.rsh_and_hybrid_coeff(source.xc, source.mol.spin)
    get_jk, get_j, get_k = grad.get_jk, grad.get_j, grad.get_k
    lr_coeff = alpha-hyb if omega != 0 else 0.
    with_exchange = hyb != 0 or lr_coeff != 0

    def exchange(mol, dm, hermi):
        # DF get_k calls self.get_jk. Use the captured native method to avoid
        # recursing into weighted_jk after the instance adapter is installed.
        if hasattr(source, 'with_df'):
            return get_jk(mol, dm, hermi=hermi, with_j=False)[1]
        return get_k(mol, dm, hermi=hermi)

    def weighted_jk(mol, dm, hermi=1):
        kdm = dm[2:]
        vj = get_j(mol, dm[:2], hermi=hermi)
        raw = exchange(mol, kdm, hermi) if hyb != 0 else None
        vk = np.asarray(raw)*hyb if raw is not None else None
        aux = np.asarray(raw.aux)*hyb if hasattr(raw, 'aux') else None
        if lr_coeff != 0:
            # Retain the native combined J/K path for long-range exchange.
            _, lr = get_jk(mol, kdm, hermi=hermi, omega=omega)
            vk = lr_coeff*np.asarray(lr) if vk is None else vk+lr_coeff*np.asarray(lr)
            if hasattr(lr, 'aux'):
                aux = lr_coeff*np.asarray(lr.aux) if aux is None else aux+lr_coeff*np.asarray(lr.aux)
        # Keep the shared HF contraction's index layout, but evaluate J
        # for only the two total densities and K for four spin densities.
        # Zero padding is bookkeeping, not an additional integral call.
        jfull = np.zeros((len(dm),)+vj.shape[1:])
        kfull = np.zeros((len(dm),)+vk.shape[1:])
        jfull[:2], kfull[2:] = vj, vk
        if hasattr(vj, 'aux'):
            jaux = np.zeros((len(dm), len(dm))+vj.aux.shape[2:])
            kaux = np.zeros_like(jaux)
            jaux[:2, :2], kaux[2:, 2:] = vj.aux, aux
            jfull = lib.tag_array(jfull, aux=jaux)
            aux = kaux
        vj, vk = jfull, kfull
        return vj, lib.tag_array(vk, aux=aux) if aux is not None else vk

    grad.get_jk = weighted_jk
    hjk = uhf.grad_fock_direct(grad, Co, Cv, zvec, dm0, atmlst,
                             with_k=with_exchange)
    q = np.asarray([v @ z @ o.T for o, v, z in zip(Co, Cv, zvec)])
    q = q + q.transpose(0, 2, 1)
    xc = -grad_xc_fock(source, dm0, q, atmlst,
                       grid_response=mf_grad.source_grid_response)
    mf_grad.de_source_hjk, mf_grad.de_source_xc = hjk, xc
    mf_grad.fock_deriv_method_used = 'direct_source_ks'
    mf_grad.fock_deriv_intermediate_bytes = grad.fock_deriv_intermediate_bytes
    return hjk + xc
