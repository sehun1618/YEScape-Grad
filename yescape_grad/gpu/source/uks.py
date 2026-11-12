# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''GPU UKS source Fock derivative at fixed AO density.

For σ in {α, β}, Q_σ=Cv_σ Z_σ Co_σ.T+h.c., so -sum_σ Q_σ:F_σ^lambda
=-2 sum_σ Z_σ:F_σ,vo^lambda. Coulomb uses total density, exchange same spin.
Full-range exchange uses hyb and long-range exchange alpha-hyb.
XC includes the source mixed density/nuclear derivative, not target XC.
'''

import cupy as cp
from pyscf import df
from gpu4pyscf.grad import rhf, tdrhf
from gpu4pyscf.scf.jk import _VHFOpt
from gpu4pyscf.df.df_jk import _make_factorized_dm
from gpu4pyscf.df.grad import tdrhf as df_tdrhf
from yescape_grad.gpu.df.grad.memory import jk_energies_per_atom_bounded
from yescape_grad.gpu.source.xc import grad_xc_fock


def grad_fock(mf_grad, Co, Cv, zvec, dm0, atmlst):
    '''Contract the explicit UKS source derivative on the GPU.

    AO derivatives and contractions use GPU arrays. Native LibXC may use
    its host fallback for functionals unsupported by its CUDA library.
    '''
    if dm0.ndim != 3 or dm0.shape[0] != 2:
        raise ValueError('UKS source density has the wrong spin layout')
    if mf_grad.fock_deriv_method != 'direct':
        raise NotImplementedError('DFA source requires fock_deriv_method=direct')
    source, mol = mf_grad.base.mf_source, mf_grad.mol
    df_source = getattr(source, 'with_df', None)
    omega, alpha, hyb = source._numint.rsh_and_hybrid_coeff(source.xc, mol.spin)
    with_exchange = hyb != 0 or (omega != 0 and alpha != hyb)
    de_h = cp.zeros((mol.natm, 3))
    qs, ps = [], []
    for co, cv, z in zip(Co, Cv, zvec):
        c = cp.concatenate((co, cv), axis=1)
        occ = cp.zeros(c.shape[1])
        occ[:co.shape[1]] = 1
        h1 = rhf.get_grad_hcore(source.nuc_grad_method(), mo_coeff=c, mo_occ=occ)
        # Hcore is spin independent; the native RHF helper accepts occupation 1.
        de_h -= 2*cp.einsum('ai,kxai->kx', z, h1[:, :, co.shape[1]:])
        if not with_exchange:
            continue  # No spin-specific exchange densities for pure UKS.
        if df_source is None:
            q = cv @ z @ co.T
            qs.append(q+q.T)
            ps.append(co @ co.T)
        else:
            qs.append(_make_factorized_dm(cv @ z, co, symmetrize=1))
            ps.append(_make_factorized_dm(co, co, symmetrize=0))
    co = cp.concatenate(Co, axis=1)
    left = cp.concatenate([v @ z for v, z in zip(Cv, zvec)], axis=1)
    if df_source is None:
        qtotal = left @ co.T
        qtotal = qtotal + qtotal.T
        ptotal = co @ co.T
    else:
        qtotal = _make_factorized_dm(left, co, symmetrize=1)
        ptotal = _make_factorized_dm(co, co, symmetrize=0)
    pairs, jf, kf = [[qtotal, ptotal]], [2.], [0.]
    if with_exchange:
        pairs += [[qs[0], ps[0]], [qs[1], ps[1]]]
        jf += [0., 0.]
        kf += [4., 4.]

    def contract_jk(omega, jfactor, kfactor):
        # Remove the total-J pair from the LR exchange contribution,
        # and exchange pairs from the full-range J of pure long-range hybrids.
        # Zero J coefficients do not request a dedicated DF K-only backend.
        active = [i for i, (j, k) in enumerate(zip(jfactor, kfactor)) if j != 0 or k != 0]
        active_pairs = [pairs[i] for i in active]
        jfactor, kfactor = [jfactor[i] for i in active], [kfactor[i] for i in active]
        if df_source is None:
            opt = source._opt_gpu.get(omega)
            if opt is None:
                with mol.with_range_coulomb(omega):
                    opt = _VHFOpt(mol, source.direct_scf_tol).build()
                source._opt_gpu[omega] = opt
            return tdrhf._jk_energies_per_atom(
                opt, active_pairs, j_factor=jfactor, k_factor=kfactor,
                sum_results=True, verbose=mf_grad.verbose)
        auxmol = df_source.auxmol
        if auxmol is None:
            auxmol = df.addons.make_auxmol(mol, df_source.auxbasis)
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            opt = df_tdrhf.Int3c2eOpt(mol, auxmol).build()
        return jk_energies_per_atom_bounded(
            mf_grad, opt, active_pairs, j_factor=jfactor, k_factor=kfactor,
            sum_results=True, verbose=mf_grad.verbose)

    hjk = de_h-cp.asarray(contract_jk(0., jf, [hyb*k for k in kf]))
    if omega != 0 and alpha != hyb:
        hjk -= cp.asarray(contract_jk(omega, [0.]*len(jf),
                                      [(alpha-hyb)*k for k in kf]))
    q = cp.stack([v @ z @ o.T for o, v, z in zip(Co, Cv, zvec)])
    q = q+q.transpose(0, 2, 1)
    xc = -grad_xc_fock(source, dm0, q, atmlst,
                       grid_response=mf_grad.source_grid_response)
    mf_grad.de_source_hjk = cp.asnumpy(hjk)
    mf_grad.de_source_xc = cp.asnumpy(xc)
    mf_grad.fock_deriv_method_used = 'direct_source_ks'
    return hjk+xc
