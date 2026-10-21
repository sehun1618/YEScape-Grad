# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''GPU RKS source Fock derivative at fixed AO density.

Q=2(Cv Z Co.T+h.c.), so -Q:F^lambda=-4 Z:F_vo^lambda.
Full-range exchange uses hyb and long-range exchange alpha-hyb.
XC includes the source mixed density/nuclear derivative, not target XC.
'''

import cupy as cp
from pyscf import df
from gpu4pyscf.grad import rhf, tdrhf
from gpu4pyscf.scf.jk import _VHFOpt
from gpu4pyscf.df.df_jk import _make_factorized_dm
from gpu4pyscf.df.grad import tdrhf as df_tdrhf
from dcforge.gpu.df.grad.memory import jk_energies_per_atom_bounded
from dcforge.gpu.source.xc import grad_xc_fock


def grad_fock(mf_grad, Co, Cv, zvec, dm0, atmlst):
    '''Contract the explicit RKS source derivative on the GPU.

    AO derivatives and contractions use GPU arrays. Native LibXC may use
    its host fallback for functionals unsupported by its CUDA library.
    '''
    if dm0.ndim != 2:
        raise ValueError('RKS source density has the wrong spin layout')
    if mf_grad.fock_deriv_method != 'direct':
        raise NotImplementedError('DFA source requires fock_deriv_method=direct')
    source, mol = mf_grad.base.mf_source, mf_grad.mol
    df_source = getattr(source, 'with_df', None)
    omega, alpha, hyb = source._numint.rsh_and_hybrid_coeff(source.xc, mol.spin)
    c = cp.concatenate((Co, Cv), axis=1)
    occ = cp.zeros(c.shape[1])
    occ[:Co.shape[1]] = 2
    h1 = rhf.get_grad_hcore(source.nuc_grad_method(), mo_coeff=c, mo_occ=occ)
    de_h = -4*cp.einsum('ai,kxai->kx', zvec, h1[:, :, Co.shape[1]:])
    if df_source is None:
        q = Cv @ zvec @ Co.T
        q = q+q.T
        p = 2*Co @ Co.T
    else:
        q = _make_factorized_dm(Cv @ zvec, Co, symmetrize=1)
        p = _make_factorized_dm(2*Co, Co, symmetrize=0)
    pairs, jf, kf = [[q, p]], [4.], [4.]

    def contract_jk(omega, jfactor, kfactor):
        # Retain density pairs with a nonzero J or K contribution.
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
    q = Cv @ zvec @ Co.T
    q = 2*(q+q.T)
    xc = -grad_xc_fock(source, dm0, q, atmlst,
                       grid_response=mf_grad.source_grid_response)
    mf_grad.de_source_hjk = cp.asnumpy(hjk)
    mf_grad.de_source_xc = cp.asnumpy(xc)
    mf_grad.fock_deriv_method_used = 'direct_source_ks'
    return hjk+xc
