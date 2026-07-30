'''Density-fitted GPU source-RHF Fock derivative contractions.'''

import cupy as cp
from gpu4pyscf.df.df_jk import _make_factorized_dm
from gpu4pyscf.df.grad import tdrhf as df_tdrhf_grad
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.lib.cupy_helper import contract
from pyscf import df


def grad_fock_direct(mf_grad, Co, Cv, zvec, dm0, atmlst):
    r'''Return the DF ``-4 Z F^HF,lambda_vo`` direct contraction.

    The public argument order mirrors the conventional GPU and CPU routes.
    Internally, exact occupied and Z-transition factors are passed to the
    GPU4PySCF three-center cross-density kernel, avoiding both ``make_h1`` and
    dense-density SVD factorization.

    Args:
        mf_grad : dcforge.gpu.df.grad.rhf_dft.Gradients
            Active density-fitted GPU restricted gradient object.
        Co : cupy.ndarray of shape (nao, nocc)
            Occupied source-RHF MO coefficients.
        Cv : cupy.ndarray of shape (nao, nvir)
            Virtual source-RHF MO coefficients.
        zvec : cupy.ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        dm0 : cupy.ndarray of shape (nao, nao)
            Closed-shell source density. Its shape is checked; exact low-rank
            factors are reconstructed from ``Co``.
        atmlst : sequence of int
            Nuclear centers included in the gradient. All atoms are required.

    Returns:
        cupy.ndarray of shape (N_atom, 3)
            Directly contracted DF source-Fock term in Ha/Bohr.
    '''
    mol = mf_grad.mol
    mf_hf = mf_grad.base.mf_hf
    if dm0.shape != (mol.nao, mol.nao):
        raise ValueError('dm0 must be one closed-shell AO density matrix')
    if not hasattr(mf_hf, 'with_df'):
        raise TypeError('GPU DF direct contraction requires a DF source')
    if list(atmlst) != list(range(mol.natm)):
        raise NotImplementedError(
            'GPU DF direct source-Fock contraction requires all atoms'
        )

    mo_coeff = cp.concatenate((Co, Cv), axis=1)
    mo_occ = cp.zeros(mo_coeff.shape[1])
    mo_occ[:Co.shape[1]] = 2
    hcore1 = rhf_grad.get_grad_hcore(
        mf_hf.nuc_grad_method(), mo_coeff=mo_coeff, mo_occ=mo_occ)
    de_hcore = -4 * contract(
        'ai,kxai->kx', zvec.conj(), hcore1[:, :, Co.shape[1]:, :]
    ).real

    z_left = Cv @ zvec
    dz = _make_factorized_dm(z_left, Co, symmetrize=1)
    dm0_factor = _make_factorized_dm(Co * 2.0, Co, symmetrize=0)
    auxmol = mf_hf.with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(
            mol, auxbasis=mf_hf.with_df.auxbasis)
    int3c2e_opt = df_tdrhf_grad.Int3c2eOpt(mol, auxmol).build()
    de_jk = df_tdrhf_grad._jk_energies_per_atom(
        int3c2e_opt,
        [[dz, dm0_factor]],
        j_factor=[4.0],
        k_factor=[4.0],
        sum_results=True,
        verbose=mf_grad.verbose,
    )

    de_jk = -cp.asarray(de_jk)
    mf_grad.de_fock_hcore = cp.asnumpy(de_hcore)
    mf_grad.de_fock_jk = cp.asnumpy(de_jk)
    mf_grad.fock_deriv_intermediate_bytes = (
        hcore1.nbytes + z_left.nbytes + dz.nbytes + dm0_factor.nbytes
    )
    return de_hcore + de_jk
