# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''RHF source Fock nuclear derivatives; SCF remains upstream.'''

import cupy as cp
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf.jk import _VHFOpt


def _get_vhfopt(mf_source):
    '''Return the conventional four-center GPU J/K optimizer.'''
    vhfopt = mf_source._opt_gpu.get(None)
    if vhfopt is None:
        vhfopt = mf_source._opt_gpu[None] = _VHFOpt(
            mf_source.mol, mf_source.direct_scf_tol).build()
    return vhfopt


def grad_fock_make_h1(mf_grad, Co, Cv, zvec, mo_coeff, mo_occ,
                      atmlst):
    '''Return the source-Fock term through GPU4PySCF ``make_h1``.

    This reference route materializes ``(N_atom,3,nmo,nocc)`` before selecting
    its VO rows.  It remains available as the numerical reference for the DF
    direct contracted implementation.

    The tensor is the explicit source-HF operator derivative at the fixed
    reference density, not the total self-consistent derivative including
    the induced density response.  Overlap and response terms are assembled
    separately in the common gradient layer.

    Args:
        mf_grad : dcforge.gpu.grad.rnsc.Gradients
            Active GPU gradient object.
        Co : cupy.ndarray of shape (nao, nocc)
            Occupied source MO coefficients.
        Cv : cupy.ndarray of shape (nao, nvir)
            Virtual source MO coefficients.
        zvec : cupy.ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        mo_coeff : cupy.ndarray of shape (nao, nmo)
            Complete source MO coefficients.
        mo_occ : cupy.ndarray of shape (nmo,)
            Exact restricted source occupations.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        cupy.ndarray of shape (len(atmlst), 3)
            The ``-4 Z F^HF,lambda_vo`` contribution in Ha/Bohr.
    '''
    mf_hess = mf_grad.base.mf_source.Hessian()
    if hasattr(mf_hess, 'auxbasis_response'):
        mf_hess.auxbasis_response = 2 * int(mf_grad.auxbasis_response)
    h1mo = cp.asarray(mf_hess.make_h1(mo_coeff, mo_occ))
    mf_grad.fock_deriv_intermediate_bytes = h1mo.nbytes
    f1vo = h1mo[atmlst][:, :, mo_occ == 0, :]
    return -4 * contract('ai,kxai->kx', zvec.conj(), f1vo).real


def grad_fock_direct(mf_grad, Co, Cv, zvec, dm0, atmlst):
    r'''Return the conventional ``-4 Z F^HF,lambda_vo`` contraction.

    The Hermitian transition density ``D_Z=C_v Z C_o^dagger+h.c.`` is paired
    with the source density in GPU4PySCF's four-center derivative J/K kernel.
    The complete ``(N_atom,3,nmo,nocc)`` first-order Fock is never built.

    Args:
        mf_grad : dcforge.gpu.grad.rnsc.Gradients
            Active conventional GPU restricted gradient object.
        Co : cupy.ndarray of shape (nao, nocc)
            Occupied source MO coefficients.
        Cv : cupy.ndarray of shape (nao, nvir)
            Virtual source MO coefficients.
        zvec : cupy.ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        dm0 : cupy.ndarray of shape (nao, nao)
            Closed-shell source-HF density.
        atmlst : sequence of int
            Nuclear centers included in the gradient. All atoms are required.

    Returns:
        cupy.ndarray of shape (N_atom, 3)
            Directly contracted source-Fock term in Ha/Bohr.

    Notes:
        ``_jk_energies_per_atom`` is a private GPU4PySCF 1.8.1 API. The
        ``make_h1`` reference guards its cross-density coefficients.
    '''
    mol = mf_grad.mol
    mf_source = mf_grad.base.mf_source
    if dm0.shape != (mol.nao, mol.nao):
        raise ValueError('dm0 must be one closed-shell AO density matrix')
    if hasattr(mf_source, 'with_df'):
        raise TypeError('conventional GPU direct contraction received a DF source')
    if list(atmlst) != list(range(mol.natm)):
        raise NotImplementedError(
            'GPU direct source-Fock contraction requires all atoms'
        )

    # Reconstruct the canonical occupied-then-virtual arrays expected by
    # GPU4PySCF. Keeping them local makes the public CPU/GPU term signature
    # identical without changing the optimized hcore projection.
    mo_coeff = cp.concatenate((Co, Cv), axis=1)
    mo_occ = cp.zeros(mo_coeff.shape[1])
    mo_occ[:Co.shape[1]] = 2
    hcore1 = rhf_grad.get_grad_hcore(
        mf_source.nuc_grad_method(),
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
    )
    de_hcore = -4 * contract(
        'ai,kxai->kx', zvec.conj(), hcore1[:, :, Co.shape[1]:, :]
    ).real

    z_left = Cv @ zvec
    dz = z_left @ Co.conj().T
    dz += dz.conj().T
    de_jk = tdrhf_grad._jk_energies_per_atom(
        _get_vhfopt(mf_source),
        [[dz, dm0]],
        j_factor=[4.0],
        k_factor=[4.0],
        sum_results=True,
        verbose=mf_grad.verbose,
    )

    # The cross kernel returns +4 Z:(J^lambda-K^lambda/2)_vo.
    de_jk = -cp.asarray(de_jk)
    mf_grad.de_fock_hcore = cp.asnumpy(de_hcore)
    mf_grad.de_fock_jk = cp.asnumpy(de_jk)
    mf_grad.fock_deriv_intermediate_bytes = (
        hcore1.nbytes + z_left.nbytes + dz.nbytes
    )
    return de_hcore + de_jk
