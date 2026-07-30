'''Density-fitted GPU source-UHF Fock derivative contractions.'''

import cupy as cp
from gpu4pyscf.df.df_jk import _make_factorized_dm
from gpu4pyscf.df.grad import tdrhf as df_tdrhf_grad
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.lib.cupy_helper import contract
from pyscf import df


def grad_fock_direct(mf_grad, Co, Cv, zvec, dm0, atmlst):
    r'''Return the DF-UHF source-Fock term by direct contraction.

    One total-spin Coulomb pair and two same-spin exchange pairs evaluate

    .. math::

        -D_Z:J^\lambda[P]
        +D_Z^\alpha:K^\lambda[P^\alpha]
        +D_Z^\beta:K^\lambda[P^\beta]

    without materializing spin-resolved first-order Fock tensors.

    Args:
        mf_grad : dcforge.gpu.df.grad.uhf_dft.Gradients
            Active density-fitted GPU unrestricted gradient object.
        Co, Cv : tuple of two cupy.ndarray
            Alpha/beta occupied and virtual source-UHF coefficients.
        zvec : tuple of two cupy.ndarray
            Positive-RHS alpha/beta Z-vectors.
        dm0 : cupy.ndarray of shape (2, nao, nao)
            Spin source densities. Exact low-rank factors are reconstructed
            from ``Co`` rather than recovered from these dense matrices.
        atmlst : sequence of int
            Nuclear centers included in the gradient. All atoms are required.

    Returns:
        cupy.ndarray of shape (N_atom, 3)
            Directly contracted DF-UHF source-Fock term in Ha/Bohr.
    '''
    mol = mf_grad.mol
    mf_hf = mf_grad.base.mf_hf
    if dm0.shape != (2, mol.nao, mol.nao):
        raise ValueError('dm0 must contain alpha and beta AO density matrices')
    if not hasattr(mf_hf, 'with_df'):
        raise TypeError('GPU DF-UHF direct contraction requires a DF source')
    if list(atmlst) != list(range(mol.natm)):
        raise NotImplementedError(
            'GPU DF-UHF direct source-Fock contraction requires all atoms'
        )

    de_hcore = cp.zeros((mol.natm, 3))
    hcore_bytes = 0
    for z_s, Co_s, Cv_s in zip(zvec, Co, Cv):
        coeff_s = cp.concatenate((Co_s, Cv_s), axis=1)
        occ_s = cp.zeros(coeff_s.shape[1])
        occ_s[:Co_s.shape[1]] = 1
        hcore1_s = rhf_grad.get_grad_hcore(
            mf_hf.nuc_grad_method(), mo_coeff=coeff_s, mo_occ=occ_s)
        de_hcore -= 2 * contract(
            'ai,kxai->kx', z_s.conj(),
            hcore1_s[:, :, Co_s.shape[1]:, :],
        ).real
        hcore_bytes += hcore1_s.nbytes

    Coa, Cob = Co
    Cva, Cvb = Cv
    za, zb = zvec
    za_left = Cva @ za
    zb_left = Cvb @ zb
    dza = _make_factorized_dm(za_left, Coa, symmetrize=1)
    dzb = _make_factorized_dm(zb_left, Cob, symmetrize=1)
    pa = _make_factorized_dm(Coa, Coa, symmetrize=0)
    pb = _make_factorized_dm(Cob, Cob, symmetrize=0)

    Co_total = cp.concatenate((Coa, Cob), axis=1)
    z_left = cp.concatenate((za_left, zb_left), axis=1)
    dz = _make_factorized_dm(z_left, Co_total, symmetrize=1)
    dm = _make_factorized_dm(Co_total, Co_total, symmetrize=0)

    auxmol = mf_hf.with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(
            mol, auxbasis=mf_hf.with_df.auxbasis)
    int3c2e_opt = df_tdrhf_grad.Int3c2eOpt(mol, auxmol).build()
    de_jk = df_tdrhf_grad._jk_energies_per_atom(
        int3c2e_opt,
        [[dz, dm], [dza, pa], [dzb, pb]],
        j_factor=[2.0, 0.0, 0.0],
        k_factor=[0.0, 4.0, 4.0],
        sum_results=True,
        verbose=mf_grad.verbose,
    )

    de_jk = -cp.asarray(de_jk)
    mf_grad.de_fock_hcore = cp.asnumpy(de_hcore)
    mf_grad.de_fock_jk = cp.asnumpy(de_jk)
    mf_grad.fock_deriv_intermediate_bytes = (
        hcore_bytes + za_left.nbytes + zb_left.nbytes
        + dza.nbytes + dzb.nbytes + pa.nbytes + pb.nbytes
        + dz.nbytes + dm.nbytes
    )
    return de_hcore + de_jk
