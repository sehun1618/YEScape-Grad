# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''UHF source Fock nuclear derivatives; SCF remains upstream.'''

import cupy as cp
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.lib.cupy_helper import contract
from yescape_grad.gpu.source.rhf import _get_vhfopt


def grad_fock_make_h1(mf_grad, Co, Cv, zvec, mo_coeff, mo_occ,
                      atmlst):
    '''Return the GPU UHF source-Fock term through ``Hessian.make_h1``.

    GPU4PySCF returns one MO/occupied-column tensor per spin with shape
    ``(N_atom,3,nmo,nocc_σ)``. This reference route materializes both tensors,
    selects their VO rows, and contracts them with the two Z-vectors.

    Args:
        mf_grad : yescape_grad.gpu.grad.unsc.Gradients
            Active GPU unrestricted gradient object.
        Co, Cv : tuple of two cupy.ndarray
            Alpha/beta occupied and virtual source MO coefficients.
        zvec : tuple of two cupy.ndarray
            Positive-RHS alpha/beta Z-vectors.
        mo_coeff : cupy.ndarray of shape (2, nao, nmo)
            Complete source MO coefficients.
        mo_occ : cupy.ndarray of shape (2, nmo)
            Exact unrestricted source occupations.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        cupy.ndarray of shape (len(atmlst), 3)
            ``-2 sum_σ Z^σ F^UHF,σ,lambda_vo`` in Ha/Bohr.
    '''
    mf_hess = mf_grad.base.mf_source.Hessian()
    if hasattr(mf_hess, 'auxbasis_response'):
        mf_hess.auxbasis_response = 2 * int(mf_grad.auxbasis_response)
    h1mo = mf_hess.make_h1(mo_coeff, mo_occ, atmlst=atmlst)
    mf_grad.fock_deriv_intermediate_bytes = sum(x.nbytes for x in h1mo)

    de = cp.zeros((len(atmlst), 3))
    for z_s, h1_s, occ_s in zip(zvec, h1mo, mo_occ):
        f1vo = h1_s[atmlst][:, :, occ_s == 0, :]
        de -= 2 * contract('ai,kxai->kx', z_s.conj(), f1vo).real
    return de


def grad_fock_direct(mf_grad, Co, Cv, zvec, dm0, atmlst):
    r'''Return the conventional GPU UHF source-Fock direct contraction.

    This function has the same mathematical responsibility and public
    signature as :func:`yescape_grad.cpu.source.uhf.grad_fock_direct`. With
    ``P=P^alpha+P^beta`` and ``D_Z=D_Z^alpha+D_Z^beta``, its two-electron
    contribution is

    .. math::

        -D_Z:J^\lambda[P]
        +D_Z^\alpha:K^\lambda[P^\alpha]
        +D_Z^\beta:K^\lambda[P^\beta].

    Args:
        mf_grad : yescape_grad.gpu.grad.unsc.Gradients
            Active conventional GPU unrestricted gradient object.
        Co, Cv : tuple of two cupy.ndarray
            Alpha/beta occupied and virtual source MO coefficients.
        zvec : tuple of two cupy.ndarray
            Positive-RHS alpha/beta Z-vectors.
        dm0 : cupy.ndarray of shape (2, nao, nao)
            Spin-resolved source-UHF densities.
        atmlst : sequence of int
            Nuclear centers included in the gradient. All atoms are required.

    Returns:
        cupy.ndarray of shape (N_atom, 3)
            Directly contracted source-UHF Fock term in Ha/Bohr.

    Notes:
        The source-specific term is assembled by the common GPU gradient driver. The
        private conventional four-center kernel is guarded by the retained
        ``make_h1`` reference.
    '''
    mol = mf_grad.mol
    mf_source = mf_grad.base.mf_source
    if dm0.shape != (2, mol.nao, mol.nao):
        raise ValueError('dm0 must contain alpha and beta AO density matrices')
    if hasattr(mf_source, 'with_df'):
        raise TypeError('conventional GPU UHF direct contraction received a DF source')
    if list(atmlst) != list(range(mol.natm)):
        raise NotImplementedError(
            'GPU direct source-UHF Fock contraction requires all atoms'
        )

    de_hcore = cp.zeros((mol.natm, 3))
    hcore_bytes = 0
    source_grad_factory = getattr(mf_source, 'nuc_grad_method', None)
    source_grad = source_grad_factory() if callable(source_grad_factory) else None
    if source_grad is NotImplemented or source_grad is None:
        # GPU4PySCF 1.8.1 has GPU ROHF SCF but no ROHF gradient driver.  The
        # one-electron derivative is spin-independent, so use the GPU RHF
        # hcore provider without invoking any CPU gradient path.
        source_grad = rhf_grad.Gradients(mf_source)
    for z_s, Co_s, Cv_s in zip(zvec, Co, Cv):
        coeff_s = cp.concatenate((Co_s, Cv_s), axis=1)
        occ_s = cp.zeros(coeff_s.shape[1])
        occ_s[:Co_s.shape[1]] = 1
        hcore1_s = rhf_grad.get_grad_hcore(
            source_grad, mo_coeff=coeff_s, mo_occ=occ_s)
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

    dza = za_left @ Coa.conj().T
    dza += dza.conj().T
    dzb = zb_left @ Cob.conj().T
    dzb += dzb.conj().T
    dz = dza + dzb
    dm = dm0[0] + dm0[1]

    de_jk = tdrhf_grad._jk_energies_per_atom(
        _get_vhfopt(mf_source),
        [[dz, dm], [dza, dm0[0]], [dzb, dm0[1]]],
        j_factor=[2.0, 0.0, 0.0],
        k_factor=[0.0, 4.0, 4.0],
        sum_results=True,
        verbose=mf_grad.verbose,
    )

    # The cross kernel returns +D_Z:J[P] - sum_σ D_Z^σ:K[P^σ].
    de_jk = -cp.asarray(de_jk)
    mf_grad.de_fock_hcore = cp.asnumpy(de_hcore)
    mf_grad.de_fock_jk = cp.asnumpy(de_jk)
    mf_grad.fock_deriv_intermediate_bytes = (
        hcore_bytes + za_left.nbytes + zb_left.nbytes
        + dza.nbytes + dzb.nbytes + dz.nbytes + dm.nbytes
    )
    return de_hcore + de_jk
