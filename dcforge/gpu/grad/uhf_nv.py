# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''GPU spin-resolved nonvariational molecular UHF-DFT gradient terms.'''

import cupy as cp
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.lib.cupy_helper import contract

# S^lambda_oo/vo projection is spin-independent. UHF applies the same kernel
# separately to alpha and beta MO coefficients.
from dcforge.gpu.grad.rhf_nv import (
    _get_vhfopt,
)
from dcforge.gpu.grad.rhf_nv import (
    make_s1_oo_and_grad_ovlp_energy as _make_s1_oo_and_grad_ovlp_energy,
)


def make_s1_oo_and_grad_ovlp_energy(
    mol, Co, Cv, e_occ, zvec, s1, atmlst,
):
    '''Build both spin ``S^lambda_oo`` blocks and stream the VO term.

    This is the unrestricted production wrapper around the restricted fused
    helper. The overlap transformation itself is spin-independent, but the
    UHF energy term is a sum of two occupation-one spin contributions. The
    wrapper therefore applies the same atom-local transformation separately
    to alpha and beta and changes the restricted factor four to factor two per
    spin.

    It exists separately from the full spin-wise ``make_s1_oo_vo`` reference for
    the same reason as the RHF helper: production retains only both ``s1oo``
    tensors, whereas a reference caller may need the actual ``s1vo`` arrays.
    Making this distinction explicit prevents an accidental
    ``(2,3N,nvir_σ,nocc_σ)`` production allocation.

    Args:
        mol : pyscf.gto.Mole
            Molecule defining atom-local AO slices.
        Co, Cv : tuple of two cupy.ndarray
            Alpha/beta occupied and virtual source coefficients.
        e_occ : tuple of two cupy.ndarray
            Alpha/beta occupied source orbital energies.
        zvec : tuple of two cupy.ndarray
            Positive-RHS alpha/beta Z-vectors.
        s1 : cupy.ndarray of shape (3, nao, nao)
            One-center AO overlap derivative.
        atmlst : sequence of int
            Requested nuclear centers.

    Returns:
        s1oo : tuple of two cupy.ndarray
            Alpha/beta occupied overlap blocks.
        de_ovlp_energy : cupy.ndarray of shape (len(atmlst), 3)
            Immediately contracted spin-summed overlap-energy term.
    '''
    s1oo = []
    de = cp.zeros((len(atmlst), 3))
    for Co_s, Cv_s, e_s, z_s in zip(Co, Cv, e_occ, zvec):
        s1oo_s, de_s = _make_s1_oo_and_grad_ovlp_energy(
            mol, Co_s, Cv_s, e_s, z_s, s1, atmlst)
        s1oo.append(s1oo_s)
        # The restricted helper includes factor four.  Each unrestricted spin
        # channel carries factor two, hence one half of its returned term.
        de += .5 * de_s
    return tuple(s1oo), de


def grad_fock_make_h1(mf_grad, Co, Cv, zvec, mo_coeff, mo_occ,
                      atmlst):
    '''Return the GPU UHF source-Fock term through ``Hessian.make_h1``.

    GPU4PySCF returns one MO/occupied-column tensor per spin with shape
    ``(N_atom,3,nmo,nocc_σ)``. This reference route materializes both tensors,
    selects their VO rows, and contracts them with the two Z-vectors.

    Args:
        mf_grad : dcforge.gpu.grad.uhf_dft.Gradients
            Active GPU unrestricted gradient object.
        Co, Cv : tuple of two cupy.ndarray
            Alpha/beta occupied and virtual source-UHF MO coefficients.
        zvec : tuple of two cupy.ndarray
            Positive-RHS alpha/beta Z-vectors.
        mo_coeff : cupy.ndarray of shape (2, nao, nmo)
            Complete source-UHF MO coefficients.
        mo_occ : cupy.ndarray of shape (2, nmo)
            Exact unrestricted source occupations.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        cupy.ndarray of shape (len(atmlst), 3)
            ``-2 sum_σ Z^σ F^UHF,σ,lambda_vo`` in Ha/Bohr.
    '''
    mf_hess = mf_grad.base.mf_hf.Hessian()
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
    signature as :func:`dcforge.cpu.grad.uhf_nv.grad_fock_direct`. With
    ``P=P^alpha+P^beta`` and ``D_Z=D_Z^alpha+D_Z^beta``, its two-electron
    contribution is

    .. math::

        -D_Z:J^\lambda[P]
        +D_Z^\alpha:K^\lambda[P^\alpha]
        +D_Z^\beta:K^\lambda[P^\beta].

    Args:
        mf_grad : dcforge.gpu.grad.uhf_dft.Gradients
            Active conventional GPU unrestricted gradient object.
        Co, Cv : tuple of two cupy.ndarray
            Alpha/beta occupied and virtual source-UHF MO coefficients.
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
        The mathematical term belongs to the common GPU ``grad`` layer. The
        private conventional four-center kernel is guarded by the retained
        ``make_h1`` reference.
    '''
    mol = mf_grad.mol
    mf_hf = mf_grad.base.mf_hf
    if dm0.shape != (2, mol.nao, mol.nao):
        raise ValueError('dm0 must contain alpha and beta AO density matrices')
    if hasattr(mf_hf, 'with_df'):
        raise TypeError('conventional GPU UHF direct contraction received a DF source')
    if list(atmlst) != list(range(mol.natm)):
        raise NotImplementedError(
            'GPU direct source-UHF Fock contraction requires all atoms'
        )

    de_hcore = cp.zeros((mol.natm, 3))
    hcore_bytes = 0
    source_grad_factory = getattr(mf_hf, 'nuc_grad_method', None)
    source_grad = source_grad_factory() if callable(source_grad_factory) else None
    if source_grad is NotImplemented or source_grad is None:
        # GPU4PySCF 1.8.1 has GPU ROHF SCF but no ROHF gradient driver.  The
        # one-electron derivative is spin-independent, so use the GPU RHF
        # hcore provider without invoking any CPU gradient path.
        source_grad = rhf_grad.Gradients(mf_hf)
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
        _get_vhfopt(mf_hf),
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


def grad_ovlp_energy(s1vo, e_occ, zvec):
    '''Return ``+2 sum_σ Z^σ S^lambda,vo_σ epsilon^σ_o`` on the GPU.'''
    de = cp.zeros(s1vo[0].shape[:2])
    for s1vo_s, e_s, z_s in zip(s1vo, e_occ, zvec):
        se_vo = s1vo_s * e_s[None, None, None, :]
        de += 2 * contract('ai,kxai->kx', z_s.conj(), se_vo).real
    return de


def grad_ovlp_response_full(Co, Cv, zvec, s1oo, fvind):
    '''Return the UHF overlap response with the full perturbation batch.

    This ``3N`` implementation is a numerical reference for the production
    single-adjoint order and is not called by the default gradient driver.
    '''
    nset, ncart = s1oo[0].shape[:2]
    nao = Co[0].shape[0]
    dm1 = []
    for Co_s, s1oo_s in zip(Co, s1oo):
        tmp = contract('kxij,ui->kxuj', s1oo_s, Co_s)
        dm_s = contract('kxuj,vj->kxuv', tmp, Co_s.conj())
        dm1.append(dm_s + dm_s.transpose(0, 1, 3, 2).conj())
    dm1 = cp.stack(dm1).reshape(2, -1, nao, nao)
    v1ao = fvind(dm1)

    de = cp.zeros((nset, ncart))
    for Co_s, Cv_s, z_s, v1_s in zip(Co, Cv, zvec, v1ao):
        tmp = contract('Auv,ua->Aav', v1_s, Cv_s.conj())
        v1vo = contract('Aav,vi->Aai', tmp, Co_s)
        v1vo = v1vo.reshape(nset, ncart, Cv_s.shape[1], Co_s.shape[1])
        de += contract('ai,kxai->kx', z_s.conj(), v1vo).real
    return de


def get_z_response_oo(Co, Cv, zvec, fvind):
    '''Return ``(C_o^σ)^dagger G^UHF[D_Z]^σ C_o^σ`` for each spin σ.

    Unlike :func:`grad_ovlp_response_full`, this production provider applies
    the coupled UHF response to one alpha/beta Z-density pair rather than all
    ``3N`` geometry densities. It is kept distinct from the final
    ``grad_ovlp_response`` contraction so a DF subclass can replace only the
    AO-potential construction with an MO-direct implementation.
    '''
    dmz = []
    for Co_s, Cv_s, z_s in zip(Co, Cv, zvec):
        dz_s = Cv_s @ z_s @ Co_s.conj().T
        dmz.append(dz_s + dz_s.conj().T)

    vz = fvind(cp.stack(dmz))
    vzoo = []
    for Co_s, vz_s in zip(Co, vz):
        tmp = Co_s.conj().T @ vz_s
        vzoo.append(tmp @ Co_s)
    return tuple(vzoo)


def grad_ovlp_response(s1oo, vzoo):
    '''Contract the UHF overlap response from provider-owned OO blocks.

    The self-adjoint UHF response identity replaces ``3N`` pairs of
    alpha/beta perturbation densities by one pair of Z-transition densities.
    AO-to-MO projection is kept as two matrix multiplications per spin.
    '''
    de = cp.zeros(s1oo[0].shape[:2])
    for s1oo_s, vzoo_s in zip(s1oo, vzoo):
        de += contract('kxij,ij->kx', s1oo_s, vzoo_s).real
    return de


def nonvariational_correction(
    mf_grad, s1oo, de_fock, de_ovlp_energy, de_ovlp_response,
):
    '''Assemble the explicit GPU ``2 sum_σ Z^σ V^lambda_σ`` terms.

    Args:
        mf_grad : dcforge.gpu.grad.uhf_dft.Gradients
            Active gradient object receiving the host term ledger.
        s1oo : tuple of two cupy.ndarray
            Alpha/beta occupied overlap blocks retained from the single
            overlap transformation.
        de_fock : cupy.ndarray of shape (len(atmlst), 3)
            Precontracted spin-resolved source-Fock derivative term.
        de_ovlp_energy : cupy.ndarray of shape (len(atmlst), 3)
            Immediately contracted ``S^lambda_vo epsilon_o`` term.
        de_ovlp_response : cupy.ndarray of shape (len(atmlst), 3)
            Provider-independent adjoint overlap-response contraction.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Host unrestricted nonvariational correction in Ha/Bohr.
    '''
    mf_grad.s1oo = s1oo
    mf_grad.s1vo = None
    mf_grad.de_fock = cp.asnumpy(de_fock)
    mf_grad.de_ovlp_energy = cp.asnumpy(de_ovlp_energy)
    mf_grad.de_ovlp_response = cp.asnumpy(de_ovlp_response)
    mf_grad.de_nvar = (
        mf_grad.de_fock
        + mf_grad.de_ovlp_energy
        + mf_grad.de_ovlp_response
    )
    mf_grad.overlap_response_calls = 1
    mf_grad.overlap_response_nrhs = 1
    mf_grad.overlap_transform_calls = 1
    return mf_grad.de_nvar
