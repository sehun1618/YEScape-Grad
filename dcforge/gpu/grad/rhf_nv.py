# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''GPU nonvariational ``4 Z^dagger V^lambda`` gradient terms.'''

import cupy as cp
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf.jk import _VHFOpt


def _get_vhfopt(mf_hf):
    '''Return the conventional four-center GPU J/K optimizer.'''
    vhfopt = mf_hf._opt_gpu.get(None)
    if vhfopt is None:
        vhfopt = mf_hf._opt_gpu[None] = _VHFOpt(
            mf_hf.mol, mf_hf.direct_scf_tol).build()
    return vhfopt


def make_s1_oo_vo(mol, Co, Cv, s1, atmlst):
    '''Build and return the complete OO and VO overlap tensors.

    Args:
        mol : pyscf.gto.Mole
            Molecule defining atom-local AO slices.
        Co : cupy.ndarray of shape (nao, nocc)
            Occupied source MO coefficients.
        Cv : cupy.ndarray of shape (nao, nvir)
            Virtual source MO coefficients.
        s1 : cupy.ndarray of shape (3, nao, nao)
            One-center AO overlap derivative from GPU4PySCF.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        s1oo : cupy.ndarray of shape (nset, 3, nocc, nocc)
            Full moving-basis occupied-occupied overlap blocks.
        s1vo : cupy.ndarray of shape (nset, 3, nvir, nocc)
            Full moving-basis virtual-occupied overlap blocks.

    Notes:
        This is the reusable transformation/reference interface: callers that
        need the actual ``S^lambda_vo`` values receive the complete tensor.
        No virtual-virtual block or ``(3N,nmo,nmo)`` tensor is formed.

        The production gradient does not call this function because it would
        retain ``(3N,nvir,nocc)`` values used only once. It calls
        :func:`make_s1_oo_and_grad_ovlp_energy`, which fuses the same VO
        transformation with its only consumer. This function remains separate
        so the unfused algebra can be tested directly and future callers are
        not forced to provide ``e_occ`` and ``zvec`` merely to obtain overlap
        blocks.
    '''
    nocc = Co.shape[1]
    nvir = Cv.shape[1]
    s1oo = cp.zeros((len(atmlst), 3, nocc, nocc), dtype=Co.dtype)
    s1vo = cp.zeros((len(atmlst), 3, nvir, nocc), dtype=Co.dtype)
    aoslices = mol.aoslice_by_atom()

    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        s1co = contract('xpv,vi->xpi', s1[:, p0:p1], Co)
        s1oo[k] = contract('xpi,pj->xij', s1co, Co[p0:p1].conj())
        s1oo[k] += s1oo[k].transpose(0, 2, 1).conj()

        s1vo[k] = contract('xpi,pa->xai', s1co, Cv[p0:p1].conj())
        s1cv = contract('xpv,va->xpa', s1[:, p0:p1], Cv)
        s1vo[k] += contract('xpa,pi->xai', s1cv, Co[p0:p1].conj())
    return s1oo, s1vo


def make_s1_oo_and_grad_ovlp_energy(
    mol, Co, Cv, e_occ, zvec, s1, atmlst,
):
    '''Build ``S^lambda_oo`` and immediately contract the VO overlap term.

    The complete ``S^lambda_vo`` tensor scales as
    ``(3*N_atom,nvir,nocc)`` but is consumed only by
    ``4 Z^dagger S^lambda_vo epsilon_o``.  This production ordering forms one
    atom's VO block, contracts it to three Cartesian values, and discards it.
    ``S^lambda_oo`` is retained because both the target-overlap skeleton and
    the adjoint source-HF response require it.

    Args:
        mol : pyscf.gto.Mole
            Molecule defining atom-local AO slices.
        Co : cupy.ndarray of shape (nao, nocc)
            Occupied source-RHF coefficients.
        Cv : cupy.ndarray of shape (nao, nvir)
            Virtual source-RHF coefficients.
        e_occ : cupy.ndarray of shape (nocc,)
            Occupied source orbital energies.
        zvec : cupy.ndarray of shape (nvir, nocc)
            Positive-RHS Z-vector.
        s1 : cupy.ndarray of shape (3, nao, nao)
            One-center AO overlap derivative.
        atmlst : sequence of int
            Requested nuclear centers.

    Returns:
        s1oo : cupy.ndarray of shape (nset, 3, nocc, nocc)
            Full moving-basis occupied overlap blocks.
        de_ovlp_energy : cupy.ndarray of shape (nset, 3)
            ``+4 Z^dagger S^lambda_vo epsilon_o`` in Ha/Bohr.

    Notes:
        This is the production, fused counterpart of
        :func:`make_s1_oo_vo`. Both functions evaluate identical OO and VO
        projection formulae, but their return contracts differ:

        - ``make_s1_oo_vo`` returns complete ``s1oo`` and ``s1vo`` tensors;
        - this function returns ``s1oo`` and the already contracted scalar
          gradient term, never the complete ``s1vo`` tensor.

        A separate function is intentional. Folding this behavior into
        ``make_s1_oo_vo`` through optional arguments would make its return type
        depend on flags and obscure whether the memory-heavy VO tensor exists.
        The explicit names keep the materialized reference and streamed
        production paths independently testable. Streaming reduces persistent
        VO storage; it is not assumed to make the small contraction itself
        faster.
    '''
    nocc = Co.shape[1]
    s1oo = cp.zeros((len(atmlst), 3, nocc, nocc), dtype=Co.dtype)
    de = cp.zeros((len(atmlst), 3), dtype=Co.dtype)
    ze = zvec.conj() * e_occ[None, :]
    aoslices = mol.aoslice_by_atom()

    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        s1co = contract('xpv,vi->xpi', s1[:, p0:p1], Co)
        s1oo[k] = contract('xpi,pj->xij', s1co, Co[p0:p1].conj())
        s1oo[k] += s1oo[k].transpose(0, 2, 1).conj()

        s1vo = contract('xpi,pa->xai', s1co, Cv[p0:p1].conj())
        s1cv = contract('xpv,va->xpa', s1[:, p0:p1], Cv)
        s1vo += contract('xpa,pi->xai', s1cv, Co[p0:p1].conj())
        de[k] = 4 * contract('ai,xai->x', ze, s1vo).real
    return s1oo, de


def grad_fock_make_h1(mf_grad, Co, Cv, zvec, mo_coeff, mo_occ,
                      atmlst):
    '''Return the source-Fock term through GPU4PySCF ``make_h1``.

    This reference route materializes ``(N_atom,3,nmo,nocc)`` before selecting
    its VO rows.  It remains available as the numerical reference for the DF
    direct contracted implementation.

    The tensor is the explicit source-HF operator derivative at the fixed
    reference density, not the total self-consistent derivative including
    the induced density response.  Overlap and response terms are assembled
    separately below.

    Args:
        mf_grad : dcforge.gpu.grad.rhf_dft.Gradients
            Active GPU gradient object.
        Co : cupy.ndarray of shape (nao, nocc)
            Occupied source-RHF MO coefficients.
        Cv : cupy.ndarray of shape (nao, nvir)
            Virtual source-RHF MO coefficients.
        zvec : cupy.ndarray of shape (nvir, nocc)
            Canonical positive-RHS Z-vector.
        mo_coeff : cupy.ndarray of shape (nao, nmo)
            Complete source-RHF MO coefficients.
        mo_occ : cupy.ndarray of shape (nmo,)
            Exact restricted source occupations.
        atmlst : sequence of int
            Nuclear centers included in the gradient.

    Returns:
        cupy.ndarray of shape (len(atmlst), 3)
            The ``-4 Z F^HF,lambda_vo`` contribution in Ha/Bohr.
    '''
    mf_hess = mf_grad.base.mf_hf.Hessian()
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
        mf_grad : dcforge.gpu.grad.rhf_dft.Gradients
            Active conventional GPU restricted gradient object.
        Co : cupy.ndarray of shape (nao, nocc)
            Occupied source-RHF MO coefficients.
        Cv : cupy.ndarray of shape (nao, nvir)
            Virtual source-RHF MO coefficients.
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
    mf_hf = mf_grad.base.mf_hf
    if dm0.shape != (mol.nao, mol.nao):
        raise ValueError('dm0 must be one closed-shell AO density matrix')
    if hasattr(mf_hf, 'with_df'):
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
        mf_hf.nuc_grad_method(),
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
        _get_vhfopt(mf_hf),
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


def grad_ovlp_energy(s1vo, e_occ, zvec):
    '''Return ``+4 Z S^lambda_vo epsilon_o`` on the GPU.'''
    se_vo = s1vo * e_occ[None, None, None, :]
    return 4 * contract('ai,kxai->kx', zvec.conj(), se_vo).real


def grad_ovlp_response_full(Co, Cv, zvec, s1oo, fvind):
    '''Return the overlap-response term with the full perturbation batch.

    This function is the numerical reference for the production single-adjoint
    contraction.  It forms ``3N`` occupied-space AO densities and must not be
    used by the default gradient driver.
    '''
    nset, ncart = s1oo.shape[:2]
    nao = Co.shape[0]

    tmp = contract('kxij,ui->kxuj', s1oo, Co)
    dm1 = contract('kxuj,vj->kxuv', tmp, Co.conj())
    dm1 = dm1 + dm1.transpose(0, 1, 3, 2).conj()
    v1ao = fvind(dm1.reshape(-1, nao, nao)) * 2
    tmp = contract('Auv,ua->Aav', v1ao, Cv.conj())
    v1vo = contract('Aav,vi->Aai', tmp, Co)
    v1vo = v1vo.reshape(nset, ncart, Cv.shape[1], Co.shape[1])
    return 2 * contract('ai,kxai->kx', zvec.conj(), v1vo).real


def get_z_response_oo(Co, Cv, zvec, fvind):
    '''Return ``C_o^dagger G^HF[D_Z] C_o`` through the AO provider.

    This function is the conventional AO provider for the production adjoint
    contraction. It differs from :func:`grad_ovlp_response_full`, which applies
    ``fvind`` to all ``3N`` geometry perturbations. Here one Z-transition
    density is applied and only its OO block is returned.

    Keeping provider construction separate from :func:`grad_ovlp_response`
    lets conventional gradients use this AO route while DF gradients override
    it with an MO-direct provider. The final mathematical term contraction is
    shared and therefore cannot silently acquire provider-specific factors.
    '''
    dz = Cv @ zvec @ Co.conj().T
    vz = fvind(dz + dz.conj().T)
    tmp = Co.conj().T @ vz
    return tmp @ Co


def grad_ovlp_response(s1oo, vzoo):
    '''Contract the overlap response from its provider-independent MO block.

    With ``D_Z = C_v Z C_o^dagger + h.c.`` and the self-adjoint real-RHF
    response operator ``G^HF``, the ``3N`` perturbation contractions satisfy

    ``4 Z:G^HF[P^lambda_oo]_vo``
    ``= 4 S^lambda_oo : C_o^dagger G^HF[D_Z] C_o``.

    The provider has already applied the source-HF response to the single
    Z-transition density.  This function owns only the mathematical
    ``S^lambda_oo:V_Z,oo`` term, independent of whether ``vzoo`` came from a
    conventional AO response or a DF MO-direct contraction.
    '''
    return 4 * contract('kxij,ij->kx', s1oo, vzoo).real


def nonvariational_correction(
    mf_grad, s1oo, de_fock, de_ovlp_energy, de_ovlp_response,
):
    '''Assemble the three explicit GPU ``4 Z^dagger V^lambda`` terms.'''
    mf_grad.s1oo = s1oo
    # The production path deliberately never materializes the full VO tensor.
    mf_grad.s1vo = None
    mf_grad.de_fock = cp.asnumpy(de_fock)
    mf_grad.de_ovlp_energy = cp.asnumpy(de_ovlp_energy)
    mf_grad.de_ovlp_response = cp.asnumpy(de_ovlp_response)
    mf_grad.de_nvar = (
        mf_grad.de_fock
        + mf_grad.de_ovlp_energy
        + mf_grad.de_ovlp_response
    )
    mf_grad.overlap_transform_calls = 1
    return mf_grad.de_nvar
