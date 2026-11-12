# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Source-neutral overlap and nonvariational gradient assembly.'''

import cupy as cp
from gpu4pyscf.lib.cupy_helper import contract


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
    the adjoint source-SCF response require it.

    Args:
        mol : pyscf.gto.Mole
            Molecule defining atom-local AO slices.
        Co : cupy.ndarray of shape (nao, nocc)
            Occupied source coefficients.
        Cv : cupy.ndarray of shape (nao, nvir)
            Virtual source coefficients.
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
    '''Return ``C_o^dagger G^source[D_Z] C_o`` through the AO provider.

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

    With ``D_Z = C_v Z C_o^dagger + h.c.`` and the self-adjoint real HF/KS
    response operator ``G^source``, the ``3N`` perturbation contractions satisfy

    ``-4 Z:G^source[P^lambda_oo]_vo``
    ``= 4 S^lambda_oo : C_o^dagger G^source[D_Z] C_o``.

    Here ``P^lambda_oo = -2 C_o S^lambda_oo C_o^dagger``.
    The provider has already applied the source-SCF response to the single
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
