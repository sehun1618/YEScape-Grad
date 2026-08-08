'''GPU sampled-k positive-RHS Z-vector gradient terms.'''

import cupy as cp

from dcforge.gpu.pbc.grad.krhf_skeleton import (
    grad_hcore_energy,
    real_gradient_term,
)
from dcforge.gpu.pbc.grad.krhf_zvec import make_z_dm


def make_s1_oo_vo(cell, Co, Cv, s1, atmlst):
    r'''Build complete moving-basis :math:`S^\lambda_{oo}` and
    :math:`S^\lambda_{vo}` device blocks.

    Each atom-owned AO row is paired with its Hermitian-conjugate column;
    plain transpose is not valid for complex Bloch orbitals.
    '''
    Co = cp.asarray(Co)
    Cv = cp.asarray(Cv)
    nkpts = len(Co)
    nocc = Co.shape[2]
    nvir = Cv.shape[2]
    s1oo = cp.zeros(
        (len(atmlst), 3, nkpts, nocc, nocc), dtype=Co.dtype)
    s1vo = cp.zeros(
        (len(atmlst), 3, nkpts, nvir, nocc), dtype=Co.dtype)
    aoslices = cell.aoslice_by_atom()
    for iatom, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        s1row = s1[:, :, p0:p1]
        s1co = cp.einsum('kxpv,kvi->kxpi', s1row, Co)
        # Form C_o^H S_row C_o before restoring its Hermitian-conjugate AO
        # column.  The left coefficient must carry the conjugation; reversing
        # these operands returns S_oo^T for complex Bloch orbitals.
        oo = cp.einsum(
            'kpa,kxpi->kxai', Co[:, p0:p1].conj(), s1co)
        oo += oo.transpose(0, 1, 3, 2).conj()
        vo = cp.einsum(
            'kpa,kxpi->kxai', Cv[:, p0:p1].conj(), s1co)
        s1cv = cp.einsum('kxpv,kva->kxpa', s1row, Cv)
        # C_v^H S_row^H C_o = (S_row C_v)^H C_o.  Conjugating C_o here is
        # only equivalent for real Gamma orbitals and was the Water8 FD bug.
        vo += cp.einsum(
            'kxpa,kpi->kxai', s1cv.conj(), Co[:, p0:p1])
        s1oo[iatom] = oo.transpose(1, 0, 2, 3)
        s1vo[iatom] = vo.transpose(1, 0, 2, 3)
    return s1oo, s1vo


def grad_fock_direct(
        mf_grad, Co, Cv, zvec, dm0, vj1, vk1,
        vj1_z, vk1_z, atmlst):
    r'''Return ``-2/Nk Re Tr[D_Z F_HF^lambda[P]]`` on the GPU.

    ``vj1/vk1`` are complete finite-grid matrix derivatives for the source
    density.  A second derivative built from ``D_Z`` provides the reverse
    polarization identity

    ``Tr[D_Z G^lambda[P]] = Tr[P G^lambda[D_Z]]``

    as an independent numerical diagnostic; it is not averaged into the
    production value.
    '''
    nkpts = len(mf_grad.base.kpts)
    dz = make_z_dm(Co, Cv, zvec)
    de_hcore_density = grad_hcore_energy(
        mf_grad, dz, atmlst, 'de_fock_hcore_density')
    de_hcore = -2 * de_hcore_density

    vhf1 = vj1 - .5 * vk1
    vhf1_z = vj1_z - .5 * vk1_z
    de_jk = real_gradient_term(
        mf_grad,
        -2 * cp.einsum(
            'akxij,kji->ax', vhf1[atmlst], dz) / nkpts,
        'de_fock_jk')
    de_jk_reverse = real_gradient_term(
        mf_grad,
        -2 * cp.einsum(
            'akxij,kji->ax', vhf1_z[atmlst], dm0) / nkpts,
        'de_fock_jk_reverse')
    mf_grad.fock_derivative_reverse_error = float(cp.max(cp.abs(
        de_jk - de_jk_reverse)).get())
    mf_grad.de_fock_hcore = de_hcore
    mf_grad.de_fock_jk = de_jk
    mf_grad.de_fock_jk_reverse = de_jk_reverse
    return de_hcore + de_jk


def grad_fock_contracted(mf_grad, dz, de_fock_jk, atmlst):
    r'''Return GDF ``-2 Tr[D_Z h^lambda]`` plus contracted J/K response.

    GPU4PySCF's sampled-k GDF derivative endpoint returns only the complete
    source-plus-metric-plus-Ewald scalar response.  It does not expose an AO
    derivative matrix.  This function keeps the one-electron and two-electron
    terms explicit while accepting that native contracted J/K observable.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.krhf_dft.Gradients
            Active sampled-k GPU gradient object.
        dz : cupy.ndarray of shape (nkpts,nao,nao)
            Hermitian Z-transition density.
        de_fock_jk : cupy.ndarray of shape (len(atmlst),3)
            Native combined GDF J/K response with the DC-DFT sign included.
        atmlst : sequence of int
            Requested nuclear centers.

    Returns:
        cupy.ndarray
            Complete source-Fock nonvariational derivative in Ha/Bohr.
    '''
    de_hcore_density = grad_hcore_energy(
        mf_grad, dz, atmlst, 'de_fock_hcore_density')
    de_hcore = -2 * de_hcore_density
    de_jk = real_gradient_term(
        mf_grad, de_fock_jk, 'de_fock_jk')
    mf_grad.de_fock_hcore = de_hcore
    mf_grad.de_fock_jk = de_jk
    mf_grad.de_fock_jk_reverse = None
    mf_grad.fock_derivative_reverse_error = None
    return de_hcore + de_jk


def grad_fock_resolved(mf_grad, dz, dm0, response, atmlst):
    r'''Return RSDF Hcore plus source/metric/Ewald Fock response terms.

    ``response`` retains the CPU oracle's complete complex AO derivatives,
    but all final contractions and source-versus-metric term ownership are
    evaluated on the device.  The reverse density order is an independent
    polarization diagnostic and is not averaged into production terms.
    '''
    nkpts = len(mf_grad.base.kpts)
    de_hcore_density = grad_hcore_energy(
        mf_grad, dz, atmlst, 'de_fock_hcore_density')
    de_hcore = -2 * de_hcore_density

    components = {}
    for suffix in ('source', 'metric'):
        vhf1 = (
            response[f'vj1_{suffix}']
            - .5 * response[f'vk1_{suffix}'])
        components[suffix] = real_gradient_term(
            mf_grad, -2 * cp.einsum(
                'axkij,kji->ax', vhf1[atmlst], dz) / nkpts,
            f'de_fock_{suffix}')
    components['ewald'] = real_gradient_term(
        mf_grad, cp.einsum(
            'axkij,kji->ax', response['vk1_ewald'][atmlst], dz)
        / nkpts, 'de_fock_ewald')

    vhf1 = (
        response['vj1_source'] + response['vj1_metric']
        - .5 * (response['vk1_source'] + response['vk1_metric']
                + response['vk1_ewald']))
    vhf1_z = (
        response['vj1_source_z'] + response['vj1_metric_z']
        - .5 * (response['vk1_source_z'] + response['vk1_metric_z']
                + response['vk1_ewald_z']))
    forward = -2 * cp.einsum(
        'axkij,kji->ax', vhf1[atmlst], dz) / nkpts
    reverse = -2 * cp.einsum(
        'axkij,kji->ax', vhf1_z[atmlst], dm0) / nkpts
    mf_grad.fock_derivative_reverse_error = float(cp.max(cp.abs(
        forward - reverse)).get())
    mf_grad.de_fock_hcore = de_hcore
    mf_grad.de_fock_source = components['source']
    mf_grad.de_fock_metric = components['metric']
    mf_grad.de_fock_ewald = components['ewald']
    mf_grad.de_fock_jk = sum(components.values())
    mf_grad.de_fock_jk_reverse = real_gradient_term(
        mf_grad, reverse, 'de_fock_jk_reverse')
    return de_hcore + mf_grad.de_fock_jk


def grad_fock_resolved_contracted(mf_grad, dz, response_terms, atmlst):
    r'''Return RSDF source-Fock response from scalar contracted terms.

    The one-electron term remains an explicit contraction with ``D_Z``.
    Source, metric and Ewald two-electron terms were already contracted by
    :func:`make_kpts_contracted_gradient_response`; keeping them separate
    preserves the term ledger while avoiding ten persistent
    ``(natm,3,nkpts,nao,nao)`` device arrays.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.krhf_dft.Gradients
            Active sampled-k gradient object.
        dz : cupy.ndarray of shape (nkpts,nao,nao)
            Hermitian Z-transition density.
        response_terms : dict
            Device ``(len(atmlst),3)`` source, metric, Ewald and reverse
            scalar contractions.
        atmlst : sequence of int
            Requested nuclear centers.

    Returns:
        cupy.ndarray
            Complete source-Fock nonvariational derivative in Ha/Bohr.
    '''
    de_hcore_density = grad_hcore_energy(
        mf_grad, dz, atmlst, 'de_fock_hcore_density')
    de_hcore = -2 * de_hcore_density
    source = real_gradient_term(
        mf_grad, response_terms['de_fock_source'], 'de_fock_source')
    metric = real_gradient_term(
        mf_grad, response_terms['de_fock_metric'], 'de_fock_metric')
    ewald = real_gradient_term(
        mf_grad, response_terms['de_fock_ewald'], 'de_fock_ewald')
    reverse = (
        cp.asarray(response_terms['de_fock_source_reverse'])
        + cp.asarray(response_terms['de_fock_metric_reverse'])
        + cp.asarray(response_terms['de_fock_ewald_reverse']))
    forward = source + metric + ewald
    mf_grad.fock_derivative_reverse_error = float(cp.max(cp.abs(
        forward - reverse)).get())
    mf_grad.de_fock_hcore = de_hcore
    mf_grad.de_fock_source = source
    mf_grad.de_fock_metric = metric
    mf_grad.de_fock_ewald = ewald
    mf_grad.de_fock_jk = forward
    mf_grad.de_fock_jk_reverse = real_gradient_term(
        mf_grad, reverse, 'de_fock_jk_reverse')
    return de_hcore + forward


def grad_ovlp_energy(mf_grad, s1vo, e_occ, zvec, nkpts):
    r'''Return ``+4/Nk Re sum_k Z_k^H S^lambda_vo,k epsilon_o,k``.'''
    zvec = cp.asarray(zvec)
    e_occ = cp.asarray(e_occ)
    se_vo = s1vo * e_occ[None, None, :, None, :]
    value = 4 * cp.einsum(
        'kvi,txkvi->tx', zvec.conj(), se_vo) / nkpts
    return real_gradient_term(
        mf_grad, value, 'de_ovlp_energy', expect_real=False)


def grad_ovlp_response(mf_grad, Co, Cv, zvec, s1oo, fvind, nkpts):
    r'''Return ``+4/Nk Re Tr[S^lambda_oo G_HF[D_Z]_oo]``.'''
    dz = make_z_dm(Co, Cv, zvec)
    vz = cp.asarray(fvind(dz))
    vzoo = cp.asarray([
        Co_k.conj().T @ vz_k @ Co_k
        for Co_k, vz_k in zip(Co, vz, strict=True)])
    value = 4 * cp.einsum(
        'axkij,kji->ax', s1oo, vzoo) / nkpts
    return real_gradient_term(mf_grad, value, 'de_ovlp_response')


def nonvariational_correction(
        mf_grad, Co, Cv, mo_energy, mo_occ, zvec, s1,
        de_fock, fvind, atmlst):
    '''Assemble the three device positive-RHS Z-vector gradient terms.'''
    nkpts = len(Co)
    s1oo, s1vo = make_s1_oo_vo(
        mf_grad.cell, Co, Cv, s1, atmlst)
    mo_energy = cp.asarray(mo_energy)
    mo_occ = cp.asarray(mo_occ)
    e_occ = cp.asarray([
        mo_energy[k][mo_occ[k] > 0]
        for k in range(nkpts)])
    de_ovlp_energy = grad_ovlp_energy(
        mf_grad, s1vo, e_occ, zvec, nkpts)
    de_ovlp_response = grad_ovlp_response(
        mf_grad, Co, Cv, zvec, s1oo, fvind, nkpts)
    mf_grad.s1oo = s1oo
    mf_grad.s1vo = s1vo
    mf_grad.de_fock = de_fock
    mf_grad.de_ovlp_energy = de_ovlp_energy
    mf_grad.de_ovlp_response = de_ovlp_response
    mf_grad.de_nvar = de_fock + de_ovlp_energy + de_ovlp_response
    return mf_grad.de_nvar
