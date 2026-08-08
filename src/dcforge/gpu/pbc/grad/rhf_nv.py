'''Gamma-point GPU nonvariational ``4 Z^dagger V^lambda`` terms.''' 

import cupy as cp
import numpy as np

from dcforge.gpu.pbc.grad.skeleton import contract_hcore


def make_s1_oo_vo(cell, Co, Cv, s1, atmlst):
    '''Build full moving-basis ``S^lambda_oo`` and ``S^lambda_vo`` blocks.'''
    nocc = Co.shape[1]
    nvir = Cv.shape[1]
    s1oo = cp.zeros((len(atmlst), 3, nocc, nocc), dtype=Co.dtype)
    s1vo = cp.zeros((len(atmlst), 3, nvir, nocc), dtype=Co.dtype)
    aoslices = cell.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        s1co = cp.einsum('xpv,vi->xpi', s1[:, p0:p1], Co)
        s1oo[k] = cp.einsum('xpi,pj->xij', s1co, Co[p0:p1].conj())
        s1oo[k] += s1oo[k].transpose(0, 2, 1).conj()
        s1vo[k] = cp.einsum('xpi,pa->xai', s1co, Cv[p0:p1].conj())
        s1cv = cp.einsum('xpv,va->xpa', s1[:, p0:p1], Cv)
        s1vo[k] += cp.einsum('xpa,pi->xai', s1cv, Co[p0:p1].conj())
    return s1oo, s1vo


def grad_fock_direct(mf_grad, Co, Cv, zvec, dm0, hcore_gen, vj1, vk1,
                     atmlst):
    r'''Return ``-4 Z F^HF,lambda_vo`` from full GPU FFTDF derivatives.

    ``grad_jk`` returns complete AO derivatives.  Consequently the source
    Fock term is a single full-matrix contraction ``-2 Tr[D_Z F^lambda]``;
    the two atom-row cross contractions used by the CPU FFTDF implementation
    are the equivalent representation for its row-derivative tensors.
    '''
    dz = Cv @ zvec @ Co.conj().T
    dz = dz + dz.conj().T
    de_hcore = -2 * contract_hcore(
        mf_grad.cell, hcore_gen, dz, atmlst)
    vhf1 = vj1[atmlst] - .5 * vk1[atmlst]
    de_jk = -2 * cp.einsum('axij,ji->ax', vhf1, dz).real

    mf_grad.de_fock_hcore = cp.asnumpy(de_hcore)
    mf_grad.de_fock_source = cp.asnumpy(de_jk)
    mf_grad.de_fock_metric = None
    mf_grad.de_fock_ewald = np.zeros_like(mf_grad.de_fock_hcore)
    return de_hcore + de_jk


def grad_fock_gdf(mf_grad, Co, Cv, zvec, dm0, hcore_gen, atmlst,
                  hcore_derivative=None, source_derivative=None,
                  metric_derivative=None, ewald_derivative=None,
                  native_derivative=None, resolved_backend=None):
    r'''Return the one-electron part of ``-4 Z F^HF,lambda_vo`` for GDF.

    The ordinary GPU4PySCF endpoint fuses target ``E_J`` and the signed
    two-electron Z term in one native traversal.  In that default route this
    helper supplies only the uncoupled one-electron contribution
    ``-2 Tr[D_Z h^lambda]``.  When the opt-in two-channel native ABI is
    available, the caller passes the separately contracted source, metric and
    Ewald components and this helper records the full physical Fock ledger.

    Args:
        mf_grad : Gradients
            Active GDF semilocal-DFA@RHF gradient object.
        Co, Cv : cupy.ndarray
            Occupied and virtual source-HF coefficient blocks.
        zvec : cupy.ndarray
            Solved adjoint Z vector in the VO block.
        dm0 : cupy.ndarray
            Closed-shell source-HF AO density.
        hcore_gen : callable
            GPU local one-electron derivative generator.
        atmlst : sequence of int
            Requested centers.

    Kwargs:
        hcore_derivative : cupy.ndarray or None
            Precontracted ``-2 Tr[D_Z h^lambda]`` value.  GDF supplies this
            from the paired ``P,D_Z`` hcore traversal so the costly local
            matrix is not generated twice.  ``None`` retains the standalone
            reference contraction.
        source_derivative, metric_derivative, ewald_derivative :
            cupy.ndarray or None
            Separate two-electron Z-Fock components from the optional
            two-channel GDF native ABI.  Either all three are supplied or
            all three are omitted for the ordinary fused route.
        native_derivative : cupy.ndarray or None
            Already combined two-electron source-Fock derivative.  This is
            used when target J belongs to multigrid, so the standalone signed
            GDF polarization supplies only the source-HF response.  It cannot
            be combined with the three resolved component arguments.
        resolved_backend : str or None
            Ledger label for explicitly resolved source/metric/Ewald terms.
            ``None`` retains the optional two-channel native label.

    Returns:
        cupy.ndarray
        Complete source-Fock response in Ha/Bohr when the three optional
        components are supplied; otherwise its one-electron part only.

    Notes:
        The public term structure remains explicit: a fused value is never
        silently put in ``de_j`` or ``de_fock_source``.
    '''
    dz = Cv @ zvec @ Co.conj().T
    dz = dz + dz.conj().T
    if hcore_derivative is None:
        de_hcore = -2 * contract_hcore(
            mf_grad.cell, hcore_gen, dz, atmlst)
    else:
        de_hcore = cp.asarray(hcore_derivative)
        if de_hcore.shape != (len(atmlst), 3):
            raise ValueError('precontracted GDF Fock hcore has invalid shape')
    supplied = (source_derivative, metric_derivative, ewald_derivative)
    if native_derivative is not None:
        if any(value is not None for value in supplied):
            raise ValueError(
                'combined and resolved GDF Fock derivatives are exclusive')
        de_native = cp.asarray(native_derivative)
        if de_native.shape != (len(atmlst), 3):
            raise ValueError('combined GDF Fock derivative has invalid shape')
        mf_grad.de_fock_hcore = cp.asnumpy(de_hcore)
        mf_grad.de_fock_native = cp.asnumpy(de_native)
        mf_grad.de_fock_source = None
        mf_grad.de_fock_metric = None
        mf_grad.de_fock_ewald = None
        mf_grad.de_fock_backend = 'gdf_signed_source_fock_native_combined'
        return de_hcore + de_native
    if any(value is None for value in supplied):
        if any(value is not None for value in supplied):
            raise ValueError(
                'GDF source, metric and Ewald derivatives must be supplied together')
        mf_grad.de_fock_hcore = cp.asnumpy(de_hcore)
        mf_grad.de_fock_native = None
        mf_grad.de_fock_source = None
        mf_grad.de_fock_metric = None
        mf_grad.de_fock_ewald = None
        mf_grad.de_fock_backend = 'gdf_hcore_plus_fused_native_response'
        return de_hcore
    expected = (len(atmlst), 3)
    de_source, de_metric, de_ewald = tuple(
        cp.asarray(value) for value in supplied)
    if any(value.shape != expected
           for value in (de_hcore, de_source, de_metric, de_ewald)):
        raise ValueError('GDF Fock response terms have inconsistent atom shape')
    mf_grad.de_fock_hcore = cp.asnumpy(de_hcore)
    mf_grad.de_fock_native = None
    mf_grad.de_fock_source = cp.asnumpy(de_source)
    mf_grad.de_fock_metric = cp.asnumpy(de_metric)
    mf_grad.de_fock_ewald = cp.asnumpy(de_ewald)
    mf_grad.de_fock_backend = (
        'gdf_two_channel_native_adjoint'
        if resolved_backend is None else str(resolved_backend))
    return de_hcore + de_source + de_metric + de_ewald


def grad_fock_rsdf(mf_grad, hcore_derivative, source_derivative,
                   metric_derivative, ewald_derivative, atmlst):
    r'''Assemble RSDF ``-2 D_Z:F^{\lambda}`` from resolved scalars.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.rhf_dft.Gradients
            Active Gamma semilocal-DFA@RHF gradient object.
        hcore_derivative : cupy.ndarray
            Precontracted ``-2 Tr[D_Z h^lambda]`` term.
        source_derivative, metric_derivative : cupy.ndarray
            Reverse-adjoint RSDF source and metric contributions to the
            source-Fock Z bilinear.
        ewald_derivative : cupy.ndarray
            Explicit ``exxdiv='ewald'`` overlap correction, or zero when
            source HF selected ``exxdiv=None``.
        atmlst : sequence of int
            Requested atom order, used to validate result shapes.

    Returns:
        cupy.ndarray
            The complete source-Fock derivative in Ha/Bohr.

    Notes:
        Unlike GDF's current fused primitive endpoint, Gamma RSDF exposes
        source and metric adjoints independently.  They remain separately
        recorded in the public ledger rather than being hidden in one
        combined two-electron field.
    '''
    expected = (len(atmlst), 3)
    values = tuple(cp.asarray(value) for value in (
        hcore_derivative, source_derivative, metric_derivative,
        ewald_derivative))
    if any(value.shape != expected for value in values):
        raise ValueError('RSDF Fock response terms have inconsistent atom shape')
    de_hcore, de_source, de_metric, de_ewald = values
    mf_grad.de_fock_hcore = cp.asnumpy(de_hcore)
    mf_grad.de_fock_source = cp.asnumpy(de_source)
    mf_grad.de_fock_metric = cp.asnumpy(de_metric)
    mf_grad.de_fock_ewald = cp.asnumpy(de_ewald)
    mf_grad.de_fock_native = None
    mf_grad.de_fock_backend = 'rsdf_source_metric_native_adjoint'
    return de_hcore + de_source + de_metric + de_ewald


def grad_ovlp_energy(s1vo, e_occ, zvec):
    '''Return ``+4 Z S^lambda_vo epsilon_o`` in Ha/Bohr.'''
    return 4 * cp.einsum(
        'ai,kxai->kx', zvec.conj(), s1vo * e_occ[None, None, None, :]).real


def grad_ovlp_response(Co, Cv, zvec, s1oo, fvind):
    '''Return the moving-metric source-HF response with one adjoint density.'''
    vzoo = make_ovlp_response_intermediate(Co, Cv, zvec, fvind)
    return 4 * cp.einsum('kxij,ij->kx', s1oo, vzoo).real


def make_ovlp_response_intermediate(Co, Cv, zvec, fvind):
    r'''Return ``C_o^dagger V[D_Z] C_o`` while source CDERI is available.

    This AO-response action is the last nonvariational term that needs the
    source-HF J/K provider.  Keeping the small occupied-space result allows an
    opt-in large calculation to release central CDERI before explicit GDF or
    RSDF nuclear-response work starts.
    '''
    dz = Cv @ zvec @ Co.conj().T
    vz = fvind(dz + dz.conj().T)
    return Co.conj().T @ vz @ Co


def nonvariational_correction(mf_grad, Co, Cv, mo_energy, zvec, s1,
                              de_fock, fvind, atmlst,
                              ovlp_response_intermediate=None):
    '''Assemble the three periodic GPU ``4 Z^dagger V^lambda`` terms.'''
    s1oo, s1vo = make_s1_oo_vo(mf_grad.cell, Co, Cv, s1, atmlst)
    e_occ = mo_energy[:Co.shape[1]]
    de_ovlp_energy = grad_ovlp_energy(s1vo, e_occ, zvec)
    if ovlp_response_intermediate is None:
        de_ovlp_response = grad_ovlp_response(
            Co, Cv, zvec, s1oo, fvind)
    else:
        vzoo = cp.asarray(ovlp_response_intermediate)
        if vzoo.shape != (Co.shape[1], Co.shape[1]):
            raise ValueError('overlap-response intermediate has invalid shape')
        de_ovlp_response = 4 * cp.einsum(
            'kxij,ij->kx', s1oo, vzoo).real

    mf_grad.s1oo = s1oo
    mf_grad.s1vo = s1vo
    mf_grad.de_fock = cp.asnumpy(de_fock)
    mf_grad.de_ovlp_energy = cp.asnumpy(de_ovlp_energy)
    mf_grad.de_ovlp_response = cp.asnumpy(de_ovlp_response)
    mf_grad.de_nvar = (
        mf_grad.de_fock
        + mf_grad.de_ovlp_energy
        + mf_grad.de_ovlp_response)
    return mf_grad.de_nvar
