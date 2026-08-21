'''Gamma-point GPU RHF nonvariational DC-DFT strain response.'''

import cupy as cp
import numpy as np
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract

from dcforge.cpu.pbc.df.gaussian.strain import (
    kinetic_strain,
    overlap_strain,
)
from dcforge.cpu.pbc.stress.ewald import (
    gamma_ewald_exchange_strain,
)
from dcforge.gpu.pbc.df.gaussian.pp_strain import (
    contract_gdf_pseudopotential_strain,
)
from dcforge.gpu.pbc.df.gdf.stress import (
    make_gdf_strain_response,
)
from dcforge.gpu.pbc.df.rhf_response import (
    gen_vind as gen_df_vind,
)
from dcforge.gpu.pbc.df.rhf_response import (
    gen_vind_ao_oracle as gen_df_vind_ao_oracle,
)
from dcforge.gpu.pbc.df.rsdf.stress import (
    make_rsdf_strain_response,
)
from dcforge.gpu.pbc.energy.rhf_dft import (
    _target_fock_signature,
)
from dcforge.gpu.pbc.grad.rhf_state import (
    get_response_state,
    make_response_state_signature,
    store_response_state,
    use_response_state,
)
from dcforge.gpu.pbc.grad.rhf_zvec import (
    solve_z_vector,
)
from dcforge.gpu.pbc.stress.profiler import (
    stage_start,
    stage_stop,
)


def make_s1_oo_vo(Co, Cv, s1):
    r'''Project the complete strain overlap derivative into OO and VO blocks.

    The input is a full bra-plus-ket homogeneous-strain derivative.  Unlike
    an atom-owned AO-row nuclear derivative, it needs neither an extra factor
    of two nor a separate Hermitian completion.

    Args:
        Co : cupy.ndarray of shape (nao,nocc)
            Occupied source-RHF coefficients.
        Cv : cupy.ndarray of shape (nao,nvir)
            Virtual source-RHF coefficients.
        s1 : cupy.ndarray of shape (3,3,nao,nao)
            Complete AO overlap strain derivative.

    Returns:
        tuple
            ``s1oo`` and ``s1vo`` with leading raw strain axes ``(3,3)``.
    '''
    Co = cp.asarray(Co)
    Cv = cp.asarray(Cv)
    s1 = cp.asarray(s1)
    nao = int(Co.shape[0])
    if Cv.shape[0] != nao or s1.shape != (3, 3, nao, nao):
        raise ValueError('strain overlap and MO coefficient shapes differ')
    s1co = contract('xypq,qi->xypi', s1, Co)
    s1oo = contract('pi,xypj->xyij', Co.conj(), s1co)
    s1vo = contract('pa,xypi->xyai', Cv.conj(), s1co)
    return s1oo, s1vo


def make_z_density(Co, Cv, zvec):
    r'''Return :math:`D_Z=C_v Z C_o^\dagger+\mathrm{h.c.}` on the device.'''
    Co = cp.asarray(Co)
    Cv = cp.asarray(Cv)
    zvec = cp.asarray(zvec)
    if zvec.shape != (Cv.shape[1], Co.shape[1]):
        raise ValueError('Z-vector has incompatible virtual/occupied shape')
    dz = Cv @ zvec @ Co.conj().T
    return dz + dz.conj().T


def work_fock_hcore(dz, hcore1):
    r'''Return ``-2 Tr[D_Z h^(xy)]`` in Hartree on the device.'''
    dz = cp.asarray(dz)
    hcore1 = cp.asarray(hcore1)
    if hcore1.shape != (3, 3) + dz.shape:
        raise ValueError('Hcore strain and Z-density shapes differ')
    return -2 * contract('xyij,ji->xy', hcore1, dz).real


def work_fock_gdf(gdf_response):
    r'''Return the source-Fock channel of the shared GDF strain response.'''
    names = tuple(gdf_response.get('observable_names', ()))
    if names != ('target_j', 'source_fock'):
        raise ValueError('unexpected GDF strain observable ordering')
    raw = cp.asarray(gdf_response['raw_contraction'])
    if raw.shape != (2, 3, 3):
        raise ValueError('GDF strain response must have shape (2,3,3)')
    return raw[1]


def work_fock_ewald(dz, exchange1):
    r'''Return the Gamma ``exxdiv='ewald'`` source-Fock contribution.'''
    dz = cp.asarray(dz)
    exchange1 = cp.asarray(exchange1)
    if exchange1.shape != (3, 3) + dz.shape:
        raise ValueError('Ewald exchange strain and Z-density shapes differ')
    return contract('xyij,ji->xy', exchange1, dz).real


def work_ovlp_energy(s1vo, e_occ, zvec):
    r'''Return ``+4 Re <Z,S^(xy)_vo epsilon_o>`` in Hartree.'''
    s1vo = cp.asarray(s1vo)
    e_occ = cp.asarray(e_occ)
    zvec = cp.asarray(zvec)
    if s1vo.shape[:2] != (3, 3) or s1vo.shape[2:] != zvec.shape:
        raise ValueError('VO overlap strain and Z-vector shapes differ')
    if e_occ.shape != (zvec.shape[1],):
        raise ValueError('occupied energies have incompatible shape')
    return 4 * contract(
        'ai,xyai->xy', zvec.conj(),
        s1vo * e_occ[None, None, None, :]).real


def work_ovlp_response(Co, dz, s1oo, fvind):
    r'''Return the OO moving-metric source-HF response contribution.

    RHF self-adjointness moves the response kernel from all nine occupied
    metric densities onto the single central Z density.  Consequently only
    ``C_o^dagger G[D_Z] C_o`` is built before its nine small contractions.
    '''
    Co = cp.asarray(Co)
    dz = cp.asarray(dz)
    s1oo = cp.asarray(s1oo)
    if dz.shape != (Co.shape[0], Co.shape[0]):
        raise ValueError('Z-density and occupied coefficients differ')
    if s1oo.shape != (3, 3, Co.shape[1], Co.shape[1]):
        raise ValueError('OO overlap strain has incompatible shape')
    vz = cp.asarray(fvind(dz))
    vzoo = Co.conj().T @ vz @ Co
    return 4 * contract('xyij,ji->xy', s1oo, vzoo).real


def make_strain_source(Co, Cv, e_occ, s1oo, s1vo, fock1, fvind):
    r'''Build the explicit positive-RHS strain CPHF source oracle.

    The returned tensor is

    .. math::

        V^{(xy)}=S_{vo}^{(xy)}\epsilon_o-F_{vo}^{(xy)}
        -G[-2C_oS_{oo}^{(xy)}C_o^\dagger]_{vo}.

    Production uses the three scalar adjoint contractions and does not retain
    this full forward source.
    '''
    Co = cp.asarray(Co)
    Cv = cp.asarray(Cv)
    e_occ = cp.asarray(e_occ)
    s1oo = cp.asarray(s1oo)
    s1vo = cp.asarray(s1vo)
    fock1 = cp.asarray(fock1)
    nao = int(Co.shape[0])
    if Cv.shape[0] != nao:
        raise ValueError('occupied and virtual AO dimensions differ')
    if fock1.shape != (3, 3, nao, nao):
        raise ValueError('source Fock strain must have shape (3,3,nao,nao)')
    source = s1vo * e_occ[None, None, None, :]
    fock_co = contract('xypq,qi->xypi', fock1, Co)
    source -= contract('pa,xypi->xyai', Cv.conj(), fock_co)
    for x in range(3):
        for y in range(3):
            dm_oo = -2 * Co @ s1oo[x, y] @ Co.conj().T
            source[x, y] -= Cv.conj().T @ cp.asarray(fvind(dm_oo)) @ Co
    return source


def contract_strain_source(zvec, source):
    r'''Return ``+4 Re <Z,V^(xy)>`` for an explicit device source.'''
    zvec = cp.asarray(zvec)
    source = cp.asarray(source)
    if source.shape[:2] != (3, 3) or source.shape[2:] != zvec.shape:
        raise ValueError('strain source and Z-vector shapes differ')
    return 4 * contract('ai,xyai->xy', zvec.conj(), source).real


def make_z_state(
        mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
        profile=None):
    r'''Build one GPU target Fock and solve one source-HF Z-vector.

    All six strain components share the returned perturbation-independent
    state.  The target object is evaluated only through its cached or direct
    fixed-density Fock; no target DFA SCF is started.
    '''
    _progress(mf_grad, 'state_setup')
    state_stage = stage_start(profile)
    mf = mf_grad.base
    mf_hf = mf.mf_hf
    if mf.e_dft is None:
        raise RuntimeError('run the periodic GPU RHFDFT energy kernel first')
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    mo_energy = cp.asarray(mo_energy)
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    occupied = mo_occ > 0
    virtual = mo_occ == 0
    if not bool(cp.all((mo_occ == 2) | virtual)):
        raise ValueError('periodic GPU RHF stress requires exact 2/0 occupations')
    if cp.iscomplexobj(mo_coeff):
        imaginary = float(cp.max(cp.abs(mo_coeff.imag)).get())
        if imaginary > 1.e-10:
            raise NotImplementedError(
                'the first GPU stress route requires real Gamma orbitals')
        mo_coeff = mo_coeff.real
    Co = mo_coeff[:, occupied]
    Cv = mo_coeff[:, virtual]
    dm0 = cp.asarray(mf_hf.make_rdm1(mo_coeff, mo_occ))
    stage_stop(profile, 'state_setup', state_stage)

    _progress(mf_grad, 'target_fock')
    target_stage = stage_start(profile)
    signature = _target_fock_signature(mf, dm0)
    if (mf.target_fock is not None and mf.dm_hf is not None
            and mf.target_fock_signature == signature
            and bool(cp.array_equal(cp.asarray(mf.dm_hf), dm0))):
        target_fock = cp.asarray(mf.target_fock)
        target_fock_reused = True
    else:
        target_fock = cp.asarray(mf.get_fock(dm=dm0))
        target_fock_reused = False
    fock_mo = mo_coeff.conj().T @ target_fock @ mo_coeff
    fvo = fock_mo[virtual][:, occupied]
    foo = fock_mo[occupied][:, occupied]
    stage_stop(profile, 'target_fock', target_stage)

    fvind = mf_hf.gen_response(
        mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=1)
    signature = make_response_state_signature(
        mf_grad, mo_energy, mo_coeff, mo_occ, dm0)
    response_state = get_response_state(mf_grad, signature)

    _progress(mf_grad, 'z_response')
    response_stage = stage_start(profile)
    if response_state is None:
        response_mode = mf_grad.df_cphf_response_mode
        if response_mode == 'direct_vo':
            vind = gen_df_vind(
                mf_hf, Co, Cv,
                cache_mode=mf_grad.df_cphf_response_cache,
                memory_fraction=mf_grad.df_cphf_response_memory_fraction,
                minimum_reserve_bytes=(
                    mf_grad.df_cphf_response_minimum_reserve_bytes),
                profile_detail=mf_grad.profile_df_cphf_response)
        elif response_mode == 'ao_oracle':
            vind = gen_df_vind_ao_oracle(mf_hf, Co, Cv)
        else:
            raise ValueError(
                'df_cphf_response_mode must be direct_vo or ao_oracle')
        log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
        response_profile = None
        try:
            zvec, z_info = solve_z_vector(
                mo_energy, mo_occ, fvo, vind,
                conv_tol=mf_grad.cphf_conv_tol,
                krylov_tol=mf_grad.cphf_krylov_tol,
                residual_tol=mf_grad.cphf_residual_tol,
                max_cycle=mf_grad.cphf_max_cycle,
                refinement_solver=mf_grad.cphf_refinement_solver,
                initial_guess=mf_grad.z_initial_guess,
                log=log)
        finally:
            close_response = getattr(vind, 'close', None)
            if close_response is not None:
                response_profile = close_response()
            if 'z_info' not in locals():
                mf_grad.z_initial_guess = None
        response_orbitals = Cv @ zvec
        dz = make_z_density(Co, Cv, zvec)
        z_warm_start_metadata = {
            **(mf_grad.z_warm_start_projection or {}),
            'requested': z_info['initial_guess_requested'],
            'used': z_info['initial_guess_used'],
            'fallback_reason': z_info['initial_guess_fallback_reason'],
            'initial_residual_max_abs': z_info[
                'initial_residual_max_abs'],
            'diagonal_initial_residual_max_abs': z_info[
                'diagonal_initial_residual_max_abs'],
        }
        response_state = store_response_state(
            mf_grad, signature, owner='stress',
            mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ,
            Co=Co, Cv=Cv, dm0=dm0, target_fock=target_fock,
            fvo=fvo, foo=foo, zvec=zvec, z_info=z_info,
            response_orbitals=response_orbitals, dz=dz,
            z_response_profile=response_profile,
            z_warm_start_metadata=z_warm_start_metadata)
        response_profile = use_response_state(
            mf_grad, response_state, reused=False)
    else:
        zvec = response_state['zvec']
        z_info = response_state['z_info']
        response_orbitals = response_state['response_orbitals']
        dz = response_state['dz']
        response_profile = use_response_state(
            mf_grad, response_state, reused=True)
        target_fock_reused = True
    stage_stop(profile, 'z_response', response_stage)

    mf_grad.target_fock_reused = target_fock_reused
    mf_grad.z_initial_guess = None
    return {
        'mo_energy': mo_energy,
        'mo_coeff': mo_coeff,
        'mo_occ': mo_occ,
        'Co': Co,
        'Cv': Cv,
        'e_occ': mo_energy[occupied],
        'dm0': dm0,
        'target_fock': target_fock,
        'fvo': fvo,
        'foo': foo,
        'fvind': fvind,
        'zvec': zvec,
        'response_orbitals': response_orbitals,
        'dz': dz,
        'z_info': z_info,
        'z_response_profile': response_profile,
        'target_fock_reused': target_fock_reused,
        'response_state_reused': mf_grad.response_state_reused,
        'response_state_owner': mf_grad.response_state_owner,
        'response_state_token': mf_grad.response_state_token,
        'z_solve_count': mf_grad.z_solve_count,
    }


def nonvariational_correction(
        mf_grad, state=None, reciprocal_block_size=None,
        auxiliary_block_size=None,
        overlap_gdf_short_long_range=True,
        pp_reciprocal_mesh=None, pp_block_size=None,
        validate_central=False, hcore_value_tol=1.e-7,
        gdf_response=None, hcore_components=None, profile=None):
    r'''Assemble the three GPU nonvariational strain terms.

    The Z solve and reverse GDF observable construction are device resident.
    The central solved source is reconstructed from the active provider
    CDERI/metric factorization.  Reciprocal GDF strain consumers run on the
    device where GPU4PySCF exposes the required endpoint; screened SR and
    one-center Fourier moments remain explicit analytic host contractions.
    Local-part1 and local-part2 are contracted for ``P`` and ``D_Z`` by
    direct SR/LR kernels, while the nonlocal GTH term uses the provider's
    exact half integrals and analytic lattice-image moments.  No PP strain
    AO matrix or displaced Cell is formed.  The remaining execution
    boundaries are recorded on ``mf_grad``; no CPU SCF or silent production
    fallback is used.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.rhf_dft.Gradients
            Active GPU DC-DFT derivative driver.

    Kwargs:
        state : dict or None
            Perturbation-independent target-Fock/Z state. ``None`` builds it
            once; callers assembling all terms pass the shared state.
        reciprocal_block_size : int or None
            Optional G-vector block shared by GDF and local-PP reciprocal
            contractions. ``None`` uses the provider memory planner.
        auxiliary_block_size : int or None
            Reverse-GDF auxiliary block. It changes temporary storage only.
        overlap_gdf_short_long_range : bool
            Overlap the independent host screened GDF contractions with the
            corresponding device reciprocal contractions.
        pp_reciprocal_mesh, pp_block_size
            Optional pseudopotential oracle controls.
        validate_central : bool
            Require the assembled analytic Hcore value to reproduce the
            active provider Hcore within ``hcore_value_tol``.
        hcore_value_tol : float
            Central Hcore/CDERI gauge-parity threshold.
        gdf_response, hcore_components : dict or None
            Precomputed exact components used by term-level tests or callers
            sharing one derivative construction.
        profile : dict or None
            Opt-in non-overlapping stress profiler ledger.

    Returns:
        ndarray of shape (3, 3)
            Energy-like nonvariational strain work in Hartree.
    '''
    if state is None:
        state = make_z_state(mf_grad)
    mf = mf_grad.base
    cell = mf.cell
    Co = cp.asarray(state['Co'])
    Cv = cp.asarray(state['Cv'])
    dm0 = cp.asarray(state['dm0'])
    zvec = cp.asarray(state['zvec'])
    dz = cp.asarray(state['dz'])

    _progress(mf_grad, 'one_electron_overlap_kinetic')
    one_electron_stage = stage_start(profile)
    overlap = overlap_strain(cell)
    kinetic = (
        kinetic_strain(cell) if hcore_components is None
        else hcore_components['kinetic'])
    stage_stop(profile, 'one_electron_overlap_kinetic', one_electron_stage)
    if hcore_components is None:
        _progress(mf_grad, 'pseudopotential')
        pp_stage = stage_start(profile)
        pseudopotential = contract_gdf_pseudopotential_strain(
            mf.with_df, cp.stack((dm0, dz)),
            reciprocal_mesh=pp_reciprocal_mesh,
            block_size=pp_block_size,
            source_block_size=reciprocal_block_size,
            value_tol=hcore_value_tol)
        stage_stop(profile, 'pseudopotential', pp_stage)
        hcore_components = {
            'kinetic': kinetic,
            'pseudopotential': pseudopotential,
        }
    pseudopotential = hcore_components['pseudopotential']
    hcore_value = cp.asarray(
        np.asarray(kinetic['value'])
        + np.asarray(pseudopotential['value']))
    provider_hcore = cp.asarray(mf.mf_hf.get_hcore())
    hcore_value_parity = float(cp.max(cp.abs(
        hcore_value - provider_hcore)).get())
    if validate_central and hcore_value_parity > float(hcore_value_tol):
        raise RuntimeError(
            'analytic strain Hcore does not reproduce the GPU provider: '
            f'{hcore_value_parity:.3e} > {float(hcore_value_tol):.3e}')
    kinetic1 = cp.asarray(kinetic['raw_derivative'])

    if gdf_response is None:
        _progress(mf_grad, 'gaussian_df_source_metric')
        gdf_stage = stage_start(profile)
        if getattr(mf.with_df, 'backend_name', None) == 'gdf':
            gdf_response = make_gdf_strain_response(
                mf.with_df, dm0, dz,
                reciprocal_block_size=reciprocal_block_size,
                auxiliary_block_size=auxiliary_block_size,
                value_tol=hcore_value_tol,
                profile_components=profile is not None,
                overlap_short_long_range=overlap_gdf_short_long_range)
        elif getattr(mf.with_df, 'backend_name', None) == 'rsdf':
            gdf_response = make_rsdf_strain_response(
                mf.with_df, dm0, dz,
                reciprocal_block_size=reciprocal_block_size,
                auxiliary_block_size=auxiliary_block_size,
                profile_components=profile is not None,
                overlap_short_long_range=overlap_gdf_short_long_range)
        else:
            raise TypeError(
                'Gaussian-DF stress requires a GDF or RSDF provider')
        stage_stop(profile, 'gaussian_df_source_metric', gdf_stage)
    pp_contraction = cp.asarray(pseudopotential['raw_contraction'])
    if pp_contraction.shape != (2, 3, 3):
        raise ValueError(
            'contracted PP response must contain P and D_Z channels')
    work_hcore1 = (
        work_fock_hcore(dz, kinetic1) - 2 * pp_contraction[1])
    work_gdf1 = work_fock_gdf(gdf_response)
    _progress(mf_grad, 'gamma_ewald_exchange')
    ewald_stage = stage_start(profile)
    if mf.mf_hf.exxdiv == 'ewald':
        ewald_exchange = gamma_ewald_exchange_strain(
            cell, cp.asnumpy(dm0))
        work_ewald1 = work_fock_ewald(
            dz, ewald_exchange['raw_derivative'])
    elif mf.mf_hf.exxdiv is None:
        ewald_exchange = None
        work_ewald1 = cp.zeros((3, 3))
    else:
        raise NotImplementedError(
            "Gaussian-DF stress supports source exxdiv=None or 'ewald'")
    stage_stop(profile, 'gamma_ewald_exchange', ewald_stage)
    work_fock1 = work_hcore1 + work_gdf1 + work_ewald1

    _progress(mf_grad, 'nonvariational_overlap_terms')
    overlap_stage = stage_start(profile)
    s1oo, s1vo = make_s1_oo_vo(
        Co, Cv, cp.asarray(overlap['raw_derivative']))
    work_energy = work_ovlp_energy(
        s1vo, state['e_occ'], zvec)
    work_response = work_ovlp_response(
        Co, dz, s1oo, state['fvind'])
    stage_stop(profile, 'nonvariational_overlap_terms', overlap_stage)
    work_nvar = work_fock1 + work_energy + work_response

    mf_grad.s1oo_strain = s1oo
    mf_grad.s1vo_strain = s1vo
    mf_grad.z_density = dz
    mf_grad.work_fock_hcore = cp.asnumpy(work_hcore1)
    mf_grad.work_fock_gdf = cp.asnumpy(work_gdf1)
    mf_grad.work_fock_ewald = cp.asnumpy(work_ewald1)
    mf_grad.work_fock = cp.asnumpy(work_fock1)
    mf_grad.work_ovlp_energy = cp.asnumpy(work_energy)
    mf_grad.work_ovlp_response = cp.asnumpy(work_response)
    mf_grad.work_nvar = cp.asnumpy(work_nvar)
    mf_grad.gaussian_df_strain_response = gdf_response
    mf_grad.gdf_strain_response = gdf_response
    mf_grad.hcore_strain_value_parity_max_abs = hcore_value_parity
    mf_grad.hcore_strain_components = hcore_components
    mf_grad.ewald_exchange_strain = ewald_exchange
    pp_host = tuple(
        f'GDF_GTH_PP_{name}'
        for name in pseudopotential['host_contracted_primitives'])
    mf_grad.stress_host_primitive_boundary = tuple(dict.fromkeys((
        'overlap_matrix_for_nonvariational_vo_projection',
        'kinetic_matrix_for_z_hcore',
        *gdf_response['host_primitive_boundary'],
        *pp_host,
        'analytic_gamma_ewald_exchange')))
    return mf_grad.work_nvar


def _progress(mf_grad, label):
    '''Emit one flush-safe stress substage without changing any array.'''
    callback = getattr(mf_grad, 'progress_callback', None)
    if callback is not None:
        callback(f'stress/{label}')


__all__ = [
    'contract_strain_source',
    'make_s1_oo_vo',
    'make_strain_source',
    'make_z_density',
    'make_z_state',
    'nonvariational_correction',
    'work_fock_ewald',
    'work_fock_gdf',
    'work_fock_hcore',
    'work_ovlp_energy',
    'work_ovlp_response',
]
