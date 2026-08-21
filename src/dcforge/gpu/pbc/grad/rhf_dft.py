'''Gamma GPU FFTDF/GDF/RSDF DFA@RHF gradients.'''

import time

import cupy as cp
import numpy as np
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.grad import rhf as gpu_rhf_grad
from gpu4pyscf.pbc.gto import int1e
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.grad import krhf as cpu_krhf_grad

from dcforge.gpu.pbc.df.fft import FFTDF, grad_jk
from dcforge.gpu.pbc.df.gaussian.hcore import (
    contract_hcore_pair as contract_gaussian_hcore_pair,
)
from dcforge.gpu.pbc.df.gdf import (
    GDF,
    restricted_hf_bilinear_derivative_direct,
    target_j_and_fock_energy_derivative,
    target_j_fock_energy_derivative,
    target_j_fock_signed_energy_derivative,
)
from dcforge.gpu.pbc.df.rhf_mixed_response import solve_mixed_gdf_z_vector
from dcforge.gpu.pbc.df.rhf_response import (
    gen_vind as gen_df_vind,
)
from dcforge.gpu.pbc.df.rhf_response import (
    gen_vind_ao_oracle as gen_df_vind_ao_oracle,
)
from dcforge.gpu.pbc.df.rsdf import RSDF, make_gradient_response
from dcforge.gpu.pbc.energy.rhf_dft import (
    RHFDFT,
    _check_input,
    _target_fock_signature,
    _target_xc_name,
)
from dcforge.gpu.pbc.grad.memory import resolve_memory_policy
from dcforge.gpu.pbc.grad.rhf_nv import (
    grad_fock_direct,
    grad_fock_gdf,
    grad_fock_rsdf,
    make_ovlp_response_intermediate,
    nonvariational_correction,
)
from dcforge.gpu.pbc.grad.rhf_state import (
    get_response_state,
    make_response_state_signature,
    store_response_state,
    use_response_state,
)
from dcforge.gpu.pbc.grad.rhf_zvec import gen_vind, solve_z_vector
from dcforge.gpu.pbc.grad.skeleton import (
    contract_hcore_pair,
    effective_hcore_batch_size,
    get_hcore_generator,
    should_stream_hcore,
    skeleton_gradient,
)


def _profile_start(mf_grad):
    '''Synchronize and start one optional validation-only GPU stage timer.'''
    if not mf_grad.profile_gradient_stages:
        return None
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


def _profile_stop(mf_grad, profile, label, start):
    '''Synchronize and retain one optional validation-only GPU stage time.'''
    if start is None:
        return
    cp.cuda.Stream.null.synchronize()
    profile[label] = time.perf_counter() - start


def _gdf_response_boundary_start(mf_grad):
    '''Start the opt-in profiler-off GDF response boundary timer.

    This validation timer introduces exactly one synchronization before the
    native response.  Unlike ``profile_gradient_stages``, it does not enable
    any timer inside the GDF implementation.
    '''
    if not mf_grad.measure_gdf_response_boundary:
        return None
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


def _gdf_response_boundary_stop(mf_grad, start):
    '''Finish the opt-in GDF response boundary with one synchronization.'''
    if start is None:
        return
    cp.cuda.Stream.null.synchronize()
    mf_grad.gdf_response_boundary_seconds = time.perf_counter() - start


def _progress(mf_grad, label):
    '''Publish one optional non-scientific queue/profiler stage marker.'''
    callback = getattr(mf_grad, 'progress_callback', None)
    if callback is not None:
        callback(str(label))


def _gdf_workspace_metadata(metadata):
    '''Return only bounded-workspace diagnostics from native GDF metadata.'''
    names = (
        'three_center_workspace_plan',
        'short_range_source_workspace_plan',
        'metric_solve_workspace_plan',
        'metric_solve_workspace_plans',
        'solved_three_center_bytes',
        'dense_solved_three_center_bytes_avoided',
        'signed_state_full_rank_bytes_avoided',
        'reciprocal_pair_cache',
    )
    workspace = {
        name: metadata[name] for name in names if name in metadata}
    adjoint = metadata.get('adjoint')
    if isinstance(adjoint, dict):
        # Detailed times already live in ``gdf_response_profile``.  Retain
        # only the execution plan here so the queue can relate a measured
        # pass count to auxiliary block size without duplicating timers.
        workspace['adjoint_plan'] = {
            name: value for name, value in adjoint.items()
            if name != 'profile'
        }
    return workspace


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None):
    '''Evaluate the electronic Gamma-point GPU DFA@RHF gradient.

    The term ledger is identical to the CPU PBC implementation:

    ``de_elec = de_hcore + de_veff + de_ovlp``
    ``        + de_fock + de_ovlp_energy + de_ovlp_response``.

    Args:
        mf_grad : Gradients
            Active GPU PBC gradient object.

    Kwargs:
        mo_energy, mo_coeff, mo_occ : cupy.ndarray or None
            Canonical source-RHF orbitals; converged values are used by
            default.
        atmlst : sequence of int or None
            Requested centers.  Stage 2 implements all-atom gradients only.

    Returns:
        ndarray of shape (natm, 3)
            Electronic energy derivative in Ha/Bohr.
    '''
    mf = mf_grad.base
    mf_hf = mf.mf_hf
    cell = mf_grad.cell
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    _check_input(mf)
    resolve_memory_policy(mf_grad, required_property='gradient')
    if not isinstance(mf.with_df, (FFTDF, GDF, RSDF)):
        raise NotImplementedError('unsupported GPU PBC analytic-gradient provider')
    if mf.e_dft is None:
        raise RuntimeError('run the periodic RHFDFT energy kernel first')
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    if atmlst is None: atmlst = range(cell.natm)
    atmlst = list(atmlst)
    if atmlst != list(range(cell.natm)):
        raise NotImplementedError('GPU PBC gradients currently require all atoms')

    # GPU kernels are normally asynchronous.  Stage synchronization is
    # deliberately opt-in because it is validation telemetry, not part of
    # the production mathematical path or a performance optimization.
    profile = {} if mf_grad.profile_gradient_stages else None
    mf_grad.z_response_profile = None
    mf_grad.response_state_reused = False
    mf_grad.z_warm_start_metadata = None
    mf_grad.hcore_stage_profile = None
    mf_grad.hcore_pp_loc_part1_plan = None
    mf_grad.de_hcore_components = None
    mf_grad.de_fock_hcore_components = None
    mf_grad.gdf_adjoint_backend_used = None
    mf_grad.gdf_adjoint_backend_fallback_reason = None
    mf_grad.gdf_response_boundary_seconds = None
    if (mf_grad.measure_gdf_response_boundary
            and mf_grad.profile_gradient_stages):
        raise ValueError(
            'measure_gdf_response_boundary requires '
            'profile_gradient_stages=False')
    if (mf_grad.measure_gdf_response_boundary
            and not isinstance(mf.with_df, GDF)):
        raise NotImplementedError(
            'measure_gdf_response_boundary requires a GPU GDF provider')
    if mf_grad.hcore_method not in ('auto', 'gdf', 'grid'):
        raise ValueError("hcore_method must be 'auto', 'gdf', or 'grid'")
    if mf_grad.hcore_method == 'gdf' and not isinstance(mf.with_df, GDF):
        raise NotImplementedError(
            "hcore_method='gdf' requires a GPU GDF provider; "
            "use 'auto' or 'grid'")
    precision_policy = str(mf_grad.precision_policy).lower()
    if precision_policy not in ('fp64', 'mixed', 'mixed_lowmem'):
        raise ValueError(
            'precision_policy must be fp64, mixed, or mixed_lowmem')
    if precision_policy != 'fp64' and not isinstance(mf.with_df, GDF):
        raise NotImplementedError(
            'mixed precision currently requires real-Gamma GDF')
    derivative_precision = str(
        mf_grad.gdf_derivative_precision).lower()
    if derivative_precision not in ('fp64', 'mixed'):
        raise ValueError(
            'gdf_derivative_precision must be fp64 or mixed')
    if derivative_precision != 'fp64' and not isinstance(mf.with_df, GDF):
        raise NotImplementedError(
            'mixed GDF derivative contractions require real-Gamma GDF')
    if isinstance(mf.with_df, GDF):
        mf.with_df.gdf_derivative_precision = derivative_precision
    mf_grad.precision_policy_effective = precision_policy
    mf_grad.precision_fallback_reason = None
    mf_grad.mixed_precision_report = None
    total_start = _profile_start(mf_grad)

    mo_energy = cp.asarray(mo_energy)
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    so = mo_occ > 0
    sv = mo_occ == 0
    if not bool(cp.all((mo_occ == 2) | sv)):
        raise ValueError('periodic GPU RHF-DFT requires exact 2/0 occupations')
    if cp.iscomplexobj(mo_coeff):
        raise NotImplementedError('the first GPU Gamma gradient requires real orbitals')
    Co = mo_coeff[:, so]
    Cv = mo_coeff[:, sv]
    dm0 = mf_hf.make_rdm1(mo_coeff, mo_occ)

    _progress(mf_grad, 'target_fock')
    stage_start = _profile_start(mf_grad)
    target_signature = _target_fock_signature(mf, dm0)
    if (mf.target_fock is not None and mf.dm_hf is not None
            and mf.target_fock_signature == target_signature):
        fock = cp.asarray(mf.target_fock)
        mf_grad.target_fock_reused = True
    else:
        fock = mf.get_fock(dm=dm0)
        mf_grad.target_fock_reused = False
    _profile_stop(mf_grad, profile, 'target_fock_seconds', stage_start)
    fock_mo = mo_coeff.conj().T @ fock @ mo_coeff
    fvo = fock_mo[sv][:, so]
    foo = fock_mo[so][:, so]

    fvind = mf_hf.gen_response(
        mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=1)
    response_signature = make_response_state_signature(
        mf_grad, mo_energy, mo_coeff, mo_occ, dm0)
    response_state = get_response_state(mf_grad, response_signature)
    _progress(mf_grad, 'z_vector')
    stage_start = _profile_start(mf_grad)
    if response_state is None:
        if precision_policy != 'fp64':
            if mf_grad.df_cphf_response_mode != 'direct_vo':
                raise ValueError(
                    "precision_policy='mixed' requires "
                    "df_cphf_response_mode='direct_vo'")
            zvec, z_info, mixed_profile = solve_mixed_gdf_z_vector(
                mf_hf, mo_energy, mo_occ, Co, Cv, fvo,
                conv_tol=mf_grad.cphf_conv_tol,
                residual_tol=mf_grad.cphf_residual_tol,
                max_cycle=mf_grad.cphf_max_cycle,
                refinement_solver=mf_grad.cphf_refinement_solver,
                initial_guess=mf_grad.z_initial_guess,
                require_hdf5=precision_policy == 'mixed_lowmem',
                log=log)
            mf_grad.z_response_profile = mixed_profile
            mf_grad.mixed_precision_report = mixed_profile
            fallback = mixed_profile.get('fallback')
            if fallback is not None:
                mf_grad.precision_policy_effective = 'fp64_fallback'
                mf_grad.precision_fallback_reason = fallback['reason']
        elif isinstance(mf.with_df, (GDF, RSDF)):
            # Keep GPU4PySCF's AO-density response as the oracle used by the
            # moving-overlap term below.  Repeated Z-vector actions can
            # exploit their known cross-factor density without changing the
            # PySCF ``gen_response`` contract or the Hessian equation.
            response_mode = mf_grad.df_cphf_response_mode
            if response_mode == 'direct_vo':
                vind = gen_df_vind(
                    mf_hf, Co, Cv,
                    cache_mode=mf_grad.df_cphf_response_cache,
                    memory_fraction=(
                        mf_grad.df_cphf_response_memory_fraction),
                    minimum_reserve_bytes=(
                        mf_grad.df_cphf_response_minimum_reserve_bytes),
                    profile_detail=(
                        mf_grad.profile_gradient_stages
                        or mf_grad.profile_df_cphf_response))
            elif response_mode == 'ao_oracle':
                vind = gen_df_vind_ao_oracle(mf_hf, Co, Cv)
            else:
                raise ValueError(
                    'df_cphf_response_mode must be direct_vo or ao_oracle')
        else:
            vind = gen_vind(Co, Cv, fvind)
        if precision_policy == 'fp64':
            try:
                zvec, z_info = solve_z_vector(
                    mo_energy, mo_occ, fvo, vind,
                    conv_tol=mf_grad.cphf_conv_tol,
                    krylov_tol=mf_grad.cphf_krylov_tol,
                    residual_tol=mf_grad.cphf_residual_tol,
                    max_cycle=mf_grad.cphf_max_cycle,
                    refinement_solver=mf_grad.cphf_refinement_solver,
                    initial_guess=mf_grad.z_initial_guess,
                    log=log,
                )
            finally:
                close_response = getattr(vind, 'close', None)
                if close_response is not None:
                    mf_grad.z_response_profile = close_response()
                if 'z_info' not in locals():
                    mf_grad.z_initial_guess = None
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
        response_orbitals = Cv @ zvec
        dz = response_orbitals @ Co.conj().T
        dz = dz + dz.conj().T
        response_state = store_response_state(
            mf_grad, response_signature, owner='gradient',
            mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ,
            Co=Co, Cv=Cv, dm0=dm0, target_fock=fock, fvo=fvo, foo=foo,
            zvec=zvec, z_info=z_info,
            response_orbitals=response_orbitals, dz=dz,
            z_response_profile=mf_grad.z_response_profile,
            z_warm_start_metadata=z_warm_start_metadata)
        use_response_state(mf_grad, response_state, reused=False)
    else:
        zvec = response_state['zvec']
        z_info = response_state['z_info']
        response_orbitals = response_state['response_orbitals']
        dz = response_state['dz']
        use_response_state(mf_grad, response_state, reused=True)
        z_warm_start_metadata = response_state.get(
            'z_warm_start_metadata')
    if precision_policy != 'fp64' and mf_grad.mixed_precision_report is None:
        # A second gradient/stress call may reuse the bounded response state
        # without rebuilding FP32 CDERI.  Preserve the original precision
        # ledger so reuse is never mislabeled as an ordinary FP64 solve.
        mf_grad.mixed_precision_report = mf_grad.z_response_profile
        fallback = (mf_grad.mixed_precision_report or {}).get('fallback')
        if fallback is not None:
            mf_grad.precision_policy_effective = 'fp64_fallback'
            mf_grad.precision_fallback_reason = fallback['reason']
    _profile_stop(mf_grad, profile, 'z_vector_seconds', stage_start)
    mf_grad.z_warm_start_metadata = z_warm_start_metadata
    # A projected guess belongs to exactly one geometry and one solve.
    mf_grad.z_initial_guess = None
    # ``U=C_v Z`` is both the natural response-orbital factor of
    # ``D_Z=U C_o^T+C_o U^T`` and the existing first half of the AO-density
    # construction.  Retain it so the optional MO-factorized GDF adjoint can
    # consume the exact physical factors without refactorizing ``D_Z``.
    ovlp_response_intermediate = None
    release_cderi = (
        isinstance(mf.with_df, (GDF, RSDF))
        and bool(mf.with_df.release_cderi_after_response))
    if release_cderi:
        _progress(mf_grad, 'overlap_response_intermediate')
        stage_start = _profile_start(mf_grad)
        ovlp_response_intermediate = make_ovlp_response_intermediate(
            Co, Cv, zvec, fvind)
        _profile_stop(
            mf_grad, profile, 'overlap_response_intermediate_seconds',
            stage_start)

    if release_cderi and isinstance(mf.with_df, GDF):
        # Native GDF nuclear response rebuilds source/metric primitives from
        # ``auxcell`` and no longer consumes the SCF CDERI.  Releasing here
        # prevents mapped host factors from overlapping that large stage.
        mf_grad.cderi_release_metadata = mf.with_df.release_cderi()

    if isinstance(mf.with_df, FFTDF):
        # FFTDF uses the public full-matrix Hcore derivative route.
        hcore_gen = get_hcore_generator(cell)
        s1 = -int1e.int1e_ipovlp(cell, np.zeros((1, 3)))[0]
        # This GPU kernel returns full AO derivatives.  They are shared by the
        # target Hartree skeleton and source-Fock Z contraction.
        _progress(mf_grad, 'fftdf_jk_derivative')
        stage_start = _profile_start(mf_grad)
        vj1, vk1 = grad_jk(
            cell, dm0, exxdiv=mf_hf.exxdiv, with_df=mf.with_df)
        _profile_stop(mf_grad, profile, 'fftdf_jk_derivative_seconds',
                      stage_start)
        mf_grad.vj1 = vj1
        mf_grad.vk1 = vk1
        _progress(mf_grad, 'skeleton')
        stage_start = _profile_start(mf_grad)
        de_skel = skeleton_gradient(
            mf_grad, dm0, foo, Co, s1, hcore_gen, vj1, atmlst)
        _profile_stop(mf_grad, profile, 'skeleton_seconds', stage_start)
        _progress(mf_grad, 'source_fock')
        stage_start = _profile_start(mf_grad)
        de_fock = grad_fock_direct(
            mf_grad, Co, Cv, zvec, dm0, hcore_gen, vj1, vk1, atmlst)
        _profile_stop(mf_grad, profile, 'source_fock_seconds', stage_start)
    elif isinstance(mf.with_df, GDF):
        # Native GDF fuses target J and the signed source-Fock two-electron
        # response in one primitive traversal.  It is an exact algebraic
        # replacement for separate ``de_j`` and ``de_fock_native`` values,
        # but is deliberately retained as its own coupled ledger field.
        mf_grad.vj1 = None
        mf_grad.vk1 = None
        response_mode = mf_grad.gdf_response_mode
        if response_mode not in ('auto', 'fused', 'signed', 'separate'):
            raise ValueError(
                'gdf_response_mode must be auto, fused, signed, or separate')
        adjoint_backend = mf_grad.gdf_adjoint_backend
        if adjoint_backend not in (
                'auto', 'packed', 'mo_factorized', 'mo_direct'):
            raise ValueError(
                'gdf_adjoint_backend must be auto, packed, mo_factorized, '
                'or mo_direct')
        if (adjoint_backend in ('mo_factorized', 'mo_direct')
                and response_mode in ('signed', 'separate')):
            raise ValueError(
                f"gdf_adjoint_backend='{adjoint_backend}' requires the direct "
                "gdf_response_mode='auto' or 'fused' route")
        if (derivative_precision == 'mixed'
                and response_mode not in ('auto', 'fused')):
            raise ValueError(
                "gdf_derivative_precision='mixed' requires "
                "gdf_response_mode='auto' or 'fused'")
        # ``P`` and ``D_Z`` contract the identical local H1.  ``auto`` uses
        # the analytic GDF-operator SR/LR pseudopotential representation and
        # never materializes the all-atom H1 tensor.  ``grid`` retains the
        # uniform-grid GPU4PySCF representation as an explicit oracle.
        use_gdf_hcore = mf_grad.hcore_method in ('auto', 'gdf')
        hcore_streamed = should_stream_hcore(cell)
        hcore_gen = (
            None if use_gdf_hcore or hcore_streamed
            else get_hcore_generator(cell))
        hcore_batch_size = effective_hcore_batch_size(
            cell, mf_grad.hcore_batch_size, streamed=hcore_streamed)
        mf_grad.hcore_backend = (
            'gdf_native_sr_lr_density_contracted' if use_gdf_hcore else
            ('gdf_streamed_density_contracted' if hcore_streamed
             else 'gpu4pyscf_hcore_generator_batched'))
        mf_grad.hcore_effective_batch_size = (
            None if use_gdf_hcore else hcore_batch_size)
        _progress(mf_grad, 'hcore_pair')
        stage_start = _profile_start(mf_grad)
        hcore_profile = {} if profile is not None else None
        if use_gdf_hcore:
            de_hcore, de_fock_hcore, hcore_components = (
                contract_gaussian_hcore_pair(
                    mf.with_df, dm0, dz, atmlst, profile=hcore_profile))
            local1_plan = mf.with_df.get_pp_loc_part1_plan()
            mf_grad.hcore_pp_loc_part1_plan = {
                'omega': local1_plan['omega'],
                'ke_cutoff': local1_plan['ke_cutoff'],
                'mesh': local1_plan['mesh'].tolist(),
                'mode': local1_plan['mode'],
            }
            mf_grad.de_hcore_components = {
                name: cp.asnumpy(value[0])
                for name, value in hcore_components.items()}
            mf_grad.de_fock_hcore_components = {
                name: cp.asnumpy(-2 * value[1])
                for name, value in hcore_components.items()}
        else:
            de_hcore, de_fock_hcore = contract_hcore_pair(
                cell, hcore_gen, dm0, dz, atmlst,
                atom_batch_size=hcore_batch_size, profile=hcore_profile)
            mf_grad.de_hcore_components = None
            mf_grad.de_fock_hcore_components = None
        _profile_stop(mf_grad, profile, 'hcore_pair_seconds', stage_start)
        mf_grad.hcore_stage_profile = hcore_profile
        if mf.multigrid_with_j:
            # Target J belongs to MG.  The Gaussian provider therefore owns
            # only the source-HF Z-Fock derivative.  A direct ``P,D_Z``
            # adjoint is algebraically identical to signed polarization but
            # avoids building the two full-rank ``P+D_Z``/``P-D_Z`` tensors.
            _progress(mf_grad, 'gdf_native_response')
            stage_start = _profile_start(mf_grad)
            boundary_start = _gdf_response_boundary_start(mf_grad)
            try:
                direct_fock, response_memory = (
                    restricted_hf_bilinear_derivative_direct(
                        mf_hf, dm0, dz, atmlst, return_metadata=True,
                        profile_detail=mf_grad.profile_gradient_stages,
                        occupied_coefficients=Co,
                        response_orbitals=response_orbitals,
                        adjoint_backend=adjoint_backend))
            finally:
                _gdf_response_boundary_stop(mf_grad, boundary_start)
            _profile_stop(
                mf_grad, profile, 'gdf_native_response_seconds', stage_start)
            s1 = -int1e.int1e_ipovlp(cell, np.zeros((1, 3)))[0]
            _progress(mf_grad, 'skeleton')
            stage_start = _profile_start(mf_grad)
            de_skel = skeleton_gradient(
                mf_grad, dm0, foo, Co, s1, None, None, atmlst,
                de_hcore_energy=de_hcore)
            _profile_stop(mf_grad, profile, 'skeleton_seconds', stage_start)
            mf_grad.de_j_fock_native = None
            mf_grad.de_j_fock_source = None
            mf_grad.de_j_fock_metric = None
            mf_grad.de_j_fock_ewald = None
            mf_grad.gdf_term_partition = (
                'multigrid_target_j_xc_plus_direct_gdf_source_fock')
            mf_grad.gdf_reciprocal_memory_plan = response_memory[
                'reciprocal_plan']
            mf_grad.gdf_reciprocal_zero_order = response_memory[
                'reciprocal_zero_order']
            mf_grad.gdf_reciprocal_derivative = response_memory[
                'reciprocal_derivative']
            mf_grad.gdf_response_profile = response_memory['profile']
            mf_grad.gdf_response_workspace = _gdf_workspace_metadata(
                response_memory)
            adjoint_metadata = response_memory['adjoint']
            mf_grad.gdf_adjoint_backend_used = adjoint_metadata[
                'backend_resolved']
            mf_grad.gdf_adjoint_backend_fallback_reason = adjoint_metadata[
                'backend_fallback_reason']
            components = response_memory['term_components']
            mf_grad.gdf_response_range_components = response_memory.get(
                'term_components_by_range')
            _progress(mf_grad, 'source_fock')
            stage_start = _profile_start(mf_grad)
            de_fock = grad_fock_gdf(
                mf_grad, Co, Cv, zvec, dm0, None, atmlst,
                hcore_derivative=de_fock_hcore,
                source_derivative=-cp.asarray(components['source']),
                metric_derivative=-cp.asarray(components['metric']),
                ewald_derivative=-cp.asarray(components['ewald']),
                resolved_backend='gdf_direct_source_fock_adjoint')
            mf_grad.de_j_fock_term_residual_max_abs = float(np.max(np.abs(
                cp.asnumpy(-direct_fock)
                - mf_grad.de_fock_source - mf_grad.de_fock_metric
                - mf_grad.de_fock_ewald)))
            _profile_stop(
                mf_grad, profile, 'source_fock_seconds', stage_start)
            _progress(mf_grad, 'nonvariational')
            stage_start = _profile_start(mf_grad)
            de_nvar = nonvariational_correction(
                mf_grad, Co, Cv, mo_energy, zvec, s1,
                de_fock, fvind, atmlst,
                ovlp_response_intermediate=ovlp_response_intermediate)
            _profile_stop(
                mf_grad, profile, 'nonvariational_seconds', stage_start)
            _profile_stop(mf_grad, profile, 'total_seconds', total_start)
            mf_grad.gradient_stage_profile = profile
            return de_skel + de_nvar
        target_xc = _target_xc_name(mf)
        use_signed = response_mode == 'signed'
        use_separate = not use_signed and (
            response_mode == 'separate'
            or (response_mode == 'auto'
                and target_xc != 'R2SCAN'
                and mf.with_df.native_multi_observable_library is not None))
        if target_xc == 'R2SCAN' and response_mode == 'separate':
            raise NotImplementedError(
                'GPU GDF R2SCAN separate target-J/Z-Fock channels have not '
                'passed the optional two-channel process-safety gate; use '
                'auto, signed, or the coupled direct-adjoint fused route')
        _progress(mf_grad, 'gdf_native_response')
        stage_start = _profile_start(mf_grad)
        boundary_start = _gdf_response_boundary_start(mf_grad)
        try:
            if use_separate:
                gdf_terms, response_memory = target_j_and_fock_energy_derivative(
                    mf_hf, dm0, dz, atmlst)
            elif use_signed:
                de_j_fock_native, response_memory = (
                    target_j_fock_signed_energy_derivative(
                        mf_hf, dm0, dz, atmlst, return_metadata=True))
            else:
                de_j_fock_native, response_memory = (
                    target_j_fock_energy_derivative(
                        mf_hf, dm0, dz, atmlst, return_metadata=True,
                        profile_detail=mf_grad.profile_gradient_stages,
                        occupied_coefficients=Co,
                        response_orbitals=response_orbitals,
                        adjoint_backend=adjoint_backend))
        finally:
            _gdf_response_boundary_stop(mf_grad, boundary_start)
        _profile_stop(mf_grad, profile, 'gdf_native_response_seconds',
                      stage_start)
        mf_grad.gdf_response_workspace = _gdf_workspace_metadata(
            response_memory)
        # The high-mesh Hcore and primitive GDF response above do not depend
        # on overlap derivatives.  Build ``S^lambda`` only after them, which
        # prevents this independent GPU4PySCF integral workspace from being
        # live during the large local-GTH grid traversal.
        s1 = -int1e.int1e_ipovlp(cell, np.zeros((1, 3)))[0]
        _progress(mf_grad, 'skeleton')
        stage_start = _profile_start(mf_grad)
        de_skel = skeleton_gradient(
            mf_grad, dm0, foo, Co, s1, None, None, atmlst,
            de_j_energy=(gdf_terms['de_j'] if use_separate
                         else cp.zeros((len(atmlst), 3))),
            de_hcore_energy=de_hcore)
        _profile_stop(mf_grad, profile, 'skeleton_seconds', stage_start)
        if use_separate:
            mf_grad.de_j = cp.asnumpy(gdf_terms['de_j'])
            mf_grad.de_j_source = cp.asnumpy(gdf_terms['de_j_source'])
            mf_grad.de_j_metric = cp.asnumpy(gdf_terms['de_j_metric'])
            mf_grad.de_j_native = None
            mf_grad.de_j_backend = 'gdf_two_channel_native_adjoint'
            mf_grad.de_j_fock_native = None
            mf_grad.de_j_fock_source = cp.asnumpy(
                gdf_terms['de_j_source'] + gdf_terms['de_fock_source'])
            mf_grad.de_j_fock_metric = cp.asnumpy(
                gdf_terms['de_j_metric'] + gdf_terms['de_fock_metric'])
            mf_grad.de_j_fock_ewald = cp.asnumpy(
                gdf_terms['de_j_ewald'] + gdf_terms['de_fock_ewald'])
            mf_grad.de_j_fock_term_residual_max_abs = max(
                response_memory['target_term_component_residual_max_abs'],
                response_memory['fock_term_component_residual_max_abs'])
            mf_grad.gdf_term_partition = (
                'target_j_and_z_fock_separate_source_metric_ewald')
            mf_grad.gdf_reciprocal_memory_plan = response_memory['reciprocal_plan']
            mf_grad.gdf_reciprocal_zero_order = response_memory['reciprocal_zero_order']
            mf_grad.gdf_reciprocal_derivative = {
                'target': response_memory['target_reciprocal_derivative'],
                'fock': response_memory['fock_reciprocal_derivative'],
            }
            mf_grad.gdf_response_profile = response_memory['profile']
            _progress(mf_grad, 'source_fock')
            stage_start = _profile_start(mf_grad)
            de_fock = grad_fock_gdf(
                mf_grad, Co, Cv, zvec, dm0, None, atmlst,
                hcore_derivative=de_fock_hcore,
                source_derivative=gdf_terms['de_fock_source'],
                metric_derivative=gdf_terms['de_fock_metric'],
                ewald_derivative=gdf_terms['de_fock_ewald'])
            _profile_stop(mf_grad, profile, 'source_fock_seconds', stage_start)
        else:
            if not use_signed:
                adjoint_metadata = response_memory['adjoint']
                mf_grad.gdf_adjoint_backend_used = adjoint_metadata[
                    'backend_resolved']
                mf_grad.gdf_adjoint_backend_fallback_reason = (
                    adjoint_metadata['backend_fallback_reason'])
            mf_grad.de_j = None
            mf_grad.de_j_native = None
            mf_grad.de_j_backend = (
                'gdf_signed_into_de_j_fock_native' if use_signed
                else 'gdf_fused_into_de_j_fock_native')
            mf_grad.de_j_fock_native = cp.asnumpy(de_j_fock_native)
            # The GDF target-J and source-Fock observables must remain fused in
            # one native traversal.  The primitive derivative itself is still
            # linear in ``dB``, ``dM`` and the Ewald overlap correction, so expose
            # that exact subterm partition without pretending to have dispatched
            # separately cached target-J and source-Fock kernels.
            components = response_memory['term_components']
            mf_grad.gdf_response_range_components = response_memory.get(
                'term_components_by_range')
            mf_grad.de_j_fock_source = np.asarray(components['source'])
            mf_grad.de_j_fock_metric = np.asarray(components['metric'])
            mf_grad.de_j_fock_ewald = np.asarray(components['ewald'])
            mf_grad.de_j_fock_term_residual_max_abs = response_memory[
                'term_component_residual_max_abs']
            mf_grad.gdf_term_partition = (
                'target_j_and_z_fock_signed_source_metric_ewald'
                if use_signed else
                'target_j_and_z_fock_coupled_source_metric_ewald')
            mf_grad.gdf_reciprocal_memory_plan = response_memory['reciprocal_plan']
            mf_grad.gdf_reciprocal_zero_order = response_memory['reciprocal_zero_order']
            mf_grad.gdf_reciprocal_derivative = response_memory['reciprocal_derivative']
            mf_grad.gdf_response_profile = response_memory['profile']
            _progress(mf_grad, 'source_fock')
            stage_start = _profile_start(mf_grad)
            de_fock = grad_fock_gdf(
                mf_grad, Co, Cv, zvec, dm0, None, atmlst,
                hcore_derivative=de_fock_hcore)
            _profile_stop(mf_grad, profile, 'source_fock_seconds', stage_start)
    else:
        # RSDF retains independent source and metric range-separation
        # settings.  Its reverse-adjoint response therefore exposes the
        # target-J and Z-Fock observables as four separately contracted
        # scalars.  The only non-DF piece is Ewald exchange, which remains
        # an explicit overlap correction in the same ledger.
        mf_grad.vj1 = None
        mf_grad.vk1 = None
        use_analytic_hcore = mf_grad.hcore_method == 'auto'
        hcore_streamed = should_stream_hcore(cell)
        hcore_gen = (
            None if use_analytic_hcore or hcore_streamed
            else get_hcore_generator(cell))
        hcore_batch_size = effective_hcore_batch_size(
            cell, mf_grad.hcore_batch_size, streamed=hcore_streamed)
        mf_grad.hcore_backend = (
            'rsdf_native_sr_lr_density_contracted' if use_analytic_hcore else
            ('rsdf_streamed_density_contracted' if hcore_streamed
             else 'gpu4pyscf_hcore_generator_batched'))
        mf_grad.hcore_effective_batch_size = (
            None if use_analytic_hcore else hcore_batch_size)
        _progress(mf_grad, 'hcore_pair')
        stage_start = _profile_start(mf_grad)
        hcore_profile = {} if profile is not None else None
        if use_analytic_hcore:
            de_hcore, de_fock_hcore, hcore_components = (
                contract_gaussian_hcore_pair(
                    mf.with_df, dm0, dz, atmlst, profile=hcore_profile))
            local1_plan = mf.with_df.get_pp_loc_part1_plan()
            mf_grad.hcore_pp_loc_part1_plan = {
                'omega': local1_plan['omega'],
                'ke_cutoff': local1_plan['ke_cutoff'],
                'mesh': local1_plan['mesh'].tolist(),
                'mode': local1_plan['mode'],
            }
            mf_grad.de_hcore_components = {
                name: cp.asnumpy(value[0])
                for name, value in hcore_components.items()}
            mf_grad.de_fock_hcore_components = {
                name: cp.asnumpy(-2 * value[1])
                for name, value in hcore_components.items()}
        else:
            de_hcore, de_fock_hcore = contract_hcore_pair(
                cell, hcore_gen, dm0, dz, atmlst,
                atom_batch_size=hcore_batch_size, profile=hcore_profile)
            mf_grad.de_hcore_components = None
            mf_grad.de_fock_hcore_components = None
        _profile_stop(mf_grad, profile, 'hcore_pair_seconds', stage_start)
        mf_grad.hcore_stage_profile = hcore_profile
        _progress(mf_grad, 'rsdf_native_response')
        stage_start = _profile_start(mf_grad)
        rsdf_terms, rsdf_metadata = make_gradient_response(
            mf.with_df, dm0, dz, exxdiv=mf_hf.exxdiv,
            with_target_j=not mf.multigrid_with_j)
        _profile_stop(mf_grad, profile, 'rsdf_native_response_seconds',
                      stage_start)
        if release_cderi:
            # RSDF reverse-Cholesky needs central C, unlike the native GDF
            # primitive route.  Its final source/metric scalars mark the
            # earliest exact release point.
            mf_grad.cderi_release_metadata = mf.with_df.release_cderi()
        # Keep the overlap primitive out of the large RSDF CUDA response
        # lifetime.  ``S^lambda`` is independent of source/metric
        # derivatives and is only needed by skeleton/nonvariational terms.
        s1 = -int1e.int1e_ipovlp(cell, np.zeros((1, 3)))[0]
        de_j_source = (
            None if mf.multigrid_with_j else rsdf_terms['de_j_source'])
        de_j_metric = (
            None if mf.multigrid_with_j else rsdf_terms['de_j_metric'])
        _progress(mf_grad, 'skeleton')
        stage_start = _profile_start(mf_grad)
        de_skel = skeleton_gradient(
            mf_grad, dm0, foo, Co, s1, None, None, atmlst,
            de_j_energy=(
                None if de_j_source is None
                else de_j_source + de_j_metric),
            de_hcore_energy=de_hcore)
        _profile_stop(mf_grad, profile, 'skeleton_seconds', stage_start)
        mf_grad.de_j_source = (
            None if de_j_source is None else cp.asnumpy(de_j_source))
        mf_grad.de_j_metric = (
            None if de_j_metric is None else cp.asnumpy(de_j_metric))
        mf_grad.de_j_native = None
        mf_grad.de_j_backend = 'rsdf_source_metric_native_adjoint'
        mf_grad.rsdf_response_metadata = rsdf_metadata
        mf_grad.rsdf_source_reciprocal = rsdf_metadata['source'][
            'reciprocal_memory']
        _progress(mf_grad, 'source_fock')
        stage_start = _profile_start(mf_grad)
        de_fock = grad_fock_rsdf(
            mf_grad, de_fock_hcore, rsdf_terms['de_fock_source'],
            rsdf_terms['de_fock_metric'], rsdf_terms['de_fock_ewald'],
            atmlst)
        _profile_stop(mf_grad, profile, 'source_fock_seconds', stage_start)
    _progress(mf_grad, 'nonvariational')
    stage_start = _profile_start(mf_grad)
    de_nvar = nonvariational_correction(
        mf_grad, Co, Cv, mo_energy, zvec, s1, de_fock, fvind, atmlst,
        ovlp_response_intermediate=ovlp_response_intermediate)
    _profile_stop(mf_grad, profile, 'nonvariational_seconds', stage_start)
    _profile_stop(mf_grad, profile, 'total_seconds', total_start)
    mf_grad.gradient_stage_profile = profile
    if isinstance(mf.with_df, GDF) and mf_grad.de_j_fock_native is not None:
        return de_skel + cp.asnumpy(de_j_fock_native) + de_nvar
    return de_skel + de_nvar


class Gradients(gpu_rhf_grad.GradientsBase):
    '''Analytic nuclear gradient of Gamma-point GPU ``E_DFA[P_RHF]``.'''

    _keys = gpu_rhf_grad.GradientsBase._keys | {
        'cphf_conv_tol', 'cphf_krylov_tol', 'cphf_residual_tol',
        'cphf_max_cycle',
        'cphf_refinement_solver', 'df_cphf_response_mode',
        'precision_policy', 'precision_policy_effective',
        'precision_fallback_reason', 'mixed_precision_report',
        'gdf_derivative_precision',
        'df_cphf_response_cache',
        'df_cphf_response_memory_fraction',
        'df_cphf_response_minimum_reserve_bytes',
        'profile_df_cphf_response', 'z_response_profile',
        'response_state', 'response_state_signature',
        'response_state_token', 'response_state_reused',
        'response_state_owner', 'response_state_reuse_count',
        'z_solve_count',
        'z_warm_start', 'z_warm_start_min_subspace_overlap',
        'z_initial_guess', 'z_warm_start_projection',
        'z_warm_start_metadata',
        'hcore_method', 'hcore_batch_size',
        'hcore_backend', 'hcore_effective_batch_size',
        'hcore_stage_profile', 'hcore_pp_loc_part1_plan',
        'de_hcore_components',
        'de_fock_hcore_components',
        'profile_gradient_stages', 'gradient_stage_profile',
        'profile_stress_stages', 'stress_stage_profile',
        'stress_memory_preflight', 'stress_reference_options',
        'stress_report',
        'gaussian_df_strain_response', 'gdf_strain_response',
        'progress_callback',
        'grid_response', 'target_fock_reused', 'xc_deriv_backend',
        'de_j_backend', 'de_j_native', 'de_j_source', 'de_j_metric',
        'de_j_xc_multigrid',
        'de_fock_backend', 'de_fock_native',
        'de_j_fock_native', 'gdf_term_partition',
        'gdf_response_mode', 'gdf_adjoint_backend',
        'gdf_adjoint_backend_used',
        'gdf_adjoint_backend_fallback_reason',
        'measure_gdf_response_boundary', 'gdf_response_boundary_seconds',
        'de_j_fock_source', 'de_j_fock_metric', 'de_j_fock_ewald',
        'de_j_fock_term_residual_max_abs',
        'gdf_reciprocal_memory_plan', 'gdf_reciprocal_zero_order',
        'gdf_reciprocal_derivative', 'gdf_response_profile',
        'gdf_response_workspace', 'gdf_response_range_components',
        'z_primary_wall_seconds', 'z_refinement_wall_seconds',
        'z_refinement_iterations', 'z_wall_seconds',
        'z_current_wall_seconds', 'z_response_backend',
        'z_requested_conv_tol', 'z_requested_krylov_tol',
        'z_effective_krylov_tol', 'z_krylov_tolerance_policy',
        'z_physical_residual_tol', 'z_cleanup_krylov_tol',
        'z_cleanup_wall_seconds', 'z_cleanup_iterations',
        'rsdf_response_metadata', 'rsdf_source_reciprocal',
        'cderi_release_metadata',
        'memory_policy', 'property_sequence', 'resolved_memory_policy',
    }

    def __init__(self, mf):
        if not isinstance(mf, RHFDFT):
            raise TypeError('GPU PBC Gradients requires dcforge.gpu.pbc.energy.RHFDFT')
        if not isinstance(mf.with_df, (FFTDF, GDF, RSDF)):
            raise NotImplementedError('unsupported GPU PBC analytic-gradient provider')
        gpu_rhf_grad.GradientsBase.__init__(self, mf)
        self.grid_response = True
        self.cphf_conv_tol = 1e-9
        # ``None`` preserves the established matrix-free safety cap
        # ``min(cphf_conv_tol, 1e-10)``.  A positive value is an explicit
        # tolerance-study override and remains subordinate to the separately
        # reconstructed physical residual gate.
        self.cphf_krylov_tol = None
        self.cphf_residual_tol = 1e-8
        self.cphf_max_cycle = 200
        # At real Gamma the restricted orbital Hessian is self-adjoint.  The
        # MINRES route avoids restarted-GMRES storage; ``rhf_zvec`` retries
        # GMRES if its independently reconstructed residual is not accepted.
        self.cphf_refinement_solver = 'minres'
        # The public default remains the fully FP64 response.  ``mixed`` is
        # an explicit real-Gamma GDF opt-in that accelerates only repeated
        # direct-VO products and accepts the result through an FP64 physical
        # residual.  ``mixed_lowmem`` has the same arithmetic but requires an
        # HDF5-backed FP64 source factor so the large FP64 CDERI is not also
        # resident in mapped host memory.
        self.precision_policy = 'fp64'
        self.precision_policy_effective = 'fp64'
        self.precision_fallback_reason = None
        self.mixed_precision_report = None
        # The public GDF precision profiles may lower only the large MO-source
        # products in the native nuclear response.  Primitive derivative
        # integrals, metric derivative and atom reduction remain FP64.
        self.gdf_derivative_precision = 'fp64'
        # GDF/RSDF production accumulates only ``(J-K/2) C_occ`` before the
        # virtual projection.  The former complete-AO J/K route is retained
        # as an explicit numerical oracle, never selected automatically.
        self.df_cphf_response_mode = 'direct_vo'
        # Adaptive Gaussian-DF Z-response cache shared by GDF and RSDF.
        # ``recompute`` remains the exact storage oracle.  Production
        # ``auto`` selects only a bounded cache that leaves both the predicted
        # action workspace and an explicit device reserve; any allocation
        # failure falls back to recomputation.
        self.df_cphf_response_cache = 'auto'
        self.df_cphf_response_memory_fraction = 0.8
        self.df_cphf_response_minimum_reserve_bytes = 4 * 1024**3
        self.profile_df_cphf_response = False
        self.z_response_profile = None
        # ``custom`` preserves every existing fine-grained option.  Users may
        # select fast/balanced/low_memory explicitly and must declare whether
        # stress follows gradient so CDERI lifetime is planned globally.
        self.memory_policy = 'custom'
        self.property_sequence = ('gradient', 'stress')
        self.resolved_memory_policy = None
        # Nuclear gradients and homogeneous strain share the same target
        # adjoint equation.  Cache only its bounded perturbation-independent
        # solution, never the large direct-VO response workspace.
        self.response_state = None
        self.response_state_signature = None
        self.response_state_token = None
        self.response_state_reused = False
        self.response_state_owner = None
        self.response_state_reuse_count = 0
        self.z_solve_count = 0
        # Scanner-only response warm start.  The previous raw VO amplitude is
        # never reused directly: :class:`RHFDFTGradScanner` transports it
        # through current/previous AO cross-overlaps and the solver accepts it
        # only when its current physical residual improves on the gap guess.
        self.z_warm_start = False
        self.z_warm_start_min_subspace_overlap = 0.5
        self.z_initial_guess = None
        self.z_warm_start_projection = None
        self.z_warm_start_metadata = None
        # For GDF, ``auto`` uses the source-consistent analytic SR/LR
        # pseudopotential representation.  FFTDF and RSDF keep their existing
        # uniform-grid representation.  Explicit ``grid`` is the GDF
        # numerical oracle; explicit ``gdf`` rejects non-GDF providers.
        self.hcore_method = 'auto'
        # This requests a few atom centers per local-Hcore traversal.  In the
        # ordinary public-matrix route, one gives the upstream-style
        # per-atom reference path.  At a large enough mesh the code switches
        # representation only to avoid an all-atom SI allocation.  The
        # streamed route preserves this requested batch size because its AO
        # density and atom-row work now use one cell-wide grid traversal;
        # batching controls only atom-specific local-GTH FFT storage.
        self.hcore_batch_size = 4
        self.hcore_backend = None
        self.hcore_effective_batch_size = None
        self.hcore_stage_profile = None
        self.hcore_pp_loc_part1_plan = None
        self.de_hcore_components = None
        self.de_fock_hcore_components = None
        # Keep production execution asynchronous.  Validation drivers may
        # request stream-synchronized stage timings through this opt-in flag.
        self.profile_gradient_stages = False
        self.gradient_stage_profile = None
        # Stress profiling is separately opt-in because each stage boundary
        # synchronizes the CUDA stream and records memory snapshots.  It does
        # not alter any mesh, block policy, solver tolerance or term owner.
        self.profile_stress_stages = False
        self.stress_stage_profile = None
        self.stress_memory_preflight = None
        self.stress_reference_options = None
        self.stress_report = None
        self.gaussian_df_strain_response = None
        self.gdf_strain_response = None
        # Optional queue/profiler callback.  It receives stage labels only;
        # it cannot alter arrays, solver tolerances, or the term ledger.
        self.progress_callback = None
        self.target_fock_reused = False
        self.xc_deriv_backend = None
        self.de_j_backend = None
        self.de_j_native = None
        self.de_j_source = None
        self.de_j_metric = None
        self.de_j_xc_multigrid = None
        self.de_fock_backend = None
        self.de_fock_native = None
        self.de_j_fock_native = None
        self.gdf_term_partition = None
        # ``auto`` selects the direct primitive-adjoint route for PBE and
        # R2SCAN.  An explicit two-channel native library may replace the PBE
        # route.  ``fused``, ``signed`` and ``separate`` pin those
        # representations for auditing;
        # signed remains the algebraic R2SCAN oracle/fallback.
        self.gdf_response_mode = 'auto'
        # ``auto`` selects the exact packed-source-free MO-direct adjoint for
        # supported closed-shell real-Gamma GDF response.  The established
        # density-based ``packed`` route remains the explicit numerical
        # oracle and the exact fallback when an internal auto caller cannot
        # supply real ``C_o`` and ``U=C_v Z`` factors.  ``mo_factorized`` is
        # retained as the isolated packed-backed algebra experiment.
        self.gdf_adjoint_backend = 'auto'
        self.gdf_adjoint_backend_used = None
        self.gdf_adjoint_backend_fallback_reason = None
        # Validation-only production boundary.  When enabled it performs one
        # synchronization immediately before and after the complete native
        # response while leaving every internal detailed profiler disabled.
        self.measure_gdf_response_boundary = False
        self.gdf_response_boundary_seconds = None
        self.de_j_fock_source = None
        self.de_j_fock_metric = None
        self.de_j_fock_ewald = None
        self.de_j_fock_term_residual_max_abs = None
        self.gdf_reciprocal_memory_plan = None
        self.gdf_reciprocal_zero_order = None
        self.gdf_reciprocal_derivative = None
        self.gdf_response_profile = None
        self.gdf_response_workspace = None
        self.gdf_response_range_components = None
        self.rsdf_response_metadata = None
        self.rsdf_source_reciprocal = None
        self.cderi_release_metadata = None
        self.z_primary_wall_seconds = None
        self.z_refinement_wall_seconds = None
        self.z_refinement_iterations = None
        self.z_requested_conv_tol = None
        self.z_requested_krylov_tol = None
        self.z_effective_krylov_tol = None
        self.z_krylov_tolerance_policy = None
        self.z_physical_residual_tol = None
        self.z_cleanup_krylov_tol = None
        self.z_cleanup_wall_seconds = None
        self.z_cleanup_iterations = None
        self.z_wall_seconds = None
        self.z_current_wall_seconds = None
        self.z_response_backend = None

    def reset(self, cell=None):
        '''Reset the active Cell and discard all derivative result caches.

        GPU4PySCF's base reset delegates the geometry change to the source,
        target and shared provider.  DC-Forge additionally owns a target Fock,
        Z-vector, GDF strain response, multigrid task ledger and assembled
        stress terms.  None of those quantities is transferable to a new
        geometry or lattice, so a scanner must clear them before reconverging
        the source state.
        '''
        gpu_rhf_grad.GradientsBase.reset(self, cell)
        for name in (
                'zvec', 'z_info', 'z_response_profile',
                'z_rhs_max_abs', 'z_residual_max_abs',
                'z_residual_relative', 'z_solver', 'z_dimension',
                'z_primary_wall_seconds', 'z_refinement_wall_seconds',
                'z_refinement_iterations', 'z_requested_conv_tol',
                'z_requested_krylov_tol', 'z_effective_krylov_tol',
                'z_krylov_tolerance_policy', 'z_physical_residual_tol',
                'z_cleanup_krylov_tol', 'z_cleanup_wall_seconds',
                'z_cleanup_iterations', 'z_wall_seconds',
                'z_current_wall_seconds', 'z_response_backend',
                'mixed_precision_report',
                'response_state', 'response_state_signature',
                'response_state_token', 'response_state_owner',
                'gdf_response_range_components',
                'gaussian_df_strain_response', 'gdf_strain_response',
                'hcore_strain_components',
                'target_xc_strain_metadata', 'ewald_strain',
                'work_hcore', 'work_j', 'work_xc', 'work_ovlp',
                'work_ewald', 'work_fock', 'work_ovlp_energy',
                'work_ovlp_response', 'work_skeleton', 'work_nvar',
                'work_raw', 'stress_raw', 'stress_tensor', 'stress_voigt',
                'pressure', 'virial_tensor', 'stress_terms',
                'stress_owner_ledger', 'stress_execution_boundary',
                'stress_symmetry_residual',
                'stress_recomposition_residual',
                'stress_memory_preflight', 'stress_stage_profile',
                'stress_report',
                'production_analytic', 'gpu_native_complete',
                'low_level_numerical_components'):
            setattr(self, name, None)
        self.target_fock_reused = False
        self.precision_policy_effective = str(self.precision_policy)
        self.precision_fallback_reason = None
        self.response_state_reused = False
        self.response_state_reuse_count = 0
        self.z_solve_count = 0
        return self

    def dump_flags(self, verbose=None):
        '''Log GPU PBC response controls in PySCF style.'''
        gpu_rhf_grad.GradientsBase.dump_flags(self, verbose)
        logger.info(self, 'grid_response = %s', self.grid_response)
        logger.info(self, 'cphf_conv_tol = %g', self.cphf_conv_tol)
        logger.info(self, 'cphf_krylov_tol = %s', self.cphf_krylov_tol)
        logger.info(self, 'cphf_residual_tol = %g', self.cphf_residual_tol)
        logger.info(self, 'cphf_max_cycle = %d', self.cphf_max_cycle)
        logger.info(self, 'cphf_refinement_solver = %s',
                    self.cphf_refinement_solver)
        logger.info(self, 'precision_policy = %s', self.precision_policy)
        logger.info(self, 'precision_policy_effective = %s',
                    self.precision_policy_effective)
        logger.info(self, 'precision_fallback_reason = %s',
                    self.precision_fallback_reason)
        logger.info(self, 'gdf_derivative_precision = %s',
                    self.gdf_derivative_precision)
        logger.info(self, 'df_cphf_response_mode = %s',
                    self.df_cphf_response_mode)
        logger.info(self, 'df_cphf_response_cache = %s',
                    self.df_cphf_response_cache)
        logger.info(self, 'df_cphf_response_memory_fraction = %.3f',
                    self.df_cphf_response_memory_fraction)
        logger.info(self, 'df_cphf_response_minimum_reserve_bytes = %d',
                    self.df_cphf_response_minimum_reserve_bytes)
        logger.info(self, 'profile_df_cphf_response = %s',
                    self.profile_df_cphf_response)
        logger.info(self, 'memory_policy = %s', self.memory_policy)
        logger.info(self, 'property_sequence = %s', self.property_sequence)
        logger.info(self, 'z_warm_start = %s', self.z_warm_start)
        logger.info(self, 'z_warm_start_min_subspace_overlap = %.3f',
                    self.z_warm_start_min_subspace_overlap)
        logger.info(self, 'z_response_backend = %s', self.z_response_backend)
        logger.info(self, 'response_state_reused = %s',
                    self.response_state_reused)
        logger.info(self, 'z_solve_count = %d', self.z_solve_count)
        logger.info(self, 'hcore_batch_size = %d', self.hcore_batch_size)
        logger.info(self, 'hcore_method = %s', self.hcore_method)
        logger.info(self, 'gdf_response_mode = %s', self.gdf_response_mode)
        logger.info(self, 'gdf_adjoint_backend = %s',
                    self.gdf_adjoint_backend)
        logger.info(self, 'measure_gdf_response_boundary = %s',
                    self.measure_gdf_response_boundary)
        logger.info(self, 'profile_gradient_stages = %s',
                    self.profile_gradient_stages)
        logger.info(self, 'profile_stress_stages = %s',
                    self.profile_stress_stages)
        return self

    def set_property_sequence(self, *properties):
        '''Declare the ordered property lifecycle for memory planning.'''
        if len(properties) == 1 and isinstance(properties[0], (list, tuple)):
            properties = tuple(properties[0])
        self.property_sequence = tuple(properties)
        self.resolved_memory_policy = None
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None,
               atmlst=None):
        '''Return the complete nuclear energy derivative ``dE/dR`` in Ha/Bohr.'''
        if not self.grid_response:
            raise NotImplementedError('GPU PBC gradient requires grid_response=True')
        de_elec = grad_elec(self, mo_energy, mo_coeff, mo_occ, atmlst)
        self.atmlst = list(range(self.cell.natm)) if atmlst is None else list(atmlst)
        self.de_nuc = np.asarray(cpu_krhf_grad.grad_nuc(self.cell, self.atmlst))
        self.de = np.asarray(de_elec) + self.de_nuc
        if (isinstance(self.base.with_df, GDF)
                and self.de_j_fock_native is not None):
            self.de_skel = (
                self.de_skel_elec + self.de_j_fock_native + self.de_nuc)
            self.de_terms = (
                self.de_hcore + self.de_veff + self.de_extra + self.de_ovlp
                + self.de_nuc + self.de_j_fock_native + self.de_fock
                + self.de_ovlp_energy + self.de_ovlp_response)
        else:
            self.de_skel = self.de_skel_elec + self.de_nuc
            self.de_terms = (
                self.de_hcore + self.de_veff + self.de_extra + self.de_ovlp
                + self.de_nuc + self.de_fock + self.de_ovlp_energy
                + self.de_ovlp_response)
        return self.de

    grad_elec = grad_elec
    grad = kernel

    def get_stress(self):
        '''Return the homogeneous-strain derivative of ``E_DFA[P_RHF]``.

        The implementation is routed through the DC-Forge stress namespace so
        GPU4PySCF's conventional FFTDF-RKS stress cannot be returned for a
        nonvariational GDF target by inheritance or accidental delegation.
        '''
        if getattr(self.base, 'gdf_precision_profile', 'fp64') != 'fp64':
            raise NotImplementedError(
                'GDF mixed-precision profiles are validated for energy and '
                'nuclear gradient only; rebuild with profile=\'fp64\' before '
                'analytic stress')
        from dcforge.gpu.pbc.stress.rhf_dft import kernel

        return kernel(self)

    def as_scanner(self):
        '''Return a PySCF-style energy-and-gradient scanner.'''
        return as_scanner(self)

    def as_stress_scanner(self, **reference_options):
        '''Return a production analytic energy-and-stress scanner.'''
        if getattr(self.base, 'gdf_precision_profile', 'fp64') != 'fp64':
            raise NotImplementedError(
                'GDF mixed-precision profiles do not support stress scanners')
        from dcforge.gpu.pbc.stress.rhf_dft import as_stress_scanner

        return as_stress_scanner(self, **reference_options)

    def to_cpu(self):
        raise NotImplementedError('GPU PBC gradient state conversion is not implemented')


Grad = Gradients


def as_scanner(mf_grad):
    '''Create a callable GPU periodic energy-and-gradient scanner.

    Args:
        mf_grad : Gradients
            Configured fixed-density analytic-gradient driver.  Its response,
            Hcore, batching, and profiling options are retained by the scanner.

    Returns:
        RHFDFTGradScanner
            Callable scanner returning ``(energy, gradient)`` where energy is
            in Hartree and gradient is ``dE/dR`` in Ha/Bohr.

    Notes:
        ``lib.GradScanner`` converts the base workflow through
        :meth:`RHFDFT.as_scanner`.  Each call therefore reconverges source RHF,
        evaluates the fixed-density target energy, and only then evaluates the
        analytic Z-vector gradient for that same geometry and provider state.
    '''
    if isinstance(mf_grad, lib.GradScanner):
        return mf_grad
    if not isinstance(mf_grad, Gradients):
        raise TypeError('GPU PBC gradient scanner requires Gradients')

    logger.info(mf_grad, 'Create scanner for %s', mf_grad.__class__)
    name = mf_grad.__class__.__name__ + RHFDFTGradScanner.__name_mixin__
    return lib.set_class(
        RHFDFTGradScanner(mf_grad),
        (RHFDFTGradScanner, mf_grad.__class__), name)


class RHFDFTGradScanner(lib.GradScanner):
    '''Stateful scanner for GPU Gamma-point ``E_DFA[P_RHF]`` and ``dE/dR``.'''

    def __init__(self, mf_grad):
        lib.GradScanner.__init__(self, mf_grad)

    def __call__(self, cell_or_geom, **kwargs):
        '''Evaluate fixed-density energy and analytic gradient at one geometry.'''
        if isinstance(cell_or_geom, gto.Cell):
            cell = cell_or_geom
        else:
            cell = self.cell.set_geom_(cell_or_geom, inplace=False)

        previous_response = _capture_previous_z_response(self)

        # Follow PySCF's gradient-scanner order.  ``reset`` invalidates this
        # driver's old term ledger; the base energy scanner then rebuilds the
        # source/provider state before ``kernel`` consumes it.
        self.reset(cell)
        energy = self.base(cell)
        self.z_initial_guess = None
        self.z_warm_start_projection = {
            'available': False,
            'fallback_reason': 'warm start disabled or no previous Z-vector',
        }
        if previous_response is not None:
            try:
                guess, projection = _project_previous_z_response(
                    previous_response, self.base, cell,
                    self.z_warm_start_min_subspace_overlap)
                self.z_initial_guess = guess
                self.z_warm_start_projection = projection
            except (ValueError, RuntimeError, NotImplementedError) as error:
                self.z_warm_start_projection = {
                    'available': False,
                    'fallback_reason': str(error),
                }
        gradient = self.kernel(**kwargs)
        return energy, gradient


def _capture_previous_z_response(scanner):
    '''Retain one old-geometry VO response only when warm start is requested.'''
    if not bool(getattr(scanner, 'z_warm_start', False)):
        return None
    zvec = getattr(scanner, 'zvec', None)
    mo_coeff = getattr(scanner.base, 'mo_coeff', None)
    mo_occ = getattr(scanner.base, 'mo_occ', None)
    if zvec is None or mo_coeff is None or mo_occ is None:
        return None
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    return {
        'cell': scanner.cell,
        'provider_class': type(scanner.base.with_df),
        'Co': mo_coeff[:, mo_occ > 0].copy(),
        'Cv': mo_coeff[:, mo_occ == 0].copy(),
        'zvec': cp.asarray(zvec).copy(),
    }


def _project_previous_z_response(previous, current_mf, current_cell,
                                 minimum_overlap):
    r'''Project an old VO response through the current--old AO cross overlap.

    If ``S^{new,old}`` is the cross-geometry AO overlap, this constructs

    ``Z_new = (Cv_new^T S_new,old Cv_old)``
    ``        Z_old (Co_old^T S_old,new Co_new)``.

    Thus MO phases, occupied rotations and virtual rotations are transported
    explicitly.  The previous raw ``Z_ai`` is never interpreted in the new
    orbital basis without this transformation.
    '''
    old_cell = previous['cell']
    if type(current_mf.with_df) is not previous['provider_class']:
        raise ValueError('DF provider class changed between scanner calls')
    if not np.allclose(
            old_cell.lattice_vectors(), current_cell.lattice_vectors(),
            atol=1.e-12, rtol=0.):
        raise NotImplementedError('Z warm start requires a fixed lattice')
    if not np.array_equal(old_cell.ao_loc, current_cell.ao_loc):
        raise ValueError('AO topology changed between scanner calls')
    mo_coeff = cp.asarray(current_mf.mo_coeff)
    mo_occ = cp.asarray(current_mf.mo_occ)
    Co = mo_coeff[:, mo_occ > 0]
    Cv = mo_coeff[:, mo_occ == 0]
    if Co.shape[1] != previous['Co'].shape[1]:
        raise ValueError('occupied topology changed between scanner calls')
    if Cv.shape[1] != previous['Cv'].shape[1]:
        raise ValueError('virtual topology changed between scanner calls')
    overlap = np.asarray(gto.intor_cross(
        'int1e_ovlp', current_cell, old_cell, kpt=np.zeros(3)))
    imaginary = float(np.max(np.abs(overlap.imag)))
    if imaginary > 1.e-10:
        raise NotImplementedError(
            f'Gamma cross overlap has imaginary component {imaginary:.3e}')
    overlap = cp.asarray(overlap.real)
    virtual_transport = Cv.T @ overlap @ previous['Cv']
    occupied_transport = previous['Co'].T @ overlap.T @ Co
    virtual_singular = cp.linalg.svd(
        virtual_transport, compute_uv=False)
    occupied_singular = cp.linalg.svd(
        occupied_transport, compute_uv=False)
    minimum_virtual = float(cp.min(virtual_singular).get())
    minimum_occupied = float(cp.min(occupied_singular).get())
    quality = min(minimum_virtual, minimum_occupied)
    if quality < float(minimum_overlap):
        raise ValueError(
            f'MO-subspace overlap {quality:.3f} is below '
            f'{float(minimum_overlap):.3f}')
    guess = virtual_transport @ previous['zvec'] @ occupied_transport
    return guess, {
        'available': True,
        'projection': 'cross_ao_overlap_virtual_and_occupied',
        'minimum_virtual_subspace_overlap': minimum_virtual,
        'minimum_occupied_subspace_overlap': minimum_occupied,
        'minimum_required_subspace_overlap': float(minimum_overlap),
    }
