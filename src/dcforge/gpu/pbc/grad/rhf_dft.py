'''Gamma-point GPU FFTDF/GDF/RSDF analytic DFA@RHF nuclear gradients.'''

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
    target_j_and_fock_energy_derivative,
    target_j_fock_energy_derivative,
    target_j_fock_signed_energy_derivative,
)
from dcforge.gpu.pbc.df.rsdf import RSDF, make_gradient_response
from dcforge.gpu.pbc.energy.rhf_dft import (
    RHFDFT,
    _check_input,
    _target_fock_signature,
    _target_xc_name,
)
from dcforge.gpu.pbc.grad.rhf_nv import (
    grad_fock_direct,
    grad_fock_gdf,
    grad_fock_rsdf,
    nonvariational_correction,
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


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None):
    '''Evaluate the electronic Gamma-point GPU FFTDF/GDF DFA@RHF gradient.

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
    mf_grad.hcore_stage_profile = None
    mf_grad.hcore_pp_loc_part1_plan = None
    mf_grad.de_hcore_components = None
    mf_grad.de_fock_hcore_components = None
    if mf_grad.hcore_method not in ('auto', 'gdf', 'grid'):
        raise ValueError("hcore_method must be 'auto', 'gdf', or 'grid'")
    if mf_grad.hcore_method == 'gdf' and not isinstance(mf.with_df, GDF):
        raise NotImplementedError(
            "hcore_method='gdf' requires a GPU GDF provider; "
            "use 'auto' or 'grid'")
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
    vind = gen_vind(Co, Cv, fvind)
    stage_start = _profile_start(mf_grad)
    zvec, z_info = solve_z_vector(
        mo_energy, mo_occ, fvo, vind,
        conv_tol=mf_grad.cphf_conv_tol,
        residual_tol=mf_grad.cphf_residual_tol,
        max_cycle=mf_grad.cphf_max_cycle,
        refinement_solver=mf_grad.cphf_refinement_solver,
        log=log,
    )
    _profile_stop(mf_grad, profile, 'z_vector_seconds', stage_start)
    mf_grad.zvec = zvec
    mf_grad.z_rhs_max_abs = z_info['rhs_max_abs']
    mf_grad.z_residual_max_abs = z_info['residual_max_abs']
    mf_grad.z_residual_relative = z_info['residual_relative']
    mf_grad.z_solver = z_info['solver']
    mf_grad.z_dimension = z_info['dimension']
    mf_grad.z_primary_wall_seconds = z_info['primary_wall_seconds']
    mf_grad.z_refinement_wall_seconds = z_info['refinement_wall_seconds']
    mf_grad.z_refinement_iterations = z_info['refinement_iterations']
    mf_grad.z_wall_seconds = z_info['wall_seconds']
    dz = Cv @ zvec @ Co.conj().T
    dz = dz + dz.conj().T

    if isinstance(mf.with_df, FFTDF):
        # FFTDF uses the public full-matrix Hcore derivative route.
        hcore_gen = get_hcore_generator(cell)
        s1 = -int1e.int1e_ipovlp(cell, np.zeros((1, 3)))[0]
        # This GPU kernel returns full AO derivatives.  They are shared by the
        # target Hartree skeleton and source-Fock Z contraction.
        stage_start = _profile_start(mf_grad)
        vj1, vk1 = grad_jk(
            cell, dm0, exxdiv=mf_hf.exxdiv, with_df=mf.with_df)
        _profile_stop(mf_grad, profile, 'fftdf_jk_derivative_seconds',
                      stage_start)
        mf_grad.vj1 = vj1
        mf_grad.vk1 = vk1
        stage_start = _profile_start(mf_grad)
        de_skel = skeleton_gradient(
            mf_grad, dm0, foo, Co, s1, hcore_gen, vj1, atmlst)
        _profile_stop(mf_grad, profile, 'skeleton_seconds', stage_start)
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
        response_mode = mf_grad.gdf_response_mode
        if response_mode not in ('auto', 'fused', 'signed', 'separate'):
            raise ValueError(
                'gdf_response_mode must be auto, fused, signed, or separate')
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
        stage_start = _profile_start(mf_grad)
        if use_separate:
            gdf_terms, response_memory = target_j_and_fock_energy_derivative(
                mf_hf, dm0, dz, atmlst)
        elif use_signed:
            de_j_fock_native, response_memory = (
                target_j_fock_signed_energy_derivative(
                    mf_hf, dm0, dz, atmlst, return_metadata=True))
        else:
            de_j_fock_native, response_memory = target_j_fock_energy_derivative(
                mf_hf, dm0, dz, atmlst, return_metadata=True)
        _profile_stop(mf_grad, profile, 'gdf_native_response_seconds',
                      stage_start)
        # The high-mesh Hcore and primitive GDF response above do not depend
        # on overlap derivatives.  Build ``S^lambda`` only after them, which
        # prevents this independent GPU4PySCF integral workspace from being
        # live during the large local-GTH grid traversal.
        s1 = -int1e.int1e_ipovlp(cell, np.zeros((1, 3)))[0]
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
            stage_start = _profile_start(mf_grad)
            de_fock = grad_fock_gdf(
                mf_grad, Co, Cv, zvec, dm0, None, atmlst,
                hcore_derivative=de_fock_hcore,
                source_derivative=gdf_terms['de_fock_source'],
                metric_derivative=gdf_terms['de_fock_metric'],
                ewald_derivative=gdf_terms['de_fock_ewald'])
            _profile_stop(mf_grad, profile, 'source_fock_seconds', stage_start)
        else:
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
        stage_start = _profile_start(mf_grad)
        rsdf_terms, rsdf_metadata = make_gradient_response(
            mf.with_df, dm0, dz, exxdiv=mf_hf.exxdiv)
        _profile_stop(mf_grad, profile, 'rsdf_native_response_seconds',
                      stage_start)
        # Keep the overlap primitive out of the large RSDF CUDA response
        # lifetime.  ``S^lambda`` is independent of source/metric
        # derivatives and is only needed by skeleton/nonvariational terms.
        s1 = -int1e.int1e_ipovlp(cell, np.zeros((1, 3)))[0]
        de_j_source = rsdf_terms['de_j_source']
        de_j_metric = rsdf_terms['de_j_metric']
        stage_start = _profile_start(mf_grad)
        de_skel = skeleton_gradient(
            mf_grad, dm0, foo, Co, s1, None, None, atmlst,
            de_j_energy=de_j_source + de_j_metric,
            de_hcore_energy=de_hcore)
        _profile_stop(mf_grad, profile, 'skeleton_seconds', stage_start)
        mf_grad.de_j_source = cp.asnumpy(de_j_source)
        mf_grad.de_j_metric = cp.asnumpy(de_j_metric)
        mf_grad.de_j_native = None
        mf_grad.de_j_backend = 'rsdf_source_metric_native_adjoint'
        mf_grad.rsdf_response_metadata = rsdf_metadata
        mf_grad.rsdf_source_reciprocal = rsdf_metadata['source'][
            'reciprocal_memory']
        stage_start = _profile_start(mf_grad)
        de_fock = grad_fock_rsdf(
            mf_grad, de_fock_hcore, rsdf_terms['de_fock_source'],
            rsdf_terms['de_fock_metric'], rsdf_terms['de_fock_ewald'],
            atmlst)
        _profile_stop(mf_grad, profile, 'source_fock_seconds', stage_start)
    stage_start = _profile_start(mf_grad)
    de_nvar = nonvariational_correction(
        mf_grad, Co, Cv, mo_energy, zvec, s1, de_fock, fvind, atmlst)
    _profile_stop(mf_grad, profile, 'nonvariational_seconds', stage_start)
    _profile_stop(mf_grad, profile, 'total_seconds', total_start)
    mf_grad.gradient_stage_profile = profile
    if isinstance(mf.with_df, GDF) and mf_grad.de_j_fock_native is not None:
        return de_skel + cp.asnumpy(de_j_fock_native) + de_nvar
    return de_skel + de_nvar


class Gradients(gpu_rhf_grad.GradientsBase):
    '''Analytic nuclear gradient of Gamma-point GPU ``E_DFA[P_RHF]``.'''

    _keys = gpu_rhf_grad.GradientsBase._keys | {
        'cphf_conv_tol', 'cphf_residual_tol', 'cphf_max_cycle',
        'cphf_refinement_solver', 'hcore_method', 'hcore_batch_size',
        'hcore_backend', 'hcore_effective_batch_size',
        'hcore_stage_profile', 'hcore_pp_loc_part1_plan',
        'de_hcore_components',
        'de_fock_hcore_components',
        'profile_gradient_stages', 'gradient_stage_profile',
        'grid_response', 'target_fock_reused', 'xc_deriv_backend',
        'de_j_backend', 'de_j_native', 'de_j_source', 'de_j_metric',
        'de_fock_backend', 'de_fock_native',
        'de_j_fock_native', 'gdf_term_partition',
        'gdf_response_mode',
        'de_j_fock_source', 'de_j_fock_metric', 'de_j_fock_ewald',
        'de_j_fock_term_residual_max_abs',
        'gdf_reciprocal_memory_plan', 'gdf_reciprocal_zero_order',
        'gdf_reciprocal_derivative', 'gdf_response_profile',
        'z_primary_wall_seconds', 'z_refinement_wall_seconds',
        'z_refinement_iterations', 'z_wall_seconds',
        'rsdf_response_metadata', 'rsdf_source_reciprocal',
    }

    def __init__(self, mf):
        if not isinstance(mf, RHFDFT):
            raise TypeError('GPU PBC Gradients requires dcforge.gpu.pbc.energy.RHFDFT')
        if not isinstance(mf.with_df, (FFTDF, GDF, RSDF)):
            raise NotImplementedError('unsupported GPU PBC analytic-gradient provider')
        gpu_rhf_grad.GradientsBase.__init__(self, mf)
        self.grid_response = True
        self.cphf_conv_tol = 1e-9
        self.cphf_residual_tol = 1e-8
        self.cphf_max_cycle = 200
        # At real Gamma the restricted orbital Hessian is self-adjoint.  The
        # MINRES route avoids restarted-GMRES storage; ``rhf_zvec`` retries
        # GMRES if its independently reconstructed residual is not accepted.
        self.cphf_refinement_solver = 'minres'
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
        self.target_fock_reused = False
        self.xc_deriv_backend = None
        self.de_j_backend = None
        self.de_j_native = None
        self.de_j_source = None
        self.de_j_metric = None
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
        self.de_j_fock_source = None
        self.de_j_fock_metric = None
        self.de_j_fock_ewald = None
        self.de_j_fock_term_residual_max_abs = None
        self.gdf_reciprocal_memory_plan = None
        self.gdf_reciprocal_zero_order = None
        self.gdf_reciprocal_derivative = None
        self.gdf_response_profile = None
        self.rsdf_response_metadata = None
        self.rsdf_source_reciprocal = None
        self.z_primary_wall_seconds = None
        self.z_refinement_wall_seconds = None
        self.z_refinement_iterations = None
        self.z_wall_seconds = None

    def dump_flags(self, verbose=None):
        '''Log GPU PBC response controls in PySCF style.'''
        gpu_rhf_grad.GradientsBase.dump_flags(self, verbose)
        logger.info(self, 'grid_response = %s', self.grid_response)
        logger.info(self, 'cphf_conv_tol = %g', self.cphf_conv_tol)
        logger.info(self, 'cphf_residual_tol = %g', self.cphf_residual_tol)
        logger.info(self, 'cphf_max_cycle = %d', self.cphf_max_cycle)
        logger.info(self, 'cphf_refinement_solver = %s',
                    self.cphf_refinement_solver)
        logger.info(self, 'hcore_batch_size = %d', self.hcore_batch_size)
        logger.info(self, 'hcore_method = %s', self.hcore_method)
        logger.info(self, 'gdf_response_mode = %s', self.gdf_response_mode)
        logger.info(self, 'profile_gradient_stages = %s',
                    self.profile_gradient_stages)
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

    def as_scanner(self):
        '''Return a PySCF-style energy-and-gradient scanner.'''
        return as_scanner(self)

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

        # Follow PySCF's gradient-scanner order.  ``reset`` invalidates this
        # driver's old term ledger; the base energy scanner then rebuilds the
        # source/provider state before ``kernel`` consumes it.
        self.reset(cell)
        energy = self.base(cell)
        gradient = self.kernel(**kwargs)
        return energy, gradient
