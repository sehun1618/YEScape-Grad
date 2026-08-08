'''GPU sampled-k semilocal-DFA@KRHF analytic nuclear gradients.'''

import time

import cupy as cp
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.grad import rhf as gpu_rhf_grad

from dcforge.cpu.pbc.kpts import IMAGINARY_RESIDUAL_TOL
from dcforge.gpu.pbc.df.fft import FFTDF, grad_jk
from dcforge.gpu.pbc.df.gdf import GDF
from dcforge.gpu.pbc.df.gdf import (
    make_gradient_response as make_gdf_gradient_response,
)
from dcforge.gpu.pbc.df.rsdf import (
    RSDF,
    make_kpts_contracted_gradient_response,
    make_kpts_full_coulomb_response_probe,
    make_kpts_gradient_response,
)
from dcforge.gpu.pbc.energy.krhf_dft import (
    KRHFDFT,
    _check_input,
    _target_fock_signature,
)
from dcforge.gpu.pbc.grad.krhf_nv import (
    grad_fock_contracted,
    grad_fock_direct,
    grad_fock_resolved,
    grad_fock_resolved_contracted,
    nonvariational_correction,
)
from dcforge.gpu.pbc.grad.krhf_skeleton import (
    get_ovlp_deriv,
    prepare_gaussian_hcore_pair,
    real_gradient_term,
    skeleton_gradient,
)
from dcforge.gpu.pbc.grad.krhf_zvec import (
    gen_hf_response,
    make_fvo,
    make_mo_partitions,
    make_z_dm,
    solve_z_vector,
    solve_z_vector_matrix_free,
)


def _profile_start(mf_grad):
    '''Synchronize and start one optional validation-only stage timer.'''
    if not mf_grad.profile_gradient_stages:
        return None
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


def _profile_stop(profile, label, start):
    '''Synchronize and store one optional GPU/host stage wall time.'''
    if start is None:
        return
    cp.cuda.Stream.null.synchronize()
    profile[label] = time.perf_counter() - start


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None):
    r'''Evaluate the electronic GPU full-BZ semilocal-DFA@KRHF gradient.

    The term ledger mirrors the CPU implementation:

    ``de_elec = de_hcore + de_j + de_vxc + de_ovlp``
    ``        + de_fock + de_ovlp_energy + de_ovlp_response``.

    FFTDF materializes complete device J/K matrix derivatives for ``P`` and
    ``D_Z``.  GDF uses a native density-contracted response.  RSDF uses the
    provider-exact resolved contracted response and retains a full-matrix
    oracle; its one-split GPU4PySCF response is exposed only as a disqualified
    full-Coulomb probe.  The production response solver is matrix-free, while
    the dense real-linear solve remains the small-system numerical oracle.
    '''
    mf = mf_grad.base
    mf_hf = mf.mf_hf
    cell = mf.cell
    _check_input(mf)
    if not isinstance(mf.with_df, (FFTDF, GDF, RSDF)):
        raise NotImplementedError(
            'GPU sampled-k gradient supports FFTDF, GDF and RSDF only')
    if mf.e_dft is None:
        raise RuntimeError('run the GPU sampled-k energy kernel first')
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    if atmlst is None:
        atmlst = range(cell.natm)
    atmlst = [int(ia) for ia in atmlst]
    profile = {} if mf_grad.profile_gradient_stages else None
    total_start = _profile_start(mf_grad)

    mo_energy = cp.asarray(mo_energy)
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    Co, Cv, _e_ai = make_mo_partitions(
        mo_energy, mo_coeff, mo_occ)
    Co_array = cp.asarray(Co)
    dm0 = cp.asarray(mf_hf.make_rdm1(mo_coeff, mo_occ))

    stage_start = _profile_start(mf_grad)
    signature = _target_fock_signature(mf, dm0)
    if (mf.target_fock is not None and mf.dm_hf is not None
            and mf.target_fock_signature == signature
            and bool(cp.array_equal(mf.dm_hf, dm0).get())):
        fock = cp.asarray(mf.target_fock)
        mf_grad.target_fock_reused = True
    else:
        h1e = mf.get_hcore(cell, mf.kpts)
        fock = cp.asarray(h1e) + cp.asarray(
            mf.get_veff(cell, dm0, kpts=mf.kpts))
        mf_grad.target_fock_reused = False
    fvo = make_fvo(mf, Co, Cv)
    foo = cp.asarray([
        Co_k.conj().T @ fock_k @ Co_k
        for Co_k, fock_k in zip(Co, fock, strict=True)])
    _profile_stop(profile, 'target_fock_and_projection', stage_start)

    stage_start = _profile_start(mf_grad)
    fvind = gen_hf_response(mf_hf)
    dimension = 2 * sum(int(block.size) for block in fvo)
    zvec_method = mf_grad.zvec_method
    if zvec_method == 'auto':
        zvec_method = (
            'dense' if dimension <= mf_grad.dense_oracle_size
            else 'matrix_free')
    if zvec_method == 'dense':
        zvec, z_info = solve_z_vector(
            mo_energy, mo_occ, Co, Cv, fvo, fvind,
            residual_tol=mf_grad.cphf_residual_tol,
            dense_oracle_size=mf_grad.dense_oracle_size,
            kpts_weights=mf.kpts_weights)
    else:
        zvec, z_info = solve_z_vector_matrix_free(
            mo_energy, mo_occ, Co, Cv, fvo, fvind,
            conv_tol=mf_grad.cphf_conv_tol,
            residual_tol=mf_grad.cphf_residual_tol,
            max_cycle=mf_grad.cphf_max_cycle,
            kpts_weights=mf.kpts_weights,
            self_adjoint_tol=mf_grad.self_adjoint_tol,
            restart=mf_grad.gmres_restart)
    mf_grad.zvec_method_used = zvec_method
    mf_grad.zvec = cp.asarray(zvec)
    mf_grad.z_rhs_max_abs = z_info['rhs_max_abs']
    mf_grad.z_residual_max_abs = z_info['residual_max_abs']
    mf_grad.z_residual_relative = z_info['residual_relative']
    mf_grad.z_solver = z_info['solver']
    mf_grad.z_dimension = z_info['dimension']
    mf_grad.z_condition_number = z_info['condition_number']
    mf_grad.z_matrix_asymmetry_max_abs = (
        z_info['matrix_asymmetry_max_abs'])
    mf_grad.z_direct_adjoint_max_abs = z_info['direct_adjoint_max_abs']
    mf_grad.z_self_adjoint_probe_error = z_info.get(
        'self_adjoint_probe_error')
    mf_grad.z_iterations = z_info.get('iterations')
    mf_grad.z_restart_cycles = z_info.get('restart_cycles')
    mf_grad.z_operator_calls = z_info.get('operator_calls')
    mf_grad.z_preconditioner_calls = z_info.get('preconditioner_calls')
    mf_grad.z_dense_matrix_bytes = z_info.get('dense_matrix_bytes')
    mf_grad.z_workspace_estimate_bytes = z_info.get(
        'workspace_estimate_bytes')
    mf_grad.z_krylov_host_control = z_info.get('krylov_host_control')
    mf_grad.z_krylov_host_control_bytes_estimate = z_info.get(
        'krylov_host_control_bytes_estimate')
    mf_grad.z_wall_seconds = z_info.get('wall_seconds')
    dz = make_z_dm(Co, Cv, zvec)
    mf_grad.dz = dz
    _profile_stop(profile, 'z_vector', stage_start)

    stage_start = _profile_start(mf_grad)
    s1 = get_ovlp_deriv(mf_grad)
    _profile_stop(profile, 'overlap_derivative', stage_start)
    stage_start = _profile_start(mf_grad)
    if isinstance(mf.with_df, (GDF, RSDF)):
        prepare_gaussian_hcore_pair(
            mf_grad, dm0, dz, atmlst)
    _profile_stop(profile, 'gaussian_hcore_pair', stage_start)
    stage_start = _profile_start(mf_grad)
    if isinstance(mf.with_df, GDF):
        gdf_terms, gdf_metadata = make_gdf_gradient_response(
            mf_hf, dm0, dz, atmlst=atmlst)
        vj1 = vk1 = vj1_z = vk1_z = None
        mf_grad.de_j = gdf_terms['de_j']
        mf_grad.de_j_gdf_native = gdf_terms['de_j']
        mf_grad.de_fock_gdf_native = gdf_terms['de_fock']
        mf_grad.gdf_response_metadata = gdf_metadata
        mf_grad.provider_response_backend = (
            gdf_metadata['response_backend'])
        mf_grad.gdf_native_host_result_bytes = int(
            gdf_metadata['native_host_result_bytes'])
        vj1_for_skeleton = None
    elif isinstance(mf.with_df, RSDF):
        if mf_grad.rsdf_response_mode == 'full_coulomb_probe':
            rsdf_response, rsdf_metadata = (
                make_kpts_full_coulomb_response_probe(
                    mf_hf, dm0, dz, atmlst=atmlst))
            vj1 = vk1 = vj1_z = vk1_z = None
            mf_grad.de_j = real_gradient_term(
                mf_grad, rsdf_response['de_j'], 'de_j')
            mf_grad.de_j_rsdf_native = mf_grad.de_j
            mf_grad.de_fock_rsdf_native = rsdf_response['de_fock']
            mf_grad.de_j_source = None
            mf_grad.de_j_metric = None
            mf_grad.rsdf_response_storage_used = 'native_combined'
        elif (mf_grad.rsdf_response_mode == 'oracle'
              and mf_grad.rsdf_response_storage == 'contracted'):
            rsdf_response, rsdf_metadata = (
                make_kpts_contracted_gradient_response(
                    mf.with_df, dm0, dz, s1, atmlst=atmlst,
                    exxdiv=mf_hf.exxdiv))
            vj1 = vk1 = vj1_z = vk1_z = None
            mf_grad.de_j_source = real_gradient_term(
                mf_grad, rsdf_response['de_j_source'], 'de_j_source')
            mf_grad.de_j_metric = real_gradient_term(
                mf_grad, rsdf_response['de_j_metric'], 'de_j_metric')
            mf_grad.rsdf_response_storage_used = 'contracted'
        elif (mf_grad.rsdf_response_mode == 'oracle'
              and mf_grad.rsdf_response_storage == 'full'):
            rsdf_response, rsdf_metadata = make_kpts_gradient_response(
                mf.with_df, dm0, dz, s1, exxdiv=mf_hf.exxdiv)
            for name, value in rsdf_response.items():
                setattr(mf_grad, name, value)
            vj1 = (
                rsdf_response['vj1_source']
                + rsdf_response['vj1_metric'])
            vk1 = (
                rsdf_response['vk1_source']
                + rsdf_response['vk1_metric']
                + rsdf_response['vk1_ewald'])
            vj1_z = (
                rsdf_response['vj1_source_z']
                + rsdf_response['vj1_metric_z'])
            vk1_z = (
                rsdf_response['vk1_source_z']
                + rsdf_response['vk1_metric_z']
                + rsdf_response['vk1_ewald_z'])
            nkpts = len(mf.kpts)
            mf_grad.de_j_source = real_gradient_term(
                mf_grad, .5 * cp.einsum(
                    'axkij,kji->ax', rsdf_response['vj1_source'], dm0)
                / nkpts, 'de_j_source')
            mf_grad.de_j_metric = real_gradient_term(
                mf_grad, .5 * cp.einsum(
                    'axkij,kji->ax', rsdf_response['vj1_metric'], dm0)
                / nkpts, 'de_j_metric')
            mf_grad.rsdf_response_storage_used = 'full'
        else:
            raise ValueError(
                "rsdf_response_mode must be 'oracle' or "
                "'full_coulomb_probe', and "
                "rsdf_response_storage must be 'contracted' or 'full'")
        if mf_grad.rsdf_response_mode == 'oracle':
            mf_grad.de_j = mf_grad.de_j_source + mf_grad.de_j_metric
        mf_grad.rsdf_response = (
            rsdf_response
            if mf_grad.rsdf_response_storage_used == 'full' else None)
        mf_grad.rsdf_response_metadata = rsdf_metadata
        mf_grad.provider_response_backend = (
            rsdf_metadata['response_backend'])
        mf_grad.rsdf_host_input_bytes = int(
            rsdf_metadata.get('host_input_bytes', 0))
        mf_grad.rsdf_host_result_bytes = int(
            rsdf_metadata.get(
                'host_result_bytes',
                rsdf_metadata.get('native_host_result_bytes', 0)))
        mf_grad.rsdf_host_workspace_bytes = int(
            rsdf_metadata.get('host_workspace_bytes', 0))
        mf_grad.rsdf_host_peak_jk_matrix_bytes_estimate = int(
            rsdf_metadata.get('host_peak_jk_matrix_bytes_estimate', 0))
        mf_grad.rsdf_device_full_matrix_bytes_avoided = int(
            rsdf_metadata.get('device_full_matrix_bytes_avoided', 0))
        if mf_grad.rsdf_response_mode == 'oracle':
            mf_grad.scientific_host_fallback = True
            mf_grad.host_fallback_components = tuple(dict.fromkeys(
                mf_grad.host_fallback_components + (
                    'DC-Forge CPU sampled-k RSDF derivative oracle',)))
        vj1_for_skeleton = None
    else:
        vj1, vk1 = grad_jk(
            cell, dm0, exxdiv=mf_hf.exxdiv, with_df=mf.with_df,
            kpts=mf.kpts)
        vj1_z, vk1_z = grad_jk(
            cell, dz, exxdiv=mf_hf.exxdiv, with_df=mf.with_df,
            kpts=mf.kpts)
        mf_grad.provider_response_backend = (
            'device_full_matrix_correctness')
        vj1_for_skeleton = vj1
    mf_grad.vj1 = vj1
    mf_grad.vk1 = vk1
    mf_grad.vj1_z = vj1_z
    mf_grad.vk1_z = vk1_z
    _profile_stop(profile, 'provider_response', stage_start)

    stage_start = _profile_start(mf_grad)
    de_skel = skeleton_gradient(
        mf_grad, dm0, foo, Co_array, s1, vj1_for_skeleton, atmlst)
    _profile_stop(profile, 'skeleton', stage_start)
    stage_start = _profile_start(mf_grad)
    if isinstance(mf.with_df, GDF):
        de_fock = grad_fock_contracted(
            mf_grad, dz, gdf_terms['de_fock'], atmlst)
    elif isinstance(mf.with_df, RSDF):
        if mf_grad.rsdf_response_storage_used == 'native_combined':
            de_fock = grad_fock_contracted(
                mf_grad, dz, rsdf_response['de_fock'], atmlst)
        elif mf_grad.rsdf_response_storage_used == 'contracted':
            de_fock = grad_fock_resolved_contracted(
                mf_grad, dz, rsdf_response, atmlst)
        else:
            de_fock = grad_fock_resolved(
                mf_grad, dz, dm0, rsdf_response, atmlst)
    else:
        de_fock = grad_fock_direct(
            mf_grad, Co, Cv, zvec, dm0, vj1, vk1,
            vj1_z, vk1_z, atmlst)
    _profile_stop(profile, 'source_fock_contraction', stage_start)
    stage_start = _profile_start(mf_grad)
    de_nvar = nonvariational_correction(
        mf_grad, Co, Cv, mo_energy, mo_occ, zvec, s1,
        de_fock, fvind, atmlst)
    _profile_stop(profile, 'nonvariational', stage_start)
    mf_grad.fftdf_response_backend = (
        mf_grad.provider_response_backend
        if isinstance(mf.with_df, FFTDF) else None)
    _profile_stop(profile, 'electronic_total', total_start)
    mf_grad.gradient_stage_timings = profile
    return de_skel + de_nvar


class Gradients(gpu_rhf_grad.GradientsBase):
    '''Analytic fixed-cell GPU gradient of sampled-k ``E_DFA[P_KRHF]``.'''

    _keys = gpu_rhf_grad.GradientsBase._keys | {
        'cphf_conv_tol', 'cphf_residual_tol', 'cphf_max_cycle',
        'dense_oracle_size', 'zvec_method', 'zvec_method_used',
        'self_adjoint_tol', 'gmres_restart', 'grid_response',
        'profile_gradient_stages', 'gradient_stage_timings',
        'target_fock_reused', 'xc_deriv_backend',
        'de_j_xc_multigrid',
        'zvec', 'dz', 'z_rhs_max_abs', 'z_residual_max_abs',
        'z_residual_relative', 'z_solver', 'z_dimension',
        'z_condition_number', 'z_matrix_asymmetry_max_abs',
        'z_direct_adjoint_max_abs', 'z_dense_matrix_bytes',
        'z_self_adjoint_probe_error', 'z_iterations',
        'z_restart_cycles', 'z_operator_calls',
        'z_preconditioner_calls', 'z_workspace_estimate_bytes',
        'z_krylov_host_control',
        'z_krylov_host_control_bytes_estimate', 'z_wall_seconds',
        'vj1', 'vk1', 'vj1_z', 'vk1_z',
        'fock_derivative_reverse_error', 'fftdf_response_backend',
        'provider_response_backend', 'gdf_response_metadata',
        'gdf_native_host_result_bytes', 'de_j_gdf_native',
        'de_fock_gdf_native', 'de_j_rsdf_native',
        'de_fock_rsdf_native',
        'rsdf_response', 'rsdf_response_metadata',
        'rsdf_response_mode', 'rsdf_response_storage',
        'rsdf_response_storage_used',
        'rsdf_host_input_bytes', 'rsdf_host_result_bytes',
        'rsdf_host_workspace_bytes',
        'rsdf_host_peak_jk_matrix_bytes_estimate',
        'rsdf_device_full_matrix_bytes_avoided',
        'vj1_source', 'vk1_source', 'vj1_metric', 'vk1_metric',
        'vk1_ewald', 'vj1_source_z', 'vk1_source_z',
        'vj1_metric_z', 'vk1_metric_z', 'vk1_ewald_z',
        'de_j_source', 'de_j_metric', 'de_fock_source',
        'de_fock_metric', 'de_fock_ewald',
        'imaginary_residual_tol', 'term_imaginary_residuals',
        'term_imaginary_residual_max_abs',
        'term_raw_imaginary_components',
        'term_raw_imaginary_component_max_abs',
        'nonlocal_pseudo_host_result_bytes',
        'xc_derivative_host_result_bytes', 'final_host_result_bytes',
        'gaussian_hcore_host_calls',
        'gaussian_hcore_contraction_calls', 'gaussian_hcore_method',
        'gaussian_hcore_pair', 'gaussian_hcore_components',
        'gaussian_hcore_host_input_bytes',
        'gaussian_hcore_host_workspace_bytes',
        'gaussian_hcore_host_result_bytes',
        'gaussian_hcore_full_matrix_bytes_avoided',
        'gaussian_hcore_host_owner',
        'scientific_host_fallback', 'host_fallback_components',
        'translation_residual', 'translation_residual_max_abs',
        'term_recomposition_max_abs',
    }

    def __init__(self, mf):
        if not isinstance(mf, KRHFDFT):
            raise TypeError(
                'GPU sampled-k Gradients requires GPU KRHFDFT')
        if not isinstance(mf.with_df, (FFTDF, GDF, RSDF)):
            raise NotImplementedError(
                'GPU sampled-k Gradients supports FFTDF, GDF and RSDF only')
        gpu_rhf_grad.GradientsBase.__init__(self, mf)
        self.cphf_conv_tol = 1e-9
        self.cphf_residual_tol = 1e-8
        self.cphf_max_cycle = 200
        self.dense_oracle_size = 512
        self.zvec_method = 'auto'
        self.zvec_method_used = None
        self.self_adjoint_tol = 1e-10
        self.gmres_restart = 80
        self.grid_response = True
        self.profile_gradient_stages = False
        self.gradient_stage_timings = None
        self.target_fock_reused = False
        self.xc_deriv_backend = None
        self.zvec = None
        self.dz = None
        self.z_rhs_max_abs = None
        self.z_residual_max_abs = None
        self.z_residual_relative = None
        self.z_solver = None
        self.z_dimension = None
        self.z_condition_number = None
        self.z_matrix_asymmetry_max_abs = None
        self.z_direct_adjoint_max_abs = None
        self.z_self_adjoint_probe_error = None
        self.z_iterations = None
        self.z_restart_cycles = None
        self.z_operator_calls = None
        self.z_preconditioner_calls = None
        self.z_dense_matrix_bytes = None
        self.z_workspace_estimate_bytes = None
        self.z_krylov_host_control = None
        self.z_krylov_host_control_bytes_estimate = None
        self.z_wall_seconds = None
        self.vj1 = self.vk1 = None
        self.vj1_z = self.vk1_z = None
        self.fock_derivative_reverse_error = None
        self.fftdf_response_backend = None
        self.provider_response_backend = None
        self.gdf_response_metadata = None
        self.gdf_native_host_result_bytes = 0
        self.de_j_gdf_native = None
        self.de_fock_gdf_native = None
        self.de_j_rsdf_native = None
        self.de_j_xc_multigrid = None
        self.de_fock_rsdf_native = None
        self.rsdf_response = None
        self.rsdf_response_metadata = None
        self.rsdf_response_mode = 'oracle'
        self.rsdf_response_storage = 'contracted'
        self.rsdf_response_storage_used = None
        self.rsdf_host_input_bytes = 0
        self.rsdf_host_result_bytes = 0
        self.rsdf_host_workspace_bytes = 0
        self.rsdf_host_peak_jk_matrix_bytes_estimate = 0
        self.rsdf_device_full_matrix_bytes_avoided = 0
        self.vj1_source = self.vk1_source = None
        self.vj1_metric = self.vk1_metric = None
        self.vk1_ewald = None
        self.vj1_source_z = self.vk1_source_z = None
        self.vj1_metric_z = self.vk1_metric_z = None
        self.vk1_ewald_z = None
        self.de_j_source = self.de_j_metric = None
        self.de_fock_source = self.de_fock_metric = None
        self.de_fock_ewald = None
        self.imaginary_residual_tol = IMAGINARY_RESIDUAL_TOL
        self.term_imaginary_residuals = {}
        self.term_imaginary_residual_max_abs = 0.0
        self.term_raw_imaginary_components = {}
        self.term_raw_imaginary_component_max_abs = 0.0
        self.nonlocal_pseudo_host_result_bytes = 0
        self.xc_derivative_host_result_bytes = 0
        self.final_host_result_bytes = 0
        self.gaussian_hcore_host_calls = 0
        self.gaussian_hcore_contraction_calls = 0
        self.gaussian_hcore_method = 'contracted'
        self.gaussian_hcore_pair = None
        self.gaussian_hcore_components = None
        self.gaussian_hcore_host_input_bytes = 0
        self.gaussian_hcore_host_workspace_bytes = 0
        self.gaussian_hcore_host_result_bytes = 0
        self.gaussian_hcore_full_matrix_bytes_avoided = 0
        self.gaussian_hcore_host_owner = None
        self.scientific_host_fallback = mf.scientific_host_fallback
        self.host_fallback_components = mf.host_fallback_components
        self.translation_residual = None
        self.translation_residual_max_abs = None
        self.term_recomposition_max_abs = None

    def dump_flags(self, verbose=None):
        gpu_rhf_grad.GradientsBase.dump_flags(self, verbose)
        logger.info(self, 'zvec_method = %s', self.zvec_method)
        logger.info(self, 'dense_oracle_size = %d', self.dense_oracle_size)
        logger.info(self, 'cphf_conv_tol = %g', self.cphf_conv_tol)
        logger.info(
            self, 'cphf_residual_tol = %g', self.cphf_residual_tol)
        logger.info(self, 'cphf_max_cycle = %d', self.cphf_max_cycle)
        logger.info(self, 'self_adjoint_tol = %g', self.self_adjoint_tol)
        logger.info(self, 'gmres_restart = %d', self.gmres_restart)
        logger.info(
            self, 'profile_gradient_stages = %s',
            self.profile_gradient_stages)
        logger.info(self, 'rsdf_response_mode = %s', self.rsdf_response_mode)
        logger.info(
            self, 'rsdf_response_storage = %s', self.rsdf_response_storage)
        logger.info(
            self, 'scientific_host_fallback = %s',
            self.scientific_host_fallback)
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None,
               atmlst=None):
        '''Evaluate sampled-k analytic ``dE/dR`` in Ha/Bohr.'''
        if not self.grid_response:
            raise NotImplementedError(
                'GPU sampled-k target derivative requires grid_response=True')
        if self.zvec_method not in ('auto', 'dense', 'matrix_free'):
            raise ValueError(
                "zvec_method must be 'auto', 'dense' or 'matrix_free'")
        if self.cphf_conv_tol <= 0:
            raise ValueError('cphf_conv_tol must be positive')
        if self.cphf_residual_tol <= 0:
            raise ValueError('cphf_residual_tol must be positive')
        if self.cphf_max_cycle < 1:
            raise ValueError('cphf_max_cycle must be positive')
        if self.dense_oracle_size < 1:
            raise ValueError('dense_oracle_size must be positive')
        if self.self_adjoint_tol <= 0:
            raise ValueError('self_adjoint_tol must be positive')
        if self.gmres_restart < 1:
            raise ValueError('gmres_restart must be positive')
        if self.rsdf_response_mode not in (
                'oracle', 'full_coulomb_probe'):
            raise ValueError(
                "rsdf_response_mode must be 'oracle' or "
                "'full_coulomb_probe'")
        if self.rsdf_response_storage not in ('contracted', 'full'):
            raise ValueError(
                "rsdf_response_storage must be 'contracted' or 'full'")
        if atmlst is None:
            atmlst = list(range(self.cell.natm))
        else:
            atmlst = [int(ia) for ia in atmlst]
        self.atmlst = atmlst

        de_elec = grad_elec(
            self, mo_energy=mo_energy, mo_coeff=mo_coeff,
            mo_occ=mo_occ, atmlst=atmlst)
        de_nuc = cp.asarray(self.grad_nuc(atmlst=atmlst))
        de = de_elec + de_nuc
        self.de_nuc = de_nuc
        self.de_skel = self.de_skel_elec + de_nuc
        self.de_terms = (
            self.de_hcore + self.de_veff + self.de_ovlp
            + self.de_fock + self.de_ovlp_energy
            + self.de_ovlp_response + self.de_nuc)
        self.term_recomposition_max_abs = float(cp.max(cp.abs(
            de - self.de_terms)).get())
        self.translation_residual = cp.sum(de, axis=0)
        self.translation_residual_max_abs = float(cp.max(cp.abs(
            self.translation_residual)).get())

        self.final_host_result_bytes += int(de.nbytes)
        self.de = cp.asnumpy(de)
        self._move_term_ledger_to_host()
        self._finalize()
        return self.de

    grad_elec = grad_elec

    def as_scanner(self):
        raise NotImplementedError('GPU sampled-k scanner is deferred')

    def _move_term_ledger_to_host(self):
        names = (
            'de_hcore', 'de_j', 'de_vxc', 'de_veff', 'de_ovlp',
            'de_grid', 'de_extra', 'de_skel_elec', 'de_fock_hcore',
            'de_fock_jk', 'de_fock_jk_reverse', 'de_fock',
            'de_j_gdf_native', 'de_fock_gdf_native',
            'de_j_rsdf_native', 'de_fock_rsdf_native',
            'de_j_source', 'de_j_metric', 'de_fock_source',
            'de_fock_metric', 'de_fock_ewald',
            'de_ovlp_energy', 'de_ovlp_response', 'de_nvar',
            'de_nuc', 'de_skel', 'de_terms', 'translation_residual',
        )
        for name in names:
            value = getattr(self, name, None)
            if isinstance(value, cp.ndarray):
                self.final_host_result_bytes += int(value.nbytes)
                setattr(self, name, cp.asnumpy(value))


Grad = Gradients
