'''Reusable perturbation-independent Gamma RHF response state.'''

import hashlib

import cupy as cp
import numpy as np

from dcforge.gpu.pbc.energy.rhf_dft import _target_fock_signature


def _array_fingerprint(array):
    '''Return an exact, storage-independent fingerprint of one GPU array.

    Response-state reuse is much cheaper than another GDF CPHF solve, but a
    stale response is not an admissible optimization.  The fingerprint makes
    a bounded device-to-host copy of the MO state and hashes its exact bytes.
    It deliberately does not retain a second device copy of ``C``.
    '''
    host = np.ascontiguousarray(cp.asnumpy(cp.asarray(array)))
    digest = hashlib.sha256()
    digest.update(str(host.dtype).encode())
    digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
    digest.update(host.view(np.uint8))
    return digest.hexdigest()


def make_response_state_signature(
        mf_grad, mo_energy, mo_coeff, mo_occ, dm0):
    '''Return the complete identity of a reusable RHF adjoint solution.

    The Z-vector is reusable among nuclear-displacement components.  It is
    not independent of the source orbitals/Hessian, target Fock, provider,
    or solver contract.
    All of those inputs are therefore included.  Performance-only direct-VO
    cache controls are included as well so a requested response benchmark is
    never silently replaced by an earlier cached solve.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.rhf_dft.Gradients
            Active derivative driver.
        mo_energy, mo_coeff, mo_occ : cupy.ndarray
            Source-RHF canonical state.
        dm0 : cupy.ndarray
            Source density defining the fixed-density target Fock.

    Returns:
        tuple
            Immutable response-state signature.
    '''
    mf = mf_grad.base
    provider = mf.with_df
    return (
        'dcforge.gpu.pbc.rhf_response_state.v2',
        _target_fock_signature(mf, dm0),
        _array_fingerprint(mo_energy),
        _array_fingerprint(mo_coeff),
        _array_fingerprint(mo_occ),
        type(provider).__module__,
        type(provider).__name__,
        id(provider),
        getattr(mf.mf_hf, 'exxdiv', None),
        getattr(provider, 'omega', None),
        float(mf_grad.cphf_conv_tol),
        (None if mf_grad.cphf_krylov_tol is None
         else float(mf_grad.cphf_krylov_tol)),
        float(mf_grad.cphf_residual_tol),
        int(mf_grad.cphf_max_cycle),
        str(mf_grad.cphf_refinement_solver),
        str(mf_grad.cphf_preconditioner),
        bool(mf_grad.cphf_preconditioner_required),
        str(mf_grad.precision_policy),
        float(mf_grad.mixed_z_inner_tol),
        float(mf_grad.mixed_z_correction_tol),
        float(mf_grad.mixed_z_quantized_residual_tol),
        float(mf_grad.mixed_z_quantized_hard_reject_tol),
        int(mf_grad.mixed_z_max_corrections),
        str(mf_grad.df_cphf_response_mode),
        str(mf_grad.df_cphf_response_cache),
        float(mf_grad.df_cphf_response_memory_fraction),
        int(mf_grad.df_cphf_response_minimum_reserve_bytes),
        bool(mf_grad.profile_df_cphf_response),
        bool(mf_grad.profile_gradient_stages),
        bool(getattr(mf_grad, 'terminal_response_reuse', False)),
    )


def response_state_token(signature):
    '''Return a compact provenance token for one response signature.'''
    return hashlib.sha256(repr(signature).encode()).hexdigest()


def get_response_state(mf_grad, signature):
    '''Return a valid cached state, or ``None`` after any strict mismatch.'''
    state = getattr(mf_grad, 'response_state', None)
    if state is None or state.get('signature') != signature:
        return None
    z_info = state.get('z_info', {})
    residual = float(z_info.get('residual_max_abs', np.inf))
    if not np.isfinite(residual):
        return None
    if residual > float(mf_grad.cphf_residual_tol):
        return None
    return state


def store_response_state(
        mf_grad, signature, *, owner, mo_energy, mo_coeff, mo_occ,
        Co, Cv, dm0, target_fock, fvo, foo, zvec, z_info,
        response_orbitals, dz, z_response_profile,
        z_warm_start_metadata=None, ovlp_response_intermediate=None,
        terminal_response_metadata=None,
        cphf_preconditioner_metadata=None):
    '''Store only bounded perturbation-independent response quantities.

    The direct-VO response object is intentionally absent.  Its multi-GiB
    optional GDF intermediates are closed by the caller immediately after the
    solve; only MO/AO factors already needed by the derivative are retained.
    '''
    solve_count = int(getattr(mf_grad, 'z_solve_count', 0)) + 1
    state = {
        'signature': signature,
        'token': response_state_token(signature),
        'owner': str(owner),
        'solve_ordinal': solve_count,
        'mo_energy': mo_energy,
        'mo_coeff': mo_coeff,
        'mo_occ': mo_occ,
        'Co': Co,
        'Cv': Cv,
        'dm0': dm0,
        'target_fock': target_fock,
        'fvo': fvo,
        'foo': foo,
        'zvec': zvec,
        'z_info': dict(z_info),
        'response_orbitals': response_orbitals,
        'dz': dz,
        'z_response_profile': z_response_profile,
        'z_warm_start_metadata': z_warm_start_metadata,
        # At most one occupied-occupied matrix is retained.  It is tied to the
        # complete response-state signature above and therefore cannot cross a
        # provider, orbital, precision-policy or geometry change.
        'ovlp_response_intermediate': ovlp_response_intermediate,
        'terminal_response_metadata': terminal_response_metadata,
        'cphf_preconditioner_metadata': cphf_preconditioner_metadata,
    }
    mf_grad.response_state = state
    mf_grad.response_state_signature = signature
    mf_grad.response_state_token = state['token']
    mf_grad.z_solve_count = solve_count
    return state


def use_response_state(mf_grad, state, *, reused):
    '''Publish one cached or newly solved state through PySCF-style fields.'''
    z_info = state['z_info']
    profile = state.get('z_response_profile')
    if isinstance(profile, dict):
        profile = dict(profile)
        profile['response_state_reused'] = bool(reused)
        profile['response_state_owner'] = state['owner']
        profile['current_action_count'] = 0 if reused else profile.get(
            'action_count')

    mf_grad.response_state_reused = bool(reused)
    mf_grad.response_state_owner = state['owner']
    mf_grad.response_state_token = state['token']
    if reused:
        mf_grad.response_state_reuse_count = int(getattr(
            mf_grad, 'response_state_reuse_count', 0)) + 1
    mf_grad.zvec = state['zvec']
    mf_grad.z_info = z_info
    mf_grad.z_rhs_max_abs = z_info['rhs_max_abs']
    mf_grad.z_residual_max_abs = z_info['residual_max_abs']
    mf_grad.z_residual_relative = z_info['residual_relative']
    mf_grad.z_solver = z_info['solver']
    mf_grad.z_dimension = z_info['dimension']
    mf_grad.z_primary_wall_seconds = z_info['primary_wall_seconds']
    mf_grad.z_refinement_wall_seconds = z_info['refinement_wall_seconds']
    mf_grad.z_refinement_iterations = z_info['refinement_iterations']
    mf_grad.z_requested_conv_tol = z_info['requested_conv_tol']
    mf_grad.z_requested_krylov_tol = z_info['requested_krylov_tol']
    mf_grad.z_effective_krylov_tol = z_info['effective_krylov_tol']
    mf_grad.z_krylov_tolerance_policy = z_info[
        'krylov_tolerance_policy']
    mf_grad.z_physical_residual_tol = z_info['physical_residual_tol']
    mf_grad.z_cleanup_krylov_tol = z_info['cleanup_krylov_tol']
    mf_grad.z_cleanup_wall_seconds = z_info['cleanup_wall_seconds']
    mf_grad.z_cleanup_iterations = z_info['cleanup_iterations']
    mf_grad.z_wall_seconds = z_info['wall_seconds']
    mf_grad.z_current_wall_seconds = (
        0. if reused else z_info['wall_seconds'])
    mf_grad.z_response_backend = z_info['response_backend']
    mf_grad.cphf_preconditioner_metadata = dict(
        state.get('cphf_preconditioner_metadata') or {})
    mf_grad.z_response_profile = profile
    return profile


__all__ = [
    'get_response_state',
    'make_response_state_signature',
    'response_state_token',
    'store_response_state',
    'use_response_state',
]
