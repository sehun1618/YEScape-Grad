'''Public stress gate for Gamma-point GPU ``E_DFA[P_RHF]`` workflows.'''

import time

import numpy as np
from gpu4pyscf.pbc.dft import multigrid_v2
from pyscf import lib
from pyscf.pbc import gto

from dcforge.gpu.pbc.df.gdf import GDF
from dcforge.gpu.pbc.df.rsdf import RSDF
from dcforge.gpu.pbc.energy.rhf_dft import RHFDFT, _target_xc_name
from dcforge.gpu.pbc.grad.memory import resolve_memory_policy
from dcforge.gpu.pbc.stress.memory import stress_memory_preflight
from dcforge.gpu.pbc.stress.profiler import (
    finish_profile,
    new_profile,
    stage_start,
    stage_stop,
)
from dcforge.gpu.pbc.stress.strain import (
    pressure_from_stress,
    stress_from_work,
    stress_tensor_to_voigt,
    virial_from_stress,
)


def check_stress_input(mf_grad):
    '''Validate the first GPU PBC DC-DFT stress ownership contract.'''
    mf = mf_grad.base
    cell = mf.cell
    if not isinstance(mf, RHFDFT):
        raise TypeError('GPU PBC stress requires dcforge GPU RHFDFT')
    if cell.dimension != 3:
        raise NotImplementedError('PBC stress requires a three-dimensional Cell')
    if cell.spin != 0 or cell.nelectron % 2:
        raise NotImplementedError('PBC stress requires a closed-shell Cell')
    if not cell._pseudo or cell._ecp:
        raise NotImplementedError('PBC stress requires a GTH pseudopotential')
    if not isinstance(mf.with_df, (GDF, RSDF)):
        raise NotImplementedError(
            'PBC stress requires a GDF or RSDF source/target provider')
    if mf.with_df is not mf.mf_hf.with_df:
        raise ValueError(
            'source RHF and target DFA must share one Gaussian-DF provider')
    if not isinstance(mf._numint, multigrid_v2.MultiGridNumInt):
        raise NotImplementedError(
            'the first PBC stress route requires target XC multigrid')
    if mf.multigrid_with_j:
        raise NotImplementedError(
            'the first PBC stress route requires XC-only multigrid '
            '(with_j=False)')
    if _target_xc_name(mf) not in ('PBE', 'R2SCAN'):
        raise NotImplementedError('GPU PBC stress supports PBE then r2SCAN')
    kpt = np.asarray(getattr(mf.mf_hf, 'kpt', np.zeros(3)))
    if kpt.shape != (3,) or np.max(np.abs(kpt)) > 1.e-12:
        raise NotImplementedError('the first PBC stress route requires Gamma point')


def kernel(mf_grad):
    '''Return the production analytic DC-DFT stress tensor.

    The primary return follows the PySCF stress convention requested by the
    DC-Forge contract: a symmetric ``(3,3)`` tensor in ``Ha/Bohr^3``.  The
    complete term, owner, memory and provenance report is retained as
    ``mf_grad.stress_report``.  Analytic host primitives are permitted only
    when named explicitly; a numerical Cell displacement or silent backend
    substitution fails closed.
    '''
    check_stress_input(mf_grad)
    mf = mf_grad.base
    if (not mf.mf_hf.converged
            or not bool(getattr(mf, '_source_converged', False))
            or mf.e_dft is None):
        raise RuntimeError(
            'run the source RHF and fixed-density target energy before '
            'requesting PBC stress')
    options = dict(getattr(mf_grad, 'stress_reference_options', None) or {})
    report = reference_kernel(mf_grad, **options)
    _check_production_report(mf_grad, report)
    mf_grad.stress_report = report
    return np.asarray(report['stress_tensor'])


def reference_kernel(
        mf_grad, reciprocal_block_size=None,
        auxiliary_block_size=None,
        overlap_gdf_short_long_range=True,
        pp_reciprocal_mesh=None, pp_block_size=None,
        xc_image_block_size=None, validate_central=False,
        strict_memory_preflight=False, memory_fraction=.8,
        minimum_reserve_bytes=4 * 1024 ** 3):
    r'''Return the complete hybrid GPU/host analytic stress ledger.

    One target Fock and one device Z-vector are shared by all six strain
    components.  Target-J/source-Fock GDF adjoints, overlap/kinetic
    contractions, local-part1 long-range contraction and XC-only multigrid
    work run on the GPU.  GDF ``B_LR`` AO-pair/``B_vbar`` and
    ``M_LR``/``M_G0`` contractions are also device resident.  Screened
    ``B_SR`` and ``M_SR`` use explicit C/OpenMP direct contractions on the
    host; local part 2 and nonlocal GTH use exact analytic image-resolved
    Gaussian integrals on the host without AO derivative matrices or a PP
    reciprocal mesh.  Every such boundary is reported by name.

    The XC-only multigrid term differentiates the active pair/task hierarchy
    analytically, including r2SCAN ``tau``.  The remaining GDF reciprocal
    formula, screened-integral, GTH and Ewald primitives are analytic host
    boundaries, not Cell finite differences.  They keep
    ``gpu_native_complete=False`` but do not invalidate the production
    analytic result.

    Kwargs:
        reciprocal_block_size : int or None
            Shared bounded G-vector block for transparent GDF primitives.
        auxiliary_block_size : int or None
            Reverse-GDF auxiliary block. ``None`` uses the provider policy.
        overlap_gdf_short_long_range : bool
            Run the independent host ``B_SR``/``M_SR`` work concurrently
            with device ``B_LR``/``M_LR``.  False retains the sequential
            timing oracle without changing any term.
        pp_reciprocal_mesh, pp_block_size
            Optional fixed GTH pseudopotential representation controls.
        xc_image_block_size : int or None
            Optional MG lattice-image block. ``None`` selects the automatic
            density-size-aware bound.
        validate_central : bool
            Require complete analytic Hcore value parity with the active GPU
            source provider.  GDF operator-gauge parity is always required.
        strict_memory_preflight : bool
            Abort before the Z solve when the production dense live-set
            estimate exceeds the live provider budget.  CUDA allocator
            retention and bounded reciprocal/library transactions are
            reported exclusions rather than being mislabeled as an exact
            measured peak.
        memory_fraction, minimum_reserve_bytes
            Stress-wide dense-workspace budget.  GDF reciprocal blocks keep
            the provider's separate, usually smaller, block policy.

    Returns:
        dict
            CPU-compatible raw/symmetric/Voigt stress, pressure, virial,
            term ledger, Z diagnostics, and explicit execution boundaries.
    '''
    from dcforge.gpu.pbc.stress.rhf_nv import (
        make_z_state,
        nonvariational_correction,
    )
    from dcforge.gpu.pbc.stress.rhf_skeleton import skeleton_work

    check_stress_input(mf_grad)
    mf = mf_grad.base
    resolve_memory_policy(mf_grad, required_property='stress')
    profile = new_profile(bool(getattr(
        mf_grad, 'profile_stress_stages', False)))
    total_started = time.perf_counter()
    _progress(mf_grad, 'memory_preflight')
    preflight_stage = stage_start(profile)
    memory_preflight = stress_memory_preflight(
        mf_grad, strict=strict_memory_preflight,
        memory_fraction=memory_fraction,
        minimum_reserve_bytes=minimum_reserve_bytes)
    stage_stop(profile, 'memory_preflight', preflight_stage)
    state = make_z_state(mf_grad, profile=profile)
    work_nvar = nonvariational_correction(
        mf_grad, state=state,
        reciprocal_block_size=reciprocal_block_size,
        auxiliary_block_size=auxiliary_block_size,
        overlap_gdf_short_long_range=overlap_gdf_short_long_range,
        pp_reciprocal_mesh=pp_reciprocal_mesh,
        pp_block_size=pp_block_size,
        validate_central=validate_central, profile=profile)
    work_skeleton = skeleton_work(
        mf_grad, state, mf_grad.gdf_strain_response,
        mf_grad.hcore_strain_components,
        xc_image_block_size=xc_image_block_size, profile=profile)
    _progress(mf_grad, 'assembly')
    assembly_stage = stage_start(profile)
    work = np.asarray(work_skeleton) + np.asarray(work_nvar)
    volume = float(mf_grad.cell.vol)
    stress_raw = work / volume
    stress_tensor = stress_from_work(work, volume)
    stress_voigt = stress_tensor_to_voigt(stress_tensor)
    pressure = pressure_from_stress(stress_tensor)
    virial = virial_from_stress(stress_tensor, volume)

    work_terms = {
        'work_hcore': np.asarray(mf_grad.work_hcore),
        'work_j': np.asarray(mf_grad.work_j),
        'work_xc': np.asarray(mf_grad.work_xc),
        'work_ovlp': np.asarray(mf_grad.work_ovlp),
        'work_ewald': np.asarray(mf_grad.work_ewald),
        'work_fock': np.asarray(mf_grad.work_fock),
        'work_ovlp_energy': np.asarray(mf_grad.work_ovlp_energy),
        'work_ovlp_response': np.asarray(mf_grad.work_ovlp_response),
    }
    recomposed = sum(work_terms.values())
    recomposition_residual = float(np.max(np.abs(recomposed - work)))
    symmetry_residual = float(np.max(np.abs(stress_raw - stress_raw.T)))
    stress_terms = {
        name: stress_from_work(value, volume)
        for name, value in work_terms.items()
    }
    low_level_numerical = list(
        mf_grad.target_xc_strain_metadata[
            'low_level_numerical_components'])
    gdf_metadata = mf_grad.gaussian_df_strain_response
    provider_name = str(mf.with_df.backend_name).upper()
    owner_ledger = {
        'source_hf_jk': f'GPU_{provider_name}',
        'source_hf_response': f'GPU_{provider_name}',
        'source_hcore': f'{provider_name}_GTH_representation',
        'target_hcore': f'{provider_name}_GTH_representation',
        'target_j': provider_name,
        'target_xc': mf_grad.target_xc_strain_metadata[
            'target_xc_stress_owner'],
        'target_exact_exchange': None,
        'ion_ion': 'analytic_Ewald',
        'volume_division': 'dcforge_stress_assembler',
    }
    execution_boundary = {
        'device': (
            'source_RHF', 'target_Fock', 'Z_vector',
            f'{provider_name}_metric_solve',
            f'{provider_name}_reverse_adjoint',
            'overlap_contraction', 'kinetic_contraction',
            'multigrid_XC',
            *gdf_metadata['device_native_contracted_primitives'],
            *mf_grad.stress_native_gpu_contracted_primitives),
        'host_analytic_primitives': tuple(
            mf_grad.stress_host_primitive_boundary),
        'silent_cpu_fallback': False,
        'cpu_scf_called': False,
        'target_scf_called': False,
    }
    production_analytic = bool(
        gdf_metadata['production_analytic']
        and mf_grad.target_xc_strain_metadata['production_analytic']
        and mf_grad.hcore_strain_components['pseudopotential'][
            'production_analytic']
        and not low_level_numerical)
    report = {
        'production_analytic': production_analytic,
        'gpu_native_complete': False,
        'low_level_numerical_components': low_level_numerical,
        'work_raw_hartree': work,
        'work_terms_hartree': work_terms,
        'stress_raw': stress_raw,
        'stress_tensor': stress_tensor,
        'stress_voigt': stress_voigt,
        'pressure': pressure,
        'virial_tensor': virial,
        'stress_terms': stress_terms,
        'stress_owner_ledger': owner_ledger,
        'stress_memory_preflight': memory_preflight,
        'execution_boundary': execution_boundary,
        'stress_symmetry_residual': symmetry_residual,
        'stress_recomposition_residual': recomposition_residual / volume,
        'z_residual_max_abs': state['z_info']['residual_max_abs'],
        'z_solver': state['z_info']['solver'],
        'z_dimension': state['z_info']['dimension'],
        'z_refinement_iterations': state['z_info'][
            'refinement_iterations'],
        'z_wall_seconds': state['z_info']['wall_seconds'],
        'z_current_wall_seconds': (
            0. if state['response_state_reused']
            else state['z_info']['wall_seconds']),
        'z_response_profile': state['z_response_profile'],
        'response_state_reused': state['response_state_reused'],
        'response_state_owner': state['response_state_owner'],
        'response_state_token': state['response_state_token'],
        'z_solve_count': state['z_solve_count'],
        'target_fock_reused': state['target_fock_reused'],
        'target_scf_called': False,
        'memory_policy': mf_grad.memory_policy,
        'property_sequence': list(mf_grad.property_sequence),
        'resolved_memory_policy': mf_grad.resolved_memory_policy,
        'xc_strain_metadata': mf_grad.target_xc_strain_metadata,
        'gdf_strain_metadata': {
            'provider_backend': gdf_metadata.get(
                'provider_backend', str(mf.with_df.backend_name)),
            'storage': gdf_metadata['storage'],
            'production_analytic': gdf_metadata['production_analytic'],
            'device_adjoint': gdf_metadata['device_adjoint'],
            'device_metric_solve': gdf_metadata['device_metric_solve'],
            'device_native_contracted_primitives': (
                gdf_metadata['device_native_contracted_primitives']),
            'host_primitive_boundary': (
                gdf_metadata['host_primitive_boundary']),
            'silent_cpu_fallback': gdf_metadata['silent_cpu_fallback'],
            'resolved_representation': (
                gdf_metadata['resolved_representation']),
            'cderi_gauge_aligned_parity_max_abs': (
                gdf_metadata[
                    'cderi_gauge_aligned_parity_max_abs']),
            'provider_factor_reconstruction_max_abs': (
                gdf_metadata[
                    'provider_factor_reconstruction_max_abs']),
            'provider_factor_reconstruction_tolerance': (
                gdf_metadata[
                    'provider_factor_reconstruction_tolerance']),
            'central_context': gdf_metadata['central_context'],
            'host_to_device_bytes': gdf_metadata['host_to_device_bytes'],
            'device_to_host_bytes': gdf_metadata['device_to_host_bytes'],
            'source_device_to_host': gdf_metadata[
                'source_device_to_host'],
            'overlap_short_long_range': gdf_metadata[
                'overlap_short_long_range'],
            'component_wall_seconds': gdf_metadata[
                'component_wall_seconds'],
            'primitive_component_wall_seconds': {
                'source': gdf_metadata['source'][
                    'component_wall_seconds'],
                'metric': gdf_metadata['metric'][
                    'component_wall_seconds'],
            },
            'primitive_execution_schedule': {
                'source': gdf_metadata['source']['execution_schedule'],
                'metric': gdf_metadata['metric']['execution_schedule'],
            },
            'primitive_reciprocal_memory': {
                'source': {
                    'memory_plan': gdf_metadata['source']['long_range'][
                        'memory_plan'],
                    'block_execution': gdf_metadata['source']['long_range'][
                        'block_execution'],
                    'peak_explicit_block_bytes': gdf_metadata[
                        'source']['long_range']['peak_explicit_block_bytes'],
                },
                'metric': {
                    'memory_plan': gdf_metadata['metric']['long_range'][
                        'memory_plan'],
                    'block_execution': gdf_metadata['metric']['long_range'][
                        'block_execution'],
                    'peak_explicit_block_bytes': gdf_metadata[
                        'metric']['long_range']['peak_explicit_block_bytes'],
                },
            },
            'derivative_mode': 'reverse_adjoint_direct_voigt',
            'full_forward_reference_enabled': False,
        },
        'hcore_strain_metadata': {
            'value_parity_max_abs': (
                mf_grad.hcore_strain_value_parity_max_abs),
            'native_gpu_contracted_primitives': (
                mf_grad.stress_native_gpu_contracted_primitives),
            'host_matrix_primitives': (
                mf_grad.hcore_strain_components[
                    'pseudopotential']['host_matrix_primitives']),
            'pseudopotential_value_contraction_parity_max_abs': (
                mf_grad.hcore_strain_components['pseudopotential'][
                    'value_contraction_parity_max_abs']),
            'pseudopotential_local_part2': (
                mf_grad.hcore_strain_components[
                    'pseudopotential']['local_part2']),
            'pseudopotential_local_part1': (
                mf_grad.hcore_strain_components[
                    'pseudopotential']['local_part1']),
            'pseudopotential_nonlocal': (
                mf_grad.hcore_strain_components[
                    'pseudopotential']['nonlocal']),
        },
    }
    report['gaussian_df_strain_metadata'] = report['gdf_strain_metadata']
    stage_stop(profile, 'assembly', assembly_stage)
    report['stress_stage_profile'] = finish_profile(
        profile, total_started)
    mf_grad.work_raw = work
    mf_grad.stress_raw = stress_raw
    mf_grad.stress_tensor = stress_tensor
    mf_grad.stress_voigt = stress_voigt
    mf_grad.pressure = pressure
    mf_grad.virial_tensor = virial
    mf_grad.stress_terms = stress_terms
    mf_grad.stress_owner_ledger = owner_ledger
    mf_grad.stress_memory_preflight = memory_preflight
    mf_grad.stress_execution_boundary = execution_boundary
    mf_grad.stress_symmetry_residual = symmetry_residual
    mf_grad.stress_recomposition_residual = recomposition_residual / volume
    mf_grad.production_analytic = production_analytic
    mf_grad.gpu_native_complete = False
    mf_grad.low_level_numerical_components = low_level_numerical
    mf_grad.stress_stage_profile = report['stress_stage_profile']
    mf_grad.stress_report = report
    return report


def _check_production_report(mf_grad, report):
    '''Reject an incomplete or internally inconsistent public stress result.'''
    if not report['production_analytic']:
        raise RuntimeError(
            'GPU PBC stress did not complete an analytic production ledger')
    if report['low_level_numerical_components']:
        raise RuntimeError(
            'production stress contains low-level numerical components: '
            f"{report['low_level_numerical_components']}")
    if report['execution_boundary']['silent_cpu_fallback']:
        raise RuntimeError('production stress encountered a silent CPU fallback')
    arrays = (
        report['work_raw_hartree'], report['stress_raw'],
        report['stress_tensor'], report['stress_voigt'],
        report['virial_tensor'])
    if not all(np.all(np.isfinite(np.asarray(value))) for value in arrays):
        raise RuntimeError('production stress contains NaN or Inf')
    if report['z_residual_max_abs'] > float(mf_grad.cphf_residual_tol):
        raise RuntimeError(
            'production stress Z residual exceeds cphf_residual_tol: '
            f"{report['z_residual_max_abs']:.3e} > "
            f'{float(mf_grad.cphf_residual_tol):.3e}')
    if report['stress_recomposition_residual'] > 1.e-10:
        raise RuntimeError(
            'production stress term recomposition residual is too large: '
            f"{report['stress_recomposition_residual']:.3e}")
    if report['stress_symmetry_residual'] > 1.e-10:
        raise RuntimeError(
            'production stress antisymmetric residual is too large: '
            f"{report['stress_symmetry_residual']:.3e}")


def _progress(mf_grad, label):
    '''Emit a stress substage marker consumed by the user queue monitor.'''
    callback = getattr(mf_grad, 'progress_callback', None)
    if callback is not None:
        callback(f'stress/{label}')


def as_stress_scanner(mf_grad, **reference_options):
    '''Return a production analytic energy-and-stress scanner.

    Every call reconverges source RHF for the new Cell, rebuilds the target
    fixed-density state, and invalidates GDF/CDERI, multigrid tasks, Z-vector
    and the preceding stress term ledger.  The scanner returns the complete
    report rather than only its primary tensor so geometry workflows can
    retain owner and memory provenance at every step.
    '''
    if isinstance(mf_grad, RHFDFTStressScanner):
        if reference_options:
            mf_grad.stress_reference_options.update(reference_options)
        return mf_grad
    if not hasattr(mf_grad, 'base') or not isinstance(mf_grad.base, RHFDFT):
        raise TypeError('GPU PBC stress scanner requires RHFDFT Gradients')
    name = mf_grad.__class__.__name__ + '_StressScanner'
    scanner = RHFDFTStressScanner(mf_grad, reference_options)
    return lib.set_class(
        scanner, (RHFDFTStressScanner, mf_grad.__class__), name)


class RHFDFTStressScanner(lib.GradScanner):
    '''Stateful scanner for target energy and analytic DC-DFT stress.'''

    def __init__(self, mf_grad, reference_options):
        lib.GradScanner.__init__(self, mf_grad)
        self.stress_reference_options = dict(reference_options)

    def __call__(self, cell_or_geom, stress_options=None, **kwargs):
        '''Evaluate one Cell and return ``(energy, stress_report)``.

        ``kwargs`` are source-RHF scanner options.  Development stress options
        are supplied separately through ``stress_options`` so a block-control
        keyword cannot accidentally reach the SCF kernel.
        '''
        if isinstance(cell_or_geom, gto.Cell):
            cell = cell_or_geom
        else:
            cell = self.cell.set_geom_(cell_or_geom, inplace=False)
        self.reset(cell)
        energy = self.base(cell, **kwargs)
        options = dict(self.stress_reference_options)
        if stress_options is not None:
            options.update(dict(stress_options))
        report = reference_kernel(self, **options)
        _check_production_report(self, report)
        return energy, report

    def as_scanner(self):
        '''Return this already stateful stress scanner.'''
        return self


__all__ = [
    'RHFDFTStressScanner',
    'as_stress_scanner',
    'check_stress_input',
    'kernel',
    'reference_kernel',
]
