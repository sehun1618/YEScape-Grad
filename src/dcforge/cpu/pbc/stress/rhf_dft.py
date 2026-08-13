'''Public stress gate for Gamma-point CPU ``E_DFA[P_RHF]`` workflows.'''

import numpy as np
from pyscf.pbc.dft import multigrid

from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.energy.rhf_dft import RHFDFT, _target_xc_name
from dcforge.cpu.pbc.stress.strain import (
    pressure_from_stress,
    stress_from_work,
    stress_tensor_to_voigt,
    virial_from_stress,
)


def check_stress_input(mf_grad):
    '''Validate the first CPU PBC DC-DFT stress ownership contract.

    Validation deliberately precedes every source-SCF or stress primitive so
    unsupported FFTDF/RSDF, target-MG-J, non-PBE and non-Gamma requests fail
    closed.  Passing this gate does not yet imply that the analytic GDF stress
    implementation is complete.
    '''
    mf = mf_grad.base
    cell = mf.cell
    if not isinstance(mf, RHFDFT):
        raise TypeError('CPU PBC stress requires dcforge CPU RHFDFT')
    if cell.dimension != 3:
        raise NotImplementedError('PBC stress requires a three-dimensional Cell')
    if cell.spin != 0 or cell.nelectron % 2:
        raise NotImplementedError('PBC stress requires a closed-shell Cell')
    if not cell._pseudo or cell._ecp:
        raise NotImplementedError('PBC stress requires a GTH pseudopotential')
    if not isinstance(mf.with_df, GDF):
        raise NotImplementedError(
            'the first PBC stress route requires a GDF source/target provider')
    if mf.with_df is not mf.mf_hf.with_df:
        raise ValueError('source RHF and target DFA must share one GDF provider')
    if not isinstance(mf._numint, multigrid.MultiGridNumInt2):
        raise NotImplementedError(
            'the first PBC stress route requires target XC multigrid')
    if mf.multigrid_with_j or bool(mf._numint.xc_with_j):
        raise NotImplementedError(
            'the first PBC stress route requires XC-only multigrid '
            '(with_j=False)')
    if _target_xc_name(mf) != 'PBE':
        raise NotImplementedError(
            'the first CPU PBC stress correctness release supports PBE only')
    kpt = np.asarray(getattr(mf.mf_hf, 'kpt', np.zeros(3)))
    if kpt.shape != (3,) or np.max(np.abs(kpt)) > 1.e-12:
        raise NotImplementedError('the first PBC stress route requires Gamma point')


def kernel(mf_grad):
    '''Return analytic DC-DFT stress when the full GDF ledger is available.'''
    check_stress_input(mf_grad)
    raise NotImplementedError(
        'CPU GDF + XC-only multigrid analytic DC-DFT stress is not complete; '
        'use relaxed_energy_finite_difference as the P1 validation oracle')


def reference_kernel(
        mf_grad, xc_step=1.e-4, reciprocal_block_size=None,
        pp_reciprocal_mesh=None, pp_block_size=None,
        validate_central=False):
    r'''Return the semi-analytic CPU PBE stress development ledger.

    This function joins the analytic one-electron, GTH PP, GDF, Ewald and
    one-Z-vector terms with the matched pair-multigrid XC finite-difference
    oracle.  It is intentionally separate from :func:`kernel`: the presence
    of the numerical XC component keeps ``production_analytic=False`` and
    must not open the public ``get_stress()`` route.

    Kwargs:
        xc_step : float
            Dimensionless Cell displacement used only by the pair-MG XC
            finite-difference oracle.
        reciprocal_block_size : int or None
            Bounded GDF source/metric strain block.
        pp_reciprocal_mesh, pp_block_size
            Optional fixed GTH PP reciprocal representation controls.
        validate_central : bool
            Enable central Hcore and CDERI provider-parity gates.

    Returns:
        dict
            Raw work, physical stress, term and owner ledgers, Z diagnostics
            and explicit numerical-component metadata.
    '''
    from dcforge.cpu.pbc.stress.rhf_nv import (
        make_z_state,
        nonvariational_correction,
    )
    from dcforge.cpu.pbc.stress.rhf_skeleton import skeleton_work

    check_stress_input(mf_grad)
    state = make_z_state(mf_grad)
    work_nvar = nonvariational_correction(
        mf_grad, state=state,
        reciprocal_block_size=reciprocal_block_size,
        pp_reciprocal_mesh=pp_reciprocal_mesh,
        pp_block_size=pp_block_size,
        validate_central=validate_central)
    work_skeleton = skeleton_work(
        mf_grad, state, mf_grad.gdf_strain_response,
        hcore_components=mf_grad.hcore_strain_components,
        xc_step=xc_step)
    work = np.asarray(work_skeleton + work_nvar)
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
    owner_ledger = {
        'source_hf_jk': 'GDF',
        'source_hf_response': 'GDF',
        'source_hcore': 'GDF_GTH_representation',
        'target_hcore': 'GDF_GTH_representation',
        'target_j': 'GDF',
        'target_xc': mf_grad.target_xc_strain_metadata[
            'target_xc_stress_owner'],
        'target_exact_exchange': None,
        'ion_ion': 'analytic_Ewald',
        'volume_division': 'dcforge_stress_assembler',
    }
    report = {
        'production_analytic': False,
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
        'stress_symmetry_residual': symmetry_residual,
        'stress_recomposition_residual': recomposition_residual / volume,
        'z_residual_max_abs': state['z_info']['residual_max_abs'],
        'z_solver': state['z_info']['solver'],
        'target_fock_reused': state['target_fock_reused'],
        'target_scf_called': False,
        'xc_strain_metadata': mf_grad.target_xc_strain_metadata,
        'gdf_strain_metadata': {
            'storage': mf_grad.gdf_strain_response['storage'],
            'reciprocal_block_size': (
                mf_grad.gdf_strain_response['reciprocal_block_size']),
            'production_analytic': (
                mf_grad.gdf_strain_response['production_analytic']),
            'cderi_value_parity_max_abs': (
                mf_grad.gdf_strain_response[
                    'cderi_value_parity_max_abs']),
        },
        'hcore_strain_metadata': {
            'value_parity_max_abs': (
                mf_grad.hcore_strain_value_parity_max_abs),
            'production_analytic': True,
        },
    }
    mf_grad.work_raw = work
    mf_grad.stress_raw = stress_raw
    mf_grad.stress_tensor = stress_tensor
    mf_grad.stress_voigt = stress_voigt
    mf_grad.pressure = pressure
    mf_grad.virial_tensor = virial
    mf_grad.stress_terms = stress_terms
    mf_grad.stress_owner_ledger = owner_ledger
    mf_grad.stress_symmetry_residual = symmetry_residual
    mf_grad.stress_recomposition_residual = recomposition_residual / volume
    mf_grad.production_analytic = False
    mf_grad.low_level_numerical_components = low_level_numerical
    return report


__all__ = ['check_stress_input', 'kernel', 'reference_kernel']
