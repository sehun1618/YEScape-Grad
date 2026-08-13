'''GPU GDF/RSDF + XC-only multigrid DC-DFT stress cases.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import RHFDFT


def make_cell(bond=1.4):
    '''Return one small real-Gamma GTH Cell in Bohr.'''
    cell = gto.Cell()
    cell.atom = f'H 0 0 0; H 0 0 {bond}'
    cell.a = np.asarray((
        (6.0, .2, .0),
        (.0, 6.3, .1),
        (.1, .0, 6.5),
    ))
    cell.unit = 'Bohr'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-hf'
    cell.ke_cutoff = 100
    cell.precision = 1.e-8
    cell.verbose = 0
    return cell.build()


# Case 1: one fixed-geometry DC-r2SCAN stress evaluation.
#
# Source RHF J/K and target Hartree J are owned by GDF.  Multigrid owns XC
# only; ``with_j=True`` is a different target Hamiltonian and is rejected by
# this stress route.  Named GDF/PP/Ewald host primitives are analytic rather
# than Cell finite differences, so the public tensor API is available even
# though the report keeps ``gpu_native_complete=False``.
cell = make_cell()
mf = RHFDFT(cell, xc='R2SCAN').density_fit(auxbasis='weigend')
mf = mf.multigrid_numint(with_j=False)
mf.conv_tol = 1.e-10
mf.run()

mf_grad = mf.nuc_grad_method()
mf_grad.cphf_conv_tol = 1.e-9
mf_grad.cphf_residual_tol = 1.e-8
mf_grad.profile_stress_stages = True
# The two terms in each pair are mathematically independent.  ``True`` runs
# host B_SR/M_SR while the matching device B_LR/M_LR work is active.  Set it
# to False only to obtain the sequential timing oracle; stress is unchanged.
mf_grad.stress_reference_options = {
    'overlap_gdf_short_long_range': True,
}
stress = mf_grad.get_stress()
report = mf_grad.stress_report

print('stress status = production analytic hybrid')
print('production analytic =', report['production_analytic'])
print('GPU native complete =', report['gpu_native_complete'])
print('owner ledger =', report['stress_owner_ledger'])
print('resolved GDF =', report['gdf_strain_metadata']['resolved_representation'])
print('GDF central context =', report['gdf_strain_metadata']['central_context'])
print('GDF host boundary =',
      report['gdf_strain_metadata']['host_primitive_boundary'])
print('GDF packed host staging =',
      report['gdf_strain_metadata']['source_device_to_host'])
print('MG mesh =', report['xc_strain_metadata']['mesh'])
print('Z residual =', report['z_residual_max_abs'])
print('stress [Ha/Bohr^3] =\n', stress)
print('Voigt [xx,yy,zz,yz,xz,xy] =', report['stress_voigt'])
print('pressure [Ha/Bohr^3] =', report['pressure'])
print('virial [Ha] =\n', report['virial_tensor'])
print('memory preflight =', report['stress_memory_preflight'])
print('stage profile =', report['stress_stage_profile'])


# Case 2: RSDF gradient followed by stress with one shared Z-vector.
#
# RSDF keeps separate source ``(omega, mesh_compact)`` and metric
# ``(omega_j2c, mesh_j2c)`` representations.  They are resolved by PySCF and
# differentiated independently; the user does not copy GDF omega or mesh
# values into them.  Calling both properties on the same derivative object
# reuses the perturbation-independent target Fock and Z-vector state.
mf_rs = RHFDFT(cell, xc='R2SCAN').rs_density_fit(auxbasis='weigend')
mf_rs = mf_rs.multigrid_numint(with_j=False)
mf_rs.conv_tol = 1.e-10
mf_rs.run()

mf_rs_grad = mf_rs.nuc_grad_method().set(
    cphf_conv_tol=1.e-9,
    cphf_residual_tol=1.e-8,
)
gradient_rs = mf_rs_grad.kernel()
stress_rs = mf_rs_grad.get_stress()
report_rs = mf_rs_grad.stress_report

print('RSDF gradient [Ha/Bohr] =\n', gradient_rs)
print('RSDF stress [Ha/Bohr^3] =\n', stress_rs)
print('RSDF source/metric representation =',
      report_rs['gaussian_df_strain_metadata']['resolved_representation'])
print('RSDF M_SR screening =',
      report_rs['gaussian_df_strain_metadata']['metric'][
          'short_range_screening'])
print('response state reused =', report_rs['response_state_reused'])
print('Z solve count =', report_rs['z_solve_count'])


# Case 3: a GDF Cell/geometry scanner.
#
# Each call reconverges source RHF and discards the preceding GDF/CDERI,
# multigrid task, target-Fock, Z-vector, and stress-term caches.  Stress is a
# homogeneous Cell derivative; the bond displacement here only demonstrates
# safe same-process reentry and is not a finite-strain stress check.
scanner = mf_grad.as_stress_scanner(
    overlap_gdf_short_long_range=True)
energy0, stress0 = scanner(cell)
energy1, stress1 = scanner(make_cell(bond=1.41))

print('scanner target energies [Ha] =', energy0, energy1)
print('scanner pressure [Ha/Bohr^3] =',
      stress0['pressure'], stress1['pressure'])
