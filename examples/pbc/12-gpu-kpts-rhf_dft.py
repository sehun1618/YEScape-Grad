'''GPU full-BZ sampled-k semilocal-DFA@KRHF calculations.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import KRHFDFT

cell = gto.Cell()
cell.atom = 'H 0 0 0; H 0 0 1.4'
cell.a = np.eye(3) * 6.0
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pbe'
cell.precision = 1e-8
cell.ke_cutoff = 80.0
cell.verbose = 0
cell.build()

kpts = cell.make_kpts([2, 1, 1], wrap_around=True)


def run_gradient(mf):
    '''Run one source KRHF, fixed-density target and analytic gradient.'''
    mf.set(conv_tol=1e-10, conv_tol_grad=1e-8, max_cycle=80).run()
    mf_grad = mf.nuc_grad_method().set(
        zvec_method='auto',
        cphf_conv_tol=1e-9,
        cphf_residual_tol=1e-8,
        cphf_max_cycle=200,
    )
    gradient = mf_grad.kernel()
    return mf_grad, gradient


def print_result(label, mf, mf_grad, gradient):
    '''Print the physical result and the sampled-k response diagnostics.'''
    print(f'\n[{label}]')
    print(f'Provider = {type(mf.with_df).__name__}')
    print(f'k-point weights = {mf.kpts_weights}')
    print(f'Source KRHF energy = {mf.e_hf:.15f} Ha')
    print(f'Target {mf.xc}@KRHF energy = {mf.e_dft:.15f} Ha')
    print('Gradient dE/dR [Ha/Bohr]')
    print(gradient)
    print('Force -dE/dR [Ha/Bohr]')
    print(-gradient)
    print(f'Z-vector route = {mf_grad.zvec_method_used}')
    print(f'Z physical residual = {mf_grad.z_residual_max_abs:.9e}')
    print(f'Term recomposition = '
          f'{mf_grad.term_recomposition_max_abs:.9e}')
    print(f'Provider response = {mf_grad.provider_response_backend}')
    print(f'Host fallback = {mf_grad.scientific_host_fallback}')
    print(f'Host fallback components = {mf_grad.host_fallback_components}')


#
# Case 1: GPU FFTDF R2SCAN@KRHF with k-resolved tau
#
# The explicit full-BZ k-point array and equal weights are shared by source
# KRHF, fixed-density target KRKS and every response term.  The target KRKS
# kernel is never run.
#
mf_fft = KRHFDFT(
    cell, kpts=kpts, xc='R2SCAN', kpts_wrap_around=True)
mf_fft_grad, gradient_fft = run_gradient(mf_fft)
print_result('GPU full-BZ FFTDF', mf_fft, mf_fft_grad, gradient_fft)


#
# Case 2: GPU GDF PBE@KRHF
#
# GDF uses GPU4PySCF's native sampled-k density-contracted derivative.  The
# provider records the version-pinned non-Gamma GTH Hcore host primitive and
# any conditional device-layout copy instead of hiding these boundaries.
#
mf_gdf = KRHFDFT(
    cell, kpts=kpts, xc='PBE', kpts_wrap_around=True,
).density_fit(auxbasis='gth-szv')
mf_gdf_grad, gradient_gdf = run_gradient(mf_gdf)
print_result('GPU full-BZ GDF', mf_gdf, mf_gdf_grad, gradient_gdf)


#
# Case 3: GPU RSDF PBE@KRHF
#
# Source SCF, central RSDF J/K, target energy, CPHF and Z-vector stay on the
# GPU.  GPU4PySCF 1.8 has no derivative endpoint that preserves the separate
# source and metric range-separation settings, so the contracted CPU RSDF
# derivative oracle is an explicit, measured host boundary in this release.
#
mf_rsdf = KRHFDFT(
    cell, kpts=kpts, xc='PBE', kpts_wrap_around=True,
).rs_density_fit(auxbasis='gth-szv')
mf_rsdf.with_df.omega = .3
mf_rsdf.with_df.mesh_compact = np.array([9, 9, 9])
mf_rsdf.with_df.omega_j2c = .4
mf_rsdf.with_df.mesh_j2c = np.array([11, 11, 11])
mf_rsdf_grad, gradient_rsdf = run_gradient(mf_rsdf)
print_result('GPU full-BZ RSDF', mf_rsdf, mf_rsdf_grad, gradient_rsdf)


#
# Case 4: GDF source/response plus sampled-k R2SCAN multigrid J+XC
#
# with_j=True keeps source KRHF/CPHF on GDF but moves fixed-density target J,
# XC, and their explicit derivative to one multigrid reciprocal potential.
# R2SCAN adds the k-resolved MGGA tau channel to the XC evaluation.
#
mf_mg = KRHFDFT(
    cell, kpts=kpts, xc='R2SCAN', kpts_wrap_around=True,
).density_fit(auxbasis='gth-szv').multigrid_numint(with_j=True)
mf_mg_grad, gradient_mg = run_gradient(mf_mg)
print_result(
    'GPU full-BZ GDF source + target multigrid J+XC',
    mf_mg, mf_mg_grad, gradient_mg)
print(f'XC derivative backend = {mf_mg_grad.xc_deriv_backend}')
